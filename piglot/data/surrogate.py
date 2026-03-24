"""Module for surrogate GP models."""
from typing import Literal, Optional, List, Union, Any, TypeVar
import warnings
import torch
from torch import Tensor
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models.exact_gp import ExactGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import FantasizeMixin
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.models.transforms.outcome import Standardize, OutcomeTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from piglot.data.dataset import ObjectiveDataset
from piglot.data.transforms import Standardiser, PCA, ChainTransform
from piglot.utils.readable import ReadableModel


T = TypeVar('T', bound='SurrogateSettings')


class SurrogateSettings(ReadableModel):
    """Options for surrogate model construction."""

    noise: Literal['infer', 'fixed'] = 'infer'
    noise_model: Literal['homoscedastic', 'heteroscedastic'] = 'homoscedastic'
    pca_variance: float = 1e-6
    std_tol: float = 1e-6
    min_variance: float = 1e-6
    hetero_num_var_samples: int = 512
    hetero_max_iter: int = 128
    hetero_tol_mean: float = 1e-06
    hetero_tol_var: float = 1e-06


def fit_mll_pytorch_loop(mll: ExactMarginalLogLikelihood, n_iters: int = 100) -> None:
    """Fit a GP model using a PyTorch optimisation loop.

    Parameters
    ----------
    mll : ExactMarginalLogLikelihood
        Marginal log-likelihood to optimise.
    n_iters : int, optional
        Number of iterations to optimise for, by default 100
    """
    mll.train()
    mll.model.likelihood.train()
    optimizer = torch.optim.Adam(mll.model.parameters(), lr=0.1)
    for _ in range(n_iters):
        optimizer.zero_grad()
        output = mll.model(mll.model.train_inputs[0])
        loss = -torch.mean(mll(output, mll.model.train_targets))
        loss.backward()
        optimizer.step()
    mll.model.eval()
    mll.model.likelihood.eval()


class SingleTaskGPWithNoise(SingleTaskGP):
    """Wrapper for a SingleTaskGP model with a noise model."""

    def noise_prediction(self, points: Tensor) -> Tensor:
        """Predict the noise level at the provided points.

        Parameters
        ----------
        points : Tensor
            A `batch_shape x q x d` tensor of points at which to predict the noise level.

        Returns
        -------
        Tensor
            A `batch_shape x q x m` tensor of predicted noise levels.
        """
        noise_level = (
            self.likelihood.noise.mean(dim=-1, keepdim=True)
            if isinstance(self.likelihood, FixedNoiseGaussianLikelihood)
            else self.likelihood.noise_covar.noise
        )
        noise_shape = points.shape[:-1] + noise_level.shape
        return noise_level.expand(noise_shape)


class PseudoHeteroscedasticSingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    """A pseudo-heteroscedastic GP model."""

    def __init__(  # pylint: disable=W0231
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        # Build the noise model and use it to estimate true noise levels
        noise_model = SingleTaskGP(
            train_X=train_X,
            train_Y=torch.log(train_Yvar.clamp_min(1e-6)),
            outcome_transform=Standardize(m=train_Yvar.shape[-1]),
            input_transform=input_transform,
        )
        mll = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        fit_gpytorch_mll(mll)
        with torch.no_grad():
            noise = torch.exp(noise_model.posterior(train_X).mean)
        # Hacky way to initialise the SingleTaskGP
        SingleTaskGP.__init__(  # pylint: disable=W0233
            self,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=noise,
            input_transform=input_transform,
            outcome_transform=None,
        )
        # Disable training on the noise model
        self.noise_model = noise_model
        for param in self.noise_model.parameters():
            param.requires_grad = False
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)

    def noise_prediction(self, points: Tensor) -> Tensor:
        """Predict the noise level at the provided points.

        Parameters
        ----------
        points : Tensor
            A `batch_shape x q x d` tensor of points at which to predict the noise level.

        Returns
        -------
        Tensor
            A `batch_shape x q x m` tensor of predicted noise levels.
        """
        return torch.exp(self.noise_model.posterior(points).mean)

    def forward(self, x: Tensor) -> MultivariateNormal:  # pylint: disable=W0221
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> Union[GPyTorchPosterior, TransformedPosterior]:
        # Inject heteroscedastic noise into the model
        if isinstance(observation_noise, bool) and observation_noise:
            observation_noise = self.noise_prediction(X)
        return super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "PseudoHeteroscedasticSingleTaskGP":
        r"""Fantasize the model.

        Args:
            X: A `batch_shape x q x d` tensor of points at which to fantasize.
            sampler: The sampler used to fantasize the model.
            observation_noise: A `batch_shape x q x m` tensor of observation noise
                variances. If `None`, the model's noise model is used to generate
                noise samples.

        Returns:
            A fantasized model.
        """
        # Inject heteroscedastic noise into the model
        if observation_noise is None:
            observation_noise = self.noise_prediction(X)
        return super().fantasize(
            X=X,
            sampler=sampler,
            observation_noise=observation_noise,
            **kwargs,
        )


def fit_most_likely_heteroscedastic_gp(
    train_X: Tensor,
    train_Y: Tensor,
    num_var_samples: int = 512,
    max_iter: int = 128,
    tol_mean: float = 1e-06,
    tol_var: float = 1e-06,
) -> PseudoHeteroscedasticSingleTaskGP:
    r"""Fit the Most Likely Heteroscedastic GP.

    The original algorithm is described in
    http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
            features.
        train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
            training observations.
        num_var_samples: Number of samples to draw from posterior when estimating noise.
        max_iter: Maximum number of iterations used when fitting the model.
        tol_mean: The tolerance for the mean check.
        tol_std: The tolerance for the var check.
    Returns:
        PseudoHeteroscedasticSingleTaskGP Model fit using the "most-likely" procedure.
    """

    # fit initial homoskedastic model used to estimate noise levels
    homo_model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=ScaleKernel(RBFKernel()),
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=None,
    )
    homo_mll = ExactMarginalLogLikelihood(homo_model.likelihood, homo_model)
    fit_gpytorch_mll(homo_mll)

    # get estimates of noise
    with torch.no_grad():
        homo_posterior = homo_model.posterior(train_X)
        homo_predictive_posterior = homo_model.posterior(train_X, observation_noise=True)
    sampler = SobolQMCNormalSampler(torch.Size([num_var_samples]), seed=0)
    predictive_samples = sampler(homo_predictive_posterior)
    observed_var = 0.5 * torch.square(predictive_samples - train_Y).mean(dim=0).detach()

    # save mean and variance to check if they change later
    saved_mean = homo_posterior.mean
    saved_var = homo_posterior.variance

    for i in range(max_iter):

        # now train hetero model using computed noise
        hetero_model = PseudoHeteroscedasticSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=observed_var,
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=None,
        )
        hetero_mll = ExactMarginalLogLikelihood(hetero_model.likelihood, hetero_model)
        try:
            fit_gpytorch_mll(hetero_mll)
        except Exception as e:
            warnings.warn(f"Fitting failed on iteration {i}.", e)
            raise e

        with torch.no_grad():
            hetero_posterior = hetero_model.posterior(train_X)
            hetero_predictive_posterior = hetero_model.posterior(train_X, observation_noise=True)
            new_mean = hetero_posterior.mean
            new_var = hetero_posterior.variance

        mean_error = torch.square(saved_mean - new_mean).mean()
        var_error = torch.square(saved_var - new_var).mean()

        if mean_error < tol_mean and var_error < tol_var:
            return hetero_model

        saved_mean = new_mean
        saved_var = new_var

        # get new noise estimate
        sampler = SobolQMCNormalSampler(torch.Size([num_var_samples]), seed=i + 1)
        predictive_samples = sampler(hetero_predictive_posterior)
        observed_var = 0.5 * torch.square(predictive_samples - train_Y).mean(dim=0)

    warnings.warn(
        f"Did not reach convergence after {max_iter} iterations. Returning the current model."
    )
    return hetero_model


class ObjectiveModel:
    """Surrogate model for an objective dataset using Gaussian processes."""

    def __init__(
        self,
        dataset: ObjectiveDataset,
        settings: SurrogateSettings,
    ) -> None:
        self.dataset = dataset
        self.settings = settings
        self.raw_dataset = dataset.export_raw()

        # Build the output transform: chain PCA + standardisation on composite objectives
        if dataset.objective.is_composite():
            self.output_transform = ChainTransform(
                self.raw_dataset.outputs,
                self.raw_dataset.covariances,
                [
                    (PCA, {'variance': settings.pca_variance, 'std_tol': settings.std_tol}),
                    (Standardiser, {'std_tol': settings.std_tol}),
                ]
            )
        else:
            self.output_transform = Standardiser(
                self.raw_dataset.outputs,
                self.raw_dataset.covariances,
                std_tol=settings.std_tol,
            )

        # Build the dataset for the GP model
        self.inputs = self.raw_dataset.inputs
        self.outputs, self.output_variances = self.output_transform.transform(
            self.raw_dataset.outputs, self.raw_dataset.covariances
        )

        # Sanitise variances: these are only needed if we are using a fixed noise model
        # Regardless, we need to diagonalise and clamp them to prevent GPyTorch warnings
        if settings.noise == 'fixed' and self.output_variances is None:
            raise ValueError('Fixed noise model requires provided output variances')
        if settings.noise == 'infer' and self.output_variances is not None:
            warnings.warn('Ignoring provided output variances when inferring noise levels')
            self.output_variances = None
        if self.output_variances is not None:
            self.output_variances = torch.clamp_min(
                torch.diagonal(self.output_variances, dim1=-2, dim2=-1), settings.min_variance
            )

        # Build and fit the GP model
        if settings.noise == 'infer' and settings.noise_model == 'heteroscedastic':
            self.gp = fit_most_likely_heteroscedastic_gp(
                self.inputs,
                self.outputs,
                num_var_samples=settings.hetero_num_var_samples,
                max_iter=settings.hetero_max_iter,
                tol_mean=settings.hetero_tol_mean,
                tol_var=settings.hetero_tol_var,
            )
        else:
            model_cls = (
                SingleTaskGPWithNoise if settings.noise_model == 'homoscedastic'
                else PseudoHeteroscedasticSingleTaskGP
            )
            self.gp = model_cls(
                self.inputs,
                self.outputs,
                train_Yvar=self.output_variances,
                input_transform=Normalize(d=self.inputs.shape[-1]),
                outcome_transform=None,
            )
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            try:
                fit_gpytorch_mll(mll)
            except ModelFittingError:
                warnings.warn('Optimisation of the MLL failed, falling back to PyTorch optimiser')
                fit_mll_pytorch_loop(mll)

    def _sample(
        self,
        input_data: torch.Tensor,
        sampler: Optional[MCSampler] = None,
        sample_shape: Optional[torch.Size] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Draw samples from the model at the provided input locations.

        Parameters
        ----------
        input_data : torch.Tensor
            A `(batch_shape) x q x d` tensor of input locations.
        sampler : Optional[MCSampler], optional
            Sampler to use when drawing samples. If `None`, a SobolQMCNormalSampler is used.
        sample_shape : Optional[torch.Size], optional
            Shape of the samples to draw. If `sampler` is provided, this is ignored.
        seed : Optional[int], optional
            Random seed to use when creating the default sampler. Only used if `sampler` is `None`.

        Returns
        -------
        torch.Tensor
            A `(sample_shape) x (batch_shape) x q x o` tensor of model samples.
        """
        # Sanitise sampler options
        if sampler is not None and sample_shape is not None:
            raise ValueError('Both sampler and sample_shape were provided')
        if sampler is None and sample_shape is None:
            raise ValueError('Either sampler or sample_shape must be provided')

        # Create default sampler if needed
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape, seed=seed)

        # Draw samples from the GP posterior (reduced latent space)
        posterior = self.gp.posterior(input_data)
        samples = sampler(posterior)

        # Untransform the samples back to the original space
        return self.output_transform.untransform(samples)

    def latent_samples(
        self,
        input_data: torch.Tensor,
        sampler: Optional[MCSampler] = None,
        sample_shape: Optional[torch.Size] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Draw samples from the latent space at the provided input locations.

        Only available for composite objectives.

        Parameters
        ----------
        input_data : torch.Tensor
            A `(batch_shape) x q x d` tensor of input locations.
        sampler : Optional[MCSampler], optional
            Sampler to use when drawing samples. If `None`, a SobolQMCNormalSampler is used.
        sample_shape : Optional[torch.Size], optional
            Shape of the samples to draw. If `sampler` is provided, this is ignored.
        seed : Optional[int], optional
            Random seed to use when creating the default sampler. Only used if `sampler` is `None`.

        Returns
        -------
        torch.Tensor
            A `(sample_shape) x (batch_shape) x q x o` tensor of model samples.
        """
        # Sanitise composite objective
        if not self.dataset.objective.is_composite():
            raise ValueError('Latent samples are only available for composite objectives')
        return self._sample(input_data, sampler=sampler, sample_shape=sample_shape, seed=seed)

    def objective_samples(
        self,
        input_data: torch.Tensor,
        sampler: Optional[MCSampler] = None,
        sample_shape: Optional[torch.Size] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Draw samples from the objective function model at the provided input locations.

        Parameters
        ----------
        input_data : torch.Tensor
            A `(batch_shape) x q x d` tensor of input locations.
        sampler : Optional[MCSampler], optional
            Sampler to use when drawing samples. If `None`, a SobolQMCNormalSampler is used.
        sample_shape : Optional[torch.Size], optional
            Shape of the samples to draw. If `sampler` is provided, this is ignored.
        seed : Optional[int], optional
            Random seed to use when creating the default sampler. Only used if `sampler` is `None`.

        Returns
        -------
        torch.Tensor
            A `(sample_shape) x (batch_shape) x q x o` tensor of model samples.
        """
        samples = self._sample(input_data, sampler=sampler, sample_shape=sample_shape, seed=seed)

        # If composite objective, map samples back to objective space
        if self.dataset.objective.is_composite():
            samples = self.dataset.objective.composition(samples, input_data)
        return samples

    def composition(self, samples: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the composition function of a composite objective.

        This also converts the minimisation problem to maximisation by negating the output.

        Parameters
        ----------
        samples : torch.Tensor
            A `(batch_shape) x q x o_` tensor of model samples.
        X : torch.Tensor
            A `(batch_shape) x q x d` tensor of parameters corresponding to the input locations

        Returns
        -------
        torch.Tensor
            A `(batch_shape) x q x o` tensor of composition function values.
        """
        # For non-composite objectives, just negate the samples to minimise
        if not self.dataset.objective.is_composite():
            return -samples.squeeze(-1)

        # Untransform the samples back to the original space and apply the composition function
        samples = self.output_transform.untransform(samples)
        return -self.dataset.objective.composition(samples, X)
