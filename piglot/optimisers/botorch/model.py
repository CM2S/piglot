"""Module for GP models derived from BoTorch."""
from __future__ import annotations
from typing import Optional, List, Union, Any
import warnings
import torch
from torch import Tensor
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models.exact_gp import ExactGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
import botorch
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models import SingleTaskGP
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.models.transforms.outcome import Standardize, OutcomeTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from piglot.optimisers.botorch.dataset import BayesDataset


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

    def noise_prediction(self, X: Tensor) -> Tensor:
        """Predict the noise level at the provided points.

        Parameters
        ----------
        X : Tensor
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
        noise_shape = X.shape[:-1] + noise_level.shape
        return noise_level.expand(noise_shape)


class PseudoHeteroskedasticSingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    """A pseudo-heteroskedastic GP model."""

    def __init__(
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
        # This is hacky -- this class used to inherit from SingleTaskGP, but it
        # shouldn't so this is a quick fix to enable getting rid of that
        # inheritance
        SingleTaskGP.__init__(
            # pyre-fixme[6]: Incompatible parameter type
            self,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=noise,
            input_transform=input_transform,
        )
        # Disable training on the noise model
        self.noise_model = noise_model
        for param in self.noise_model.parameters():
            param.requires_grad = False
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)

    def noise_prediction(self, X: Tensor) -> Tensor:
        """Predict the noise level at the provided points.

        Parameters
        ----------
        X : Tensor
            A `batch_shape x q x d` tensor of points at which to predict the noise level.

        Returns
        -------
        Tensor
            A `batch_shape x q x m` tensor of predicted noise levels.
        """
        return torch.exp(self.noise_model.posterior(X).mean)

    def forward(self, x: Tensor) -> MultivariateNormal:
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
        # Inject heteroskedastic noise into the model
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
    ) -> PseudoHeteroskedasticSingleTaskGP:
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
        # Inject heteroskedastic noise into the model
        if observation_noise is None:
            observation_noise = self.noise_prediction(X)
        return super().fantasize(
            X=X,
            sampler=sampler,
            observation_noise=observation_noise,
            **kwargs,
        )


def fit_most_likely_heteroskedastic_gp(
    train_X: Tensor,
    train_Y: Tensor,
    num_var_samples: int = 512,
    max_iter: int = 128,
    tol_mean: float = 1e-06,
    tol_var: float = 1e-06,
) -> PseudoHeteroskedasticSingleTaskGP:
    r"""Fit the Most Likely Heteroskedastic GP.

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
        PseudoHeteroskedasticSingleTaskGP Model fit using the "most-likely" procedure.
    """

    # fit initial homoskedastic model used to estimate noise levels
    homo_model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=ScaleKernel(RBFKernel()),
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=train_Y.shape[-1]),
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
        hetero_model = PseudoHeteroskedasticSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=observed_var,
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=Standardize(m=train_Y.shape[-1]),
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
        print(mean_error, var_error)

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


def build_gp_model(
    dataset: BayesDataset,
    infer_noise: bool = False,
    noise_model: str = "homoscedastic",
) -> Union[SingleTaskGP, PseudoHeteroskedasticSingleTaskGP]:
    """Build a GP model from a dataset.

    Parameters
    ----------
    dataset : BayesDataset
        Dataset to build the model from.
    infer_noise : bool, optional
        Whether to infer the noise level, by default False.
    noise_model : str, optional
        Type of noise model to use, by default "homoscedastic".

    Returns
    -------
    Union[SingleTaskGP, PseudoHeteroskedasticSingleTaskGP]
        GP model.
    """
    # Transform outcomes and clamp variances to prevent warnings from GPyTorch
    values, variances = dataset.transform_outcomes()
    variances = torch.clamp_min(variances, 1e-6)
    # Initialise model instance depending on noise setting
    if noise_model not in ("homoscedastic", "heteroscedastic"):
        raise ValueError(f"Unknown noise model: {noise_model}")
    homoscedastic_noise = noise_model == "homoscedastic"
    if infer_noise and not homoscedastic_noise:
        # Use our utility for the most likely heteroskedastic model
        model = fit_most_likely_heteroskedastic_gp(dataset.params, values)
    else:
        # Manually fit a GP model
        model_cls = (
            SingleTaskGPWithNoise if homoscedastic_noise else PseudoHeteroskedasticSingleTaskGP
        )
        model = model_cls(
            dataset.params,
            values,
            train_Yvar=None if infer_noise else variances,
            input_transform=Normalize(d=dataset.params.shape[-1]),
        )
        # Fit the GP (in case of trouble, fall back to an Adam-based optimiser)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except botorch.exceptions.ModelFittingError:
            warnings.warn('Optimisation of the MLL failed, falling back to PyTorch optimiser')
            fit_mll_pytorch_loop(mll)
    return model
