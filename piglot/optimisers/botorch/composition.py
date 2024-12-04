"""Module with composition sampler helpers for BoTorch models."""
from typing import Union
import torch
from torch.autograd.functional import hessian
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_normal_samples
from piglot.objective import GenericObjective
from piglot.optimisers.botorch.dataset import BayesDataset
from piglot.optimisers.botorch.model import (
    SingleTaskGPWithNoise,
    PseudoHeteroskedasticSingleTaskGP,
)
from piglot.optimisers.botorch.risk_acquisitions import RiskMeasure


class BoTorchComposition:
    """Base class for composition sampling in BoTorch."""

    def __init__(
        self,
        model: Union[SingleTaskGPWithNoise, PseudoHeteroskedasticSingleTaskGP],
        dataset: BayesDataset,
        objective: GenericObjective,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.objective = objective

    def from_original_samples(self, vals: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Compute composition samples from samples in the original space.

        Parameters
        ----------
        vals : torch.Tensor
            Samples in the original space.
        X : torch.Tensor
            Input parameters.

        Returns
        -------
        torch.Tensor
            Composition samples.
        """
        # Just negate the untransformed outcomes if no composition is available
        if not self.objective.composition:
            return -vals.squeeze(-1)
        # Otherwise, use the composition function
        return -self.objective.composition.composition_torch(vals, X)

    def from_model_samples(self, vals: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Compute composition samples from samples directly from the model.

        Parameters
        ----------
        vals : torch.Tensor
            Samples from the model (in the transformed space).
        X : torch.Tensor
            Input parameters.

        Returns
        -------
        torch.Tensor
            Composition samples.
        """
        return self.from_original_samples(self.dataset.untransform_outcomes(vals), X)

    def from_inputs(
        self,
        X: torch.Tensor,
        num_samples: int,
        seed: int = None,
        observation_noise: bool = False,
    ) -> torch.Tensor:
        """Compute composition samples from input parameters.

        Parameters
        ----------
        X : torch.Tensor
            Input parameters.
        num_samples : int
            Number of samples to generate.
        seed : int, optional
            Random seed, by default None.
        observation_noise : bool, optional
            Whether to include observation noise, by default False.

        Returns
        -------
        torch.Tensor
            Composition samples.
        """
        sampler = SobolQMCNormalSampler(torch.Size([num_samples]), seed=seed)
        with torch.no_grad():
            posterior = self.model.posterior(X, observation_noise=observation_noise)
            samples = sampler(posterior)
        return self.from_model_samples(samples, X)

    def taylor_from_inputs(self, X: torch.Tensor, observation_noise: bool = False) -> torch.Tensor:
        """Compute Taylor expansion samples from input parameters.

        Parameters
        ----------
        X : torch.Tensor
            Input parameters.
        observation_noise : bool, optional
            Whether to include observation noise, by default False.

        Returns
        -------
        torch.Tensor
            Taylor expansion samples.
        """
        # Compute the mean and covariance of the model
        with torch.no_grad():
            posterior = self.model.posterior(X, observation_noise=observation_noise)
            mean = posterior.mean
            full_cov = posterior.mvn.covariance_matrix
        # Evaluate the function for the mean
        f_mean = self.from_model_samples(mean, X)
        # Evaluate the Hessian for each point and the Taylor expansion
        n_outputs = mean.shape[-1]
        result = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
        for i in range(X.shape[0]):
            hess = hessian(lambda x: self.from_model_samples(x, X[i, :]), mean[i, :])
            cov = full_cov[i * n_outputs: (i + 1) * n_outputs, i * n_outputs: (i + 1) * n_outputs]
            result[i] = f_mean[i] + 0.5 * torch.trace(hess @ cov)
        return result


class InjectObservationNoiseComposition(BoTorchComposition):
    """Composition sampler with observation noise injection."""

    def __init__(
        self,
        model: Union[SingleTaskGPWithNoise, PseudoHeteroskedasticSingleTaskGP],
        dataset: BayesDataset,
        objective: GenericObjective,
        num_samples: int,
        seed: int,
    ) -> None:
        super().__init__(model, dataset, objective)
        self.num_samples = num_samples
        self.var_samples = draw_sobol_normal_samples(
            d=dataset.numel_latent_space(),
            n=num_samples,
            seed=seed,
            dtype=dataset.dtype,
            device=dataset.device,
        )

    def from_model_samples(self, vals: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Compute composition samples from samples directly from the model with observation
        noise injected.

        Parameters
        ----------
        vals : torch.Tensor
            Samples from the model (in the transformed space) (n_qmc x ... x n_output).
        X : torch.Tensor
            Input parameters (... x n_x).

        Returns
        -------
        torch.Tensor
            Composition samples (num_samples x n_qmc x ... x n_output)
        """
        # Reshape noise samples to proper dimensions
        noise_samples = torch.clone(self.var_samples)
        for _ in range(len(vals.shape) - 1):
            noise_samples = noise_samples.unsqueeze(1)
        # Inject observation noise (with broadcasting)
        noise_var = self.model.noise_prediction(X)
        y_samples = vals + noise_samples * noise_var.sqrt()
        return super().from_model_samples(y_samples, X)


class RiskComposition(InjectObservationNoiseComposition):
    """Composition sampler with computation of a risk measure."""

    def __init__(
        self,
        model: Union[SingleTaskGPWithNoise, PseudoHeteroskedasticSingleTaskGP],
        dataset: BayesDataset,
        objective: GenericObjective,
        num_samples: int,
        seed: int,
        risk_measure: RiskMeasure,
    ) -> None:
        super().__init__(model, dataset, objective, num_samples, seed)
        self.risk_measure = risk_measure

    def from_model_samples(self, vals: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Compute composition samples from samples directly from the model.

        Parameters
        ----------
        vals : torch.Tensor
            Samples from the model (in the transformed space).
        X : torch.Tensor
            Input parameters.

        Returns
        -------
        torch.Tensor
            Composition samples.
        """
        return self.risk_measure(super().from_model_samples(vals, X), dim=0)
