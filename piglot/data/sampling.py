"""Module for sampling methods."""
from typing import Optional, Literal
from contextlib import ExitStack
import numpy as np
import torch
import gpytorch.settings as gpts
from botorch.posteriors import Posterior
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from piglot.data.surrogate import ObjectiveModel


class FastNormalSampler(MCSampler):
    """Fast Gaussian sampler using acceleration techniques."""

    def __init__(
        self,
        sample_shape: torch.Size,
        seed: Optional[int] = None,
        strategy: Literal['cholesky', 'ciq', 'lanczos', 'rff'] = 'lanczos',
    ) -> None:
        super().__init__(sample_shape=sample_shape, seed=seed)
        self.strategy = strategy

    def forward(self, posterior: Posterior) -> torch.Tensor:
        """Generate samples from the posterior.

        Parameters
        ----------
        posterior : Posterior
            The posterior distribution to sample from.

        Returns
        -------
        torch.Tensor
            Samples from the posterior.
        """
        with ExitStack() as es:
            if self.strategy == "cholesky":
                es.enter_context(gpts.max_cholesky_size(float("inf")))
            elif self.strategy == "ciq":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(True))
                es.enter_context(gpts.minres_tolerance(2e-3))
                es.enter_context(gpts.num_contour_quadrature(15))
            elif self.strategy == "lanczos":
                es.enter_context(
                    gpts.fast_computations(
                        covar_root_decomposition=True, log_prob=True, solves=True
                    )
                )
                es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(False))
            elif self.strategy == "rff":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(torch.no_grad())

            # Sample from the posterior
            samples = posterior.rsample(self.sample_shape)

        return samples


def draw_function_samples(
    model: ObjectiveModel,
    num_points: int,
    num_samples: int,
    seed: Optional[int] = None,
    kind: Literal['latent', 'objective'] = 'objective',
    strategy: Literal['sobol', 'cholesky', 'ciq', 'lanczos', 'rff'] = 'lanczos'
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw function samples from the given model.

    Parameters
    ----------
    model : ObjectiveModel
        The model to sample from.
    num_points : int
        The number of points to draw.
    num_samples : int
        The number of samples to draw.
    seed : Optional[int], optional
        The random seed, by default None.
    kind : Literal['latent', 'objective'], optional
        The type of samples to draw, by default 'objective'.
    strategy : Literal['sobol', 'cholesky', 'ciq', 'lanczos', 'rff'], optional
        The sampling strategy to use, by default 'lanczos'.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The grid and the drawn function samples.
    """
    # Generate random grid of points
    rng = np.random.default_rng(seed)
    params = model.dataset.settings.parameters
    grid = torch.tensor([params.get_random_vector(rng).tolist() for _ in range(num_points)])

    # Set up sampler
    if strategy == 'sobol':
        sampler = SobolQMCNormalSampler(torch.Size([num_samples]), seed=seed)
    else:
        sampler = FastNormalSampler(torch.Size([num_samples]), seed=seed, strategy=strategy)

    # Draw samples
    if kind == 'latent':
        return model.latent_samples(grid, sampler=sampler)
    return grid, model.objective_samples(grid, sampler=sampler)
