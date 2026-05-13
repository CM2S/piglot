"""Module for sampling methods."""
from typing import Optional, Literal
from contextlib import ExitStack
from functools import partial
import numpy as np
import torch
import gpytorch.settings as gpts
from botorch.optim import optimize_acqf
from botorch.posteriors import Posterior
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.utils.sampling import draw_sobol_samples
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
    num_grid_points: int,
    num_samples: int,
    seed: Optional[int] = None,
    kind: Literal['latent', 'objective'] = 'objective',
    strategy: Literal['sobol', 'cholesky', 'ciq', 'lanczos', 'rff'] = 'lanczos',
    observation_noise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw function samples from the given model.

    Parameters
    ----------
    model : ObjectiveModel
        The model to sample from.
    num_grid_points : int
        The number of grid points to consider.
    num_samples : int
        The number of samples to draw.
    seed : Optional[int], optional
        The random seed, by default None.
    kind : Literal['latent', 'objective'], optional
        The type of samples to draw, by default 'objective'.
    strategy : Literal['sobol', 'cholesky', 'ciq', 'lanczos', 'rff'], optional
        The sampling strategy to use, by default 'lanczos'.
    observation_noise : bool, optional
        Whether to include observation noise in the samples, by default False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The grid and the drawn function samples.
    """
    # Generate random grid of points
    rng = np.random.default_rng(seed)
    params = model.dataset.settings.parameters
    grid = torch.tensor([params.get_random_vector(rng).tolist() for _ in range(num_grid_points)])

    # Set up sampler
    if strategy == 'sobol':
        sampler = SobolQMCNormalSampler(torch.Size([num_samples]), seed=seed)
    else:
        sampler = FastNormalSampler(torch.Size([num_samples]), seed=seed, strategy=strategy)

    # Draw samples
    if kind == 'latent':
        return model.latent_samples(grid, sampler=sampler, observation_noise=observation_noise)
    return grid, model.objective_samples(grid, sampler=sampler, observation_noise=observation_noise)


class PathwiseSamplingModel:
    """Pathwise sampling model using Matheron paths."""

    def __init__(
        self, model: ObjectiveModel, num_samples: int, observation_noise: bool = False
    ) -> None:
        self.model = model
        self.num_samples = num_samples
        self.observation_noise = observation_noise
        self.path = draw_matheron_paths(model.gp, torch.Size([num_samples]))

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        samples = self.path(points).transpose(-1, -2)
        samples = samples.reshape(self.num_samples, *points.shape[:-1], -1)
        # Inject observation noise
        if self.observation_noise:
            noise_std = self.model.gp.noise_prediction(points).sqrt()
            samples = samples + torch.randn_like(samples) * noise_std
        return self.model.composition_from_raw(samples, points)


def find_pathwise_optima(
    model: PathwiseSamplingModel,
    bounds: torch.Tensor,
    num_restarts: int = 8,
    raw_samples: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise a pseudo-acquisition function to find the optima of pathwise samples.

    We optimise the function samples of the pathwise model jointly using some tricks to emulate
    a q-batch setting. The initial conditions for the multi-start optimisation are generated
    in a greedy fashion that exploits the q-batch trick.

    Parameters
    ----------
    model : PathwiseSamplingModel
        The pathwise sampling model.
    bounds : torch.Tensor
        The bounds for the optimisation.
    num_restarts : int, optional
        The number of restarts for the optimisation, by default 8.
    raw_samples : int, optional
        The number of raw samples for the optimisation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The optimal points and their corresponding values.
    """
    # Define the pseudo-acquisition function
    def pseudo_acqf(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        q = x.shape[-2]
        n = x.shape[:-2]
        # We do some tensor acrobatics to reshape the input and output tensors:
        # i) pathwise samplers require input shape (n x d)
        # ii) `q` is equal to `model.num_samples`
        # iii) we take the diagonal to match each q-point with each sample
        # iv) we sum over the last dimension to get a scalar value for each set of samples
        values = model(x.reshape(-1, d)).reshape(q, *n, q)
        return values.diagonal(dim1=0, dim2=-1).sum(dim=-1)

    # Update heuristics
    if raw_samples is None:
        raw_samples = max(256, 16 * bounds.shape[-1] * bounds.shape[-1])

    # Build a better set of initial conditions
    initial_candidates = draw_sobol_samples(bounds=bounds, n=raw_samples, q=1).squeeze(-2)
    init_samples = model(initial_candidates)
    top_idx = torch.topk(init_samples, k=num_restarts, dim=-1).indices
    batch_initial_conditions = initial_candidates[top_idx, :].transpose(0, 1)

    # Find optima for each sample
    points, _ = optimize_acqf(
        pseudo_acqf,
        bounds=bounds,
        q=model.num_samples,
        num_restarts=num_restarts,
        batch_initial_conditions=batch_initial_conditions.contiguous(),
    )

    # Return the true value of the pathwise samples
    with torch.no_grad():
        values = -model(points).diagonal()
    return points, values
