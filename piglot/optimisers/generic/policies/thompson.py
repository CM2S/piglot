"""Module for Thompson sampling policies."""
from typing import Optional, TypeVar, Literal
import numpy as np
import torch
from botorch.generation import MaxPosteriorSampling
from piglot.data.dataset import ObjectiveDataset
from piglot.data.surrogate import ObjectiveModel
from piglot.data.sampling import draw_function_samples
from piglot.optimisers.generic.containers import OptimisationState
from piglot.optimisers.generic.campaign import CandidatePolicy
from piglot.utils.readable import readable_from_constructor


T = TypeVar('T', bound='ThompsonSamplingCandidatePolicy')


@readable_from_constructor
class ThompsonSamplingCandidatePolicy(CandidatePolicy):
    """Policy that generates candidates using Thompson sampling."""

    def __init__(
        self,
        num_points: int,
        rounds: int = 1,
        num_workers: Optional[int] = None,
        num_candidates_per_round: Optional[int] = None,
        mode: Optional[Literal['sequential', 'batched', 'async']] = None,
        strategy: Literal['sobol', 'cholesky', 'ciq', 'lanczos', 'rff'] = 'lanczos',
        seed: Optional[int] = None,
        oversample: float = 2.0
    ) -> None:
        super().__init__(rounds, True, num_workers, num_candidates_per_round, mode)
        self.num_points = num_points
        self.strategy = strategy
        self.seed = seed
        self.oversample = oversample

    def get_next_candidates(
        self,
        num_candidates: int,
        state: OptimisationState,
        dataset: ObjectiveDataset,
        pending: list[np.ndarray],
        model: Optional[ObjectiveModel],
    ) -> list[np.ndarray]:
        """Get the next candidates to evaluate based on the current state.

        Parameters
        ----------
        num_candidates : int
            Number of candidates to generate.
        state : OptimisationState
            Current state of the optimisation campaign.
        dataset : ObjectiveDataset
            Dataset containing the observations from the optimisation campaign.
        pending : list[np.ndarray]
            List of candidates that have been submitted for evaluation but have not completed yet.
        model : Optional[ObjectiveModel]
            Surrogate model for the objective function, if required.

        Returns
        -------
        list[np.ndarray]
            List of parameters for the next candidates to evaluate.
        """
        if self.seed is not None:
            self.seed += 1

        grid, samples = draw_function_samples(
            model,
            self.num_points,
            round(num_candidates * self.oversample),
            seed=self.seed,
            strategy=self.strategy,
        )
        sampler = MaxPosteriorSampling(model.gp)
        candidates = sampler.maximize_samples(grid, -samples, num_samples=num_candidates)
        return [c.to(torch.float64).cpu().numpy() for c in candidates]
