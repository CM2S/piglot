"""Module with the query policy for generic data-driven optimisation."""
from typing import Optional, TypeVar, Literal
import numpy as np
from piglot.data.dataset import ObjectiveDataset
from piglot.data.surrogate import ObjectiveModel
from piglot.optimisers.generic.containers import OptimisationState
from piglot.optimisers.generic.campaign import CandidatePolicy
from piglot.utils.readable import readable_from_constructor


T = TypeVar('T', bound='QueryCandidatePolicy')


@readable_from_constructor
class QueryCandidatePolicy(CandidatePolicy):
    """Policy that generates candidates from a list of queries."""

    def __init__(
        self,
        file: str,
        num_workers: Optional[int] = None,
        mode: Optional[Literal['sequential', 'batched', 'async']] = None,
    ) -> None:
        # Read and sanitise points
        points = np.genfromtxt(file)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        elif points.ndim != 2:
            raise ValueError(f"Query points from file '{file}' have incorrect dimensions.")

        # Assume one point per round
        super().__init__(points.shape[0], False, num_workers, 1, mode)
        self.points: list[np.ndarray] = [point for point in points]
        self.generator = iter(self.points)

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
        # Sanitise parameter dimensions
        if len(self.points[0]) != dataset.settings.parameters.num_optim_parameters():
            raise ValueError("Query points have incorrect dimensions.")

        return [next(self.generator) for _ in range(num_candidates)]
