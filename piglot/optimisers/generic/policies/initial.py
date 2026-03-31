"""Module with the policy for evaluating the initial candidates."""
from typing import Optional, TypeVar
import numpy as np
from piglot.data.dataset import ObjectiveDataset
from piglot.data.surrogate import ObjectiveModel
from piglot.optimisers.generic.containers import OptimisationState
from piglot.optimisers.generic.campaign import CandidatePolicy
from piglot.utils.readable import readable_from_constructor


T = TypeVar('T', bound='InitialCandidatePolicy')


@readable_from_constructor
class InitialCandidatePolicy(CandidatePolicy):
    """Policy that generates candidates randomly."""

    def __init__(self, rounds: int = 1) -> None:
        super().__init__(rounds, False)

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
        return [dataset.settings.parameters.get_initial_vector()] * num_candidates
