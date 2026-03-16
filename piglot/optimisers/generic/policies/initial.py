"""Module with the policy for evaluating the initial candidates."""
from typing import Optional, TypeVar, Any
import numpy as np
from piglot.data.dataset import ObjectiveDataset
from piglot.data.surrogate import ObjectiveModel
from piglot.optimisers.generic.containers import OptimisationState
from piglot.optimisers.generic.campaign import CandidatePolicy


T = TypeVar('T', bound='InitialCandidatePolicy')


class InitialCandidatePolicy(CandidatePolicy):
    """Policy that generates candidates randomly."""

    def __init__(self, rounds: int) -> None:
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
        return [np.array([p.inital_value for p in dataset.settings.parameters])] * num_candidates

    @classmethod
    def read(cls: type[T], config: dict[str, Any]) -> T:
        """Read a candidate policy from the given configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the candidate policy.

        Returns
        -------
        T
            The created candidate policy instance.
        """
        return cls(rounds=config.get('rounds', 1))
