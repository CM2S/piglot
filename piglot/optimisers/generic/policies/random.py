"""Module with the random policy for generic data-driven optimisation."""
from typing import Optional, TypeVar, Any, Literal
import numpy as np
from piglot.data.dataset import ObjectiveDataset
from piglot.data.surrogate import ObjectiveModel
from piglot.optimisers.generic.containers import OptimisationState
from piglot.optimisers.generic.campaign import CandidatePolicy


T = TypeVar('T', bound='RandomCandidatePolicy')


class RandomCandidatePolicy(CandidatePolicy):
    """Policy that generates candidates randomly."""

    def __init__(
        self,
        rounds: int,
        num_workers: Optional[int] = None,
        num_candidates_per_round: Optional[int] = None,
        mode: Optional[Literal['sequential', 'batched', 'async']] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(rounds, False, num_workers, num_candidates_per_round, mode)
        self.rng = np.random.default_rng(seed)

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
        parameters = dataset.settings.parameters
        return [
            np.array([self.rng.uniform(low=p.lbound, high=p.ubound) for p in parameters])
            for _ in range(num_candidates)
        ]

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
        return cls(
            rounds=config.get('rounds', 1),
            num_workers=config.get('num_workers'),
            num_candidates_per_round=config.get('num_candidates_per_round'),
            mode=config.get('mode'),
            seed=config.get('seed'),
        )
