"""Module with the acquisition function-based policy for generic data-driven optimisation."""
from typing import Optional, TypeVar, Any, Literal
import numpy as np
import torch
from piglot.data.dataset import ObjectiveDataset
from piglot.data.surrogate import ObjectiveModel
from piglot.optimisers.generic.containers import OptimisationState
from piglot.optimisers.generic.campaign import CandidatePolicy
from piglot.optimisers.generic.acquisitions import (
    AcquisitionSettings,
    get_acquisition,
    optimise_acquisition,
)


T = TypeVar('T', bound='CandidatePolicy')


class AcquisitionCandidatePolicy(CandidatePolicy):
    """Policy that generates candidates based on an acquisition function."""

    def __init__(
        self,
        rounds: int,
        settings: AcquisitionSettings,
        num_workers: Optional[int] = None,
        num_candidates_per_round: Optional[int] = None,
        mode: Optional[Literal['sequential', 'batched', 'async']] = None,
    ) -> None:
        super().__init__(rounds, True, num_workers, num_candidates_per_round, mode)
        self.settings = settings

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
        acq = get_acquisition(
            model,
            dataset.settings.parameters,
            self.settings,
            state,
            pending=torch.tensor(pending) if pending else None,
        )
        candidates, _ = optimise_acquisition(
            acq, dataset.settings.parameters, model, self.settings, q=num_candidates
        )
        return [c.to(torch.float64).cpu().numpy() for c in candidates]

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
        # Check for mandatory entries
        if 'type' not in config:
            raise ValueError('Missing acquisition type for acquisition candidate policy.')
        # Extract policy keywords from the configuration
        rounds = int(config.pop('rounds', 1))
        num_workers = int(config.pop('num_workers')) if 'num_workers' in config else None
        num_candidates_per_round = (
            int(config.pop('num_candidates_per_round'))
            if 'num_candidates_per_round' in config else None
        )
        mode = str(config.pop('mode')) if 'mode' in config else None
        # Read acquisition settings from the configuration: we need to inject the name
        settings = AcquisitionSettings.read({'name': config.pop('type')} | config)
        return cls(
            rounds=rounds,
            settings=settings,
            num_workers=num_workers,
            num_candidates_per_round=num_candidates_per_round,
            mode=mode,
        )
