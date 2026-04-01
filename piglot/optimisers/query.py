"""Module for a wrapper around a query optimiser."""
from typing import Optional, Literal, TypeVar, Any
from piglot.objective import Objective
from piglot.settings import Settings
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.optimisers.generic.containers import OptimisationSettings
from piglot.optimisers.generic.policies.query import QueryCandidatePolicy


T = TypeVar('T', bound='QueryOptimiser')


class QueryOptimiser(GenericOptimiser):
    """Wrapper class for a query optimiser."""

    def __init__(
        self,
        settings: Settings,
        objective: Objective,
        optim_settings: OptimisationSettings,
        file: str,
        num_workers: Optional[int] = None,
        mode: Optional[Literal['sequential', 'batched', 'async']] = None,
    ) -> None:
        policies = {
            'Query': QueryCandidatePolicy(file, num_workers=num_workers, mode=mode)
        }
        super().__init__(settings, objective, policies, optim_settings)

    @classmethod
    def read(cls: type[T], config: dict[str, Any], settings: Settings, objective: Objective) -> T:
        """Read an optimiser from the given configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary for the optimiser.
        settings : Settings
            Settings for the optimiser.
        objective : Objective
            Objective to optimise.

        Returns
        -------
        T
            The created optimiser instance.
        """
        if 'file' not in config:
            raise ValueError("Missing 'file' path for query optimiser.")
        file = config.pop('file')
        num_workers = config.pop('num_workers', None)
        mode = config.pop('mode', None)
        optim_settings = OptimisationSettings.read(config)
        return cls(settings, objective, optim_settings, file, num_workers=num_workers, mode=mode)
