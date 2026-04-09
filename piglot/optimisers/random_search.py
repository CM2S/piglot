"""Module for a wrapper around a random search optimiser."""
from typing import Optional, TypeVar, Any
from piglot.objective import Objective
from piglot.settings import Settings
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.optimisers.generic.containers import OptimisationSettings
from piglot.optimisers.generic.policies.random import RandomCandidatePolicy


T = TypeVar('T', bound='RandomSearchOptimiser')


class RandomSearchOptimiser(GenericOptimiser):
    """Wrapper class for a random search optimiser."""

    def __init__(
        self,
        settings: Settings,
        objective: Objective,
        optim_settings: OptimisationSettings,
        seed: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        # Sanity check on input data
        if settings.iters is None:
            raise ValueError("Number of iterations must be specified in the config file.")

        policies = {
            'Random': RandomCandidatePolicy(
                rounds=settings.iters,
                num_workers=num_workers,
                num_candidates_per_round=1,
                mode='sequential' if num_workers is None or num_workers == 1 else 'async',
                seed=seed or settings.seed,
            )
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
        seed = config.pop('seed', None)
        num_workers = config.pop('num_workers', None)
        optim_settings = OptimisationSettings.read(config)
        return cls(settings, objective, optim_settings, seed=seed, num_workers=num_workers)
