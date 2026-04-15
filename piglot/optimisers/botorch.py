"""Module for a wrapper around a Bayesian optimisation campaign."""
from typing import Optional, TypeVar, Any, Literal
from piglot.objective import Objective
from piglot.settings import Settings
from piglot.optimisers.generic.acquisitions import AcquisitionSettings
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.optimisers.generic.containers import OptimisationSettings
from piglot.optimisers.generic.policies.random import RandomCandidatePolicy
from piglot.optimisers.generic.policies.initial import InitialCandidatePolicy
from piglot.optimisers.generic.policies.acquisition import AcquisitionCandidatePolicy


T = TypeVar('T', bound='BoTorchOptimiser')


class BoTorchOptimiser(GenericOptimiser):
    """Wrapper class for a Bayesian optimisation using BoTorch."""

    def __init__(
        self,
        settings: Settings,
        objective: Objective,
        optim_settings: OptimisationSettings,
        acq_settings: AcquisitionSettings,
        seed: Optional[int] = None,
        mode: Optional[Literal['sequential', 'batched', 'async']] = None,
        n_initial: Optional[int] = None
    ) -> None:
        # Sanity check on input data
        if settings.iters is None:
            raise ValueError("Number of iterations must be specified in the config file.")

        # Set up initial heuristics
        num_workers = optim_settings.num_workers
        if n_initial is None:
            n_initial = max(8, 2 * settings.parameters.num_optim_parameters())

        # Set up policies
        policies = {
            'Initial guess': InitialCandidatePolicy(),
            'Random': RandomCandidatePolicy(
                rounds=n_initial,
                num_workers=num_workers,
                num_candidates_per_round=1,
                mode='sequential' if num_workers is None or num_workers == 1 else 'async',
                seed=seed or settings.seed,
            ),
            'BoTorch': AcquisitionCandidatePolicy(
                rounds=settings.iters,
                settings=acq_settings,
                num_workers=num_workers,
                mode=mode,
            ),
        }
        super().__init__(settings, objective, policies, optim_settings)

    @classmethod
    def default_acquisition_type(cls, settings: OptimisationSettings, objective: Objective) -> str:
        """Get the default acquisition type for the optimiser.

        Parameters
        ----------
        settings : OptimisationSettings
            Settings for the optimiser.
        objective : Objective
            Objective to optimise.

        Returns
        -------
        str
            The default acquisition type.
        """
        if objective.is_multi_objective():
            return 'qlognehvi' if objective.is_noisy() else 'qlogehvi'
        if objective.is_noisy():
            return 'qlognei'
        return 'qlogei'

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
        # Optional fields
        mode = config.pop('mode', None)
        seed = config.pop('seed', settings.seed)
        acquisition = config.pop('acquisition', None)
        optim_settings = OptimisationSettings.read(config)

        # Parse acquisition: missing, simple or long specification
        if acquisition is None:
            acq_settings = AcquisitionSettings.read(
                {'name': cls.default_acquisition_type(optim_settings, objective)}
            )
        elif isinstance(acquisition, str):
            acq_settings = AcquisitionSettings.read({'name': acquisition})
        else:
            acq_settings = AcquisitionSettings.read(acquisition)

        # Minimal sanity check before creating the optimiser
        if mode not in (None, 'sequential', 'batched', 'async'):
            raise ValueError(f"Invalid mode '{mode}' in optimiser configuration.")
        return cls(
            settings,
            objective,
            optim_settings,
            acq_settings,
            seed=seed,
            mode=mode,
        )
