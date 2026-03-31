"""Legacy wrapper for the BoTorch optimiser.

This module provides backwards-compatible input specification for the BoTorch-based Bayesian
optimisation workflow, while delegating all execution to the new generic optimiser infrastructure.
Legacy parameters (e.g. ``n_initial``, ``acquisition``, ``q``, ``noisy``) are translated into the
appropriate combination of :class:`~piglot.optimisers.generic.optimiser.GenericOptimiser` policies
and settings.
"""
from typing import Any, List, Optional, TypeVar
import warnings
from piglot.settings import Settings
from piglot.objective import Objective
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.optimisers.generic.campaign import CandidatePolicy, OptimisationSettings
from piglot.optimisers.generic.acquisitions import (
    AcquisitionSettings,
    default_acquisition,
)
from piglot.optimisers.generic.policies.acquisition import AcquisitionCandidatePolicy
from piglot.optimisers.generic.policies.initial import InitialCandidatePolicy
from piglot.optimisers.generic.policies.random import RandomCandidatePolicy
from piglot.data.surrogate import SurrogateSettings


T = TypeVar('T', bound='BayesianBoTorch')


class BayesianBoTorch(GenericOptimiser):
    """Legacy wrapper that translates the old BoTorch input specification into the new generic
    optimiser infrastructure.

    This class accepts the same constructor arguments and YAML keys as the original
    ``BayesianBoTorch`` optimiser and transparently builds the corresponding
    :class:`~piglot.optimisers.generic.optimiser.GenericOptimiser` with the appropriate policies.

    The translation maps the legacy parameters as follows:

    * ``skip_initial`` controls whether an :class:`InitialCandidatePolicy` is included.
    * ``n_initial`` becomes a :class:`RandomCandidatePolicy` with the given number of rounds.
    * The main BO loop becomes an :class:`AcquisitionCandidatePolicy` with the configured
      ``acquisition``, ``beta``, ``q``, etc.
    * ``noisy``, ``reference_point``, ``nadir_scale`` and ``pca_variance`` feed into
      :class:`OptimisationSettings` and :class:`SurrogateSettings`.
    """

    def __init__(  # noqa: C901
        self,
        settings: Settings,
        objective: Objective,
        n_initial: int = None,
        acquisition: str = None,
        beta: float = 1.0,
        noisy: bool = False,
        q: int = 1,
        seed: int = None,
        reference_point: Optional[List[float]] = None,
        nadir_scale: float = 0.1,
        skip_initial: bool = False,
        pca_variance: float = None,
        num_restarts: int = None,
        raw_samples: int = None,
        mc_samples: int = None,
        batch_size: int = None,
        num_fantasies: int = None,
        sequential: bool = False,
    ) -> None:

        # --- Resolve heuristic defaults (matching old BayesianBoTorch) ---
        noisy = bool(noisy)
        skip_initial = bool(skip_initial)
        sequential = bool(sequential)
        if n_initial is None:
            n_initial = max(8, 2 * settings.parameters.num_optim_parameters())
        if seed is None:
            seed = settings.seed

        # Determine the stochastic flag
        stochastic = noisy or objective.has_variance()

        # Resolve acquisition name
        if acquisition is None:
            acquisition = default_acquisition(
                objective.is_composite(),
                objective.is_multi_objective(),
                stochastic,
                q,
            )
        else:
            if not acquisition.startswith('q'):
                acquisition = 'q' + acquisition

        # Resolve PCA variance
        if pca_variance is None and objective.is_composite():
            pca_variance = 1e-6
        elif pca_variance and not (objective.is_composite() or objective.is_multi_objective()):
            warnings.warn(
                "Ignoring PCA variance for non-composite single-objective problem.",
                UserWarning,
                stacklevel=2,
            )
            pca_variance = None

        # Build surrogate, optimisation, and acquisition settings
        policies, optim_settings = self._build_config(
            noisy=noisy,
            nadir_scale=nadir_scale,
            reference_point=reference_point,
            pca_variance=pca_variance,
            acquisition=acquisition,
            q=q,
            beta=beta,
            sequential=sequential,
            seed=seed,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            mc_samples=mc_samples,
            batch_size=batch_size,
            num_fantasies=num_fantasies,
            skip_initial=skip_initial,
            n_initial=n_initial,
            n_bo_iters=settings.iters,
        )

        super().__init__(settings, objective, policies, optim_settings)

    @staticmethod
    def _build_config(
        *,
        noisy: bool,
        nadir_scale: float,
        reference_point: Optional[List[float]],
        pca_variance: Optional[float],
        acquisition: str,
        q: int,
        beta: float,
        sequential: bool,
        seed: Optional[int],
        num_restarts: Optional[int],
        raw_samples: Optional[int],
        mc_samples: Optional[int],
        batch_size: Optional[int],
        num_fantasies: Optional[int],
        skip_initial: bool,
        n_initial: int,
        n_bo_iters: int,
    ) -> tuple[dict[str, CandidatePolicy], OptimisationSettings]:
        """Build the generic-optimiser policies and settings from legacy parameters.

        Returns
        -------
        tuple[dict[str, CandidatePolicy], OptimisationSettings]
            The ordered policy mapping and the optimisation settings.
        """
        # Surrogate settings
        noise_mode = 'infer' if noisy else 'none'
        surrogate_kwargs: dict[str, Any] = {'noise': noise_mode}
        if pca_variance is not None:
            surrogate_kwargs['pca_variance'] = pca_variance
        surrogate_settings = SurrogateSettings(**surrogate_kwargs)

        # Optimisation settings
        optim_settings = OptimisationSettings(
            noisy=noisy,
            nadir_scale=nadir_scale,
            ref_point=reference_point,
            surrogate_settings=surrogate_settings,
        )

        # Acquisition settings
        acq_kwargs: dict[str, Any] = {
            'name': acquisition,
            'q': q,
            'beta': beta,
            'sequential': sequential,
            'seed': seed,
        }
        if num_restarts is not None:
            acq_kwargs['num_restarts'] = num_restarts
        if raw_samples is not None:
            acq_kwargs['raw_samples'] = raw_samples
        if mc_samples is not None:
            acq_kwargs['mc_samples'] = mc_samples
        if batch_size is not None:
            acq_kwargs['batch_size'] = batch_size
        if num_fantasies is not None:
            acq_kwargs['num_fantasies'] = num_fantasies
        acq_settings = AcquisitionSettings(**acq_kwargs)

        # Build ordered policies
        policies: dict[str, CandidatePolicy] = {}

        # 1) Initial shot (unless skipped)
        if not skip_initial:
            policies['Initial shot'] = InitialCandidatePolicy(rounds=1)

        # 2) Random initialisation points
        if n_initial > 0:
            policies['Random initialisation'] = RandomCandidatePolicy(
                rounds=n_initial,
                seed=seed,
            )

        # 3) Main BO acquisition loop
        if n_bo_iters > 0:
            policies['BoTorch acquisition'] = AcquisitionCandidatePolicy(
                rounds=n_bo_iters,
                settings=acq_settings,
                num_candidates_per_round=q if q > 1 else None,
                num_workers=q if q > 1 else None,
            )

        return policies, optim_settings

    def name(self) -> str:
        """Return the name of the optimiser.

        Returns
        -------
        str
            Name of the optimiser.
        """
        return "BoTorch"

    @classmethod
    def read(
        cls: type[T],
        config: dict[str, Any],
        settings: Settings,
        objective: Objective,
    ) -> T:
        """Read a BayesianBoTorch optimiser from a legacy configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary with legacy BoTorch keys (e.g. ``n_initial``,
            ``acquisition``, ``q``, ``noisy``, etc.).
        settings : Settings
            Global settings for the optimisation problem.
        objective : Objective
            Objective to optimise.

        Returns
        -------
        T
            The created optimiser instance.
        """
        # Convert numeric strings to proper types (matching base Optimiser.read behaviour)
        from piglot.utils.assorted import str_to_numeric
        config = {k: str_to_numeric(v) for k, v in config.items()}

        # Handle reference_point which may come as a list from YAML
        if 'reference_point' in config and config['reference_point'] is not None:
            ref = config['reference_point']
            if isinstance(ref, (list, tuple)):
                config['reference_point'] = [float(x) for x in ref]

        return cls(settings, objective, **config)
