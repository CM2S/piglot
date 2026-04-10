"""Module for acquisition functions with BoTorch."""
from typing import Optional
import torch
from botorch.acquisition import (
    AcquisitionFunction,
    qUpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qLogExpectedImprovement,
    qNoisyExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qKnowledgeGradient,
    qSimpleRegret,
)
from botorch.acquisition.objective import (
    GenericMCObjective,
)
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
    qHypervolumeKnowledgeGradient,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.models.converter import batched_to_model_list
from botorch.optim import optimize_acqf, optimize_acqf_mixed, optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler
from piglot.data.surrogate import ObjectiveModel
from piglot.parameter import ParameterSet
from piglot.optimisers.generic.containers import OptimisationState
from piglot.utils.readable import ReadableModel


class AcquisitionSettings(ReadableModel):
    """Container for settings related to acquisition functions."""
    name: str
    q: int = 1
    beta: float = 1.0
    batch_size: int = 128
    mc_samples: int = 256
    num_restarts: int = 12
    num_fantasies: int = 64
    sequential: bool = False
    seed: Optional[int] = None
    raw_samples: Optional[int] = None


AVAILABLE_ACQUISITIONS: dict[str, type[AcquisitionFunction]] = {
    # Quasi-Monte Carlo acquisitions
    'qsr': qSimpleRegret,
    'qucb': qUpperConfidenceBound,
    'qei': qExpectedImprovement,
    'qlogei': qLogExpectedImprovement,
    'qpi': qProbabilityOfImprovement,
    'qkg': qKnowledgeGradient,
    # Quasi-Monte Carlo acquisitions for noisy problems
    'qnei': qNoisyExpectedImprovement,
    'qlognei': qLogNoisyExpectedImprovement,
    # Multi-objective acquisitions
    'qehvi': qExpectedHypervolumeImprovement,
    'qnehvi': qNoisyExpectedHypervolumeImprovement,
    'qlogehvi': qLogExpectedHypervolumeImprovement,
    'qlognehvi': qLogNoisyExpectedHypervolumeImprovement,
    'qhvkg': qHypervolumeKnowledgeGradient,
}
EXACT_IMPROVEMENT_BASED: list[str] = [
    'qei',
    'qlogei',
    'qpi',
]
NOISY_IMPROVEMENT_BASED: list[str] = [
    'qnei',
    'qlognei',
    'qnehvi',
    'qlognehvi',
]
MULTI_OBJECTIVE_ACQUISITIONS: list[str] = [
    'qehvi',
    'qnehvi',
    'qlogehvi',
    'qlognehvi',
    'qhvkg',
]
MULTI_OBJECTIVE_WITH_PARTITIONING: list[str] = [
    'qehvi',
    'qlogehvi',
]
KNOWLEDGE_GRADIENT: list[str] = [
    'qkg',
    'qhvkg',
]


def default_acquisition(
    composite: bool,
    multi_objective: bool,
    stochastic: bool,
    q: int,
) -> str:
    """Return the default acquisition function for the given optimisation problem.

    Parameters
    ----------
    composite : bool, optional
        Whether the optimisation problem is a composition.
    multi_objective : bool, optional
        Whether the optimisation problem is multi-objective.
    stochastic : bool, optional
        Whether the optimisation problem is stochastic.
    q : int, optional
        Number of candidates to generate.

    Returns
    -------
    str
        Name of the default acquisition function.
    """
    if multi_objective:
        return 'qlognehvi' if (stochastic or q > 2) else 'qlogehvi'
    if stochastic:
        return 'qlognei'
    if composite or q > 1:
        return 'qlogei'
    return 'qlogei'


def get_acquisition(
    model: ObjectiveModel,
    settings: AcquisitionSettings,
    state: OptimisationState,
    pending: Optional[torch.Tensor] = None,
) -> AcquisitionFunction:
    """Build an acquisition function from the given options.

    Parameters
    ----------
    model : ObjectiveModel
        Surrogate model to use for the acquisition function.
    settings : AcquisitionSettings
        Settings for the acquisition function.
    state : OptimisationState
        Current state of the optimisation campaign.
    pending : Optional[torch.Tensor]
        Optional tensor with pending candidates.

    Returns
    -------
    AcquisitionFunction
        Acquisition function.
    """
    if settings.name not in AVAILABLE_ACQUISITIONS:
        raise RuntimeError(f"Unknown acquisition function {settings.name}.")

    # Inject acquisition options, depending on the acquisition
    acq_options = {}

    # Sampler and knowledge gradient fantasied
    sampler = SobolQMCNormalSampler(torch.Size([settings.mc_samples]), seed=settings.seed)
    if settings.name in KNOWLEDGE_GRADIENT:
        acq_options['inner_sampler'] = sampler
        acq_options['num_fantasies'] = settings.num_fantasies
    else:
        acq_options['sampler'] = sampler

    # Objective and multi-objective options
    if settings.name in MULTI_OBJECTIVE_ACQUISITIONS:
        acq_options['objective'] = GenericMCMultiOutputObjective(model.composition_from_raw)
        acq_options['ref_point'] = state.mo_state.partitioning.ref_point
        if settings.name in MULTI_OBJECTIVE_WITH_PARTITIONING:
            acq_options['partitioning'] = state.mo_state.partitioning
    else:
        acq_options['objective'] = GenericMCObjective(model.composition_from_raw)

    # Exact or noisy improvement-based acquisitions
    if settings.name in EXACT_IMPROVEMENT_BASED:
        acq_options['best_f'] = state.best_value
    elif settings.name in NOISY_IMPROVEMENT_BASED:
        acq_options['X_baseline'] = model.inputs
        acq_options['prune_baseline'] = True

    # Other acquisition-specific options
    if settings.name == 'qkg':
        acq_options['current_value'] = state.best_value
    elif settings.name == 'qucb':
        acq_options['beta'] = settings.beta

    # Inject pending candidates
    if pending is not None:
        acq_options['X_pending'] = pending

    # Check if we need to convert the model
    gp = model.gp
    if settings.name in MULTI_OBJECTIVE_ACQUISITIONS:
        gp = batched_to_model_list(gp)

    # Build and return the acquisition function
    cls = AVAILABLE_ACQUISITIONS[settings.name]
    return cls(model=gp, **acq_options)

def optimise_acquisition(
    acq: AcquisitionFunction,
    parameters: ParameterSet,
    model: ObjectiveModel,
    settings: AcquisitionSettings,
    q: int = 1,
    batch_initial_conditions: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise the given acquisition function.

    Parameters
    ----------
    acq : AcquisitionFunction
        Acquisition function to optimise.
    parameters : ParameterSet
        Parameters to optimise.
    model : ObjectiveModel
        Surrogate model to use for the acquisition function.
    settings : AcquisitionSettings
        Settings for the acquisition function.
    q : int
        Number of candidates to generate.
    batch_initial_conditions : Optional[torch.Tensor]
        Optional tensor with initial conditions for the batch.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple with the optimised candidates and their acquisition values.
    """

    # Default values for optional options
    num_discrete = parameters.num_discrete()
    num_continuous = len(parameters) - num_discrete
    raw_samples = settings.raw_samples or max(256, 16 * num_continuous * num_continuous)
    options = {
        "sample_around_best": True,
        "seed": settings.seed,
        "init_batch_limit": settings.batch_size,
    }

    # Build bounds
    bounds = torch.tensor(
        parameters.get_bounds().T, dtype=model.inputs.dtype, device=model.inputs.device
    )

    # Optimise the acquisition function: continuous case
    if num_discrete == 0:
        return optimize_acqf(
            acq,
            bounds=bounds,
            q=q,
            num_restarts=settings.num_restarts,
            raw_samples=raw_samples,
            sequential=settings.sequential,
            options=options,
            batch_initial_conditions=batch_initial_conditions,
        )

    # We have at least one discrete parameter: find all combinations
    features_list = parameters.get_discrete_combinations()

    # Mixed case
    if num_continuous > 0:
        return optimize_acqf_mixed(
            acq,
            bounds=bounds,
            q=q,
            num_restarts=settings.num_restarts,
            fixed_features_list=features_list,
            raw_samples=raw_samples,
            options=options,
            batch_initial_conditions=batch_initial_conditions,
        )

    # We have only discrete parameters
    choices = [list(map(float, f.values())) for f in features_list]
    return optimize_acqf_discrete(
        acq,
        q=q,
        choices=torch.tensor(choices, dtype=model.inputs.dtype, device=model.inputs.device),
        max_batch_size=settings.batch_size,
    )


def build_and_optimise_acquisition(
    model: ObjectiveModel,
    parameters: ParameterSet,
    settings: AcquisitionSettings,
    state: OptimisationState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Utility function to build and optimise an acquisition function.

    Parameters
    ----------
    model : ObjectiveModel
        Surrogate model to use for the acquisition function.
    parameters : ParameterSet
        Parameters to optimise.
    settings : AcquisitionSettings
        Settings for the acquisition function.
    state : OptimisationState
        Current state of the optimisation campaign.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple with the optimised candidates and their acquisition values.
    """
    acquisition = get_acquisition(model, settings, state)
    return optimise_acquisition(acquisition, parameters, model, settings)


def get_best_posterior_mean(
    model: ObjectiveModel,
    parameters: ParameterSet,
    state: OptimisationState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Utility function to get the best point according to the surrogate model's posterior mean.

    Parameters
    ----------
    model : ObjectiveModel
        Surrogate model to use for the acquisition function.
    parameters : ParameterSet
        Parameters to optimise.
    settings : AcquisitionSettings
        Settings for the acquisition function.
    state : OptimisationState
        Current state of the optimisation campaign.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple with the best candidates and their posterior mean values.
    """
    settings = AcquisitionSettings(name='qsr', q=1)
    acq = get_acquisition(model, settings, state)
    return optimise_acquisition(
        acq, parameters, model, settings, q=1
    )
