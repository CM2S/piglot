"""Module for acquisition functions with BoTorch."""
from typing import Dict, Type, Union, Tuple, Any, List, Optional
from functools import partial
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
from botorch.acquisition.objective import GenericMCObjective
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
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from piglot.optimisers.botorch.fantasy_acquisitions import (
    qFantasySimpleRegret,
    qFantasyExpectedImprovement,
    qFantasyNoisyExpectedImprovement,
    qFantasyLogExpectedImprovement,
    qFantasyLogNoisyExpectedImprovement,
)
from piglot.optimisers.botorch.risk_acquisitions import (
    get_risk_measure,
    make_risk_acquisition,
    qSimpleRegretRisk,
    qKnowledgeGradientRisk,
)
from piglot.optimisers.botorch.composition import BoTorchComposition
from piglot.optimisers.botorch.model import SingleTaskGP, PseudoHeteroskedasticSingleTaskGP
from piglot.optimisers.botorch.dataset import BayesDataset
from piglot.parameter import ParameterSet


AVAILABLE_ACQUISITIONS: Dict[str, Type[AcquisitionFunction]] = {
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
    # Fantasy-based acquisitions
    'qfantasy_sr': qFantasySimpleRegret,
    'qfantasy_ei': qFantasyExpectedImprovement,
    'qfantasy_nei': qFantasyNoisyExpectedImprovement,
    'qfantasy_logei': qFantasyLogExpectedImprovement,
    'qfantasy_lognei': qFantasyLogNoisyExpectedImprovement,
    # Multi-objective acquisitions
    'qehvi': qExpectedHypervolumeImprovement,
    'qnehvi': qNoisyExpectedHypervolumeImprovement,
    'qlogehvi': qLogExpectedHypervolumeImprovement,
    'qlognehvi': qLogNoisyExpectedHypervolumeImprovement,
    'qhvkg': qHypervolumeKnowledgeGradient,
}
FANTASY_BASED_ACQUISITIONS: List[str] = [
    'qkg',
    'qfantasy_sr',
    'qfantasy_ei',
    'qfantasy_nei',
    'qfantasy_logei',
    'qfantasy_lognei',
    'qhvkg',
]
EXACT_IMPROVEMENT_BASED: List[str] = [
    'qei',
    'qlogei',
    'qpi',
    'qfantasy_ei',
    'qfantasy_logei',
]
NOISY_IMPROVEMENT_BASED: List[str] = [
    'qnei',
    'qlognei',
    'qfantasy_nei',
    'qfantasy_lognei',
    'qnehvi',
    'qlognehvi',
]
MULTI_OBJECTIVE_ACQUISITIONS: List[str] = [
    'qehvi',
    'qnehvi',
    'qlogehvi',
    'qlognehvi',
    'qhvkg',
]
MULTI_OBJECTIVE_WITH_PARTITIONING: List[str] = [
    'qehvi',
    'qlogehvi',
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
        return 'qlognehvi' if (stochastic or q > 1) else 'qlogehvi'
    if stochastic:
        return 'qlognei'
    if composite or q > 1:
        return 'qlogei'
    return 'qlogei'


def get_acquisition(
    name: str,
    model: Union[SingleTaskGP, PseudoHeteroskedasticSingleTaskGP],
    dataset: BayesDataset,
    composition: BoTorchComposition,
    best: torch.Tensor = None,
    **options: Any,
) -> AcquisitionFunction:
    """Build an acquisition function from the given options.

    Parameters
    ----------
    name : str
        Name of the acquisition function.
    model : Union[SingleTaskGP, PseudoHeteroskedasticSingleTaskGP]
        Model to use for the acquisition.
    dataset : BayesDataset
        Dataset to use for the acquisition.
    composition : BoTorchComposition
        Composition object to use for the acquisition.
    best : torch.Tensor
        Best value in the dataset, optional.
    options : Any
        Additional options for the acquisition.

    Returns
    -------
    AcquisitionFunction
        Acquisition function.
    """

    # Default values for optional options
    seed = options.get('seed', None)
    beta = options.get('beta', 1.0)
    num_samples = options.get('mc_samples', 256)
    num_fantasies = options.get('num_fantasies', 64)
    risk_name = options.get('risk_measure', None)
    sampler = SobolQMCNormalSampler(torch.Size([num_samples]), seed=seed)

    # Find the best value in the dataset if not provided
    if best is None:
        best = torch.max(composition.from_original_samples(dataset.values, dataset.params)).item()

    # Sanitise multi-objective options
    if name in ('qehvi', 'qlogehvi', 'qnehvi', 'qlognehvi', 'qhvkg'):
        if 'ref_point' not in options:
            raise RuntimeError("Missing reference point for multi-objective acquisition")
        if name in ('qehvi', 'qlogehvi') and 'partitioning' not in options:
            raise RuntimeError("Missing partitioning for multi-objective acquisition")

    # Delegate to the correct acquisition function and handle risk measures
    acq_class = AVAILABLE_ACQUISITIONS[name]
    if risk_name is not None:
        risk_measure = get_risk_measure(risk_name, **options)
        if 'qfantasy' in name:
            acq_class = partial(make_risk_acquisition, cls=acq_class, risk_measure=risk_measure)
        elif name == 'qsr':
            acq_class = partial(qSimpleRegretRisk, risk_measure=risk_measure)
        elif name == 'qkg':
            acq_class = partial(qKnowledgeGradientRisk, risk_measure=risk_measure)
        else:
            raise RuntimeError(
                "Only fantasy-based acquisitions, simple regret "
                "and knowledge gradient support risk measures"
            )

    # Inject acquisition options, depending on the acquisition
    acq_options = {}

    # Sampler options
    if name in FANTASY_BASED_ACQUISITIONS:
        acq_options['num_fantasies'] = num_fantasies
        acq_options['inner_sampler'] = sampler
    else:
        acq_options['sampler'] = sampler

    # Objective and multi-objective options
    if name in MULTI_OBJECTIVE_ACQUISITIONS:
        acq_options['objective'] = GenericMCMultiOutputObjective(composition.from_model_samples)
        acq_options['ref_point'] = options['ref_point']
        if name in MULTI_OBJECTIVE_WITH_PARTITIONING:
            acq_options['partitioning'] = options['partitioning']
    else:
        acq_options['objective'] = GenericMCObjective(composition.from_model_samples)

    # Exact or noisy improvement-based acquisitions
    if name in EXACT_IMPROVEMENT_BASED:
        acq_options['best_f'] = best
    elif name in NOISY_IMPROVEMENT_BASED:
        acq_options['X_baseline'] = dataset.params
        acq_options['prune_baseline'] = True

    # Other acquisition-specific options
    if name == 'qkg':
        acq_options['current_value'] = best
    elif name == 'qucb':
        acq_options['beta'] = beta

    # Build and return the acquisition function
    if name not in AVAILABLE_ACQUISITIONS:
        raise RuntimeError(f"Unknown acquisition function {name}.")
    return acq_class(model=model, **acq_options)


def optimise_acquisition(
    acq: AcquisitionFunction,
    parameters: ParameterSet,
    dataset: BayesDataset,
    max_q: Optional[int] = None,
    **options: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimise the given acquisition function.

    Parameters
    ----------
    acquisition : AcquisitionFunction
        Acquisition function to optimise.
    parameters : ParameterSet
        Parameters to optimise.
    dataset : BayesDataset
        Dataset to use for the optimisation.
    max_q : int
        Maximum number of candidates to generate, optional.
    options : Any
        Additional options for the optimisation.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple with the optimised candidates and their acquisition values.
    """

    # Default values for optional options
    n_dim = len(parameters)
    q = options.get('q', 1)
    seed = options.get('seed', None)
    num_restarts = options.get('num_restarts', 12)
    raw_samples = options.get('raw_samples', max(256, 16 * n_dim * n_dim))
    sequential = options.get('sequential', False)
    batch_size = options.get('batch_size', 128)

    # Build bounds
    bounds = torch.tensor(
        [[p.lbound for p in parameters], [p.ubound for p in parameters]],
        dtype=dataset.dtype,
        device=dataset.device,
    )

    # Optimise the acquisition function
    candidates, acq_val = optimize_acqf(
        acq,
        bounds=bounds,
        q=q if max_q is None else min(q, max_q),
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=sequential,
        options={
            "sample_around_best": True,
            "seed": seed,
            "init_batch_limit": batch_size,
        },
    )
    return candidates.cpu(), acq_val.cpu()


def build_and_optimise_acquisition(
    name: str,
    model: Union[SingleTaskGP, PseudoHeteroskedasticSingleTaskGP],
    dataset: BayesDataset,
    parameters: ParameterSet,
    composition: BoTorchComposition,
    **options: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Utility function to build and optimise an acquisition function.

    Parameters
    ----------
    name : str
        Name of the acquisition function.
    model : Union[SingleTaskGP, PseudoHeteroskedasticSingleTaskGP]
        Model to use for the acquisition.
    dataset : BayesDataset
        Dataset to use for the acquisition.
    parameters : ParameterSet
        Parameters to optimise.
    composition : BoTorchComposition
        Composition object to use for the acquisition.
    options : Any
        Additional options for the acquisition.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple with the optimised candidates and their acquisition values.
    """
    acquisition = get_acquisition(name, model, dataset, composition, **options)
    return optimise_acquisition(acquisition, parameters, dataset, **options)
