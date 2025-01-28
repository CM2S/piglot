"""Module for acquisition functions with BoTorch."""
from typing import Dict, Type, Union, Tuple, Any, List, Optional, Callable
from contextlib import ExitStack
from functools import partial
import torch
import gpytorch.settings as gpts
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
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
    MCAcquisitionFunction,
)
from botorch.acquisition.objective import (
    GenericMCObjective,
    MCAcquisitionObjective,
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
from botorch.exceptions.errors import UnsupportedError
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.base import MCSampler
from botorch.utils.sampling import draw_sobol_samples
from piglot.optimisers.botorch.fantasy_acquisitions import (
    qFantasyAcqusition,
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


class ThompsonSampling(AcquisitionFunction):
    """Thompson sampling pseudo-acquisition function."""

    def __init__(
        self,
        model: SingleTaskGP,
        num_points: int,
        objective: Optional[MCAcquisitionObjective] = None,
        seed: Optional[int] = None,
        sampling_strategy: str = "lanczos",
        observation_noise: bool = False,
        mean_of_samples: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.num_points = num_points
        self.objective = objective
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.observation_noise = observation_noise
        self.mean_of_samples = mean_of_samples

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Cannot directly call Thompson sampling's forward.")

    def _sample(self, X: torch.Tensor, q: int) -> torch.Tensor:
        """Draw q Thompson samples from the model.

        Parameters
        ----------
        X : torch.Tensor
            Grid to use for the sampling.
        q : int
            Number of samples to draw.

        Returns
        -------
        torch.Tensor
            Model samples.
        """
        # Set the sampler context
        with ExitStack() as es:
            if self.sampling_strategy == "cholesky":
                es.enter_context(gpts.max_cholesky_size(float("inf")))
            elif self.sampling_strategy == "ciq":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(True))
                es.enter_context(
                    gpts.minres_tolerance(2e-3)
                )  # Controls accuracy and runtime
                es.enter_context(gpts.num_contour_quadrature(15))
            elif self.sampling_strategy == "lanczos":
                es.enter_context(
                    gpts.fast_computations(
                        covar_root_decomposition=True, log_prob=True, solves=True
                    )
                )
                es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(False))
            elif self.sampling_strategy == "rff":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(torch.no_grad())

            # Sample from the posterior
            posterior = self.model.posterior(X, observation_noise=self.observation_noise)
            samples = posterior.rsample(torch.Size([q]))

        return samples

    def draw(self, bounds: torch.Tensor, q: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw q Thompson samples from the model.

        Parameters
        ----------
        bounds : torch.Tensor
            Bounds of the domain.
        q : int
            Number of samples to draw.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Samples and their objective values.
        """
        # Draw the grid for this evaluation
        if self.seed is not None:
            self.seed += self.model.train_targets.shape[0]
        X = draw_sobol_samples(bounds, n=self.num_points, q=1, seed=self.seed).squeeze(-2)

        # Sample from the posterior
        samples = self._sample(X, q)
        if self.mean_of_samples:
            samples = samples.mean(dim=0, keepdim=True)

        # Compute the objective (if needed)
        if self.objective is not None:
            samples = self.objective(samples, X=X)
        if len(samples.shape) > 2:
            samples = samples.squeeze(2)

        # Get the best samples
        vals, idxs = torch.max(samples, dim=-1)
        return X[idxs], vals


class FantasyThompsonSampling(ThompsonSampling):
    """Fantasy-based Thompson sampling pseudo-acquisition function."""

    def __init__(
        self,
        model: SingleTaskGP,
        num_points: int,
        seed: Optional[int] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        inner_sampler: Optional[MCSampler] = None,
        observation_noise: bool = False,
        mean_of_samples: bool = False,
        noiseless_fantasies: bool = False,
        reduction: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            num_points=num_points,
            objective=objective,
            seed=seed,
            observation_noise=observation_noise,
            mean_of_samples=mean_of_samples,
            **kwargs,
        )
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        elif objective is not None and not isinstance(objective, MCAcquisitionObjective):
            raise UnsupportedError(
                "Objectives that are not an `MCAcquisitionObjective` are not supported."
            )
        if reduction is None:
            reduction = torch.mean
        self.noiseless_fantasies = noiseless_fantasies
        self.reduction = reduction
        self.inner_sampler = inner_sampler

    def draw(self, bounds: torch.Tensor, q: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw q Thompson samples from the model.

        Parameters
        ----------
        bounds : torch.Tensor
            Bounds of the domain.
        q : int
            Number of samples to draw.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Samples and their objective values.
        """

        # Draw the grid for this evaluation
        if self.seed is not None:
            self.seed += self.model.train_targets.shape[0]
        X = draw_sobol_samples(bounds, n=self.num_points, q=1, seed=self.seed).squeeze(-2)

        # Sample from the posterior
        samples = self._sample(X, q)

        # Construct the observation noise for this fantasisation
        if self.noiseless_fantasies:
            noise = torch.zeros(X.shape[:-1] + torch.Size([self.model.num_outputs])).to(X)
        if hasattr(self.model, 'noise_prediction'):
            noise = self.model.noise_prediction(X)
        elif isinstance(self.model.likelihood, FixedNoiseGaussianLikelihood):
            if self.num_outputs > 1:
                # make noise ... x n x m
                noise = self.model.likelihood.noise.transpose(-1, -2)
            else:
                noise = self.model.likelihood.noise.unsqueeze(-1)
            noise = noise.mean(dim=-2, keepdim=True)

        # # Build and sample from the fantasy model
        # samples = samples.unsqueeze(-2)
        # fantasy_model = self.model.condition_on_observations(
        #     self.model.transform_inputs(X.unsqueeze(-2).expand(*samples.shape[:-1], X.shape[-1])),
        #     samples,
        #     noise=noise.unsqueeze(-2).expand_as(samples),
        # )
        # fantasy_posterior = fantasy_model.posterior(X, observation_noise=noise)
        # fantasy_samples = self.inner_sampler(fantasy_posterior)
        # fantasy_samples = torch.diagonal(fantasy_samples, dim1=-3, dim2=-2).transpose(-1, -2)

        # Build and sample from the fantasy model
        fantasies = []
        for i, X_curr in enumerate(X):
            curr_samples = samples[:, i, ...]
            fantasy_model = self.model.condition_on_observations(
                X_curr.unsqueeze(0).expand(*curr_samples.shape[:-1], X.shape[-1]),
                curr_samples,
                noise=noise[i].unsqueeze(0).expand_as(curr_samples),
            )
            fantasy_posterior = fantasy_model.posterior(X_curr, observation_noise=noise[i])
            fantasies.append(self.inner_sampler(fantasy_posterior))
        fantasy_samples = torch.stack(fantasies, dim=-2)

        # Compute the objective
        if self.mean_of_samples:
            fantasy_samples = fantasy_samples.mean(dim=0, keepdim=True)
        fantasy_obj = self.objective(samples=fantasy_samples, X=X)
        samples = self.reduction(fantasy_obj, dim=0)

        # Get the best samples
        vals, idxs = torch.max(samples, dim=-1)
        return X[idxs], vals


class RandomSampling(ThompsonSampling):
    """Randomly sample q points from the design space."""

    def __init__(
        self,
        model: SingleTaskGP,
        num_points: int,
        seed: Optional[int] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        observation_noise: bool = False,
        mean_of_samples: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            num_points=num_points,
            objective=objective,
            seed=seed,
            observation_noise=observation_noise,
            mean_of_samples=mean_of_samples,
            **kwargs,
        )

    def draw(self, bounds: torch.Tensor, q: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw q Thompson samples from the model.

        Parameters
        ----------
        bounds : torch.Tensor
            Bounds of the domain.
        q : int
            Number of samples to draw.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Samples and their objective values.
        """
        if self.seed is not None:
            self.seed += self.model.train_targets.shape[0]
        points = draw_sobol_samples(bounds, n=1, q=q, seed=self.seed).squeeze(0)
        return points, 0 * torch.sum(points, dim=-1)


AVAILABLE_ACQUISITIONS: Dict[str, Type[AcquisitionFunction]] = {
    'qrandom': RandomSampling,
    # Thompson sampling
    'qts': ThompsonSampling,
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
    'qfantasy_ts': FantasyThompsonSampling,
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
THOMPSON_SAMPLING: List[str] = [
    'qrandom',
    'qts',
    'qfantasy_ts',
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


def adjust_sampling_distribution(
    acquisition: MCAcquisitionFunction,
    observation_noise: bool,
    mean_response: bool,
) -> MCAcquisitionFunction:
    """Inject observation noise into the acquisition function.

    Parameters
    ----------
    acquisition : MCAcquisitionFunction
        Acquisition function to inject noise into.
    observation_noise : bool
        Whether to use observation noise.
    mean_response : bool
        Whether to use the mean of the posteriors.

    Returns
    -------
    MCAcquisitionFunction
        Acquisition function with observation noise.
    """

    if isinstance(acquisition, (qFantasyAcqusition, ThompsonSampling)):
        acquisition.observation_noise = observation_noise
        acquisition.mean_of_samples = mean_response
        return acquisition

    # TODO: add support for knowledge gradient

    def new_f(
        self: MCAcquisitionFunction,
        X: torch.Tensor,  # pylint: disable=C0103
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior = self.model.posterior(
            X=X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        if mean_response:
            samples = samples.mean(dim=0, keepdim=True)
        obj = self.objective(samples=samples, X=X)
        return samples, obj

    acquisition._get_samples_and_objectives = partial(new_f, acquisition)  # pylint: disable=W0212
    return acquisition


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
    n_dim = dataset.n_dim
    seed = options.get('seed', None)
    beta = options.get('beta', 1.0)
    num_samples = options.get('mc_samples', 256)
    num_fantasies = options.get('num_fantasies', 64)
    risk_name = options.get('risk_measure', None)
    sampler = SobolQMCNormalSampler(torch.Size([num_samples]), seed=seed)
    thompson_grid = options.get('thompson_grid', max(128, min(1024, 16 * n_dim * n_dim)))

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
        if 'qfantasy' in name or 'qrandom' in name:
            acq_class = partial(make_risk_acquisition, cls=acq_class, risk_measure=risk_measure)
        elif name == 'qsr':
            acq_class = partial(qSimpleRegretRisk, risk_measure=risk_measure)
        elif name == 'qkg':
            acq_class = partial(qKnowledgeGradientRisk, risk_measure=risk_measure)
        elif name == 'qfantasy_ts':
            acq_class = partial(FantasyThompsonSampling, reduction=risk_measure)
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
    elif name in THOMPSON_SAMPLING:
        acq_options['num_points'] = thompson_grid
        acq_options['seed'] = seed
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
    instance = acq_class(model=model, **acq_options)

    # Adjust the distribution to sample from
    distribution = options.get('distribution', 'mean')
    instance = adjust_sampling_distribution(
        instance,
        observation_noise=distribution == 'observation',
        mean_response=distribution == 'mean_response',
    )
    return instance


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

    # No optimisation required for Thompson sampling
    if isinstance(acq, ThompsonSampling):
        candidates, acq_val = acq.draw(bounds, q)
        return candidates.cpu(), acq_val.cpu()

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
