"""Main optimiser classes for using BoTorch with piglot"""
from typing import Tuple, List, Union, Type
from multiprocessing.pool import ThreadPool as Pool
import os
import warnings
import dataclasses
import numpy as np
import torch
from scipy.stats import qmc
from gpytorch.mlls import ExactMarginalLogLikelihood
import botorch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.converter import batched_to_model_list
from botorch.acquisition import (
    AcquisitionFunction,
    UpperConfidenceBound,
    qUpperConfidenceBound,
    ExpectedImprovement,
    qExpectedImprovement,
    ProbabilityOfImprovement,
    qProbabilityOfImprovement,
    LogExpectedImprovement,
    qLogExpectedImprovement,
    qNoisyExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qKnowledgeGradient,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
    qHypervolumeKnowledgeGradient,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.sampling import SobolQMCNormalSampler
from piglot.objective import (
    Objective,
    GenericObjective,
    ObjectiveResult,
)
from piglot.optimiser import Optimiser
from piglot.optimisers.botorch.dataset import BayesDataset


AVAILABLE_ACQUISITIONS = [
    # Analytical acquisitions
    'ucb', 'ei', 'logei', 'pi',
    # Quasi-Monte Carlo acquisitions
    'qucb', 'qei', 'qlogei', 'qpi', 'qkg',
    # Analytical and quasi-Monte Carlo acquisitions for noisy problems
    'qnei', 'qlognei',
    # Multi-objective acquisitions
    'qehvi', 'qnehvi', 'qlogehvi', 'qlognehvi', 'qhvkg'
]


def default_acquisition(
    composite: bool,
    multi_objective: bool,
    noisy: bool,
    q: int,
) -> str:
    """Return the default acquisition function for the given optimisation problem.

    Parameters
    ----------
    composite : bool, optional
        Whether the optimisation problem is a composition.
    multi_objective : bool, optional
        Whether the optimisation problem is multi-objective.
    noisy : bool, optional
        Whether the optimisation problem is noisy.
    q : int, optional
        Number of candidates to generate.

    Returns
    -------
    str
        Name of the default acquisition function.
    """
    if multi_objective:
        return 'qlognehvi' if noisy else 'qlogehvi'
    if noisy:
        return 'qlognei'
    if composite or q > 1:
        return 'qlogei'
    return 'logei'


def fit_mll_pytorch_loop(mll: ExactMarginalLogLikelihood, n_iters: int = 100) -> None:
    """Fit a GP model using a PyTorch optimisation loop.

    Parameters
    ----------
    mll : ExactMarginalLogLikelihood
        Marginal log-likelihood to optimise.
    n_iters : int, optional
        Number of iterations to optimise for, by default 100
    """
    mll.train()
    mll.model.likelihood.train()
    optimizer = torch.optim.Adam(mll.model.parameters(), lr=0.1)
    for _ in range(n_iters):
        optimizer.zero_grad()
        output = mll.model(mll.model.train_inputs[0])
        loss = -torch.mean(mll(output, mll.model.train_targets))
        loss.backward()
        optimizer.step()
    mll.model.eval()
    mll.model.likelihood.eval()


@dataclasses.dataclass
class MultiObjectiveData:
    """Data class for multi-objective optimisation."""
    ref_point: torch.Tensor
    mapping: List[List[int]]
    partitioning: NondominatedPartitioning
    partitioning_type: Type[NondominatedPartitioning]

    def __init__(
        self,
        mapping: List[List[int]],
        partitioning_type: Type[NondominatedPartitioning],
        ref_point: torch.Tensor = None,
        partitioning: NondominatedPartitioning = None,
    ) -> None:
        self.ref_point = ref_point
        self.mapping = mapping
        self.partitioning = partitioning
        self.partitioning_type = partitioning_type


class BayesianBoTorch(Optimiser):
    """Driver for optimisation using BoTorch."""

    def __init__(
        self,
        objective: Objective,
        n_initial: int = 8,
        n_test: int = 0,
        acquisition: str = None,
        beta: float = 1.0,
        noisy: float = False,
        q: int = 1,
        seed: int = 1,
        load_file: str = None,
        export: str = None,
        device: str = 'cpu',
    ) -> None:
        if not isinstance(objective, GenericObjective):
            raise RuntimeError("Bayesian optimiser requires a GenericObjective")
        super().__init__('BoTorch', objective)
        self.objective = objective
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.beta = beta
        self.noisy = bool(noisy) or objective.stochastic
        self.q = q
        self.seed = seed
        self.load_file = load_file
        self.export = export
        self.n_test = n_test
        self.device = device
        if acquisition is None:
            self.acquisition = default_acquisition(
                objective.composition,
                objective.multi_objective,
                self.noisy,
                self.q,
            )
        elif self.acquisition not in AVAILABLE_ACQUISITIONS:
            raise RuntimeError(f"Unkown acquisition function {self.acquisition}")
        if not self.acquisition.startswith('q') and self.q != 1:
            raise RuntimeError("Can only use q != 1 for quasi-Monte Carlo acquisitions")
        self.mo_data: MultiObjectiveData = None
        torch.set_num_threads(1)

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective

        Parameters
        ----------
        objective : Objective
            Objective to optimise
        """

    def _acq_func(
        self,
        dataset: BayesDataset,
        model: Model,
        n_dim: int,
    ) -> Tuple[AcquisitionFunction, int, int]:
        if self.objective.multi_objective:
            return self._build_acquisition_mo(dataset, model, n_dim)
        if self.objective.composition:
            return self._build_acquisition_composite(dataset, model, n_dim)
        return self._build_acquisition_scalar(dataset, model, n_dim)

    def _build_model(self, std_dataset: BayesDataset) -> Model:
        # Fetch data
        params = std_dataset.params
        values = std_dataset.values
        variances = std_dataset.variances
        # Clamp variances to prevent warnings from GPyTorch
        variances = torch.clamp_min(variances, 1e-6)
        # Initialise model instance depending on noise setting
        model = SingleTaskGP(params, values, train_Yvar=None if self.noisy else variances)
        # Fit the GP (in case of trouble, fall back to an Adam-based optimiser)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except botorch.exceptions.ModelFittingError:
            warnings.warn('Optimisation of the MLL failed, falling back to PyTorch optimiser')
            fit_mll_pytorch_loop(mll)
        return batched_to_model_list(model) if self.objective.multi_objective else model

    def _get_candidates(
            self,
            n_dim,
            dataset: BayesDataset,
            test_dataset: BayesDataset,
            ) -> Tuple[np.ndarray, float]:

        # Build model on unit-cube and standardised data
        std_dataset = dataset.standardised()
        model = self._build_model(std_dataset)

        # Evaluate GP performance with the test dataset
        cv_error = None
        if self.n_test > 0:
            std_test_params = test_dataset.normalise(test_dataset.params)
            std_test_values, _ = dataset.standardise(
                test_dataset.values,
                test_dataset.variances,
            )
            with torch.no_grad():
                posterior = model.posterior(std_test_params)
                cv_error = (posterior.mean - std_test_values).square().mean().item()

        # Build the acquisition function
        acq, num_restarts, raw_samples = self._acq_func(dataset, model, n_dim)

        # Optimise acquisition to find next candidate(s)
        candidates, _ = optimize_acqf(
            acq,
            bounds=torch.stack((std_dataset.lbounds, std_dataset.ubounds)),
            q=self.q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={
                "sample_around_best": True,
                "seed": self.seed,
            },
        )

        # Re-map to original space
        for i in range(self.q):
            candidates[i, :] = dataset.denormalise(candidates[i, :])
        return candidates.cpu().numpy(), cv_error

    def _eval_candidates(self, candidates: np.ndarray) -> List[ObjectiveResult]:
        # Single candidate case
        if self.q == 1:
            return [self.objective(candidate) for candidate in candidates]
        # Multi-candidate: run cases in parallel
        with Pool(self.q) as pool:
            results = pool.map(lambda x: self.objective(x, concurrent=True), candidates)
        return results

    def _get_random_points(
            self,
            n_points: int,
            n_dim: int,
            seed: int,
            bound: np.ndarray,
            ) -> List[np.ndarray]:
        points = qmc.Sobol(n_dim, seed=seed).random(n_points)
        return [point * (bound[:, 1] - bound[:, 0]) + bound[:, 0] for point in points]

    def _result_to_dataset(self, result: ObjectiveResult) -> Tuple[np.ndarray, np.ndarray]:
        if self.objective.multi_objective:
            values = np.concatenate(result.values) if self.objective.composition else result.values
            variances = np.zeros_like(values)
        elif self.objective.composition:
            values = np.concatenate(result.values)
            if self.objective.stochastic:
                variances = np.concatenate(result.variances)
            else:
                variances = np.zeros_like(values)
        else:
            if self.objective.stochastic:
                values, variances = ObjectiveResult.scalarise_stochastic(result)
            else:
                values, variances = (ObjectiveResult.scalarise(result), 0.0)
            values, variances = np.array([values]), np.array([variances])
        return values, variances

    def _value_to_scalar(self, value: Union[np.ndarray, torch.Tensor]) -> float:
        if self.objective.multi_objective:
            return value[0].item()
        if self.objective.composition:
            if isinstance(value, np.ndarray):
                return self.objective.composition.composition(value)
            return self.objective.composition.composition_torch(value).cpu().item()
        return value.item()

    def _init_mo_data(self, init_result: ObjectiveResult, n_outputs: int) -> None:
        # Select the partitioning scheme according to the number of objectives
        # See https://github.com/pytorch/botorch/pull/846
        partitioning_type = FastNondominatedPartitioning
        if n_outputs > 4:
            partitioning_type = NondominatedPartitioning
        # Under composite multi-objective optimisation, store the composition mapping
        mapping = []
        if self.objective.composition:
            pos = 0
            for length in [len(result) for result in init_result.values]:
                mapping.append(list(range(pos, pos + length)))
                pos += length
        self.mo_data = MultiObjectiveData(mapping, partitioning_type)

    def _update_mo_data(self, dataset: BayesDataset):
        # TODO: check how to properly handle the ref_point and best_value
        y_points = self.__mo_mc_objective(dataset.values)
        self.mo_data.ref_point = torch.min(y_points, dim=0).values
        self.mo_data.partitioning = self.mo_data.partitioning_type(
            ref_point=self.mo_data.ref_point,
            Y=y_points,
        )
        hypervolume = self.mo_data.partitioning.compute_hypervolume().item()
        pareto = self.mo_data.partitioning.pareto_Y
        with open(os.path.join(self.output_dir, "pareto_front"), 'w', encoding='utf8') as file:
            file.write('\t'.join(
                [f'{"Objective_" + str(i + 1):>15}' for i in range(pareto.shape[1])]) + '\n'
            )
            for point in pareto:
                file.write('\t'.join([f'{-x.item():>15.8f}' for x in point]) + '\n')
        # TODO: after updating the parameter set, write the parameters and hash for each point
        return -np.log(hypervolume)

    def _build_acquisition_scalar(
        self,
        dataset: BayesDataset,
        model: Model,
        n_dim: int,
    ) -> Tuple[AcquisitionFunction, int, int]:
        # Default values for multi-restart optimisation
        num_restarts = 12
        raw_samples = max(256, 16 * n_dim * n_dim)
        sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)
        # Delegate acquisition building
        std_dataset = dataset.standardised()
        best = torch.min(std_dataset.values).item()
        mc_objective = GenericMCObjective(lambda vals, X: -vals.squeeze(-1))
        if self.acquisition == 'ucb':
            acq = UpperConfidenceBound(
                model,
                self.beta,
                maximize=False,
            )
        elif self.acquisition == 'ei':
            acq = ExpectedImprovement(
                model,
                best,
                maximize=False,
            )
        elif self.acquisition == 'logei':
            acq = LogExpectedImprovement(
                model,
                best,
                maximize=False,
            )
        elif self.acquisition == 'pi':
            acq = ProbabilityOfImprovement(
                model,
                best,
                maximize=False,
            )
        elif self.acquisition == 'qucb':
            acq = qUpperConfidenceBound(
                model,
                self.beta,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qei':
            acq = qExpectedImprovement(
                model,
                best,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qlogei':
            acq = qLogExpectedImprovement(
                model,
                best,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qnei':
            acq = qNoisyExpectedImprovement(
                model,
                std_dataset.params,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qlognei':
            acq = qLogNoisyExpectedImprovement(
                model,
                std_dataset.params,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qpi':
            acq = qProbabilityOfImprovement(
                model,
                best,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qkg':
            # Knowledge gradient is quite expensive: use less samples
            num_restarts = 6
            raw_samples = 128
            sampler = SobolQMCNormalSampler(torch.Size([64]), seed=self.seed)
            acq = qKnowledgeGradient(model, sampler=sampler, objective=mc_objective)
        else:
            raise RuntimeError(f"Unknown acquisition {self.acquisition}")
        return acq, num_restarts, raw_samples

    def __mo_mc_objective(
        self,
        vals: torch.Tensor,
    ) -> torch.Tensor:
        # If no composition, simply return the values
        if not self.objective.composition:
            return -vals.squeeze(-1)
        # Composition: start by evaluating each objective
        # The objective mappings are used to select the right columns for each objective
        # We need to reshape the input to support arbitrary shapes for every dimension but the last
        vals_view = vals.view(-1, vals.shape[-1])
        objectives = [
            self.objective.composition.composition_torch(vals_view[:, indices])
            for indices in self.mo_data.mapping
        ]
        # Build the proper result tensor: stack the objectives and reshape to the original shape
        result = torch.stack(objectives, dim=-1)
        return -result.view(vals.shape[:-1] + result.shape[-1:])

    def _build_acquisition_mo(
        self,
        dataset: BayesDataset,
        model: Model,
        n_dim: int,
    ) -> Tuple[AcquisitionFunction, int, int]:
        # Default values for multi-restart optimisation
        num_restarts = 12
        raw_samples = max(256, 16 * n_dim * n_dim)
        sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)
        # Prepare standardised dataset
        std_dataset = dataset.standardised()
        _, y_avg, y_std = dataset.get_obervation_stats()
        # Delegate acquisition building
        mc_objective = GenericMCMultiOutputObjective(
            lambda vals, X: self.__mo_mc_objective(
                dataset.expand_observations(vals * y_std + y_avg)
            )
        )
        if self.acquisition == 'qehvi':
            acq = qExpectedHypervolumeImprovement(
                model,
                self.mo_data.ref_point,
                self.mo_data.partitioning,
                objective=mc_objective,
                sampler=sampler,
            )
        elif self.acquisition == 'qlogehvi':
            acq = qLogExpectedHypervolumeImprovement(
                model,
                self.mo_data.ref_point,
                self.mo_data.partitioning,
                objective=mc_objective,
                sampler=sampler,
            )
        elif self.acquisition == 'qnehvi':
            acq = qNoisyExpectedHypervolumeImprovement(
                model,
                self.mo_data.ref_point,
                std_dataset.params,
                objective=mc_objective,
                sampler=sampler,
            )
        elif self.acquisition == 'qlognehvi':
            acq = qLogNoisyExpectedHypervolumeImprovement(
                model,
                self.mo_data.ref_point,
                std_dataset.params,
                objective=mc_objective,
                sampler=sampler,
            )
        elif self.acquisition == 'qhvkg':
            acq = qHypervolumeKnowledgeGradient(
                model,
                self.mo_data.ref_point,
                objective=mc_objective,
            )
        return acq, num_restarts, raw_samples

    def _build_acquisition_composite(
        self,
        dataset: BayesDataset,
        model: Model,
        n_dim: int,
    ) -> Tuple[AcquisitionFunction, int, int]:
        # Default values for multi-restart optimisation
        num_restarts = 12
        raw_samples = max(256, 16 * n_dim * n_dim)
        sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)
        # Build composite MC objective
        _, y_avg, y_std = dataset.get_obervation_stats()
        mc_objective = GenericMCObjective(
            lambda vals, X: -self.objective.composition.composition_torch(
                dataset.expand_observations(vals * y_std + y_avg)
            )
        )
        # Delegate acquisition building
        std_dataset = dataset.standardised()
        best = torch.max(mc_objective(std_dataset.values)).item()
        if self.acquisition == 'qucb':
            acq = qUpperConfidenceBound(
                model,
                self.beta,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qei':
            acq = qExpectedImprovement(
                model,
                best,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qlogei':
            acq = qLogExpectedImprovement(
                model,
                best,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qnei':
            acq = qNoisyExpectedImprovement(
                model,
                std_dataset.params,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qlognei':
            acq = qLogNoisyExpectedImprovement(
                model,
                std_dataset.params,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qpi':
            acq = qProbabilityOfImprovement(
                model,
                best,
                sampler=sampler,
                objective=mc_objective,
            )
        elif self.acquisition == 'qkg':
            # Knowledge gradient is quite expensive: use less samples
            num_restarts = 6
            raw_samples = 128
            sampler = SobolQMCNormalSampler(torch.Size([64]), seed=self.seed)
            acq = qKnowledgeGradient(model, sampler=sampler, objective=mc_objective)
        else:
            raise RuntimeError(f"Unknown acquisition {self.acquisition}")
        return acq, num_restarts, raw_samples

    def _optimise(
        self,
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ):
        """
        Parameters
        ----------
        func : callable
            function to optimize
        n_dim : integer
            dimension, i.e., number of parameters to optimize
        n_iter : integer
            maximum number of iterations
        bound : array
            first column corresponding to the lower bound, and second column to the
            upper bound
        init_shot : list
            initial shot for the optimization problem

        Returns
        -------
        best_value : float
            best loss function value
        best_solution : list
            best parameter solution
        """

        # Evaluate initial shot and use it to infer number of dimensions
        init_result = self.objective(init_shot)
        init_values, init_variances = self._result_to_dataset(init_result)
        n_outputs = len(init_values)

        # Addtional tasks for multi-objective optimisation
        if self.objective.multi_objective:
            self._init_mo_data(init_result, n_outputs)

        # Build initial dataset with the initial shot
        dataset = BayesDataset(n_dim, n_outputs, bound, export=self.export, device=self.device)
        dataset.push(init_shot, init_values, init_variances)

        # If requested, sample some random points before starting (in parallel if possible)
        random_points = self._get_random_points(self.n_initial, n_dim, self.seed, bound)
        results = self._eval_candidates(random_points)
        for i, result in enumerate(results):
            values, variances = self._result_to_dataset(result)
            dataset.push(random_points[i], values, variances)

        # If specified, load data from the input file
        if self.load_file:
            dataset.load(self.load_file)

        # Build test dataset (in parallel if possible)
        test_dataset = BayesDataset(n_dim, n_outputs, bound, device=self.device)
        if self.n_test > 0:
            test_points = self._get_random_points(self.n_test, n_dim, self.seed + 1, bound)
            test_results = self._eval_candidates(test_points)
            for i, result in enumerate(test_results):
                values, variances = self._result_to_dataset(result)
                test_dataset.push(test_points[i], values, variances)

        # Find current best point to return to the driver
        best_params, best_result = dataset.min(self._value_to_scalar)
        best_value = self._value_to_scalar(best_result)
        if self.objective.multi_objective:
            best_value = self._update_mo_data(dataset)
            best_params = None
        self._progress_check(0, best_value, best_params)

        # Optimisation loop
        for i_iter in range(n_iter):
            # Generate candidates: catch CUDA OOM errors and fall back to CPU
            try:
                candidates, cv_error = self._get_candidates(n_dim, dataset, test_dataset)
            except torch.cuda.OutOfMemoryError:
                warnings.warn('CUDA out of memory: falling back to CPU')
                self.device = 'cpu'
                dataset = dataset.to(self.device)
                test_dataset = test_dataset.to(self.device)
                candidates, cv_error = self._get_candidates(n_dim, dataset, test_dataset)

            # Evaluate candidates (in parallel if possible)
            results = self._eval_candidates(candidates)

            # Update dataset
            values_batch = []
            for i, result in enumerate(results):
                values, variances = self._result_to_dataset(result)
                values_batch.append(self._value_to_scalar(values))
                dataset.push(candidates[i, :], values, variances)

            # Find best observation for this batch
            best_idx = np.argmin(values_batch)
            best_value = values_batch[best_idx]
            best_params = candidates[best_idx, :]
            if self.objective.multi_objective:
                best_value = self._update_mo_data(dataset)
                best_params = None

            # Update progress (with extra data if available)
            extra = None
            if cv_error:
                extra = f'Val. {cv_error:6.4}'
                if self.objective.multi_objective:
                    extra += f'  Num Pareto: {self.mo_data.partitioning.pareto_Y.shape[0]}'
            elif self.objective.multi_objective:
                extra = f'Num Pareto: {self.mo_data.partitioning.pareto_Y.shape[0]}'
            if self._progress_check(i_iter + 1, best_value, best_params, extra_info=extra):
                break

        # Return optimisation result
        best_params, best_result = dataset.min(self._value_to_scalar)
        best_result = self._value_to_scalar(best_result)
        if self.objective.multi_objective:
            best_result = self._update_mo_data(dataset)
            best_params = None
        return best_params, best_result
