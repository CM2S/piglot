"""Main optimiser classes for using BoTorch with piglot"""
from typing import Tuple, List, Union, Type, Dict, Callable
from multiprocessing.pool import ThreadPool as Pool
import os
import warnings
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
from botorch.models.transforms import Normalize
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
from botorch.acquisition.objective import GenericMCObjective, UnstandardizePosteriorTransform
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
    qHypervolumeKnowledgeGradient,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
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


AVAILABLE_ACQUISITIONS: Dict[str, Type[AcquisitionFunction]] = {
    # Analytical acquisitions
    'ucb': UpperConfidenceBound,
    'ei': ExpectedImprovement,
    'logei': LogExpectedImprovement,
    'pi': ProbabilityOfImprovement,
    # Quasi-Monte Carlo acquisitions
    'qucb': qUpperConfidenceBound,
    'qei': qExpectedImprovement,
    'qlogei': qLogExpectedImprovement,
    'qpi': qProbabilityOfImprovement,
    'qkg': qKnowledgeGradient,
    # Analytical and quasi-Monte Carlo acquisitions for noisy problems
    'qnei': qNoisyExpectedImprovement,
    'qlognei': qLogNoisyExpectedImprovement,
    # Multi-objective acquisitions
    'qehvi': qExpectedHypervolumeImprovement,
    'qnehvi': qNoisyExpectedHypervolumeImprovement,
    'qlogehvi': qLogExpectedHypervolumeImprovement,
    'qlognehvi': qLogNoisyExpectedHypervolumeImprovement,
    'qhvkg': qHypervolumeKnowledgeGradient
}


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
        reference_point: List[float] = None,
        nadir_scale: float = 0.1,
    ) -> None:
        if not isinstance(objective, GenericObjective):
            raise RuntimeError("Bayesian optimiser requires a GenericObjective")
        if bool(noisy) and objective.stochastic:
            warnings.warn("Noisy setting with stochastic objective - ignoring objective variance")
        super().__init__('BoTorch', objective)
        self.objective = objective
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.beta = beta
        self.noisy = bool(noisy)
        self.q = q
        self.seed = seed
        self.load_file = load_file
        self.export = export
        self.n_test = n_test
        self.device = device
        self.partitioning: FastNondominatedPartitioning = None
        self.adjusted_ref_point = reference_point is None
        self.ref_point = None if reference_point is None else -torch.tensor(reference_point)
        self.nadir_scale = nadir_scale
        if acquisition is None:
            self.acquisition = default_acquisition(
                objective.composition,
                objective.multi_objective,
                bool(noisy) or objective.stochastic,
                self.q,
            )
        elif self.acquisition not in AVAILABLE_ACQUISITIONS:
            raise RuntimeError(f"Unkown acquisition function {self.acquisition}")
        if not self.acquisition.startswith('q') and self.q != 1:
            raise RuntimeError("Can only use q != 1 for quasi-Monte Carlo acquisitions")
        torch.set_num_threads(1)

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective

        Parameters
        ----------
        objective : Objective
            Objective to optimise
        """

    def _build_model(self, dataset: BayesDataset) -> Model:
        # Transform outcomes and clamp variances to prevent warnings from GPyTorch
        values, variances = dataset.transform_outcomes(dataset.values, dataset.variances)
        variances = torch.clamp_min(variances, 1e-6)
        # Initialise model instance depending on noise setting
        model = SingleTaskGP(
            dataset.params,
            values,
            train_Yvar=None if self.noisy else variances,
            input_transform=Normalize(d=dataset.params.shape[-1]),
        )
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
            bounds: np.ndarray,
            dataset: BayesDataset,
            test_dataset: BayesDataset,
            ) -> Tuple[np.ndarray, float]:

        # Build model
        model = self._build_model(dataset)

        # Evaluate GP performance with the test dataset
        cv_error = None
        if self.n_test > 0:
            std_test_values, _ = dataset.transform_outcomes(
                test_dataset.values,
                test_dataset.variances,
            )
            with torch.no_grad():
                posterior = model.posterior(test_dataset.params)
                cv_error = (posterior.mean - std_test_values).square().mean().item()

        # Build the acquisition function
        acq, num_restarts, raw_samples = self._acq_func(dataset, model, n_dim)

        # Optimise acquisition to find next candidate(s)
        candidates, _ = optimize_acqf(
            acq,
            bounds=torch.from_numpy(bounds.T).to(self.device).to(dataset.dtype),
            q=self.q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={
                "sample_around_best": True,
                "seed": self.seed,
            },
        )

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
        if self.objective.composition or self.objective.multi_objective:
            values = result.values
            variances = result.variances if self.objective.stochastic else np.zeros_like(values)
        else:
            if self.objective.stochastic:
                values, variances = ObjectiveResult.scalarise_stochastic(result)
            else:
                values, variances = (ObjectiveResult.scalarise(result), 0.0)
            values, variances = np.array([values]), np.array([variances])
        return values, variances

    def _value_to_scalar(
        self,
        value: Union[np.ndarray, torch.Tensor],
        params: Union[np.ndarray, torch.Tensor],
    ) -> float:
        if self.objective.multi_objective:
            raise RuntimeError("Cannot convert multi-objective value to scalar")
        if self.objective.composition:
            if isinstance(value, np.ndarray):
                return self.objective.composition.composition(value, params)
            return self.objective.composition.composition_torch(value, params).cpu().item()
        return value.item()

    def _composition(self, vals: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Just negate the untransformed outcomes if no composition is available
        if not self.objective.composition:
            return -vals.squeeze(-1)
        # Otherwise, use the composition function
        return -self.objective.composition.composition_torch(vals, params)

    def _update_mo_data(self, dataset: BayesDataset) -> float:
        y_points = self._composition(dataset.values, dataset.params)
        # Compute reference point if needed
        if self.adjusted_ref_point:
            nadir = torch.min(y_points, dim=0).values
            self.ref_point = nadir - self.nadir_scale * (torch.max(y_points, dim=0).values - nadir)
        # Update partitioning and Pareto front
        self.partitioning = FastNondominatedPartitioning(self.ref_point, Y=y_points)
        hypervolume = self.partitioning.compute_hypervolume().item()
        pareto = self.partitioning.pareto_Y
        # Map each Pareto point to the original parameter space
        param_indices = [
            torch.argmin((y_points - pareto[i, :]).norm(dim=1)).item()
            for i in range(pareto.shape[0])
        ]
        # Dump the Pareto front to a file
        with open(os.path.join(self.output_dir, "pareto_front"), 'w', encoding='utf8') as file:
            # Write header
            num_obj = pareto.shape[1]
            file.write('\t'.join([f'{"Objective_" + str(i + 1):>15}' for i in range(num_obj)]))
            file.write('\t' + '\t'.join([f'{param.name:>15}' for param in self.parameters]) + '\n')
            # Write each point
            for i, idx in enumerate(param_indices):
                file.write('\t'.join([f'{-x.item():>15.8f}' for x in pareto[i, :]]) + '\t')
                file.write('\t'.join([f'{x.item():>15.8f}' for x in dataset.params[idx, :]]) + '\n')
        return -np.log(hypervolume)

    def _acq_func(
        self,
        dataset: BayesDataset,
        model: Model,
        n_dim: int,
    ) -> Tuple[AcquisitionFunction, int, int]:
        # Default values for multi-restart optimisation
        num_restarts = 12
        raw_samples = max(256, 16 * n_dim * n_dim)
        sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)

        # Find best value for the acquisition
        best = torch.max(self._composition(dataset.values, dataset.params)).item()

        # For analytical acquisitions, manually compute the destandardisation constants
        y_avg = torch.mean(dataset.values, dim=0)
        y_std = torch.std(dataset.values, dim=0)

        # Build composite MC objective
        def mc_objective(vals: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            return self._composition(dataset.untransform_outcomes(vals), X)

        # Delegate to the correct acquisition function
        # The arguments for each acquisition are different, so we group them into families
        acq_class = AVAILABLE_ACQUISITIONS[self.acquisition]
        # Analytical acquisitions
        if self.acquisition == 'ucb':
            acq = acq_class(
                model,
                self.beta,
                maximize=False,
                posterior_transform=UnstandardizePosteriorTransform(y_avg, y_std),
            )
        elif self.acquisition in ('ei', 'logei', 'pi'):
            acq = acq_class(
                model,
                best,
                maximize=False,
                posterior_transform=UnstandardizePosteriorTransform(y_avg, y_std),
            )
        # Quasi-Monte Carlo acquisitions
        elif self.acquisition == 'qucb':
            acq = acq_class(
                model,
                self.beta,
                sampler=sampler,
                objective=GenericMCObjective(mc_objective),
            )
        elif self.acquisition in ('qei', 'qlogei', 'qpi'):
            acq = acq_class(
                model,
                best,
                sampler=sampler,
                objective=GenericMCObjective(mc_objective),
            )
        elif self.acquisition in ('qnei', 'qlognei'):
            acq = acq_class(
                model,
                dataset.params,
                sampler=sampler,
                objective=GenericMCObjective(mc_objective),
            )
        elif self.acquisition == 'qkg':
            # Knowledge gradient is quite expensive: use less samples
            num_restarts = 6
            raw_samples = 128
            sampler = SobolQMCNormalSampler(torch.Size([64]), seed=self.seed)
            acq = acq_class(model, sampler=sampler, objective=GenericMCObjective(mc_objective))
        # Quasi-Monte Carlo multi-objective acquisitions
        elif self.acquisition in ('qehvi', 'qlogehvi'):
            acq = acq_class(
                model,
                self.ref_point,
                self.partitioning,
                objective=GenericMCMultiOutputObjective(mc_objective),
                sampler=sampler,
            )
        elif self.acquisition in ('qnehvi', 'qlognehvi'):
            acq = acq_class(
                model,
                self.ref_point,
                dataset.params,
                objective=GenericMCMultiOutputObjective(mc_objective),
                sampler=sampler,
            )
        elif self.acquisition == 'qhvkg':
            acq = acq_class(
                model,
                self.ref_point,
                objective=GenericMCMultiOutputObjective(mc_objective),
            )
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

        # Build initial dataset with the initial shot
        dataset = BayesDataset(n_dim, n_outputs, export=self.export, device=self.device)
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
        test_dataset = BayesDataset(n_dim, n_outputs, device=self.device)
        if self.n_test > 0:
            test_points = self._get_random_points(self.n_test, n_dim, self.seed + 1, bound)
            test_results = self._eval_candidates(test_points)
            for i, result in enumerate(test_results):
                values, variances = self._result_to_dataset(result)
                test_dataset.push(test_points[i], values, variances)

        # Find current best point to return to the driver
        if self.objective.multi_objective:
            best_value = self._update_mo_data(dataset)
            best_params = None
        else:
            best_params, best_result = dataset.min(self._value_to_scalar)
            best_value = self._value_to_scalar(best_result, best_params)
        self._progress_check(0, best_value, best_params)

        # Optimisation loop
        for i_iter in range(n_iter):
            # Generate candidates: catch CUDA OOM errors and fall back to CPU
            try:
                candidates, cv_error = self._get_candidates(n_dim, bound, dataset, test_dataset)
            except torch.cuda.OutOfMemoryError:
                warnings.warn('CUDA out of memory: falling back to CPU')
                self.device = 'cpu'
                dataset = dataset.to(self.device)
                test_dataset = test_dataset.to(self.device)
                candidates, cv_error = self._get_candidates(n_dim, bound, dataset, test_dataset)

            # Evaluate candidates (in parallel if possible)
            results = self._eval_candidates(candidates)

            # Update dataset
            results_batch = []
            for i, result in enumerate(results):
                values, variances = self._result_to_dataset(result)
                results_batch.append(values)
                dataset.push(candidates[i, :], values, variances)

            # Find best observation for this batch
            if self.objective.multi_objective:
                best_value = self._update_mo_data(dataset)
                best_params = None
            else:
                values_batch = [
                    self._value_to_scalar(values, candidates[i, :])
                    for i, values in enumerate(results_batch)
                ]
                best_idx = np.argmin(values_batch)
                best_value = values_batch[best_idx]
                best_params = candidates[best_idx, :]

            # Update progress (with extra data if available)
            extra = None
            if cv_error:
                extra = f'Val. {cv_error:6.4}'
                if self.objective.multi_objective:
                    extra += f'  Num Pareto: {self.partitioning.pareto_Y.shape[0]}'
            elif self.objective.multi_objective:
                extra = f'Num Pareto: {self.partitioning.pareto_Y.shape[0]}'
            if self._progress_check(i_iter + 1, best_value, best_params, extra_info=extra):
                break

        # Return optimisation result
        if self.objective.multi_objective:
            best_result = self._update_mo_data(dataset)
            best_params = None
        else:
            best_params, best_result = dataset.min(self._value_to_scalar)
            best_result = self._value_to_scalar(best_result, best_params)
        return best_params, best_result
