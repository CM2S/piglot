"""Main optimiser classes for using BoTorch with piglot"""
from typing import Tuple, List, Union
from multiprocessing.pool import ThreadPool as Pool
import warnings
import numpy as np
import torch
from scipy.stats import qmc
from gpytorch.mlls import ExactMarginalLogLikelihood
import botorch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.acquisition import ProbabilityOfImprovement, qProbabilityOfImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.sampling import SobolQMCNormalSampler
from piglot.objective import Objective, GenericObjective, ObjectiveResult
from piglot.optimisers.botorch.dataset import BayesDataset
from piglot.optimiser import Optimiser


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
        acquisition: str = 'ucb',
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
        self.noisy = bool(noisy)
        self.q = q
        self.seed = seed
        self.load_file = load_file
        self.export = export
        self.n_test = n_test
        self.device = device
        if self.acquisition not in ('ucb', 'ei', 'pi', 'kg', 'qucb', 'qei', 'qpi', 'qkg'):
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

    def _acq_func(
        self,
        dataset: BayesDataset,
        model: Model,
        n_dim: int,
    ) -> Tuple[AcquisitionFunction, int, int]:
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
        return model

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
        if self.objective.composition:
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
        if self.objective.composition:
            if isinstance(value, np.ndarray):
                return self.objective.composition.composition(value)
            return self.objective.composition.composition_torch(value).cpu().item()
        return value.item()

    def _build_acquisition_scalar(
        self,
        dataset: BayesDataset,
        model: Model,
        n_dim: int,
    ) -> Tuple[AcquisitionFunction, int, int]:
        # Default values for multi-restart optimisation
        num_restarts = 12
        raw_samples = max(256, 16 * n_dim * n_dim)
        # Delegate acquisition building
        best = torch.min(dataset.values).item()
        mc_objective = GenericMCObjective(lambda vals, X: -vals.squeeze(-1))
        sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)
        if self.acquisition == 'ucb':
            acq = UpperConfidenceBound(model, self.beta, maximize=False)
        elif self.acquisition == 'qucb':
            acq = qUpperConfidenceBound(model, self.beta, sampler=sampler, objective=mc_objective)
        elif self.acquisition == 'ei':
            acq = ExpectedImprovement(model, best, maximize=False)
        elif self.acquisition == 'qei':
            acq = qExpectedImprovement(model, best, sampler=sampler, objective=mc_objective)
        elif self.acquisition == 'pi':
            acq = ProbabilityOfImprovement(model, best, maximize=False)
        elif self.acquisition == 'qpi':
            acq = qProbabilityOfImprovement(model, best, sampler=sampler, objective=mc_objective)
        elif self.acquisition in ('kg', 'qkg'):
            # Knowledge gradient is quite expensive: use less samples
            num_restarts = 6
            raw_samples = 128
            sampler = SobolQMCNormalSampler(torch.Size([64]), seed=self.seed)
            acq = qKnowledgeGradient(model, sampler=sampler, objective=mc_objective)
        else:
            raise RuntimeError(f"Unknown acquisition {self.acquisition}")
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
        # Build composite MC objective
        _, y_avg, y_std = dataset.get_obervation_stats()
        mc_objective = GenericMCObjective(
            lambda vals, X: -self.objective.composition.composition_torch(vals * y_std + y_avg)
        )
        # Delegate acquisition building
        best = torch.max(dataset.values).item()
        sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)
        if self.acquisition in ('ucb', 'qucb'):
            acq = qUpperConfidenceBound(model, self.beta, sampler=sampler, objective=mc_objective)
        elif self.acquisition in ('ei', 'qei'):
            acq = qExpectedImprovement(model, best, sampler=sampler, objective=mc_objective)
        elif self.acquisition in ('pi', 'qpi'):
            acq = qProbabilityOfImprovement(model, best, sampler=sampler, objective=mc_objective)
        elif self.acquisition in ('kg', 'qkg'):
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
        self._progress_check(0, self._value_to_scalar(best_result), best_params)

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

            # Update progress (with CV data if available)
            extra = f'Val. {cv_error:6.4}' if cv_error else None
            if self._progress_check(i_iter + 1, best_value, best_params, extra_info=extra):
                break

        # Return optimisation result
        best_params, best_result = dataset.min(self._value_to_scalar)
        return best_params, self._value_to_scalar(best_result)
