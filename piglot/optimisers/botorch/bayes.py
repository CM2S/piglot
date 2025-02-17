"""Main optimiser classes for using BoTorch with piglot"""
from typing import Tuple, List, Type, Dict
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
    qUpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
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


def get_default_torch_device() -> str:
    """Utility to return the default PyTorch device (pre Pytorch v2.3).

    Returns
    -------
    str
        Name of the default PyTorch device.
    """
    return str(torch.tensor([0.0]).device)


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
    return 'qlogei'


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
        n_initial: int = None,
        n_test: int = 0,
        acquisition: str = None,
        beta: float = 1.0,
        noisy: float = False,
        q: int = 1,
        seed: int = 1,
        load_file: str = None,
        export: str = None,
        device: str = None,
        reference_point: List[float] = None,
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
        self.device = get_default_torch_device() if device is None else device
        self.skip_initial = bool(skip_initial)
        self.partitioning: FastNondominatedPartitioning = None
        self.adjusted_ref_point = reference_point is None
        self.ref_point = None if reference_point is None else -torch.tensor(reference_point)
        self.nadir_scale = nadir_scale
        self.pca_variance = pca_variance
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.mc_samples = mc_samples
        self.batch_size = batch_size
        self.sequential = bool(sequential)
        self.num_fantasies = num_fantasies
        if acquisition is None:
            self.acquisition = default_acquisition(
                objective.composition,
                objective.multi_objective,
                bool(noisy) or objective.stochastic,
                self.q,
            )
        else:
            orig_name = self.acquisition
            if not self.acquisition.startswith('q'):
                self.acquisition = 'q' + self.acquisition
            if self.acquisition not in AVAILABLE_ACQUISITIONS:
                raise RuntimeError(f"Unkown acquisition function {orig_name}")
        if self.pca_variance and not (objective.composition or objective.multi_objective):
            warnings.warn("Ignoring PCA variance for non-composite single-objective problem")
            self.pca_variance = None
        elif self.pca_variance is None and objective.composition:
            self.pca_variance = 1e-6
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
        values, variances = dataset.transform_outcomes(dataset.values, dataset.covariances)
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
        # MOBO requires a model list (except when there is only one output)
        if self.objective.multi_objective and values.shape[-1] > 1:
            return batched_to_model_list(model)
        return model

    def _get_candidates(
            self,
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
                test_dataset.covariances,
            )
            with torch.no_grad():
                posterior = model.posterior(test_dataset.params)
                cv_error = (posterior.mean - std_test_values).square().mean().item()

        # Build the acquisition function
        acq = self._acq_func(dataset, model)

        # Optimise acquisition to find next candidate(s)
        candidates, _ = optimize_acqf(
            acq,
            bounds=torch.from_numpy(bounds.T).to(self.device).to(dataset.dtype),
            q=self.q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            sequential=self.sequential,
            options={
                "sample_around_best": True,
                "seed": self.seed,
                "init_batch_limit": self.batch_size,
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
        return list(results)

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
        covariances = (
            result.covariances
            if self.objective.stochastic
            else np.diag(np.zeros_like(result.values))
        )
        return result.values, covariances

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
    ) -> AcquisitionFunction:
        # Default values for multi-restart optimisation
        sampler = SobolQMCNormalSampler(torch.Size([self.mc_samples]), seed=self.seed)

        # Find best value for the acquisition
        best = torch.max(self._composition(dataset.values, dataset.params)).item()

        # Build composite MC objective
        def mc_objective(vals: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            return self._composition(dataset.untransform_outcomes(vals), X)

        # Delegate to the correct acquisition function
        # The arguments for each acquisition are different, so we group them into families
        acq_class = AVAILABLE_ACQUISITIONS[self.acquisition]
        if self.acquisition == 'qucb':
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
            acq = acq_class(
                model,
                num_fantasies=self.num_fantasies,
                inner_sampler=sampler,
                objective=GenericMCObjective(mc_objective),
            )
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
                num_fantasies=self.num_fantasies,
                inner_sampler=sampler,
                objective=GenericMCMultiOutputObjective(mc_objective),
            )
        else:
            raise RuntimeError(f"Unknown acquisition {self.acquisition}")
        return acq

    def _init_dataset(self, n_dim: int, bound: np.ndarray, init_shot: np.ndarray) -> BayesDataset:
        # Can we load straight from the input file?
        if self.load_file:
            return BayesDataset.load(self.load_file)

        # Evaluate initial shot and use it to infer number of dimensions
        if not self.skip_initial:
            init_result = self.objective(init_shot)
            init_values, init_covariances = self._result_to_dataset(init_result)
            n_outputs = len(init_values)

        # If requested, sample some random points before starting (in parallel if possible)
        random_points = self._get_random_points(self.n_initial, n_dim, self.seed, bound)
        results = self._eval_candidates(random_points)

        # Infer number of points to store when skipping initial shot
        if self.skip_initial:
            values, covariances = self._result_to_dataset(results[0])
            n_outputs = len(values)

        # Build initial dataset with the initial shot (if available)
        dataset = BayesDataset(
            n_dim,
            n_outputs,  # pylint: disable=E0606
            export=self.export,
            device=self.device,
            pca_variance=self.pca_variance,
        )
        if not self.skip_initial:
            dataset.push(init_shot, init_values, init_covariances, init_result.scalar_value)

        # Add random points to the dataset
        for i, result in enumerate(results):
            values, covariances = self._result_to_dataset(result)
            dataset.push(random_points[i], values, covariances, result.scalar_value)
        return dataset

    def _get_extra_info(self, cv_error: float, dataset: BayesDataset) -> str:
        extra = None
        if cv_error:
            extra = f'Val. {cv_error:6.4}'
            if self.objective.multi_objective:
                extra += f'  Num Pareto: {self.partitioning.pareto_Y.shape[0]}'
            if self.pca_variance:
                extra += f'  Num PCA: {dataset.pca.num_components}'
        elif self.objective.multi_objective:
            extra = f'Num Pareto: {self.partitioning.pareto_Y.shape[0]}'
            if self.pca_variance:
                extra += f'  Num PCA: {dataset.pca.num_components}'
        elif self.pca_variance:
            extra = f'Num PCA: {dataset.pca.num_components}'
        return extra

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

        # Initialise heuristic variables
        self.n_initial = self.n_initial or max(8, 2 * n_dim)
        self.num_restarts = self.num_restarts or 12
        self.raw_samples = self.raw_samples or max(256, 16 * n_dim * n_dim)
        self.mc_samples = self.mc_samples or 512
        self.batch_size = self.batch_size or 128
        self.num_fantasies = self.num_fantasies or 16

        # Build initial dataset
        dataset = self._init_dataset(n_dim, bound, init_shot)

        # Build test dataset (in parallel if possible)
        test_dataset = BayesDataset(n_dim, dataset.n_outputs, device=self.device)
        if self.n_test > 0:
            test_points = self._get_random_points(self.n_test, n_dim, self.seed + 1, bound)
            test_results = self._eval_candidates(test_points)
            for i, result in enumerate(test_results):
                values, covariances = self._result_to_dataset(result)
                test_dataset.push(test_points[i], values, covariances, result.scalar_value)

        # Find current best point to return to the driver
        if self.objective.multi_objective:
            best_value = self._update_mo_data(dataset)
            best_params = None
        else:
            best_params, best_value = dataset.min()
        self._progress_check(0, best_value, best_params)

        # Optimisation loop
        for i_iter in range(n_iter):
            # Generate candidates and catch CUDA OOM errors
            candidates = None
            while candidates is None:
                try:
                    candidates, cv_error = self._get_candidates(bound, dataset, test_dataset)
                except torch.cuda.OutOfMemoryError:
                    if self.batch_size > 1:
                        warnings.warn(
                            f'CUDA out of memory: halving batch size to {self.batch_size // 2}'
                        )
                        self.batch_size //= 2
                    else:
                        warnings.warn('CUDA out of memory: falling back to CPU')
                        self.device = 'cpu'
                        torch.set_default_device('cpu')
                        dataset = dataset.to(self.device)
                        test_dataset = test_dataset.to(self.device)
                        if self.objective.multi_objective:
                            self.ref_point = self.ref_point.to(self.device)
                            self._update_mo_data(dataset)

            # Evaluate candidates (in parallel if possible)
            results = self._eval_candidates(candidates)

            # Update dataset
            values_batch = []
            for i, result in enumerate(results):
                values, covariances = self._result_to_dataset(result)
                values_batch.append(result.scalar_value)
                dataset.push(candidates[i, :], values, covariances, result.scalar_value)

            # Find best observation for this batch
            if self.objective.multi_objective:
                best_value = self._update_mo_data(dataset)
                best_params = None
            else:
                best_idx = np.argmin(values_batch)
                best_value = values_batch[best_idx]
                best_params = candidates[best_idx, :]

            # Update progress (with extra data if available)
            if self._progress_check(
                i_iter + 1,
                best_value,
                best_params,
                extra_info=self._get_extra_info(cv_error, dataset),
            ):
                break

        # Return optimisation result
        if self.objective.multi_objective:
            best_result = self._update_mo_data(dataset)
            best_params = None
        else:
            best_params, best_result = dataset.min()
        return best_params, best_result
