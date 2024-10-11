"""Main optimiser classes for using BoTorch with piglot"""
from __future__ import annotations
from typing import Tuple, List, Union, Dict, Any
import os
import warnings
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import torch
from scipy.stats import qmc
from botorch.models.converter import batched_to_model_list
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from piglot.parameter import ParameterSet
from piglot.objective import (
    Objective,
    GenericObjective,
    ObjectiveResult,
)
from piglot.optimiser import Optimiser
from piglot.optimisers.botorch.acquisitions import (
    AVAILABLE_ACQUISITIONS,
    default_acquisition,
    get_acquisition,
    optimise_acquisition,
)
from piglot.optimisers.botorch.dataset import BayesDataset
from piglot.optimisers.botorch.model import (
    SingleTaskGPWithNoise,
    PseudoHeteroskedasticSingleTaskGP,
    build_gp_model
)
from piglot.optimisers.botorch.composition import (
    BoTorchComposition,
    RiskComposition,
)
from piglot.optimisers.botorch.risk_acquisitions import (
    AVALIALBE_RISK_MEASURES,
    get_risk_measure,
)


def get_default_torch_device() -> str:
    """Utility to return the default PyTorch device (pre Pytorch v2.3).

    Returns
    -------
    str
        Name of the default PyTorch device.
    """
    return str(torch.tensor([0.0]).device)


def draw_multivariate_samples(
    means: torch.Tensor,
    covariances: torch.Tensor,
    num_samples: int,
    seed: int = None,
) -> torch.Tensor:
    """Draw samples from a multivariate normal distribution.

    Parameters
    ----------
    means : torch.Tensor
        Means of the distribution (num_batch x num_dim).
    covariances : torch.Tensor
        Covariance matrices of the distribution (num_batch x num_dim x num_dim).
    num_samples : int
        Number of samples to draw.
    seed : int, optional
        Random seed, by default None

    Returns
    -------
    torch.Tensor
        Samples from the distribution (num_samples x num_batch x num_dim).
    """
    return torch.stack([
        MultivariateNormalQMCEngine(mean, cov, seed=seed).draw(num_samples)
        for mean, cov in zip(means, covariances)
    ], dim=1)


@dataclass
class BoTorchSettingsData:
    """Container for settings used during a Bayesian optimisation run."""
    objective: GenericObjective
    n_initial: int = None
    n_test: int = 0
    acquisition: str = None
    beta: float = 1.0
    q: int = 1
    seed: int = 1
    load_file: str = None
    export: str = None
    device: str = None
    reference_point: List[float] = None
    adjusted_ref_point: bool = None
    nadir_scale: float = 0.1
    skip_initial: bool = False
    pca_variance: float = 1e-6
    num_restarts: int = None
    raw_samples: int = None
    mc_samples: int = None
    batch_size: int = None
    num_fantasies: int = None
    sequential: bool = False
    noise_model: str = 'homoscedastic'
    infer_noise: bool = None
    risk_measure: str = None
    alpha: float = None
    composite_risk_measure: str = None

    def __post_init__(self):
        # Set default device
        self.device = self.device or get_default_torch_device()

        # Sanitise the reference point
        if self.reference_point is not None:
            if self.adjusted_ref_point is True:
                raise RuntimeError("Cannot adjust the reference point when it is provided")
            self.reference_point = -torch.tensor(self.reference_point)
        elif self.adjusted_ref_point is None:
            self.adjusted_ref_point = True
        elif self.adjusted_ref_point is False:
            raise RuntimeError("Either provide a reference point or set adjusted_ref_point to True")

        # Sanitise the noise model name (to avoid the endless debate of the c vs k)
        self.noise_model.replace('k', 'c')
        if self.noise_model not in ('homoscedastic', 'heteroscedastic'):
            raise RuntimeError(f"Unknown noise model {self.noise_model}")

        # Check if we need to infer the noise level
        if self.infer_noise is None:
            self.infer_noise = self.objective.noisy and not self.objective.stochastic

        # Select acquisition
        if self.acquisition is None:
            self.acquisition = default_acquisition(
                self.objective.composition,
                self.objective.multi_objective,
                self.objective.noisy,
                self.q,
            )
        elif self.acquisition not in AVAILABLE_ACQUISITIONS:
            raise RuntimeError(f"Unkown acquisition function {self.acquisition}")
        if not self.acquisition.startswith('q') and self.q != 1:
            raise RuntimeError("Can only use q != 1 for quasi-Monte Carlo acquisitions")
        if not self.acquisition.startswith('q') and self.objective.composition:
            raise RuntimeError("Cannot use analytical acquisitions with composition")

        # Sanitise risk measure
        if self.risk_measure is not None and self.composite_risk_measure is not None:
            raise RuntimeError("Cannot use both a risk measure and a composite risk measure")
        risk_measure = self.risk_measure or self.composite_risk_measure
        if risk_measure is not None:
            if risk_measure not in AVALIALBE_RISK_MEASURES:
                raise RuntimeError(f"Unknown risk measure {risk_measure}")
            if self.alpha is None:
                raise RuntimeError("Must provide alpha when using a risk measure")

        # PCA variance is only relevant for composite problems
        if self.pca_variance and not self.objective.composition:
            self.pca_variance = None

    def get_dict(self) -> Dict[str, Any]:
        """Return the settings as a dictionary."""
        return self.__dict__

    def update(self, n_dim: int) -> None:
        """Update the settings with the number of dimensions.

        Parameters
        ----------
        n_dim : int
            Number of dimensions in the problem.
        """
        self.n_initial = self.n_initial or max(8, 2 * n_dim)
        self.num_restarts = self.num_restarts or 12
        self.raw_samples = self.raw_samples or max(256, 16 * n_dim * n_dim)
        self.mc_samples = self.mc_samples or (64 if 'fantasy' in self.acquisition else 512)
        self.batch_size = self.batch_size or 128
        self.num_fantasies = self.num_fantasies or (128 if 'fantasy' in self.acquisition else 64)


@dataclass
class BoTorchMultiObjectiveStateData:
    """Container for data used during a multi-objective Bayesian optimisation run."""
    ref_point: torch.Tensor = None
    partitioning: FastNondominatedPartitioning = None
    hypervolume: float = None
    pareto_x: torch.Tensor = None
    pareto_y: torch.Tensor = None
    pareto_yvar: torch.Tensor = None

    def update(
        self,
        settings: BoTorchSettingsData,
        x_points: torch.Tensor,
        y_points: torch.Tensor,
        yvar_points: torch.Tensor = None,
    ) -> None:
        """Update the MO state data with the latest dataset.

        Parameters
        ----------
        settings : BoTorchSettingsData
            Optimisation settings.
        x_points : torch.Tensor
            Parameter points.
        y_points : torch.Tensor
            Objective points.
        yvar_points : torch.Tensor
            Objective variance points. Set to None if not available.
        """
        # Compute reference point if needed
        if settings.adjusted_ref_point:
            nadir = torch.min(y_points, dim=0).values
            ideal = torch.max(y_points, dim=0).values
            self.ref_point = nadir - settings.nadir_scale * (ideal - nadir)
        elif self.ref_point is None:
            self.ref_point = settings.reference_point

        # Update partitioning and Pareto front
        self.partitioning = FastNondominatedPartitioning(self.ref_point, Y=y_points)
        self.hypervolume = self.partitioning.compute_hypervolume().item()
        self.pareto_y = self.partitioning.pareto_Y

        # Map each Pareto point to the original parameter space
        param_indices = [
            torch.argmin((y_points - self.pareto_y[i, :]).norm(dim=1)).item()
            for i in range(self.pareto_y.shape[0])
        ]
        self.pareto_x = x_points[param_indices, :]
        if yvar_points is not None:
            self.pareto_yvar = yvar_points[param_indices, :]

    def dump(self, output_file: str, parameters: ParameterSet) -> None:
        """Dump the Pareto front to a file.

        Parameters
        ----------
        output_file : str
            File to write the Pareto front to.
        parameters : ParameterSet
            Parameter set for the problem.
        """
        with open(output_file, 'w', encoding='utf8') as file:
            # Write header
            num_obj = self.pareto_y.shape[1]
            file.write('\t'.join([f'{"Objective_" + str(i + 1):>15}' for i in range(num_obj)]))
            file.write('\t' + '\t'.join([f'{param.name:>15}' for param in parameters]) + '\n')
            # Write each point
            for i in range(self.pareto_y.shape[0]):
                file.write('\t'.join([f'{-x.item():>15.8f}' for x in self.pareto_y[i, :]]) + '\t')
                file.write('\t'.join([f'{x.item():>15.8f}' for x in self.pareto_x[i, :]]) + '\n')

    def to(self, device: str) -> BoTorchMultiObjectiveStateData:
        """Move the MO state data to a new device.

        Parameters
        ----------
        device : str
            Device to move the data to.
        """
        self.ref_point = self.ref_point.to(device)
        self.pareto_x = self.pareto_x.to(device)
        self.pareto_y = self.pareto_y.to(device)
        self.pareto_yvar = self.pareto_yvar.to(device)
        self.partitioning = self.partitioning.to(device)
        return self


@dataclass
class BoTorchStateData:
    """Container for data used during a Bayesian optimisation run."""
    settings: BoTorchSettingsData
    dataset: BayesDataset = None
    test_dataset: BayesDataset = None
    model: Union[SingleTaskGPWithNoise, PseudoHeteroskedasticSingleTaskGP] = None
    composition: BoTorchComposition = None
    device: str = get_default_torch_device()
    extra_info: Dict[str, str] = None
    mo_data: BoTorchMultiObjectiveStateData = None
    best_params: np.ndarray = None
    best_value: float = None
    conf_interval: Tuple[float, float] = None

    def __train_model(self) -> Union[SingleTaskGPWithNoise, PseudoHeteroskedasticSingleTaskGP]:
        """Train the GP model."""
        model = build_gp_model(
            self.dataset,
            self.settings.infer_noise,
            self.settings.noise_model,
        )
        # MOBO requires a model list (except when there is only one output)
        if self.settings.objective.multi_objective and model.num_outputs > 1:
            model = batched_to_model_list(model)
        return model

    def update(self, parameters: ParameterSet, output_dir: str) -> None:
        """Update the state data with the latest dataset.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for the problem.
        output_dir : str
            Directory where to save the results.
        """
        # Build the GP model
        self.model = self.__train_model()

        # Build composition depending on the objective
        if self.settings.composite_risk_measure is not None:
            self.composition = RiskComposition(
                self.model,
                self.dataset,
                self.settings.objective,
                self.settings.mc_samples // 2,
                self.settings.seed,
                get_risk_measure(self.settings.composite_risk_measure, alpha=self.settings.alpha),
            )
        else:
            self.composition = BoTorchComposition(self.model, self.dataset, self.settings.objective)

        # Populate extra info with the number of PCA components
        self.extra_info = {}
        if self.settings.pca_variance:
            self.extra_info["Num PCA"] = str(self.dataset.pca.num_components.item())

        # Evaluate GP performance with the test dataset
        if self.test_dataset is not None:
            std_test_values, _ = self.dataset.transform_outcomes(
                self.test_dataset.values,
                self.test_dataset.covariances,
            )
            with torch.no_grad():
                posterior = self.model.posterior(self.test_dataset.params)
                f = std_test_values
                y = posterior.mean
                y_bar = y.mean(dim=0, keepdim=True)
                cv_error = (y - f).square().mean(dim=0)
                r_squared = 1 - cv_error / (y - y_bar).square().mean(dim=0)
            self.extra_info["CV Error"] = f'{cv_error.mean().item():.4e}'
            self.extra_info["R^2"] = f'{r_squared.mean().item():.4f}'

        # Under multi-objective optimisation, update the MO data and return the hypervolume
        if self.settings.objective.multi_objective:
            values = self.composition.from_original_samples(
                self.dataset.values,
                self.dataset.params,
            )
            self.mo_data.update(self.settings, self.dataset.params, values)
            self.mo_data.dump(os.path.join(output_dir, 'pareto_front'), parameters)
            self.best_params = None
            self.best_value = -np.log(self.mo_data.hypervolume)
            self.extra_info["Num Pareto"] = str(self.mo_data.pareto_y.shape[0])

        # Under noisy single-objective optimisation, find the best posterior mean
        elif self.settings.objective.noisy:
            acquisition = get_acquisition(
                'qsr',
                self.model,
                self.dataset,
                self.composition,
                best=None if self.best_value is None else -self.best_value,  # pylint: disable=E1130
                **self.settings.get_dict(),
            )
            candidates, acq_value = optimise_acquisition(
                acquisition,
                parameters,
                self.dataset,
                **self.settings.get_dict(),
            )
            self.best_params = candidates[0, :].cpu().numpy()
            self.best_value = -acq_value.item()
            # Compute the confidence interval
            with torch.no_grad():
                samples = self.composition.from_inputs(
                    candidates,
                    self.settings.mc_samples,
                    seed=self.settings.seed,
                )
                self.conf_interval = (
                    -torch.quantile(samples, 0.975).item(),
                    -torch.quantile(samples, 0.025).item(),
                )
            # Extract signal-to-noise ratio
            samples = draw_multivariate_samples(self.dataset.values, self.dataset.covariances, 1024)
            obj_samples = self.composition.from_original_samples(samples, self.dataset.params)
            obj_var = obj_samples.var(dim=0).mean().item()
            mean_var = obj_samples.mean(dim=0).var().item()
            self.extra_info["SNR"] = f'{10 * np.log10(mean_var / obj_var):.1f} dB'

        # Under exact single-objective optimisation, return the best value found
        else:
            obj = self.composition.from_original_samples(self.dataset.values, self.dataset.params)
            idx_best = torch.argmax(obj).item()
            self.best_params = self.dataset.params[idx_best, :].cpu().numpy()
            self.best_value = -obj[idx_best].item()

    def to(self, device: str) -> BoTorchStateData:
        """Move the state data to a new device.

        Parameters
        ----------
        device : str
            Device to move the data to.
        """
        self.device = device
        self.dataset = self.dataset.to(device)
        self.test_dataset = self.test_dataset.to(device)
        self.mo_data = self.mo_data.to(device) if self.mo_data else None
        # Rebuild the model and composition on the new device
        self.model = self.__train_model()
        self.composition = BoTorchComposition(self.model, self.dataset, self.settings.objective)
        return self


class BayesianBoTorch(Optimiser):
    """Driver for optimisation using BoTorch."""

    def __init__(self, objective: Objective, **kwargs) -> None:
        if not isinstance(objective, GenericObjective):
            raise RuntimeError("Bayesian optimiser requires a GenericObjective")
        super().__init__('BoTorch', objective)
        self.settings = BoTorchSettingsData(objective, **kwargs)
        torch.set_num_threads(1)

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective

        Parameters
        ----------
        objective : Objective
            Objective to optimise
        """

    def _get_candidates(self, state: BoTorchStateData) -> np.ndarray:
        acquisition = get_acquisition(
            self.settings.acquisition,
            state.model,
            state.dataset,
            state.composition,
            best=-state.best_value,
            ref_point=state.mo_data.ref_point if state.mo_data else None,
            partitioning=state.mo_data.partitioning if state.mo_data else None,
            **self.settings.get_dict(),
        )
        candidates, _ = optimise_acquisition(
            acquisition,
            self.parameters,
            state.dataset,
            **self.settings.get_dict(),
        )
        return candidates.cpu().numpy()

    def _eval_candidates(self, candidates: np.ndarray) -> List[ObjectiveResult]:
        # Single candidate case
        if self.settings.q == 1:
            return [self.objective(candidate) for candidate in candidates]
        # Multi-candidate: run cases in parallel
        with Pool(self.settings.q) as pool:
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

    def _update_dataset(
        self,
        dataset: BayesDataset,
        candidates: List[np.ndarray],
        results: List[ObjectiveResult],
        composition: BoTorchComposition = None,
    ) -> Tuple[np.ndarray, float]:
        # Update dataset with the new results
        results_batch = []
        for i, result in enumerate(results):
            zero_covar = np.zeros_like(np.diag(result.values))
            dataset.push(
                candidates[i],
                result.values,
                result.covariances if result.covariances is not None else zero_covar,
                result.scalar_value,
            )
            results_batch.append(result.values)

        # Nothing more to do for multi-objective problems or when we don't have a composition
        if self.settings.objective.multi_objective or composition is None:
            return None, None

        # For single-objective problems, find the best observation in this batch
        objectives = composition.from_original_samples(
            torch.from_numpy(np.array(results_batch)).to(dataset.device),
            torch.from_numpy(np.array(candidates)).to(dataset.device),
        )
        idx_best = torch.argmax(objectives).item()
        return candidates[idx_best], -objectives[idx_best].item()

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

        # Initialise heuristic variables and state data
        self.settings.update(n_dim)
        state = BoTorchStateData(self.settings)
        if self.objective.multi_objective:
            state.mo_data = BoTorchMultiObjectiveStateData()

        # Select initial dataset strategy: loading from file or generating random points
        if self.settings.load_file:
            state.dataset = BayesDataset.load(self.settings.load_file)

            # Sanity check: ensure the loaded dataset has the correct number of dimensions
            if state.dataset.n_dim != n_dim:
                raise RuntimeError(
                    f"Loaded dataset has {state.dataset.n_dim} dimensions, expected {n_dim}"
                )
        else:
            # Build initial dataset: initial shot + random points
            initial_points = [init_shot] if not self.settings.skip_initial else []
            if self.settings.n_initial > 1:
                initial_points += self._get_random_points(
                    self.settings.n_initial,
                    n_dim,
                    self.settings.seed,
                    bound,
                )
            if len(initial_points) == 0:
                raise RuntimeError("No initial points to evaluate")

            # Evaluate initial dataset (in parallel if possible) and infer number of outputs
            results = self._eval_candidates(initial_points)
            n_outputs = results[0].values.size

            # Build dataset and add initial points
            state.dataset = BayesDataset(
                n_dim,
                n_outputs,
                export=os.path.join(self.output_dir, 'dataset.pt'),
                device=self.settings.device,
                pca_variance=self.settings.pca_variance,
            )
            self._update_dataset(state.dataset, initial_points, results)

        # Build test dataset (in parallel if possible)
        if self.settings.n_test > 0:
            state.test_dataset = BayesDataset(n_dim, n_outputs, device=self.settings.device)
            test_points = self._get_random_points(
                self.settings.n_test,
                n_dim,
                self.settings.seed + 1,
                bound,
            )
            test_results = self._eval_candidates(test_points)
            self._update_dataset(state.test_dataset, test_points, test_results)

        # Update state data with the latest dataset and get the current best point
        state.update(self.parameters, self.output_dir)
        if self._progress_check(
            0,
            state.best_value,
            state.best_params,
            state.extra_info,
            state.conf_interval,
        ):
            return state.best_params, state.best_value

        # Optimisation loop
        for i_iter in range(n_iter):
            # Generate candidates and catch CUDA OOM errors
            candidates = None
            while candidates is None:
                try:
                    candidates = list(self._get_candidates(state))
                except torch.cuda.OutOfMemoryError:
                    if self.settings.batch_size > 1:
                        self.settings.batch_size //= 2
                        warnings.warn(
                            f'CUDA out of memory: halving batch size to {self.settings.batch_size}'
                        )
                    else:
                        warnings.warn('CUDA out of memory: falling back to CPU')
                        torch.set_default_device('cpu')
                        self.settings.device = 'cpu'
                        state = state.to('cpu')

            # Evaluate candidates (in parallel if possible)
            results = self._eval_candidates(candidates)

            # Update dataset and state data with the latest results
            best_params, best_value = self._update_dataset(
                state.dataset,
                candidates,
                results,
                composition=state.composition,
            )
            state.update(self.parameters, self.output_dir)

            # Return state best instead of batch best for noisy or multi-objective problems
            if self.settings.objective.multi_objective or self.settings.objective.noisy:
                best_params, best_value = state.best_params, state.best_value

            # Update progress and check for early stopping
            if self._progress_check(
                i_iter + 1,
                best_value,
                best_params,
                state.extra_info,
                state.conf_interval,
            ):
                break

        # Return optimisation result
        state.update(self.parameters, self.output_dir)
        return state.best_params, state.best_value
