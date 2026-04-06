"""Module containing optimisation objective primites"""
import os
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar
from threading import Lock
from dataclasses import dataclass
from functools import partial
import numpy as np
import torch
from matplotlib.figure import Figure
from piglot.settings import Settings
from piglot.parameter import ParameterValues
from piglot.utils.composition import ConcatUtility
from piglot.utils.tabular import TabularFile, TabularStringColumn, TabularFloatColumn


T = TypeVar('T', bound='Objective')


# @dataclass
# class ScalarValue:
#     """Container for a scalar value with optional variance."""
#     value: float
#     variance: Optional[float] = None


# @dataclass
# class VectorValue:
#     """Container for a vector value with optional covariances."""
#     values: np.ndarray
#     covariances: Optional[np.ndarray] = None


@dataclass
class IndividualObjectiveResult:
    """Container for individual objective results."""
    value: float
    variance: Optional[float] = None
    latent_values: Optional[np.ndarray] = None
    latent_covariances: Optional[np.ndarray] = None


@dataclass
class ObjectiveResult:
    """Container for objective results."""
    results: list[IndividualObjectiveResult]
    obj_values: np.ndarray
    obj_variances: Optional[np.ndarray] = None
    scalar_value: Optional[float] = None
    scalar_variance: Optional[float] = None
    latent_values: Optional[np.ndarray] = None
    latent_covariances: Optional[np.ndarray] = None


@dataclass
class FunctionCallsData:
    """Container for function calls data."""
    start_times: np.ndarray
    run_times: np.ndarray
    params: np.ndarray
    hashes: list[str]
    obj_values: np.ndarray
    obj_variances: Optional[np.ndarray]
    scalar_values: Optional[np.ndarray]
    scalar_variances: Optional[np.ndarray]


class IndividualObjective(ABC):
    """Base class for individual objectives for generic optimisation problems."""

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        maximise: bool = False,
        variance: bool = False,
        composite: bool = False,
        bounds: tuple[float, float] = None,
    ) -> None:
        self.name = name
        self.maximise = maximise
        self.weight = float(weight)
        self.variance = variance
        self.composite = composite
        self.bounds = None
        if bounds is not None:
            self.bounds = tuple(float(b) for b in bounds)
            if self.bounds[0] > self.bounds[1]:
                raise ValueError(f"Invalid bounds {self.bounds}.")
            if self.maximise:
                self.bounds = (-self.bounds[1], -self.bounds[0])

    def is_composite(self) -> bool:
        """Check if this objective supports composition.

        Returns
        -------
        bool
            True if this objective supports composition, False otherwise.
        """
        return self.composite

    def composition(
        self, latent: torch.Tensor, params: dict[str, torch.Tensor]  # pylint: disable=W0613
    ) -> torch.Tensor:
        """Composition function for this objective, if supported.

        Parameters
        ----------
        latent : torch.Tensor
            Latent space values from the inner function.
        params : dict[str, torch.Tensor]
            Named parameters for the given result.

        Returns
        -------
        torch.Tensor
            Composition result.
        """
        # Under non-composite objectives, the composition is just the identity function
        if not self.is_composite():
            return latent
        raise NotImplementedError("Composition function not implemented for this objective.")

    def latent_size(self) -> int:
        """Return the size of the latent space for this objective.

        Returns
        -------
        int
            Size of the latent space.
        """
        # Under non-composite objectives, assume a size of 1 for the scalar value of the objective
        if not self.is_composite():
            return 1
        raise NotImplementedError("Latent size not implemented for this objective.")

    def has_variance(self) -> bool:
        """Check if this objective has variance information.

        Returns
        -------
        bool
            Whether this objective has variance information.
        """
        return self.variance

    def sign(self) -> int:
        """Get the sign for this objective, which is -1 for maximisation and 1 for minimisation.

        Returns
        -------
        int
            The sign for this objective.
        """
        return -1 if self.maximise else 1

    def get_obj_value(self, result: IndividualObjectiveResult) -> float:
        """Get the objective value from the result.

        Parameters
        ----------
        result : IndividualObjectiveResult
            The result containing the objective value.

        Returns
        -------
        float
            The objective value, negated if this is a maximisation objective.
        """
        return self.sign() * result.value

    def get_obj_variance(self, result: IndividualObjectiveResult) -> float:
        """Get the variance of the objective value from the result, if available.

        Parameters
        ----------
        result : IndividualObjectiveResult
            The result containing the objective value and variance information.

        Returns
        -------
        float
            The variance of the objective value, or 0 if not available.
        """
        return result.variance if self.has_variance() else 0.0

    def get_latent_values(self, result: IndividualObjectiveResult) -> np.ndarray:
        """Get the latent values from the result, if available.

        Parameters
        ----------
        result : IndividualObjectiveResult
            The result containing the objective value and latent information.

        Returns
        -------
        np.ndarray
            The latent values, or the objective value if not available.
        """
        if self.is_composite():
            if result.latent_values is None:
                raise ValueError("Latent values are required for composite objectives.")
            return result.latent_values
        return np.array([self.get_obj_value(result)])

    def get_latent_covariances(self, result: IndividualObjectiveResult) -> np.ndarray:
        """Get the latent covariances from the result, if available.

        Parameters
        ----------
        result : IndividualObjectiveResult
            The result containing the objective value and latent information.

        Returns
        -------
        np.ndarray
            The latent covariances. Four cases are possible:
            - Composite with variance: the latent covariances from the result.
            - Composite without variance: a zero matrix of appropriate size.
            - Non-composite with variance: a 1x1 matrix with the variance.
            - Non-composite without variance: a 1x1 zero matrix.
        """
        if self.is_composite() and self.has_variance():
            return result.latent_covariances
        if self.is_composite():
            return np.zeros((len(result.latent_values), len(result.latent_values)))
        if self.has_variance():
            return np.array([[result.variance]])
        return np.zeros((1, 1))

    def plot_case(self, case_hash: str, **kwargs) -> list[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting function

        Returns
        -------
        list[Figure]
            List of figures with the plot
        """
        raise NotImplementedError("Single case plotting not implemented for this objective")


class FunctionCallsFileManager:
    """Manager for the function calls file."""

    def __init__(
        self,
        file_path: str,
        settings: Settings,
        objectives: list[IndividualObjective],
        scalarisation: bool,
    ) -> None:
        self.file_path = file_path
        self.settings = settings
        self.objectives = objectives
        self.scalarisation = scalarisation

        # Base output formats
        time_spec = partial(TabularFloatColumn, width=15, notation='e', precision=8)
        obj_spec = partial(TabularFloatColumn, width=15, notation='e', precision=8)
        param_spec = partial(TabularFloatColumn, width=15, notation='f', precision=6)
        hash_spec = partial(TabularStringColumn, width=64)

        # Objective columns
        obj_columns = []
        if len(self.objectives) > 1:
            for i, objective in enumerate(self.objectives):
                obj_columns.append(obj_spec(f"Objective_{i + 1}"))
                if objective.has_variance():
                    obj_columns.append(obj_spec(f"Variance_{i + 1}"))
        # Scalar objective value and variance (if available)
        if self.scalarisation is not None or len(self.objectives) == 1:
            obj_columns.append(obj_spec("Objective"))
            if any(obj.has_variance() for obj in self.objectives):
                obj_columns.append(obj_spec("Variance"))

        # Parameter columns
        param_columns = [param_spec(name) for name in self.settings.parameters.get_scalar_names()]

        # Build the full column list
        self.file = TabularFile(
            file_path,
            columns=[
                time_spec("Start Time /s"),
                time_spec("Run Time /s"),
                *obj_columns,
                *param_columns,
                hash_spec("Hash"),
            ],
        )

    def prepare(self) -> None:
        """Prepare the function calls file."""
        self.file.prepare()

    def write(
        self,
        begin_time: float,
        run_time: float,
        result: ObjectiveResult,
        params: ParameterValues,
    ) -> None:
        """Write the function call information to the function calls file.

        Parameters
        ----------
        begin_time : float
            Start time of the function call.
        run_time : float
            Run time of the function call.
        result : ObjectiveResult
            Result of the objective evaluation.
        params : ParameterValues
            Named set of parameter values at which the objective was evaluated.
        """
        # Objective values and variances
        obj_values = []
        if len(self.objectives) > 1:
            for i in range(len(self.objectives)):
                obj_values.append(result.obj_values[i])
                if self.objectives[i].has_variance():
                    obj_values.append(result.obj_variances[i])
        elif not self.scalarisation:
            obj_values.append(result.obj_values[0])
            if self.objectives[0].has_variance():
                obj_values.append(result.obj_variances[0])

        # Scalar objective value and variance (if available)
        if self.scalarisation and len(self.objectives) > 1:
            obj_values.append(result.scalar_value)
            if any(obj.has_variance() for obj in self.objectives):
                obj_values.append(result.scalar_variance)

        # Write to file
        self.file.write_row([
            begin_time,
            run_time,
            *obj_values,
            *params.scalar_values.values(),
            params.param_hash,
        ])

    def read(self) -> FunctionCallsData:
        """Read and parse the function calls file into a dictionary.

        Returns
        -------
        FunctionCallsData
            Data of the function calls file.
        """
        # Read the function calls file
        data = self.file.read()

        # Parse mandatory fields
        start_times = np.array(data["Start Time /s"])
        run_times = np.array(data["Run Time /s"])
        param_names = self.settings.parameters.get_scalar_names(include_computed=False)
        params = np.array([
            [data[name][i] for name in param_names] for i in range(len(data["Hash"]))
        ])
        hashes = data["Hash"]

        # Parse optional fields
        obj_variances = None
        scalar_values = None
        scalar_variances = None
        if len(self.objectives) > 1:
            obj_values = np.array([
                data[f"Objective_{i + 1}"] for i in range(self.num_objectives())
            ]).T
            if self.has_variance():
                obj_variances = np.array([
                    data[f"Variance_{i + 1}"] for i in range(self.num_objectives())
                ]).T
        else:
            obj_values = np.array(data["Objective"])
            if self.has_variance():
                obj_variances = np.array(data["Variance"])
            if self.scalarisation is not None:
                scalar_values = np.array(data["Objective"])
                if self.has_variance():
                    scalar_variances = np.array(data["Variance"])

        return FunctionCallsData(
            start_times=start_times,
            run_times=run_times,
            params=params,
            hashes=hashes,
            obj_values=obj_values,
            obj_variances=obj_variances,
            scalar_values=scalar_values,
            scalar_variances=scalar_variances,
        )


class Scalarisation(ABC):
    """Base class for scalarisations."""

    def __init__(self, objectives: list[IndividualObjective]) -> None:
        self.objectives = objectives
        self.weights = torch.tensor([obj.weight for obj in objectives], dtype=torch.float64)
        self.bounds = (
            torch.tensor([obj.bounds for obj in objectives], dtype=torch.float64)
            if all(obj.bounds is not None for obj in objectives)
            else None
        )

    def scalarise(
        self,
        values: np.ndarray,
        variances: Optional[np.ndarray] = None,
    ) -> tuple[float, Optional[float]]:
        """Scalarise a set of objectives.

        Parameters
        ----------
        values : np.ndarray
            Mean objective values.
        variances : Optional[np.ndarray]
            Optional variances of the objectives.

        Returns
        -------
        tuple[float, Optional[float]]
            Mean and variance of the scalarised objective.
        """
        torch_mean, torch_var = self.scalarise_torch(
            torch.from_numpy(values),
            torch.from_numpy(variances) if variances is not None else None,
        )
        if torch_var is None:
            return torch_mean.numpy(force=True).item(), None
        return torch_mean.item(), torch_var.item()

    @abstractmethod
    def scalarise_torch(
        self,
        values: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scalarise a set of objectives with gradients.

        Parameters
        ----------
        values : torch.Tensor
            Mean objective values.
        variances : Optional[torch.Tensor]
            Optional variances of the objectives.

        Returns
        -------
        tuple[torch.Tensor, Optional[torch.Tensor]]
            Mean and variance of the scalarised objective.
        """


class Objective(ABC):
    """Abstract class for optimisation objectives"""

    def __init__(
        self,
        settings: Settings,
        objectives: list[IndividualObjective],
        scalarisation: Scalarisation = None,
        composite: bool = False,
    ) -> None:
        self.settings = settings
        self.objectives = objectives
        self.scalarisation = scalarisation
        self.composite = composite or any(obj.is_composite() for obj in objectives)
        self.num_calls = 0
        self.begin_time = time.perf_counter()
        self.mutex = Lock()

        # Set up the concat utility for composition if necessary
        if self.is_composite():
            self.concat_utility = ConcatUtility([obj.latent_size() for obj in objectives])

        # Set up the output file for function calls
        self.func_calls_file = None
        if self.settings.output_dir:
            self.func_calls_file = FunctionCallsFileManager(
                os.path.join(self.settings.output_dir, "func_calls"),
                settings,
                objectives,
                scalarisation is not None,
            )

    def prepare(self) -> None:
        """Prepare output files before optimising the problem. This creates output files."""
        if self.func_calls_file is not None:
            self.func_calls_file.prepare()

    def is_composite(self) -> bool:
        """Check if this objective supports composition.

        Returns
        -------
        bool
            True if this objective supports composition, False otherwise.
        """
        return self.composite

    def num_objectives(self) -> int:
        """Get the number of objectives.

        Returns
        -------
        int
            The number of objectives.
        """
        return len(self.objectives)

    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective problem.

        Returns
        -------
        bool
            Whether this is a multi-objective problem.
        """
        return self.num_objectives() > 1 and self.scalarisation is None

    def has_variance(self) -> bool:
        """Check if this objective has variance information.

        Returns
        -------
        bool
            Whether this objective has variance information.
        """
        return any(obj.has_variance() for obj in self.objectives)

    def latent_size(self) -> int:
        """Return the size of the latent space for this objective.

        Returns
        -------
        int
            Size of the latent space.
        """
        if not self.is_composite():
            raise RuntimeError("Latent size is not defined for non-composite objectives.")
        return self.concat_utility.length()

    def get_objective_value(self, result: ObjectiveResult) -> float:
        """Get the scalar objective value from the result, if available.

        Parameters
        ----------
        result : ObjectiveResult
            The result containing the objective values and scalarised value.

        Returns
        -------
        float
            The scalar objective value.
        """
        if self.is_multi_objective():
            raise RuntimeError(
                "Scalarised objective value is not defined for multi-objective problems."
            )

        if self.scalarisation is None:
            return result.obj_values.item()
        return result.scalar_value

    def get_objective_variance(self, result: ObjectiveResult) -> float:
        """Get the scalar objective variance from the result, if available.

        Parameters
        ----------
        result : ObjectiveResult
            The result containing the objective values and scalarised value.

        Returns
        -------
        float
            The scalar objective variance.
        """
        if self.is_multi_objective():
            raise RuntimeError(
                "Scalarised objective variance is not defined for multi-objective problems."
            )

        if self.scalarisation is None:
            if result.obj_variances is None:
                raise RuntimeError("Objective variance is not available.")
            return result.obj_variances.item()
        if result.scalar_variance is None:
            raise RuntimeError("Scalarised objective variance is not available.")
        return result.scalar_variance

    def composition(self, latent: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Composition function for this objective, if supported.

        Parameters
        ----------
        latent : torch.Tensor
            Latent space values from the inner function.
        params : torch.Tensor
            Parameters for the given result.

        Returns
        -------
        torch.Tensor
            Composition result.
        """
        if not self.is_composite():
            raise RuntimeError("Composition function is not defined for non-composite objectives.")

        # Split the latent space into the individual objectives
        latent_responses = self.concat_utility.split(latent)

        # Unpack the parameters
        params_dict = self.settings.parameters.to_torch_dict(params)

        # When the inner objective is non-composite, we treat the scalar value as a
        # single-dimensional latent space
        objectives = torch.stack([
            (obj.composition(lat, params_dict) if obj.is_composite() else lat) * obj.sign()
            for lat, obj in zip(latent_responses, self.objectives)
        ], dim=-1)

        # Scalarise the objectives if necessary
        if self.scalarisation is not None:
            objectives, _ = self.scalarisation.scalarise_torch(objectives)
        elif not self.is_multi_objective():
            # For single-objective problems without scalarisation, we need to squeeze the last dim
            objectives = objectives.squeeze(-1)

        return objectives

    def __build_objective_result(self, results: list[IndividualObjectiveResult]) -> ObjectiveResult:
        """Build the full objective result from the individual objective results and parameters.

        Parameters
        ----------
        results : list[IndividualObjectiveResult]
            List of individual objective results.

        Returns
        -------
        ObjectiveResult
            The full objective result.
        """
        # Extract objective values and variances
        obj_values = np.array([
            obj.get_obj_value(res) for res, obj in zip(results, self.objectives)
        ])
        obj_variances = None
        if self.has_variance():
            obj_variances = np.array([
                obj.get_obj_variance(res) for res, obj in zip(results, self.objectives)
            ])

        # Scalarise if necessary
        scalar_value, scalar_variance = None, None
        if self.scalarisation is not None:
            scalar_value, scalar_variance = self.scalarisation.scalarise(obj_values, obj_variances)

        # Under composition, build the latent space values and covariances:
        # For composite individual objectives, we use the latent values and covariances directly.
        # For non-composite individual objectives, we treat the scalar value as a single-dimensional
        # latent space and the variance as its covariance.
        latent_values, latent_covariances = None, None
        if self.is_composite():
            latent_values = self.concat_utility.concat([
                obj.get_latent_values(res) for res, obj in zip(results, self.objectives)
            ])
            if self.has_variance():
                latent_covariances = self.concat_utility.concat_covar([
                    obj.get_latent_covariances(res) for res, obj in zip(results, self.objectives)
                ])

        return ObjectiveResult(
            results=results,
            obj_values=obj_values,
            obj_variances=obj_variances,
            scalar_value=scalar_value,
            scalar_variance=scalar_variance,
            latent_values=latent_values,
            latent_covariances=latent_covariances,
        )

    def __call__(self, params: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for the outside world.

        Handles scalarisation, composition and output file writing.

        Parameters
        ----------
        params : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        ObjectiveResult
            Objective result.
        """
        # Convert parameter values
        param_values = self.settings.parameters.to_values(params)

        # Evaluate objective(s) and build the full result
        begin_time = time.perf_counter()
        individual_obj = self._objective(param_values, concurrent=concurrent)
        result = self.__build_objective_result(individual_obj)
        end_time = time.perf_counter()

        # Update outputs
        with self.mutex:
            self.num_calls += 1
            if self.func_calls_file is not None:
                self.func_calls_file.write(
                    begin_time - self.begin_time, end_time - begin_time, result, param_values
                )
        return result

    def read_func_calls(self) -> FunctionCallsData:
        """Read and parse the function calls file into a dictionary.

        Returns
        -------
        FunctionCallsData
            Data of the function calls file.
        """
        if self.func_calls_file is None:
            raise RuntimeError("No function calls file available for this objective.")
        return self.func_calls_file.read()

    def plot_case(self, case_hash: str, **kwargs) -> list[Figure]:
        """Plot a given function call given the parameter hash.

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting function

        Returns
        -------
        list[Figure]
            List of figures with the plot
        """
        return [fig for obj in self.objectives for fig in obj.plot_case(case_hash, **kwargs)]

    def plot_best(self) -> list[Figure]:
        """Plot the current best case

        Returns
        -------
        list[Figure]
            List of figures with the plot
        """
        data = self.read_func_calls()

        # Single-objective case: find best case based on the objective value
        if not self.is_multi_objective():
            obj_vals = data.obj_values if self.scalarisation is None else data.scalar_values
            return self.plot_case(data.hashes[np.argmin(obj_vals)])

        # Multi-objective case: we plot the best individual objective for each objective
        figures = []
        for i, obj in enumerate(self.objectives):
            best_hash = data.hashes[np.argmin(data.obj_values[:, i])]
            figures.extend(
                self.plot_case(best_hash, title=f"Best case for objective {i + 1} ({obj.name})")
            )
        return figures

    @abstractmethod
    def _objective(
        self, params: ParameterValues, concurrent: bool = False
    ) -> list[IndividualObjectiveResult]:
        """Abstract method for objective computation.

        Parameters
        ----------
        params : ParameterValues
            Named set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        list[IndividualObjectiveResult]
            List of individual objective results.
        """

    @classmethod
    @abstractmethod
    def read(
        cls: type[T],
        config: dict[str, Any],
        settings: Settings,
    ) -> T:
        """Read the objective from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Terms from the configuration dictionary.
        settings : Settings
            Global settings for this piglot run.

        Returns
        -------
        Objective
            Objective function to optimise.
        """
