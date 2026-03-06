"""Module containing optimisation objective primites"""
from __future__ import annotations
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
from piglot.settings import Settings
from piglot.utils.composition import ConcatUtility, CompositionMixin
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


class IndividualObjective(CompositionMixin, ABC):
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
        super().__init__(composite)
        self.name = name
        self.maximise = maximise
        self.weight = float(weight)
        self.variance = variance
        self.bounds = None
        if bounds is not None:
            self.bounds = tuple(float(b) for b in bounds)
            if self.bounds[0] > self.bounds[1]:
                raise ValueError(f"Invalid bounds {self.bounds}.")
            if self.maximise:
                self.bounds = (-self.bounds[1], -self.bounds[0])

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


class Objective(CompositionMixin, ABC):
    """Abstract class for optimisation objectives"""

    def __init__(
        self,
        settings: Settings,
        objectives: list[IndividualObjective],
        scalarisation: Scalarisation = None,
        composite: bool = False,
    ) -> None:
        super().__init__(composite or any(obj.is_composite() for obj in objectives))
        self.settings = settings
        self.objectives = objectives
        self.scalarisation = scalarisation
        self.num_calls = 0
        self.begin_time = time.perf_counter()
        self.mutex = Lock()

        # Set up the concat utility for composition if necessary
        if self.is_composite():
            self.concat_utility = ConcatUtility([obj.latent_size() for obj in objectives])

        # Set up the output file for function calls
        self.func_calls_file = None
        if self.settings.output_dir:
            self.func_calls_file = self.__prepare_func_calls(
                os.path.join(self.settings.output_dir, "func_calls")
            )

    def prepare(self) -> None:
        """Prepare output files before optimising the problem. This creates output files."""
        if self.func_calls_file is not None:
            self.func_calls_file.prepare()

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

        # When the inner objective is non-composite, we treat the scalar value as a
        # single-dimensional latent space
        objectives = torch.stack([
            (obj.composition(lat, params) if obj.is_composite() else lat) * obj.sign()
            for lat, obj in zip(latent_responses, self.objectives)
        ], dim=-1)

        # Scalarise the objectives if necessary
        if self.scalarisation is not None:
            objectives, _ = self.scalarisation.scalarise_torch(objectives)
        elif not self.is_multi_objective():
            # For single-objective problems without scalarisation, we need to squeeze the last dim
            objectives = objectives.squeeze(-1)

        return objectives

    def __call__(self, params: np.ndarray, *args: Any, **kwargs: Any) -> ObjectiveResult:
        """Objective computation for the outside world.

        Handles scalarisation, composition and output file writing.

        Parameters
        ----------
        params : np.ndarray
            Set of parameters to evaluate the objective for.
        *args, **kwargs
            Additional arguments to pass to the evaluation function.

        Returns
        -------
        ObjectiveResult
            Objective result.
        """
        begin_time = time.perf_counter()

        # Evaluate objective(s)
        results = self._objective(params, *args, **kwargs)

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

        # Construct the full objective result
        result = ObjectiveResult(
            results=results,
            obj_values=obj_values,
            obj_variances=obj_variances,
            scalar_value=scalar_value,
            scalar_variance=scalar_variance,
            latent_values=latent_values,
            latent_covariances=latent_covariances,
        )

        # Update outputs
        end_time = time.perf_counter()
        with self.mutex:
            self.num_calls += 1
            self.__dump_call(begin_time - self.begin_time, end_time - begin_time, result, params)
        return result

    def __prepare_func_calls(self, file_path: str) -> TabularFile:
        """Prepare the function calls file based on the objectives and settings.

        Returns
        -------
        TabularFile
            list of columns for the function calls file.
        """
        # Base output formats
        time_spec = partial(TabularFloatColumn, width=15, notation='e', precision=8)
        obj_spec = partial(TabularFloatColumn, width=15, notation='e', precision=8)
        param_spec = partial(TabularFloatColumn, width=15, notation='f', precision=6)
        hash_spec = partial(TabularStringColumn, width=64)

        # Objective columns
        obj_columns = []
        if self.num_objectives() > 1:
            for i, objective in enumerate(self.objectives):
                obj_columns.append(obj_spec(f"Objective_{i + 1}"))
                if objective.has_variance():
                    obj_columns.append(obj_spec(f"Variance_{i + 1}"))
        # Scalar objective value and variance (if available)
        if self.scalarisation is not None or self.num_objectives() == 1:
            obj_columns.append(obj_spec("Objective"))
            if self.has_variance():
                obj_columns.append(obj_spec("Variance"))

        # Parameter columns
        param_columns = [param_spec(param.name) for param in self.settings.parameters]

        # Build the full column list
        return TabularFile(
            file_path,
            columns=[
                time_spec("Start Time /s"),
                time_spec("Run Time /s"),
                *obj_columns,
                *param_columns,
                hash_spec("Hash"),
            ],
        )

    def __dump_call(
        self,
        begin_time: float,
        run_time: float,
        result: ObjectiveResult,
        params: np.ndarray,
    ) -> None:
        """Dump the function call information to the function calls file.

        Parameters
        ----------
        begin_time : float
            Start time of the function call.
        run_time : float
            Run time of the function call.
        result : ObjectiveResult
            Result of the objective evaluation.
        params : np.ndarray
            Parameters at which the objective was evaluated.
        """
        if self.func_calls_file is not None:
            # Objective values and variances
            obj_values = []
            if self.num_objectives() > 1:
                for i in range(self.num_objectives()):
                    obj_values.append(result.obj_values[i])
                    if self.objectives[i].has_variance():
                        obj_values.append(result.obj_variances[i])
            else:
                obj_values.append(result.obj_values[0])
                if self.objectives[0].has_variance():
                    obj_values.append(result.obj_variances[0])

            # Scalar objective value and variance (if available)
            if self.scalarisation is not None:
                obj_values.append(result.scalar_value)
                if self.has_variance():
                    obj_values.append(result.scalar_variance)

            # Write to file
            self.func_calls_file.write_row([
                begin_time,
                run_time,
                *obj_values,
                *params,
                self.settings.parameters.hash(params),
            ])

    @abstractmethod
    def _objective(
        self, params: np.ndarray, concurrent: bool = False
    ) -> list[IndividualObjectiveResult]:
        """Abstract method for objective computation.

        Parameters
        ----------
        params : np.ndarray
            Set of parameters to evaluate the objective for.
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
