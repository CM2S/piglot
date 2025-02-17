"""Module containing optimisation objective primites"""
from __future__ import annotations
import os
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple
from threading import Lock
from dataclasses import dataclass
import numpy as np
import torch
import pandas as pd
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet


T = TypeVar('T', bound='Objective')


class Composition(ABC):
    """Abstract class for defining composition functionals with gradients"""

    def composition(self, inner: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Abstract method for computing the outer function of the composition

        Parameters
        ----------
        inner : np.ndarray
            Return value from the inner function
        params : np.ndarray
            Parameters for the given responses

        Returns
        -------
        np.ndarray
            Composition result
        """
        result = self.composition_torch(torch.from_numpy(inner), torch.from_numpy(params))
        return result.numpy(force=True)

    @abstractmethod
    def composition_torch(self, inner: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Abstract method for computing the outer function of the composition with gradients

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function
        params : torch.Tensor
            Parameters for the given responses

        Returns
        -------
        torch.Tensor
            Composition result
        """


class DynamicPlotter(ABC):
    """Abstract class for dynamically-updating plots"""

    @abstractmethod
    def update(self) -> None:
        """Update the plot with the most recent data"""


@dataclass
class ObjectiveResult:
    """Container for objective results."""
    params: np.ndarray
    values: np.ndarray
    obj_values: np.ndarray
    covariances: Optional[np.ndarray] = None
    obj_variances: Optional[np.ndarray] = None
    scalar_value: Optional[float] = None
    scalar_variance: Optional[float] = None


class IndividualObjective(ABC):
    """Base class for individual objectives for generic optimisation problems."""

    def __init__(
        self,
        maximise: bool = False,
        weight: float = 1.0,
        bounds: Tuple[float, float] = None,
    ) -> None:
        self.maximise = maximise
        self.weight = float(weight)
        self.bounds = None
        if bounds is not None:
            self.bounds = tuple(float(b) for b in bounds)
            if self.bounds[0] > self.bounds[1]:
                raise ValueError(f"Invalid bounds {self.bounds}.")
            if self.maximise:
                self.bounds = (-self.bounds[1], -self.bounds[0])


class Scalarisation(ABC):
    """Base class for scalarisations."""

    def __init__(self, objectives: List[IndividualObjective]) -> None:
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
    ) -> Tuple[float, Optional[float]]:
        """Scalarise a set of objectives.

        Parameters
        ----------
        values : np.ndarray
            Mean objective values.
        variances : Optional[np.ndarray]
            Optional variances of the objectives.

        Returns
        -------
        Tuple[float, Optional[float]]
            Mean and variance of the scalarised objective.
        """
        torch_mean, torch_var = self.scalarise_torch(
            torch.from_numpy(values),
            torch.from_numpy(variances) if variances is not None else None,
        )
        if torch_var is None:
            return torch_mean.numpy(force=True), None
        return torch_mean.item(), torch_var.item()

    @abstractmethod
    def scalarise_torch(
        self,
        values: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scalarise a set of objectives with gradients.

        Parameters
        ----------
        values : torch.Tensor
            Mean objective values.
        variances : Optional[torch.Tensor]
            Optional variances of the objectives.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Mean and variance of the scalarised objective.
        """


class Objective(ABC):
    """Abstract class for optimisation objectives"""

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> ObjectiveResult:
        """Objective computation for the outside world"""

    @abstractmethod
    def prepare(self) -> None:
        """Generic method to prepare output files before optimising the problem"""

    @classmethod
    @abstractmethod
    def read(
        cls: Type[T],
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> T:
        """Read the objective from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.
        parameters : ParameterSet
            Set of parameters for this problem.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        Objective
            Objective function to optimise.
        """

    def plot_case(self, case_hash: str, options: Dict[str, Any] = None) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        options : Dict[str, Any], optional
            Options to pass to the plotting function, by default None

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        raise NotImplementedError("Single case plotting not implemented for this objective")

    def plot_best(self) -> List[Figure]:
        """Plot the current best case

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        raise NotImplementedError("Best case plotting not implemented for this objective")

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of a updatable plots
        """
        raise NotImplementedError("Current case plotting not implemented for this objective")

    def get_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of objective history
        """
        raise NotImplementedError("Objective history not implemented for this objective")


class GenericObjective(Objective):
    """Class for generic objectives."""

    def __init__(
        self,
        parameters: ParameterSet,
        stochastic: bool = False,
        composition: Composition = None,
        scalarisation: Scalarisation = None,
        num_objectives: int = 1,
        multi_objective: bool = False,
        output_dir: str = None,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.output_dir = output_dir
        self.stochastic = stochastic
        self.scalarisation = scalarisation
        self.composition = composition
        self.num_objectives = num_objectives
        self.multi_objective = multi_objective
        self.func_calls = 0
        self.begin_time = time.perf_counter()
        self.__mutex = Lock()
        self.func_calls_file = os.path.join(output_dir, "func_calls") if output_dir else None

    def prepare(self) -> None:
        """Prepare output directories for the optimsation."""
        super().prepare()
        if self.output_dir:
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}')
                # When we have multiple objectives, write each one
                if self.num_objectives > 1:
                    for i in range(self.num_objectives):
                        file.write(f'\t{"Objective_" + str(i + 1):>15}')
                        if self.stochastic:
                            file.write(f'\t{"Variance_" + str(i + 1):>15}')
                # But we also write the scalarised objective if possible
                if not self.multi_objective:
                    file.write(f'\t{"Objective":>15}')
                    if self.stochastic:
                        file.write(f'\t{"Variance":>15}')
                for param in self.parameters:
                    file.write(f"\t{param.name:>15}")
                file.write(f'\t{"Hash":>64}\n')

    @abstractmethod
    def _objective(self, params: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Abstract method for objective computation.

        Parameters
        ----------
        params : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        ObjectiveResult
            Objective result.
        """

    def __call__(self, params: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for the outside world. Also handles output file writing.

        Parameters
        ----------
        params : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        ObjectiveResult
            Objective result.
        """
        # Evaluate objective
        self.func_calls += 1
        begin_time = time.perf_counter()
        result = self._objective(params, concurrent=concurrent)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(os.path.join(self.func_calls_file), 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    # Write out each objective value
                    if self.num_objectives > 1:
                        if self.stochastic:
                            for val, var in zip(result.obj_values, result.obj_variances):
                                file.write(f'{val:>15.8e}\t{var:>15.8e}\t')
                        else:
                            for val in result.obj_values:
                                file.write(f'{val:>15.8e}\t')
                    # Write out the scalarised objective value
                    if not self.multi_objective:
                        file.write(f'{result.scalar_value:>15.8e}\t')
                        if self.stochastic:
                            file.write(f'{result.scalar_variance:>15.8e}\t')
                    for val in params:
                        file.write(f"{val:>15.6f}\t")
                    file.write(f'{self.parameters.hash(params)}\n')
        return result

    def plot_best(self) -> List[Figure]:
        """Plot the current best case.

        Returns
        -------
        List[Figure]
            List of figures with the plot.
        """
        # Build the objective list
        objective_list = ["Objective"]
        if self.multi_objective:
            objective_list = [f"Objective_{i + 1}" for i in range(self.num_objectives)]
        # Plot the best case for each objective
        figures = []
        for i, objective in enumerate(objective_list):
            # Find hash associated with the best case
            df = pd.read_table(self.func_calls_file)
            df.columns = df.columns.str.strip()
            min_series = df.iloc[df[objective].idxmin()]
            call_hash = str(min_series["Hash"])
            # Use the single case plotting utility
            options = {'append_title': f'Best {objective} eval'} if self.multi_objective else None
            figures += self.plot_case(call_hash, options=options)
            # Also display the best case
            print(f"Best run{' (' + objective + ')'}:")
            print(min_series.drop(objective_list + ["Hash"]))
            print(f"Hash: {call_hash}")
            print(f"{objective}: {min_series[objective]:15.8e}\n")
        return figures

    def get_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the objective history.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of objective history.
        """
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        x_axis = df["Start Time /s"] + df["Run Time /s"]
        params = df[[param.name for param in self.parameters]]
        param_hash = df["Hash"].to_list()
        # Multi-objective case
        if self.multi_objective:
            values = [df[f"Objective_{i + 1}"] for i in range(self.num_objectives)]
            if self.stochastic:
                variances = [df[f"Variance_{i + 1}"] for i in range(self.num_objectives)]
            return_dict = {}
            for i in range(self.num_objectives):
                result = {
                    "time": x_axis.to_numpy(),
                    "values": values[i],
                    "params": params.to_numpy(),
                    "hashes": param_hash,
                }
                if self.stochastic:
                    result["variances"] = variances[i]
                return_dict[f"Objective_{i + 1}"] = result
            return return_dict
        # Single objective case
        result = {
            "time": x_axis.to_numpy(),
            "values": df["Objective"].to_numpy(),
            "params": params.to_numpy(),
            "hashes": param_hash,
        }
        if self.stochastic:
            result["variances"] = df["Variance"].to_numpy()
        return {"Objective": result}
