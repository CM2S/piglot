"""Module containing optimisation objective primites"""
from __future__ import annotations
import os
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from threading import Lock
from dataclasses import dataclass
import numpy as np
import torch
import pandas as pd
from scipy.stats import norm
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet


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


class Objective(ABC):
    """Abstract class for optimisation objectives"""

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Objective computation for the outside world"""

    @abstractmethod
    def prepare(self) -> None:
        """Generic method to prepare output files before optimising the problem"""

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
            ) -> Objective:
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
        raise NotImplementedError("Reading not implemented for this objective")

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


@dataclass
class ObjectiveResult:
    """Container for objective results."""
    params: np.ndarray
    values: List[np.ndarray]
    scalarisation: str
    variances: Optional[List[np.ndarray]] = None
    weights: Optional[np.ndarray] = None
    bounds: Optional[np.ndarray] = None
    types: Optional[List[bool]] = None

    def __mc_variance(
        self,
        composition: Composition,
        num_samples: int = 1024,
    ) -> float:
        """Compute the objective variance using Monte Carlo (using fixed base samples).

        Parameters
        ----------
        composition : Composition
            Composition functional to use.
        num_samples : int, optional
            Number of Monte Carlo samples, by default 1000.

        Returns
        -------
        float
            Estimated variance of the objective.
        """
        biased = [norm.rvs(loc=0, scale=1) for _ in range(num_samples)]
        mc_objectives = [
            composition.composition(
                self.values + bias * np.sqrt(self.variances),
                self.params,
            ) for bias in biased
        ]
        return np.var(mc_objectives, axis=0)

    @staticmethod
    def normalise_objective(objective: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Normalise the objective.

        Parameters
        ----------
        objective : np.ndarray
            Objective to normalise.
        bounds : np.ndarray
            Bounds for the objectives.

        Returns
        -------
        np.ndarray
            Normalised objective.
        """
        # Normalise the objective
        return (objective - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    def scalarise(self, composition: Composition = None) -> float:
        """Scalarise the result under noise-free single-objective optimisation.

        Parameters
        ----------
        composition : Composition, optional
            Composition functional to use, by default None.

        Returns
        -------
        float
            Scalarised result.
        """
        if composition is None:
            # Sanitise scalarisation method
            if self.scalarisation not in ('mean', 'stch', 'linear'):
                raise ValueError(
                    f"Invalid scalarisation '{self.scalarisation}'. Use 'mean', 'stch' or 'linear'."
                )
            if self.scalarisation in ('stch', 'linear'):
                # Sanitise the weights
                weights = np.array(self.weights)
                if np.sum(weights) != 1:
                    raise ValueError(f'Weights must sum to 1.0, got {np.sum(weights)}.')
                # Set all the objectives to be positive
                values = abs(np.array(self.values))
                # Set the bounds and types
                bounds = np.array(self.bounds)
                types = np.array(self.types)
                # Calculate the costs
                costs = np.where(types, -1, 1)
                # Calculate the normalised objective values
                norm_funcs = self.normalise_objective(values, bounds)
                if self.scalarisation == 'stch':
                    # Calculate the ideal point
                    ideal_point = np.where(types, 1, 0)
                    # Smoothing parameter for STCH
                    u = 0.01
                    # Calculate the Tchebycheff function value
                    tch_values = (np.abs((norm_funcs - ideal_point) * costs) / u) * weights
                    return np.log(np.sum(np.exp(tch_values))) * u
                return np.sum((norm_funcs*costs)*weights)
            return np.mean(self.values)
        return composition.composition(self.values, self.params).item()

    def scalarise_mo(self, composition: Composition = None) -> List[float]:
        """Pseudo-scalarise the result under noise-free multi-objective optimisation.

        Parameters
        ----------
        composition : Composition, optional
            Composition functional to use, by default None.

        Returns
        -------
        List[float]
            Pseudo-scalarised result.
        """
        if composition is None:
            return [np.mean(vals) for vals in self.values]
        return [val.item() for val in composition.composition(self.values, self.params)]

    def scalarise_stochastic(
        self,
        composition: Composition = None,
    ) -> Tuple[float, float]:
        """Scalarise the result.

        Parameters
        ----------
        composition : Composition, optional
            Composition functional to use, by default None.

        Returns
        -------
        Tuple[float, float]
            Scalarised mean and variance.
        """
        if composition is None:
            return np.mean(self.values), np.sum(self.variances)
        return (
            composition.composition(self.values, self.params).item(),
            self.__mc_variance(composition).item()
        )

    def scalarise_mo_stochastic(
            self,
            composition: Composition = None,
    ) -> Tuple[List[float], List[float]]:
        """Pseudo-scalarise the result under stochastic multi-objective optimisation.

        Parameters
        ----------
        composition : Composition, optional
            Composition functional to use, by default None.

        Returns
        -------
        List[Tuple[float, float]]
            Pseudo-scalarised means and variances.
        """
        if composition is None:
            return [
                (np.mean(vals), np.sum(vars)) for vals, vars in zip(self.values, self.variances)
            ]
        return composition.composition(self.values, self.params), self.__mc_variance(composition)


class GenericObjective(Objective):
    """Class for generic objectives."""

    def __init__(
            self,
            parameters: ParameterSet,
            stochastic: bool = False,
            composition: Composition = None,
            num_objectives: int = 1,
            output_dir: str = None,
            ) -> None:
        super().__init__()
        self.parameters = parameters
        self.output_dir = output_dir
        self.stochastic = stochastic
        self.composition = composition
        self.num_objectives = num_objectives
        self.multi_objective = num_objectives > 1
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
                if self.multi_objective:
                    for i in range(self.num_objectives):
                        file.write(f'\t{"Objective_" + str(i + 1):>15}')
                        if self.stochastic:
                            file.write(f'\t{"Variance_" + str(i + 1):>15}')
                else:
                    file.write(f'\t{"Objective":>15}')
                    if self.stochastic:
                        file.write(f'\t{"Variance":>15}')
                for param in self.parameters:
                    file.write(f"\t{param.name:>15}")
                file.write(f'\t{"Hash":>64}\n')

    @abstractmethod
    def _objective(self, values: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Abstract method for objective computation.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        ObjectiveResult
            Objective result.
        """

    def __call__(self, values: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for the outside world. Also handles output file writing.

        Parameters
        ----------
        values : np.ndarray
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
        objective_result = self._objective(values, concurrent=concurrent)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(os.path.join(self.func_calls_file), 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    if self.multi_objective:
                        if self.stochastic:
                            vals, vars = objective_result.scalarise_mo_stochastic(self.composition)
                            for value, var in zip(vals, vars):
                                file.write(f'{value:>15.8e}\t{var:>15.8e}\t')
                        else:
                            for val in objective_result.scalarise_mo(self.composition):
                                file.write(f'{val:>15.8e}\t')
                    else:
                        if self.stochastic:
                            value, var = objective_result.scalarise_stochastic(self.composition)
                            file.write(f'{value:>15.8e}\t{var:>15.8e}\t')
                        else:
                            file.write(f'{objective_result.scalarise(self.composition):>15.8e}\t')
                    for val in values:
                        file.write(f"{val:>15.6f}\t")
                    file.write(f'{self.parameters.hash(values)}\n')
        return objective_result

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
            options = {'append_title': objective} if self.multi_objective else None
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
