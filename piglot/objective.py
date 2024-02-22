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

    @abstractmethod
    def composition(self, inner: np.ndarray) -> float:
        """Abstract method for computing the outer function of the composition

        Parameters
        ----------
        inner : np.ndarray
            Return value from the inner function

        Returns
        -------
        float
            Scalar composition result
        """

    @abstractmethod
    def composition_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Abstract method for computing the outer function of the composition with gradients

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function

        Returns
        -------
        torch.Tensor
            Scalar composition result
        """

    def __call__(self, inner: np.ndarray) -> float:
        """Compute the composition result for the outer world

        Parameters
        ----------
        inner : np.ndarray
            Return value from the inner function

        Returns
        -------
        float
            Scalar composition result
        """
        return self.composition(inner)


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
    values: List[np.ndarray]
    variances: Optional[List[np.ndarray]] = None

    def scalarise(self, composition: Composition = None) -> float:
        """Scalarise the result.

        Parameters
        ----------
        composition : Composition, optional
            Composition functional to use, by default None.

        Returns
        -------
        float
            Scalarised result.
        """
        if composition is not None:
            return composition(np.concatenate(self.values))
        return np.mean(self.values)

    def scalarise_stochastic(self, composition: Composition = None) -> Tuple[float, float]:
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
        if composition is not None:
            values = np.concatenate(self.values)
            variances = np.concatenate(self.variances)
            # Compute the objective variance using Monte Carlo (using fixed base samples)
            biased = [norm.rvs(loc=0, scale=1) for _ in range(1000)]
            mc_objectives = [composition(values + bias * np.sqrt(variances)) for bias in biased]
            return composition(values), np.var(mc_objectives)
        return np.mean(self.values), np.sum(self.variances)


class GenericObjective(Objective):
    """Class for generic objectives."""

    def __init__(
            self,
            parameters: ParameterSet,
            stochastic: bool = False,
            composition: Composition = None,
            output_dir: str = None,
            ) -> None:
        super().__init__()
        self.parameters = parameters
        self.output_dir = output_dir
        self.stochastic = stochastic
        self.composition = composition
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
                    if self.stochastic:
                        value, variance = objective_result.scalarise_stochastic(self.composition)
                        file.write(f'{value:>15.8e}\t{variance:>15.8e}')
                    else:
                        file.write(f'{objective_result.scalarise(self.composition):>15.8e}')
                    for i, param in enumerate(self.parameters):
                        file.write(f"\t{param.denormalise(values[i]):>15.6f}")
                    file.write(f'\t{self.parameters.hash(values)}\n')
        return objective_result

    def plot_best(self) -> List[Figure]:
        """Plot the current best case.

        Returns
        -------
        List[Figure]
            List of figures with the plot.
        """
        # Find hash associated with the best case
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        min_series = df.iloc[df["Objective"].idxmin()]
        call_hash = str(min_series["Hash"])
        # Use the single case plotting utility
        figures = self.plot_case(call_hash)
        # Also display the best case
        print("Best run:")
        print(min_series.drop(["Objective", "Hash"]))
        print(f"Hash: {call_hash}")
        print(f"Objective: {min_series['Objective']:15.8e}")
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
        result = {
            "time": x_axis.to_numpy(),
            "values": df["Objective"].to_numpy(),
            "params": params.to_numpy(),
            "hashes": param_hash,
        }
        if self.stochastic:
            result["variances"] = df["Variance"].to_numpy()
        return {"Objective": result}
