"""Module containing optimisation objective primites"""
from __future__ import annotations
import os
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import shutil
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



class MSEComposition(Composition):
    """Mean squared error outer composite function with gradients"""

    def composition(self, inner: np.ndarray) -> float:
        """Compute the MSE outer function of the composition

        Parameters
        ----------
        inner : np.ndarray
            Return value from the inner function

        Returns
        -------
        float
            Scalar composition result
        """
        return np.mean(np.square(inner), axis=-1)

    def composition_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Compute the MSE outer function of the composition with gradients

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function

        Returns
        -------
        torch.Tensor
            Scalar composition result
        """
        return torch.mean(torch.square(inner), dim=-1)



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
        raise NotImplementedError()

    def plot_case(self, case_hash: str, options: Dict[str, Any]=None) -> List[Figure]:
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

    def scalarise(self) -> float:
        """Scalarise the result.

        Returns
        -------
        float
            Scalarised result.
        """
        return np.mean([np.mean(val) for val in self.values])

    def scalarise_stochastic(self) -> Tuple[float, float]:
        """Scalarise the result.

        Returns
        -------
        Tuple[float, float]
            Scalarised mean and variance.
        """
        return (
            np.mean([np.mean(val) for val in self.values]),
            np.sum([np.var(val) for val in self.values]),
        )


class GenericObjective(Objective):
    """Class for generic objectives."""

    def __init__(
            self,
            parameters: ParameterSet,
            stochastic: bool=False,
            composition: Composition=None,
            output_dir: str=None,
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
    def _objective(self, values: np.ndarray, concurrent: bool=False) -> ObjectiveResult:
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

    def __call__(self, values: np.ndarray, concurrent: bool=False) -> ObjectiveResult:
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
                        value, variance = objective_result.scalarise_stochastic()
                        file.write(f'{value:>15.8e}\t{variance:>15.8e}')
                    else:
                        file.write(f'{objective_result.scalarise():>15.8e}')
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
        return {
            "Objective": {
                "time": x_axis.to_numpy(),
                "values": df["Objective"].to_numpy(),
                "params": params.to_numpy(),
                "hashes": param_hash,
            }
        }


class SingleObjective(Objective):
    """Abstract class for scalar single-objectives"""

    def __init__(self, parameters: ParameterSet, output_dir: str=None) -> None:
        super().__init__()
        self.parameters = parameters
        self.output_dir = output_dir
        self.func_calls = 0
        self.begin_time = time.perf_counter()
        self.__mutex = Lock()
        self.func_calls_file = os.path.join(output_dir, "func_calls") if output_dir else None

    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        super().prepare()
        if self.output_dir:
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}\t{"Loss":>15}')
                for param in self.parameters:
                    file.write(f"\t{param.name:>15}")
                file.write(f'\t{"Hash":>64}\n')

    @abstractmethod
    def _objective(self, values: np.ndarray, concurrent: bool=False) -> float:
        """Abstract method for loss computation

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """

    def __call__(self, values: np.ndarray, concurrent: bool=False) -> float:
        """Objective computation for the outside world - also handles output file writing

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """
        self.func_calls += 1
        begin_time = time.perf_counter()
        objective_value = self._objective(values, concurrent=concurrent)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(os.path.join(self.func_calls_file), 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    file.write(f'{objective_value:>15.8e}')
                    for i, param in enumerate(self.parameters):
                        file.write(f"\t{param.denormalise(values[i]):>15.6f}")
                    file.write(f'\t{self.parameters.hash(values)}\n')
        return objective_value

    def plot_best(self) -> List[Figure]:
        """Plot the current best case

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        # Find hash associated with the best case
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        min_series = df.iloc[df["Loss"].idxmin()]
        call_hash = str(min_series["Hash"])
        # Use the single case plotting utility
        figures = self.plot_case(call_hash)
        # Also display the best case
        print("Best run:")
        print(min_series.drop(["Loss", "Hash"]))
        print(f"Hash: {call_hash}")
        print(f"Loss: {min_series['Loss']:15.8e}")
        return figures

    def get_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of objective history
        """
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        x_axis = df["Start Time /s"] + df["Run Time /s"]
        params = df[[param.name for param in self.parameters]]
        param_hash = df["Hash"].to_list()
        return {
            "Loss": {
                "time": x_axis.to_numpy(),
                "values": df["Loss"].to_numpy(),
                "params": params.to_numpy(),
                "hashes": param_hash,
            }
        }



class SingleCompositeObjective(Objective):
    """Abstract class for composite single-objectives"""

    def __init__(
            self,
            parameters: ParameterSet,
            composition: Composition,
            output_dir: str=None,
        ) -> None:
        super().__init__()
        self.parameters = parameters
        self.composition = composition
        self.output_dir = output_dir
        self.func_calls = 0
        self.begin_time = time.perf_counter()
        self.__mutex = Lock()
        self.func_calls_file = os.path.join(output_dir, "func_calls") if output_dir else None

    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        super().prepare()
        if self.output_dir:
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}\t{"Loss":>15}\t')
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Hash":>64}\n')

    @abstractmethod
    def _inner_objective(self, values: np.ndarray, concurrent: bool=False) -> np.ndarray:
        """Abstract method for computation of the inner function of the composite objective

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        np.ndarray
            Inner function value
        """

    def __call__(self, values: np.ndarray, concurrent: bool=False) -> np.ndarray:
        """Objective computation for the outside world - also handles output file writing

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """
        self.func_calls += 1
        begin_time = time.perf_counter()
        inner_objective = self._inner_objective(values, concurrent=concurrent)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(self.func_calls_file, 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    file.write(f'{self.composition(inner_objective):>15.8e}\t')
                    for i, param in enumerate(self.parameters):
                        file.write(f"{param.denormalise(values[i]):>15.6f}\t")
                    file.write(f'{self.parameters.hash(values)}\n')
        return inner_objective

    def plot_best(self) -> List[Figure]:
        """Plot the current best case

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        # Find hash associated with the best case
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        min_series = df.iloc[df["Loss"].idxmin()]
        call_hash = str(min_series["Hash"])
        # Use the single case plotting utility
        figures = self.plot_case(call_hash)
        # Also display the best case
        print("Best run:")
        print(min_series.drop(["Loss", "Hash"]))
        print(f"Hash: {call_hash}")
        print(f"Loss: {min_series['Loss']:15.8e}")
        return figures

    def get_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of objective history
        """
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        x_axis = df["Start Time /s"] + df["Run Time /s"]
        params = df[[param.name for param in self.parameters]]
        param_hash = df["Hash"].to_list()
        return {
            "Loss": {
                "time": x_axis.to_numpy(),
                "values": df["Loss"].to_numpy(),
                "params": params.to_numpy(),
                "hashes": param_hash,
            }
        }


class StochasticSingleObjective(Objective):
    """Abstract class for scalar stochastic single-objectives"""

    def __init__(self, parameters: ParameterSet, output_dir: str=None) -> None:
        super().__init__()
        self.parameters = parameters
        self.output_dir = output_dir
        self.func_calls = 0
        self.begin_time = time.perf_counter()
        self.__mutex = Lock()
        self.func_calls_file = os.path.join(output_dir, "func_calls") if output_dir else None

    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        super().prepare()
        if self.output_dir:
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}')
                file.write(f'\t{"Loss":>15}\t{"Variance":>15}')
                for param in self.parameters:
                    file.write(f"\t{param.name:>15}")
                file.write(f'\t{"Hash":>64}\n')

    @abstractmethod
    def _objective(self, values: np.ndarray, concurrent: bool=False) -> Tuple[float, float]:
        """Abstract method for loss computation.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        Tuple[float, float]
            Objective value and variance.
        """

    def __call__(self, values: np.ndarray, concurrent: bool=False) -> Tuple[float, float]:
        """Objective computation for the outside world - also handles output file writing.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        Tuple[float, float]
            Objective value and variance.
        """
        self.func_calls += 1
        begin_time = time.perf_counter()
        value, variance = self._objective(values, concurrent=concurrent)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(os.path.join(self.func_calls_file), 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    file.write(f'{value:>15.8e}\t')
                    file.write(f'{variance:>15.8e}')
                    for i, param in enumerate(self.parameters):
                        file.write(f"\t{param.denormalise(values[i]):>15.6f}")
                    file.write(f'\t{self.parameters.hash(values)}\n')
        return value, variance

    def plot_best(self) -> List[Figure]:
        """Plot the current best case

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        # Find hash associated with the best case
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        min_series = df.iloc[df["Loss"].idxmin()]
        call_hash = str(min_series["Hash"])
        # Use the single case plotting utility
        figures = self.plot_case(call_hash)
        # Also display the best case
        print("Best run:")
        print(min_series.drop(["Loss", "Hash"]))
        print(f"Hash: {call_hash}")
        print(f"Loss: {min_series['Loss']:15.8e}")
        return figures

    def get_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of objective history
        """
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        x_axis = df["Start Time /s"] + df["Run Time /s"]
        params = df[[param.name for param in self.parameters]]
        param_hash = df["Hash"].to_list()
        return {
            "Loss": {
                "time": x_axis.to_numpy(),
                "values": df["Loss"].to_numpy(),
                "params": params.to_numpy(),
                "variances": df["Variance"].to_numpy(),
                "hashes": param_hash,
            }
        }


class StochasticSingleCompositeObjective(Objective):
    """Abstract class for scalar composite stochastic single-objectives"""

    def __init__(
            self,
            parameters: ParameterSet,
            composition: Composition,
            output_dir: str=None,
        ) -> None:
        super().__init__()
        self.parameters = parameters
        self.composition = composition
        self.output_dir = output_dir
        self.func_calls = 0
        self.begin_time = time.perf_counter()
        self.__mutex = Lock()
        self.func_calls_file = os.path.join(output_dir, "func_calls") if output_dir else None

    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        super().prepare()
        if self.output_dir:
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}')
                file.write(f'\t{"Loss":>15}\t{"Variance":>15}')
                for param in self.parameters:
                    file.write(f"\t{param.name:>15}")
                file.write(f'\t{"Hash":>64}\n')

    @abstractmethod
    def _inner_objective(
        self,
        values: np.ndarray,
        concurrent: bool=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Abstract method for computation of the inner function of the composite objective.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Inner function value and variance.
        """

    def __call__(self, values: np.ndarray, concurrent: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """Objective computation for the outside world - also handles output file writing.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Objective value and variance.
        """
        self.func_calls += 1
        begin_time = time.perf_counter()
        value, variance = self._inner_objective(values, concurrent=concurrent)
        end_time = time.perf_counter()
        # Compute the objective variance using Monte Carlo (using fixed base samples)
        biased = [norm.rvs(loc=0, scale=1) for _ in range(1000)]
        mc_objectives = [self.composition(value + bias * np.sqrt(variance)) for bias in biased]
        mc_variance = np.var(mc_objectives)
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(os.path.join(self.func_calls_file), 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    file.write(f'{self.composition(value):>15.8e}\t')
                    file.write(f'{mc_variance:>15.8e}')
                    for i, param in enumerate(self.parameters):
                        file.write(f"\t{param.denormalise(values[i]):>15.6f}")
                    file.write(f'\t{self.parameters.hash(values)}\n')
        return value, variance

    def plot_best(self) -> List[Figure]:
        """Plot the current best case

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        # Find hash associated with the best case
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        min_series = df.iloc[df["Loss"].idxmin()]
        call_hash = str(min_series["Hash"])
        # Use the single case plotting utility
        figures = self.plot_case(call_hash)
        # Also display the best case
        print("Best run:")
        print(min_series.drop(["Loss", "Hash"]))
        print(f"Hash: {call_hash}")
        print(f"Loss: {min_series['Loss']:15.8e}")
        return figures

    def get_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of objective history
        """
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        x_axis = df["Start Time /s"] + df["Run Time /s"]
        params = df[[param.name for param in self.parameters]]
        param_hash = df["Hash"].to_list()
        return {
            "Loss": {
                "time": x_axis.to_numpy(),
                "values": df["Loss"].to_numpy(),
                "params": params.to_numpy(),
                "variances": df["Variance"].to_numpy(),
                "hashes": param_hash,
            }
        }



class MultiFidelitySingleObjective(Objective):
    """Class for multi-fidelity single-objectives"""

    def __init__(
            self,
            objectives: Dict[float, Objective],
            parameters: ParameterSet,
            output_dir: str=None,
        ) -> None:
        super().__init__()
        self.parameters = parameters
        self.objectives = objectives
        self.output_dir = output_dir
        # Sanitise fidelities
        self.fidelities = np.array(list(objectives.keys()))
        if np.any(self.fidelities < 0) or np.any(self.fidelities > 1):
            raise RuntimeError("Fidelities must be contained in the interval [0,1]")
        self.func_calls = 0
        self.begin_time = time.perf_counter()
        self.__mutex = Lock()
        self.call_timings = {fidelity: [] for fidelity in self.fidelities}
        self.func_calls_file = os.path.join(output_dir, "func_calls") if output_dir else None

    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        super().prepare()
        if self.output_dir:
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t')
                file.write(f'{"Run Time /s":>15}\t')
                file.write(f'{"Loss":>15}\t')
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Fidelity":>15}\t')
                file.write(f'{"Hash":>64}\n')
        for fidelity, objective in self.objectives.items():
            output_path = os.path.join(self.output_dir, f'fidelity_{fidelity:<5.3f}')
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            os.mkdir(output_path)
            objective.prepare()

    def select_fidelity(self, fidelity: float) -> float:
        """Select the target fidelity ensuring robustness to floating point round-off errors

        Parameters
        ----------
        fidelity : float
            Input fidelity

        Returns
        -------
        float
            Closest fidelity in the model

        Raises
        ------
        RuntimeError
            When a fidelity cannot be chosen
        """
        # Select the target fidelity: ensure float comparisons are approximate
        candidates = np.nonzero(np.isclose(self.fidelities, fidelity))[0]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot select a target fidelity from the input {fidelity}")
        return self.fidelities[candidates.squeeze()]

    def cost(self, fidelity: float) -> float:
        """Return the expected cost of evaluating the objective at a given fidelity

        Parameters
        ----------
        fidelity : float
            Target fidelity

        Returns
        -------
        float
            Expected cost, in seconds
        """
        target_fidelity = self.select_fidelity(fidelity)
        if len(self.call_timings[target_fidelity]) > 0:
            return np.mean(self.call_timings[target_fidelity]) + 1e-9
        return 1e-9

    def __call__(
            self,
            values: np.ndarray,
            fidelity: float=1.0,
            concurrent: bool=False,
        ) -> np.ndarray:
        """Objective computation for the outside world - also handles output file writing

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        fidelity : float
            Fidelity to run this call at, by default 1.0
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        np.ndarray
            Objective value
        """
        # Evaluate objective at target fidelity
        self.func_calls += 1
        begin_time = time.perf_counter()
        target_fidelity = self.select_fidelity(fidelity)
        objective_value = self.objectives[target_fidelity](values, concurrent=concurrent)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(os.path.join(self.func_calls_file), 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    file.write(f'{objective_value:>15.8e}\t')
                    for i, param in enumerate(self.parameters):
                        file.write(f"{param.denormalise(values[i]):>15.6f}\t")
                    file.write(f'{target_fidelity:>15.8f}\t')
                    file.write(f'{self.parameters.hash(values)}\n')
        # Update call time history
        self.call_timings[target_fidelity].append(end_time - begin_time)
        return objective_value

    def plot_best(self) -> List[Figure]:
        """Plot the current best case

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        # Find hash associated with the best case (at highest fidelity)
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        max_fidelity = df["Fidelity"].max()
        df = df[df["Fidelity"] == max_fidelity]
        min_series = df.iloc[df["Loss"].idxmin()]
        call_hash = str(min_series["Hash"])
        # Use the single case plotting utility
        figures = self.plot_case(call_hash)
        # Also display the best case
        print("Best run:")
        print(min_series.drop(["Loss", "Hash"]))
        print(f"Hash: {call_hash}")
        print(f"Loss: {min_series['Loss']:15.8e}")
        return figures

    def get_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of objective history
        """
        result = {}
        for fidelity, objective in self.objectives.items():
            data = objective.get_history()
            for name, values in data.items():
                result[f"{name} (fidelity={fidelity})"] = values
        return result


class MultiFidelityCompositeObjective(Objective):
    """Class for multi-fidelity composite single-objectives"""

    def __init__(
            self,
            objectives: Dict[float, SingleCompositeObjective],
            parameters: ParameterSet,
            output_dir: str=None,
        ) -> None:
        super().__init__()
        self.parameters = parameters
        self.objectives = objectives
        self.output_dir = output_dir
        # Sanitise fidelities
        self.fidelities = np.array(list(objectives.keys()))
        if np.any(self.fidelities < 0) or np.any(self.fidelities > 1):
            raise RuntimeError("Fidelities must be contained in the interval [0,1]")
        # Sanitise compositions
        self.composition = self.objectives[max(self.fidelities)].composition
        # for fid, objective in self.objectives.items():
        #     if objective.composition != self.composition:
        #         raise RuntimeError("All compositions must be equal for all fidelities")
        self.func_calls = 0
        self.begin_time = time.perf_counter()
        self.__mutex = Lock()
        self.call_timings = {fidelity: [] for fidelity in self.fidelities}
        self.func_calls_file = os.path.join(output_dir, "func_calls") if output_dir else None

    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        super().prepare()
        if self.output_dir:
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t')
                file.write(f'{"Run Time /s":>15}\t')
                file.write(f'{"Loss":>15}\t')
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Fidelity":>15}\t')
                file.write(f'{"Hash":>64}\n')
        for fidelity, objective in self.objectives.items():
            output_path = os.path.join(self.output_dir, f'fidelity_{fidelity:<5.3f}')
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            os.mkdir(output_path)
            objective.prepare()

    def select_fidelity(self, fidelity: float) -> float:
        """Select the target fidelity ensuring robustness to floating point round-off errors

        Parameters
        ----------
        fidelity : float
            Input fidelity

        Returns
        -------
        float
            Closest fidelity in the model

        Raises
        ------
        RuntimeError
            When a fidelity cannot be chosen
        """
        # Select the target fidelity: ensure float comparisons are approximate
        candidates = np.nonzero(np.isclose(self.fidelities, fidelity))[0]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot select a target fidelity from the input {fidelity}")
        return self.fidelities[candidates.squeeze()]

    def cost(self, fidelity: float) -> float:
        """Return the expected cost of evaluating the objective at a given fidelity

        Parameters
        ----------
        fidelity : float
            Target fidelity

        Returns
        -------
        float
            Expected cost, in seconds
        """
        target_fidelity = self.select_fidelity(fidelity)
        return np.mean(self.call_timings[target_fidelity]) + 1e-9

    def __call__(
            self,
            values: np.ndarray,
            fidelity: float=1.0,
            concurrent: bool=False,
        ) -> np.ndarray:
        """Objective computation for the outside world - also handles output file writing

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        fidelity : float
            Fidelity to run this call at
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        np.ndarray
            Objective value
        """
        # Evaluate objective at target fidelity
        self.func_calls += 1
        begin_time = time.perf_counter()
        target_fidelity = self.select_fidelity(fidelity)
        objective = self.objectives[target_fidelity]
        inner_objective = objective._inner_objective(values, concurrent=concurrent)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(os.path.join(self.func_calls_file), 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    file.write(f'{objective.composition(inner_objective):>15.8e}\t')
                    for i, param in enumerate(self.parameters):
                        file.write(f"{param.denormalise(values[i]):>15.6f}\t")
                    file.write(f'{target_fidelity:>15.8f}\t')
                    file.write(f'{self.parameters.hash(values)}\n')
        # Update call time history
        self.call_timings[target_fidelity].append(end_time - begin_time)
        return inner_objective

    def plot_best(self) -> List[Figure]:
        """Plot the objective history

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        # Find hash associated with the best case (at highest fidelity)
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        max_fidelity = df["Fidelity"].max()
        df = df[df["Fidelity"] == max_fidelity]
        min_series = df.iloc[df["Loss"].idxmin()]
        call_hash = str(min_series["Hash"])
        # Use the single case plotting utility
        figures = self.plot_case(call_hash)
        # Also display the best case
        print("Best run:")
        print(min_series.drop(["Loss", "Hash"]))
        print(f"Hash: {call_hash}")
        print(f"Loss: {min_series['Loss']:15.8e}")
        return figures

    def get_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of objective history
        """
        result = {}
        for fidelity, objective in self.objectives.items():
            data = objective.get_history()
            for name, values in data.items():
                result[f"{name} (fidelity={fidelity})"] = values
        return result
