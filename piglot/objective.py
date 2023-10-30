"""Module containing optimisation objective primites"""
import os
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import shutil
from threading import Lock
import numpy as np
import sympy
import torch
import pandas as pd
import matplotlib.pyplot as plt
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
        return np.mean(np.square(inner))

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

    def plot_case(self, case_hash: str) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot

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
    
    def get_history(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary of objective history
        """
        raise NotImplementedError("Objective history not implemented for this objective")



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
            # Prepare output directories
            if os.path.isdir(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}\t{"Loss":>15}')
                for param in self.parameters:
                    file.write(f"\t{param.name:>15}")
                file.write(f'\t{"Hash":>64}\n')

    @abstractmethod
    def _objective(self, values: np.ndarray, parallel: bool=False) -> float:
        """Abstract method for loss computation

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        parallel : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """

    def __call__(self, values: np.ndarray, parallel: bool=False) -> float:
        """Objective computation for the outside world - also handles output file writing

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        parallel : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """
        self.func_calls += 1
        begin_time = time.perf_counter()
        objective_value = self._objective(values, parallel=parallel)
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
    
    def get_history(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
            Dictionary of objective history
        """
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        x_axis = df["Start Time /s"] + df["Run Time /s"]
        params = df[[param.name for param in self.parameters]]
        return {"Loss": (x_axis.to_numpy(), df["Loss"].to_numpy(), params.to_numpy())}



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
            # Prepare output directories
            if os.path.isdir(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}\t{"Loss":>15}\t')
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Hash":>64}\n')

    @abstractmethod
    def _inner_objective(self, values: np.ndarray, parallel: bool=False) -> np.ndarray:
        """Abstract method for computation of the inner function of the composite objective

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        parallel : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        np.ndarray
            Inner function value
        """

    def __call__(self, values: np.ndarray, parallel: bool=False) -> np.ndarray:
        """Objective computation for the outside world - also handles output file writing

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        parallel : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """
        self.func_calls += 1
        begin_time = time.perf_counter()
        inner_objective = self._inner_objective(values, parallel=parallel)
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
    
    def get_history(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary of objective history
        """
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        x_axis = df["Start Time /s"] + df["Run Time /s"]
        params = df[[param.name for param in self.parameters]]
        return {"Loss": (x_axis.to_numpy(), df["Loss"].to_numpy(), params.to_numpy())}



class AnalyticalObjective(SingleObjective):
    """Objective function derived from an analytical expression"""

    def __init__(self, parameters: ParameterSet, expression: str, output_dir: str = None) -> None:
        super().__init__(parameters, output_dir)
        symbs = sympy.symbols([param.name for param in parameters])
        self.parameters = parameters
        self.expression = sympy.lambdify(symbs, expression)

    def _objective(self, values: np.ndarray, parallel: bool=False) -> float:
        """Objective computation for analytical functions

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        parallel : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """
        return self.expression(**self.parameters.to_dict(values))

    def _objective_denorm(self, values: np.ndarray) -> float:
        """Objective computation for analytical functions (denormalised parameters)

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for (denormalised)

        Returns
        -------
        float
            Objective value
        """
        return self.expression(**self.parameters.to_dict(values, input_normalised=False))
    
    def _plot_1d(self, values: np.ndarray) -> Figure:
        """Plot the objective in 1D

        Parameters
        ----------
        values : np.ndarray
            Parameter values to plot for

        Returns
        -------
        Figure
            Figure with the plot
        """
        fig, axis = plt.subplots()
        x = np.linspace(self.parameters[0].lbound, self.parameters[0].ubound, 1000)
        y = np.array([self._objective_denorm(np.array([x_i])) for x_i in x])
        axis.plot(x, y, c="black", label="Analytical Objective")
        axis.scatter(values[0], self._objective_denorm(values), c="red", label="Case")
        axis.set_xlabel(self.parameters[0].name)
        axis.set_ylabel("Analytical Objective")
        axis.set_xlim(self.parameters[0].lbound, self.parameters[0].ubound)
        axis.legend()
        axis.grid()
        return fig
    
    def _plot_2d(self, values: np.ndarray) -> Figure:
        """Plot the objective in 2D

        Parameters
        ----------
        values : np.ndarray
            Parameter values to plot for

        Returns
        -------
        Figure
            Figure with the plot
        """
        fig, axis = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.linspace(self.parameters[0].lbound, self.parameters[0].ubound, 100)
        y = np.linspace(self.parameters[1].lbound, self.parameters[1].ubound, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self._objective_denorm(np.array([x_i, y_i])) for x_i in x] for y_i in y])
        axis.plot_surface(X, Y, Z, alpha=0.8, label="Analytical Objective")
        axis.scatter(values[0], values[1], self._objective_denorm(values), c="k", label="Case")
        axis.set_xlabel(self.parameters[0].name)
        axis.set_ylabel(self.parameters[1].name)
        axis.set_zlabel("Analytical Objective")
        axis.set_xlim(self.parameters[0].lbound, self.parameters[0].ubound)
        axis.set_ylim(self.parameters[1].lbound, self.parameters[1].ubound)
        axis.legend()
        axis.grid()
        fig.tight_layout()
        return fig
    
    def plot_case(self, case_hash: str) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        # Find parameters associated with the hash
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        df = df[df["Hash"] == case_hash]
        values = df[[param.name for param in self.parameters]].to_numpy()[0,:]
        # Plot depending on the dimensions
        if len(self.parameters) <= 0:
            raise RuntimeError("Missing dimensions.")
        if len(self.parameters) == 1:
            return [self._plot_1d(values)]
        if len(self.parameters) == 2:
            return [self._plot_2d(values)]
        raise RuntimeError("Plotting not supported for 3 or more parameters.")



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
            # Prepare output directories
            if os.path.isdir(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)
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

    def __call__(self, values: np.ndarray, fidelity: float=1.0, parallel: bool=False) -> np.ndarray:
        """Objective computation for the outside world - also handles output file writing

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        fidelity : float
            Fidelity to run this call at, by default 1.0
        parallel : bool, optional
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
        objective_value = self.objectives[target_fidelity](values, parallel=parallel)
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

    def get_history(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
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
            # Prepare output directories
            if os.path.isdir(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)
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

    def __call__(self, values: np.ndarray, fidelity: float=1.0, parallel: bool=False) -> np.ndarray:
        """Objective computation for the outside world - also handles output file writing

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        fidelity : float
            Fidelity to run this call at
        parallel : bool, optional
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
        inner_objective = objective._inner_objective(values, parallel=parallel)
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

    def get_history(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the objective history

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary of objective history
        """
        result = {}
        for fidelity, objective in self.objectives.items():
            data = objective.get_history()
            for name, values in data.items():
                result[f"{name} (fidelity={fidelity})"] = values
        return result
