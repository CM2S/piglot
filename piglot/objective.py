"""Module containing optimisation objective primites"""
import os
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any, Dict
import shutil
from threading import Lock
import numpy as np
import sympy
import torch
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



class Objective(ABC):
    """Abstract class for optimisation objectives"""

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Objective computation for the outside world"""



class SingleObjective(ABC):
    """Abstract class for scalar single-objectives"""

    def __init__(self, parameters: ParameterSet, output_dir: str=None) -> None:
        super().__init__()
        self.parameters = parameters
        self.output_dir = output_dir
        self.func_calls = 0
        self.begin_time = time.perf_counter()
        self.__mutex = Lock()
        if self.output_dir:
            # Prepare output directories
            if os.path.isdir(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)
            # Build header for function calls file
            self.func_calls_file = os.path.join(output_dir, "func_calls")
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



class SingleCompositeObjective(ABC):
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
        if self.output_dir:
            # Prepare output directories
            if os.path.isdir(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)
            # Build header for function calls file
            self.func_calls_file = os.path.join(output_dir, "func_calls")
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



class MultiFidelitySingleObjective(ABC):
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
        if self.output_dir:
            # Prepare output directories
            os.makedirs(self.output_dir, exist_ok=True)
            # Build header for function calls file
            self.func_calls_file = os.path.join(output_dir, "func_calls")
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t')
                file.write(f'{"Run Time /s":>15}\t')
                file.write(f'{"Loss":>15}\t')
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Fidelity":>15}\t')
                file.write(f'{"Hash":>64}\n')

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


class MultiFidelityCompositeObjective(ABC):
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
        if self.output_dir:
            # Prepare output directories
            os.makedirs(self.output_dir, exist_ok=True)
            # Build header for function calls file
            self.func_calls_file = os.path.join(output_dir, "func_calls")
            with open(os.path.join(self.func_calls_file), 'w', encoding='utf8') as file:
                file.write(f'{"Start Time /s":>15}\t')
                file.write(f'{"Run Time /s":>15}\t')
                file.write(f'{"Loss":>15}\t')
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Fidelity":>15}\t')
                file.write(f'{"Hash":>64}\n')

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

    def __call__(self, values: np.ndarray, fidelity: float, parallel: bool=False) -> np.ndarray:
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
