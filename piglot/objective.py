"""Module containing optimisation objective primites"""
import os
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any
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
        pass



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
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}\t{"Loss":>15}')
                for param in self.parameters:
                    file.write(f"\t{param.name:>15}")
                file.write(f'\t{"Hash":>64}\n')

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
        inner_objective = self._inner_objective(values, parallel=parallel)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(self.func_calls_file, 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    file.write(f'{self.composition(inner_objective):>15.8e}')
                    for i, param in enumerate(self.parameters):
                        file.write(f"\t{param.denormalise(values[i]):>15.6f}")
                    file.write(f'\t{self.parameters.hash(values)}\n')
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
