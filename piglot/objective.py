import os
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any
import shutil
from threading import Lock
import numpy as np
import sympy
from piglot.parameter import ParameterSet


class Objective(ABC):

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass



class SingleObjective(ABC):

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
    def scalarise(self, loss_value: Any) -> float:
        pass

    @abstractmethod
    def loss(self, values: np.ndarray, parallel: bool=False) -> Any:
        pass

    def __call__(self, values: np.ndarray, parallel: bool=False) -> Any:
        self.func_calls += 1
        begin_time = time.perf_counter()
        total_loss = self.loss(values, parallel=parallel)
        end_time = time.perf_counter()
        # Update function call history file
        if self.output_dir:
            with self.__mutex:
                with open(os.path.join(self.func_calls_file), 'a', encoding='utf8') as file:
                    file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                    file.write(f'{end_time - begin_time:>15.8e}\t')
                    file.write(f'{self.scalarise(total_loss):>15.8e}')
                    for i, param in enumerate(self.parameters):
                        file.write(f"\t{param.denormalise(values[i]):>15.6f}")
                    file.write(f'\t{self.parameters.hash(values)}\n')
        return total_loss



class AnalyticalObjective(SingleObjective):

    def __init__(self, parameters: ParameterSet, expression: str, output_dir: str = None) -> None:
        super().__init__(parameters, output_dir)
        symbs = sympy.symbols([param.name for param in parameters])
        self.parameters = parameters
        self.expression = sympy.lambdify(symbs, expression)

    def scalarise(self, loss_value: Any) -> float:
        return loss_value

    def loss(self, values: np.ndarray, parallel: bool=False) -> Any:
        return self.expression(**self.parameters.to_dict(values))
