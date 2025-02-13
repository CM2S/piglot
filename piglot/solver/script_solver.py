"""Module for script-based solvers."""
from __future__ import annotations
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import os
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import OutputResult, SingleCaseSolver
from piglot.utils.assorted import read_custom_module


class ScriptSolverCallable(ABC):
    """Interface for script-based solvers."""

    @staticmethod
    @abstractmethod
    def get_output_fields() -> List[str]:
        """Get the output fields for this solver.

        Returns
        -------
        List[str]
            Output fields for this solver.
        """

    @staticmethod
    @abstractmethod
    def solve(values: Dict[str, float]) -> Dict[str, OutputResult]:
        """Callable for script-based solvers.

        Parameters
        ----------
        values : Dict[str, float]
            Current parameters to evaluate.

        Returns
        -------
        Dict[str, OutputResult]
            Evaluated results for each output field.
        """


class ScriptSolver(SingleCaseSolver):
    """Script-based solvers."""

    def __init__(
        self,
        script: ScriptSolverCallable,
        parameters: ParameterSet,
        output_dir: str,
        tmp_dir: str,
        verbosity: str,
    ) -> None:
        """Constructor for the solver class.

        Parameters
        ----------
        output_fields : List[str]
            List of output fields.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        tmp_dir : str
            Path to the temporary directory.
        """
        super().__init__(script.get_output_fields(), parameters, output_dir, tmp_dir, verbosity)
        self.script = script

    def _solve(self, values: np.ndarray, concurrent: bool) -> Dict[str, OutputResult]:
        """Internal solver for the prescribed problems.

        Parameters
        ----------
        values : array
            Current parameters to evaluate.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        Dict[str, OutputResult]
            Evaluated results for each output field.
        """
        # Run the solver
        param_dict = self.parameters.to_dict(values)
        results = self.script.solve(param_dict)
        # Sanitise output fields before returning
        for field in self.output_fields:
            if field not in results:
                raise ValueError(f"Missing output field '{field}'.")
        for field in results:
            if field not in self.output_fields:
                raise ValueError(f"Unknown output field '{field}'.")
        return results

    @classmethod
    def read(
        cls,
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> ScriptSolver:
        """Read the solver from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        ScriptSolver
            Solver to use for this problem.
        """
        # Read and instantiate the script and the temporary directory
        script = read_custom_module(config, ScriptSolverCallable)()
        tmp_dir = os.path.join(output_dir, config.pop('tmp_dir', 'tmp'))
        verbosity = config.pop('verbosity', 'none')
        return cls(script, parameters, output_dir, tmp_dir, verbosity)
