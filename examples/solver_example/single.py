"""Module with a sample single-case solver interface."""
from __future__ import annotations
from typing import Dict, Any
import os
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import OutputResult, SingleCaseSolver


class SampleSingleCaseSolver(SingleCaseSolver):
    """Sample single-case solver interface."""

    def __init__(
        self,
        field_name: str,
        parameters: ParameterSet,
        output_dir: str,
        tmp_dir: str,
        verbosity: str,
    ) -> None:
        """Constructor for the solver class.

        Parameters
        ----------
        field_name : str
            Name of the output field.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        tmp_dir : str
            Path to the temporary directory.
        """
        super().__init__([field_name], parameters, output_dir, tmp_dir, verbosity)

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
        param_dict = self.parameters.to_dict(values)
        # Build the output response: [0, 1] time grid with the parameter values
        time = np.linspace(0, 1, len(param_dict))
        response = np.array(param_dict.values())
        return {self.output_fields[0]: OutputResult(time, response)}

    @classmethod
    def read(
        cls,
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> SampleSingleCaseSolver:
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
        SampleSingleCaseSolver
            Solver to use for this problem.
        """
        # Read the field name
        if 'name' not in config:
            raise ValueError("Missing output field name.")
        name = config.pop('name')
        # Read optional parameters
        tmp_dir = os.path.join(output_dir, config.pop('tmp_dir', 'tmp'))
        verbosity = config.pop('verbosity', 'none')
        # Instantiate the solver
        return cls(name, parameters, output_dir, tmp_dir, verbosity)
