"""Sample input file generator."""
from __future__ import annotations
import os
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.input_file_solver import InputData, InputDataGenerator, write_parameters


class SampleInputGenerator(InputDataGenerator):
    """Default input data generator for input file-based solvers."""

    def generate(self, parameters: ParameterSet, values: np.ndarray, tmp_dir: str) -> InputData:
        """Generate the input data for the given set of parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        values : np.ndarray
            Current parameters to evaluate.
        tmp_dir : str
            Temporary directory to run the problem.

        Returns
        -------
        InputData
            Input data for this problem.
        """
        param_dict = parameters.to_dict(values * 2)
        # Replace parameters in the input file
        input_file = 'input.dat'
        gen_input_file = os.path.join(tmp_dir, input_file)
        write_parameters(param_dict, input_file, gen_input_file)
        return InputData(tmp_dir, gen_input_file, [])
