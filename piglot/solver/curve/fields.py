"""Module for output fields from Curve solver."""
from __future__ import annotations
from typing import Dict, Any, Tuple
import os
import copy
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import InputData, OutputField, OutputResult
from piglot.utils.solver_utils import write_parameters, get_case_name


class CurveInputData(InputData):
    """Container for dummy input data."""

    def __init__(
            self,
            case_name: str,
            expression: str,
            parametric: str,
            bounds: Tuple[float, float],
            points: int,
            ) -> None:
        super().__init__()
        self.case_name = case_name
        self.expression = expression
        self.parametric = parametric
        self.bounds = bounds
        self.points = points
        self.input_file: str = None

    def prepare(
            self,
            values: np.ndarray,
            parameters: ParameterSet,
            tmp_dir: str = None,
            ) -> CurveInputData:
        """Prepare the input data for the simulation with a given set of parameters.

        Parameters
        ----------
        values : np.ndarray
            Parameters to run for.
        parameters : ParameterSet
            Parameter set for this problem.
        tmp_dir : str, optional
            Temporary directory to run the analyses, by default None

        Returns
        -------
        CurveInputData
            Input data prepared for the simulation.
        """
        result = copy.copy(self)
        # Write the input file (with the name placeholder)
        tmp_file = os.path.join(tmp_dir, f'{self.case_name}.tmp')
        with open(tmp_file, 'w', encoding='utf8') as file:
            file.write(f'{self.expression}')
        # Write the parameters to the input file
        result.input_file = os.path.join(tmp_dir, f'{self.case_name}.dat')
        write_parameters(parameters.to_dict(values), tmp_file, result.input_file)
        return result

    def check(self, parameters: ParameterSet) -> None:
        """Check if the input data is valid according to the given parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        """
        # Generate a dummy set of parameters (to ensure proper handling of output parameters)
        values = np.array([parameter.inital_value for parameter in parameters])
        param_dict = parameters.to_dict(values, input_normalised=False)
        for parameter in param_dict:
            if parameter not in self.expression:
                raise ValueError(f"Parameter '{parameter}' not found in expression.")

    def name(self) -> str:
        """Return the name of the input data.

        Returns
        -------
        str
            Name of the input data.
        """
        return self.case_name

    def get_current(self, target_dir: str) -> CurveInputData:
        """Get the current input data.

        Parameters
        ----------
        target_dir : str
            Target directory to copy the input file.

        Returns
        -------
        CurveInputData
            Current input data.
        """
        result = CurveInputData(os.path.join(target_dir, self.case_name), self.expression,
                                self.parametric, self.bounds, self.points)
        result.input_file = os.path.join(target_dir, self.case_name + '.dat')
        return result


class Curve(OutputField):
    """Curve output reader."""

    def check(self, input_data: CurveInputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : CurveInputData
            Input data for this case.

        """

    def get(self, input_data: CurveInputData) -> OutputResult:
        """Reads reactions from a Curve analysis.

        Parameters
        ----------
        input_data : CurveInputData
            Input data for this case.

        Returns
        -------
        array
            2D array with parametric value and corresponding expression value.
        """
        input_file = input_data.input_file
        casename = get_case_name(input_file)
        output_dir = os.path.dirname(input_file)
        output_filename = os.path.join(output_dir, f'{casename}.out')
        # Ensure the file exists
        if not os.path.exists(output_filename):
            return OutputResult(np.empty(0), np.empty(0))
        data = np.genfromtxt(output_filename)
        return OutputResult(data[:, 0], data[:, 1])

    @staticmethod
    def read(config: Dict[str, Any]) -> Curve:
        """Read the output field from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Reaction
            Output field to use for this problem.
        """
        return Curve()
