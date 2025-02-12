"""Module with a sample input file-based solver."""
from __future__ import annotations
from typing import Dict, Type, Any
import os
import shutil
import numpy as np
from piglot.solver.solver import OutputResult
from piglot.solver.input_file_solver import (
    InputFileCase,
    InputData,
    InputFileSolver,
    OutputField,
)


class SampleOutputField(OutputField):
    """Sample output field reader."""

    def check(self, input_data: InputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : InputData
            Input data for this case.
        """

    def get(self, input_data: InputData) -> OutputResult:
        """Sample get method for the output field.

        Parameters
        ----------
        input_data : InputData
            Input data for this case.

        Returns
        -------
        OutputResult
            Output data for this field.
        """
        output_file, _ = os.path.splitext(input_data.input_file)
        params = np.genfromtxt(f'{output_file}.out')
        grid = np.linspace(0, 1, len(params))
        return OutputResult(grid, params)

    @classmethod
    def read(cls, config: Dict[str, Any]) -> SampleOutputField:
        """Read the output field from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        SampleOutputField
            Output field to use for this problem.
        """
        return cls()


class SampleInputFileCase(InputFileCase):
    """Sample input file-based case class."""

    def _run_case(self, input_data: InputData, tmp_dir: str) -> bool:
        """Run the case for the given set of parameters.

        Parameters
        ----------
        input_data : InputData
            Input data for this problem.
        tmp_dir : str
            Temporary directory to run the problem.

        Returns
        -------
        bool
            Whether the case ran successfully or not.
        """
        # Simply copy the input file to an output file
        input_file = os.path.join(tmp_dir, os.path.basename(input_data.input_file))
        output_file = f'{os.path.splitext(input_file)[0]}.out'
        try:
            shutil.copy(input_file, output_file)
        except shutil.Error:
            return False
        return True

    @classmethod
    def get_supported_fields(cls) -> Dict[str, Type[OutputField]]:
        """Get the supported fields for this input file type.

        Returns
        -------
        Dict[str, Type[OutputField]]
            Names and supported fields for this input file type.
        """
        return {
            'sample': SampleOutputField,
        }


class SampleInputFileSolver(InputFileSolver):
    """Sample input file-based solver class."""

    @classmethod
    def get_case_class(cls) -> Type[InputFileCase]:
        """Get the case class to use for this solver.

        Returns
        -------
        Type[InputFileCase]
            InputFileCase class to use for this solver.
        """
        return SampleInputFileCase
