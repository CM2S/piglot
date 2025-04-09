"""Module for CRATE solver."""
from typing import Dict, Type
import os
import sys
import subprocess
from piglot.solver.input_file_solver import (
    InputDataGenerator,
    InputFileCase,
    InputData,
    InputFileSolver,
    OutputField,
)
from piglot.solver.crate.fields import HresFile
from piglot.utils.solver_utils import has_keyword


class CrateCase(InputFileCase):
    """Class for Crate cases."""

    def __init__(
        self,
        name: str,
        fields: Dict[str, OutputField],
        generator: InputDataGenerator,
        crate: str,
        python_interp: str = 'python3',
        microstructure_dir: str = '.',
    ) -> None:
        super().__init__(name, fields, generator)
        self.crate_bin = crate
        self.python_interp = python_interp
        self.microstructure_dir = microstructure_dir

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
        if not os.path.isdir(self.microstructure_dir):
            raise ValueError(f"Microstructure directory '{self.microstructure_dir}' not found.")
        command = [
            self.python_interp,
            self.crate_bin,
            input_data.input_file,
            os.path.abspath(self.microstructure_dir),
        ]
        # Run the analysis
        process_result = subprocess.run(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
            cwd=tmp_dir,
        )
        if process_result.returncode != 0:
            return False
        # Check if simulation completed
        output_dir, _ = os.path.splitext(os.path.join(input_data.tmp_dir, input_data.input_file))
        case_name = os.path.basename(output_dir)
        screen_file = os.path.join(output_dir, f'{case_name}.screen')
        return has_keyword(screen_file, "Program Completed")

    @classmethod
    def get_supported_fields(cls) -> Dict[str, Type[OutputField]]:
        """Get the supported fields for this input file type.

        Returns
        -------
        Dict[str, Type[OutputField]]
            Names and supported fields for this input file type.
        """
        return {
            'hresFile': HresFile,
        }


class CrateSolver(InputFileSolver):
    """CRATE solver class."""

    @classmethod
    def get_case_class(cls) -> Type[InputFileCase]:
        """Get the case class to use for this solver.

        Returns
        -------
        Type[InputFileCase]
            InputFileCase class to use for this solver.
        """
        return CrateCase
