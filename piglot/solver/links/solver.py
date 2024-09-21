"""Module for Links solver."""
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
from piglot.solver.links.fields import Reaction, OutFile
from piglot.utils.solver_utils import has_keyword


class LinksCase(InputFileCase):
    """Class for Links cases."""

    def __init__(
        self,
        name: str,
        fields: Dict[str, OutputField],
        generator: InputDataGenerator,
        links: str,
    ) -> None:
        super().__init__(name, fields, generator)
        self.links_bin = links

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
        process_result = subprocess.run(
            [self.links_bin, input_data.input_file],
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False
        )
        if process_result.returncode != 0:
            return False
        # Check if simulation completed
        output_dir, _ = os.path.splitext(input_data.input_file)
        case_name = os.path.basename(output_dir)
        screen_file = os.path.join(output_dir, f'{case_name}.screen')
        return has_keyword(screen_file, "Program L I N K S successfully completed.")

    @classmethod
    def get_supported_fields(cls) -> Dict[str, Type[OutputField]]:
        """Get the supported fields for this input file type.

        Returns
        -------
        Dict[str, Type[OutputField]]
            Names and supported fields for this input file type.
        """
        return {
            'Reaction': Reaction,
            'OutFile': OutFile,
        }


class LinksSolver(InputFileSolver):
    """Links solver class."""

    @classmethod
    def get_case_class(cls) -> Type[InputFileCase]:
        """Get the case class to use for this solver.

        Returns
        -------
        Type[InputFileCase]
            InputFileCase class to use for this solver.
        """
        return LinksCase
