"""Module for Links solver."""
from typing import Dict, Type, Tuple, List
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
from piglot.utils.solver_utils import has_keyword, find_keyword


class LinksCase(InputFileCase):
    """Class for Links cases."""

    def __init__(
        self,
        name: str,
        fields: Dict[str, OutputField],
        generator: InputDataGenerator,
        links: str,
        mpi_command: str = None,
    ) -> None:
        super().__init__(name, fields, generator)
        self.links_bin = links
        self.mpi_command = mpi_command

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
        command = [self.links_bin, input_data.input_file]
        # Check for a coupled analysis and add MPI command if so
        if has_keyword(input_data.input_file, "NUMBER_OF_RVE"):
            if self.mpi_command is None:
                raise RuntimeError('Need to pass the "mpi_command" option for coupled analyses')
            command = self.mpi_command.split() + command
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

    @classmethod
    def _get_file_dependencies(cls, input_file: str) -> List[str]:
        """Get the dependencies for a single given input file.

        Useful for extracting the deps from an RVE in a coupled analysis.

        Parameters
        ----------
        input_file : str
            Input file to check for dependencies.

        Returns
        -------
        List[str]
            Substitution dependencies for this input file.
        """
        if not os.path.exists(input_file):
            raise ValueError(f'Input file "{input_file}" does not exist.')
        deps = []
        # Mesh file
        if has_keyword(input_file, 'MESH_FILE'):
            mesh_file = find_keyword(input_file, 'MESH_FILE').split()[1]
            deps.append(mesh_file)
        # Deformation gradient history
        if input_file.endswith('.rve') and has_keyword(input_file, 'DEFORMATION_GRADIENT_HISTORY'):
            fhist = find_keyword(input_file, 'DEFORMATION_GRADIENT_HISTORY').split()[1]
            deps.append(fhist)
        return deps

    @classmethod
    def get_dependencies(cls, input_file: str) -> Tuple[List[str], List[str]]:
        """Get the dependencies for a given input file.

        Parameters
        ----------
        input_file : str
            Input file to check for dependencies.

        Returns
        -------
        Tuple[List[str], List[str]]
            Substitution and copy dependencies for this input file.
        """
        if not os.path.exists(input_file):
            raise ValueError(f'Input file "{input_file}" does not exist.')
        # Start with the raw dependencies of the input file
        deps = cls._get_file_dependencies(input_file)
        # Check if we are using a coupled analysis
        if has_keyword(input_file, "NUMBER_OF_RVE"):
            # Extract all the RVEs and their dependencies
            with open(input_file, 'r', encoding='utf8') as file:
                # Locate the list of RVEs
                line = file.readline()
                while not line.lstrip().startswith('NUMBER_OF_RVE'):
                    line = file.readline()
                # Read each RVE
                num_rve = int(line.split()[1])
                for _ in range(num_rve):
                    # Ensure the path is relative to the input file
                    raw_rve_path = file.readline().split()[1]
                    rve_path = os.path.join(os.path.dirname(input_file), raw_rve_path)
                    deps += cls._get_file_dependencies(rve_path)
                    deps.append(rve_path)
        return deps, []


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
