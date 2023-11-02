"""Module for Links solver."""
from typing import Dict, Any
import os
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver, Case, OutputField, OutputResult, generic_read_cases
from piglot.solver.links.fields import links_fields_reader


class LinksCase(Case):
    """Container for Links cases."""

    def prepare(self) -> None:
        """Prepare the case for the simulation."""

    @staticmethod
    def read(name: str, config: Dict[str, Any]) -> Case:
        """Read the case from the configuration dictionary.

        Parameters
        ----------
        name : str
            Case name.
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Case
            Case to use for this problem.
        """
        return LinksCase(name)


class LinksSolver(Solver):
    """Links solver."""

    def __init__(
            self,
            cases: Dict[Case, Dict[str, OutputField]],
            parameters: ParameterSet,
            output_dir: str,
            links_bin: str,
            parallel: int,
            tmp_dir: str,
        ) -> None:
        """Constructor for the Links solver class.

        Parameters
        ----------
        cases : Dict[Case, Dict[str, OutputField]]
            Cases to be run and respective output fields.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        links_bin : str
            Path to the links binary.
        parallel : int
            Number of parallel processes to use.
        tmp_dir : str
            Path to the temporary directory.
        """
        super().__init__(cases, parameters, output_dir)
        self.links_bin = links_bin
        self.parallel = parallel
        self.tmp_dir = tmp_dir

    def solve(
            self,
            values: np.ndarray,
            concurrent: bool,
        ) -> Dict[Case, Dict[OutputField, OutputResult]]:
        """Solve all cases for the given set of parameter values.

        Parameters
        ----------
        values : array
            Current parameters to evaluate.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        Dict[Case, Dict[OutputField, OutputResult]]
            Evaluated results for each output field.
        """
        raise NotImplementedError()

    @staticmethod
    def read(config: Dict[str, Any], parameters: ParameterSet, output_dir: str) -> Solver:
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
        Solver
            Solver to use for this problem.
        """
        # Get the Links binary path
        if not 'links' in config:
            raise ValueError("Missing 'links' in solver configuration.")
        links_bin = config['links']
        # Read the parallelism and temporary directory (if present)
        parallel = int(config.get('parallel', 1))
        tmp_dir = os.path.join(output_dir, config.get('tmp_dir', 'tmp'))
        # Read the cases
        if not 'cases' in config:
            raise ValueError("Missing 'cases' in solver configuration.")
        cases = generic_read_cases(config['cases'], LinksCase, links_fields_reader)
        # Return the solver
        return LinksSolver(cases, parameters, output_dir, links_bin, parallel, tmp_dir)
