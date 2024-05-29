"""Module for Links solver."""
from typing import Dict, Any, Type
import os
import time
import shutil
import subprocess
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver, Case, CaseResult, OutputField, OutputResult, InputData
from piglot.solver.links.fields import links_fields_reader, LinksInputData
from piglot.utils.solver_utils import has_keyword, load_module_from_file


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

    def _run_case(self, values: np.ndarray, case: Case, tmp_dir: str) -> CaseResult:
        """Run a single case wth Links.

        Parameters
        ----------
        values: np.ndarray
            Current parameter values
        case : Case
            Case to run.
        tmp_dir: str
            Temporary directory to run the simulation

        Returns
        -------
        CaseResult
            Results for this case
        """
        # Copy input file replacing parameters by passed value
        input_data = case.input_data.prepare(values, self.parameters, tmp_dir=tmp_dir)
        input_file = input_data.input_file
        case_name, _ = os.path.splitext(os.path.basename(input_file))
        # Run LINKS (we don't use high precision timers here to keep track of the start time)
        begin_time = time.time()
        process_result = subprocess.run(
            [self.links_bin, input_file],
            stdout=self.stdout,
            stderr=self.stderr,
            check=False
        )
        end_time = time.time()
        # Check if simulation completed
        screen_file = os.path.join(os.path.splitext(input_file)[0], f'{case_name}.screen')
        failed_case = (process_result.returncode != 0 or
                       not has_keyword(screen_file, "Program L I N K S successfully completed."))
        # Read results from output directories
        responses = {name: field.get(input_data) for name, field in case.fields.items()}
        return CaseResult(
            begin_time,
            end_time - begin_time,
            values,
            not failed_case,
            self.parameters.hash(values),
            responses,
        )

    def _solve(
            self,
            values: np.ndarray,
            concurrent: bool,
            ) -> Dict[Case, CaseResult]:
        """Internal solver for the prescribed problems.

        Parameters
        ----------
        values : array
            Current parameters to evaluate.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        Dict[Case, CaseResult]
            Results for each case.
        """
        # Ensure tmp directory is clean
        tmp_dir = f'{self.tmp_dir}_{self.parameters.hash(values)}' if concurrent else self.tmp_dir
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

        def run_case(case: Case) -> CaseResult:
            return self._run_case(values, case, tmp_dir)
        # Run cases (in parallel if specified)
        if self.parallel > 1:
            with Pool(self.parallel) as pool:
                results = pool.map(run_case, self.cases)
        else:
            results = map(run_case, self.cases)
        # Ensure we actually resolve the map
        results = list(results)
        # Cleanup temporary directories
        if concurrent:
            shutil.rmtree(tmp_dir)
        # Build output dict
        return dict(zip(self.cases, results))

    def get_current_response(self) -> Dict[str, OutputResult]:
        """Get the responses from a given output field for all cases.

        Returns
        -------
        Dict[str, OutputResult]
            Output responses.
        """
        fields = self.get_output_fields()
        return {
            name: field.get(case.input_data.get_current(self.tmp_dir))
            for name, (case, field) in fields.items()
        }

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
        if 'links' not in config:
            raise ValueError("Missing 'links' in solver configuration.")
        links_bin = config['links']
        # Read the parallelism and temporary directory (if present)
        parallel = int(config.get('parallel', 1))
        tmp_dir = os.path.join(output_dir, config.get('tmp_dir', 'tmp'))
        # Read generator, if any
        input_data_class: Type[InputData] = LinksInputData
        if 'generator' in config:
            if 'script' not in config['generator']:
                raise ValueError("Missing 'script' in generator configuration.")
            if 'class' not in config['generator']:
                raise ValueError("Missing 'class' in generator configuration.")
            input_data_class = load_module_from_file(
                config['generator']['script'],
                config['generator']['class'],
            )
        # Read the cases
        if 'cases' not in config:
            raise ValueError("Missing 'cases' in solver configuration.")
        cases = []
        for case_name, case_config in config['cases'].items():
            if 'fields' not in case_config:
                raise ValueError(f"Missing 'fields' in case '{case_name}' configuration.")
            fields = {
                field_name: links_fields_reader(field_config)
                for field_name, field_config in case_config['fields'].items()
            }
            cases.append(Case(input_data_class(case_name), fields))
        # Return the solver
        return LinksSolver(cases, parameters, output_dir, links_bin, parallel, tmp_dir)
