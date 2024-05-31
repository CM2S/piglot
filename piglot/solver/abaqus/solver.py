"""Module for Abaqus solver."""
from typing import Dict, Any, List
import os
import time
import shutil
import subprocess
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver, Case, CaseResult, OutputField, OutputResult
from piglot.solver.abaqus.fields import abaqus_fields_reader, AbaqusInputData, FieldsOutput


class AbaqusSolver(Solver):
    """Abaqus solver."""

    def __init__(
        self,
        cases: Dict[Case, Dict[str, OutputField]],
        parameters: ParameterSet,
        output_dir: str,
        abaqus_bin: str,
        parallel: int,
        tmp_dir: str,
        extra_args: str = None,
    ) -> None:
        """Constructor for the Abaqus solver class.

        Parameters
        ----------
        cases : Dict[Case, Dict[str, OutputField]]
            Cases to be run and respective output fields.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        abaqus_bin : str
            Path to the abaqus binary.
        parallel : int
            Number of parallel processes to use.
        tmp_dir : str
            Path to the temporary directory.
        """
        super().__init__(cases, parameters, output_dir)
        self.abaqus_bin = abaqus_bin
        self.parallel = parallel
        self.tmp_dir = tmp_dir
        self.extra_args = extra_args

    def _post_proc_variables(
        self,
        input_data: AbaqusInputData,
        field_data: FieldsOutput
    ) -> Dict[str, Any]:
        """Generate the post-processing variables.

        Parameters
        ----------
        input_data : AbaqusInputData
            Input data for the simulation.
        field_data : FieldsOutput
            Field data for the simulation.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the post processing variables.
        """
        input_file, ext = os.path.splitext(os.path.basename(input_data.input_file))
        variables = {}
        variables['input_file'] = input_file + ext
        variables['job_name'] = input_data.job_name
        variables['step_name'] = input_data.step_name
        variables['instance_name'] = input_data.instance_name
        variables['set_name'] = field_data.set_name
        variables['field'] = field_data.field
        variables['x_field'] = field_data.x_field

        return variables

    def _run_case(self, values: np.ndarray, case: Case, tmp_dir: str) -> CaseResult:
        """Run single/parallel cases with Abaqus.

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

        # Create a temporary directory (the name of the directory is the case name) inside tmp_dir
        # (because of parallel runs) and copy the input file
        case_name, _ = os.path.splitext(os.path.basename(case.input_data.input_file))
        tmp_dir = os.path.join(tmp_dir, case_name)
        os.mkdir(tmp_dir)

        # Copy input file replacing parameters by passed value
        input_data = case.input_data.prepare(values, self.parameters, tmp_dir=tmp_dir)
        input_file = input_data.input_file

        # Run ABAQUS (we don't use high precision timers here to keep track of the start time)
        begin_time = time.time()
        extra_args: List[str] = self.extra_args.split() if self.extra_args else []
        run_inp = subprocess.run(
            [
                self.abaqus_bin,
                f"job={input_data.job_name}",
                f"input={os.path.basename(input_file)}",
                'interactive',
                'ask_delete=OFF',
            ] + extra_args,
            cwd=tmp_dir,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

        variables = self._post_proc_variables(input_data, list(case.fields.values())[0])
        python_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reader.py')

        run_odb = subprocess.run(
            [self.abaqus_bin, 'viewer', f"noGUI={python_script}", "--",
             f"input_file={variables['input_file']}", "--",
             f"job_name={variables['job_name']}", "--",
             f"step_name={variables['step_name']}", "--",
             f"instance_name={variables['instance_name']}", "--",
             f"set_name={variables['set_name']}", "--",
             f"field={variables['field']}", "--",
             f"x_field={variables['x_field']}"],
            cwd=tmp_dir,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        end_time = time.time()

        failed_case = (run_inp.returncode != 0 or run_odb.returncode != 0)

        responses = {name: field.get(input_data) if not failed_case else
                     OutputResult(np.empty(0), np.empty(0)) for name, field in case.fields.items()}

        return CaseResult(
            begin_time,
            end_time - begin_time,
            values,
            not failed_case,
            self.parameters.hash(values),
            responses,
        )

    def _solve(self, values: np.ndarray, concurrent: bool,) -> Dict[Case, CaseResult]:
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
        # Run cases (in parallel if specified)

        def run_case(case: Case) -> CaseResult:
            return self._run_case(values, case, tmp_dir)
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
        # Get the Abaqus binary path
        if 'abaqus_path' not in config:
            raise ValueError("Missing 'abaqus_path' in solver configuration.")
        abaqus_bin = config['abaqus_path']
        # Read the parallelism and temporary directory (if present)
        parallel = int(config.get('parallel', 1))
        tmp_dir = os.path.join(output_dir, config.get('tmp_dir', 'tmp'))
        extra_args = config.get('extra_args', None)
        # Read the cases
        if 'cases' not in config:
            raise ValueError("Missing 'cases' in solver configuration.")
        cases = []
        for case_name, case_config in config['cases'].items():
            if 'fields' not in case_config:
                raise ValueError(
                    f"Missing 'fields' in case '{case_name}' configuration.")
            fields = {
                field_name: abaqus_fields_reader(field_config)
                for field_name, field_config in case_config['fields'].items()
            }
            # If job_name, step_name and instance_name are not indicated, the default value is None
            step_name = case_config.get('step_name', None)
            instance_name = case_config.get('instance_name', None)
            cases.append(Case(AbaqusInputData(case_name, step_name, instance_name), fields))

        # Return the solver
        return AbaqusSolver(
            cases,
            parameters,
            output_dir,
            abaqus_bin,
            parallel,
            tmp_dir,
            extra_args=extra_args,
        )
