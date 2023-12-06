"""Module for Abaqus solver."""
from typing import Dict, Any
import os
import time
import shutil
import subprocess
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver, Case, CaseResult, OutputField, OutputResult
from piglot.solver.abaqus.fields import abaqus_fields_reader, AbaqusInputData


class AbaqusSolver(Solver):  # inherits from the Solver class
    """Abaqus solver."""

    def __init__(
        self,
        cases: Dict[Case, Dict[str, OutputField]],
        parameters: ParameterSet,
        output_dir: str,
        abaqus_bin: str,
        parallel: int,
        tmp_dir: str,
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
        # The constructor first calls the constructor of the superclass Solver with the cases, \
        # parameters, and output_dir arguments. Then, it sets the abaqus_bin, parallel, and \
        # tmp_dir attributes of the AbaqusSolver instance to the corresponding arguments.

    def _gen_post_proc_inputfile(self, input_data: AbaqusInputData, tmp_dir: str) -> str:
        """Generate the post-processing input file.

        Parameters
        ----------
        input_data : AbaqusInputData
            Input data for the simulation.
        tmp_dir : str
            Temporary directory to run the simulation.
        """
        input_file, ext = os.path.splitext(os.path.basename(input_data.input_file))
        post_proc_file = os.path.join(tmp_dir, f'post_{input_file}.dat')
        with open(post_proc_file, 'w', encoding='utf-8') as file:
            file.write(f"input_file = '{input_file}{ext}'\n")
            file.write(f"job_name = '{input_data.job_name}'\n")
            file.write(f"step_name = '{input_data.step_name}'\n")
            file.write(f"instance_name = '{input_data.instance_name}'\n")
        return post_proc_file

    def _run_case(self, values: np.ndarray, case: Case, tmp_dir: str) -> CaseResult:
        """Run a single case wth Abaqus.

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
        input_data = case.input_data.prepare(
            values, self.parameters, tmp_dir=tmp_dir)
        input_file = input_data.input_file
        case_name, _ = os.path.splitext(os.path.basename(input_file))
        # Run ABAQUS (we don't use high precision timers here to keep track of the start time)
        begin_time = time.time()
        run_inp = subprocess.run(
            [self.abaqus_bin, f"job={input_data.job_name}", f"input={os.path.basename(input_file)}",
                'interactive', 'ask_delete=OFF'],
            cwd=tmp_dir,
            shell=True,  # deixar o sheel=True?
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        # TODO: Verificar se a simulação foi concluída
        post_proc_file = self._gen_post_proc_inputfile(input_data, tmp_dir)
        python_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reader.py')
        # TODO: ver se da para passar a infornmação do job_name, step_name e instance_name para o 
        # abaqus
        run_odb = subprocess.run(
            [self.abaqus_bin, 'viewer', f"noGUI={python_script}", "--",
             f"input_file={os.path.basename(post_proc_file)}"],
            cwd=tmp_dir,
            shell=True,  # deixar o sheel=True?
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        # TODO: Verificar se o pos processamento foi concluído
        end_time = time.time()
        # TODO: VER DEPOIS PARA O ABAQUS
        # Check if simulation completed
        screen_file = os.path.join(os.path.splitext(
            input_file)[0], f'{case_name}.screen')
        # TODO: REFAZER!!!!!!
        failed_case = False
        responses = {name: field.get(input_data)
                     for name, field in case.fields.items()}
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
        if not 'abaqus_path' in config:
            raise ValueError("Missing 'abaqus_path' in solver configuration.")
        abaqus_bin = config['abaqus_path']
        # Read the parallelism and temporary directory (if present)
        parallel = int(config.get('parallel', 1))
        tmp_dir = os.path.join(output_dir, config.get('tmp_dir', 'tmp'))
        # Read the cases
        if not 'cases' in config:
            raise ValueError("Missing 'cases' in solver configuration.")
        cases = []
        for case_name, case_config in config['cases'].items():
            if not 'fields' in case_config:
                raise ValueError(
                    f"Missing 'fields' in case '{case_name}' configuration.")
            fields = {
                field_name: abaqus_fields_reader(field_config)
                for field_name, field_config in case_config['fields'].items()
            }
            cases.append(Case(AbaqusInputData(case_name, case_config['job_name'],
                                              case_config['step_name'],
                                              case_config['instance_name']), fields))
        # Return the solver
        return AbaqusSolver(cases, parameters, output_dir, abaqus_bin, parallel, tmp_dir)
