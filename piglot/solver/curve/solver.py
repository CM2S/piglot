"""Module for Curve solver."""
from typing import Dict, Any, List
import os
import time
import shutil
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import sympy
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver, Case, CaseResult, OutputResult
from piglot.solver.curve.fields import CurveInputData, Curve


class CurveSolver(Solver):
    """Curve solver."""

    def __init__(
            self,
            cases: List[Case],
            parameters: ParameterSet,
            output_dir: str,
            parallel: int = 1,
            tmp_dir: str = 'tmp',
            ) -> None:
        """Constructor for the Curve solver class.

        Parameters
        ----------
        cases : List[Case]
            Cases to be run.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        parallel : int
            Number of parallel processes to use.
        tmp_dir : str
            Path to the temporary directory.
        """
        super().__init__(cases, parameters, output_dir)
        self.parallel = parallel
        self.tmp_dir = tmp_dir

    def _run_case(self, values: np.ndarray, case: Case, tmp_dir: str) -> CaseResult:
        """Run a single case wth Curve.

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
        input_data: CurveInputData = case.input_data.prepare(values, self.parameters, tmp_dir)
        # Run dummy solver
        begin_time = time.time()
        # Read the expression from the input file
        with open(input_data.input_file, 'r', encoding='utf8') as file:
            expression_str = file.read()
        symbs = sympy.symbols(input_data.parametric)
        expression = sympy.lambdify(symbs, expression_str)
        # Evaluate the expression on the grid
        grid = np.linspace(input_data.bounds[0], input_data.bounds[1], input_data.points)
        curve = [expression(**{input_data.parametric: x}) for x in grid]
        # Write out the curve
        output_file = os.path.splitext(input_data.input_file)[0] + '.out'
        with open(output_file, 'w', encoding='utf8') as file:
            for x, y in zip(grid, curve):
                file.write(f'{x} {y}\n')
        # Read results from output directories
        responses = {name: field.get(input_data) for name, field in case.fields.items()}
        end_time = time.time()
        return CaseResult(
            begin_time,
            end_time - begin_time,
            values,
            True,
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
        # Read the parallelism and temporary directory (if present)
        parallel = int(config.get('parallel', 1))
        tmp_dir = os.path.join(output_dir, config.get('tmp_dir', 'tmp'))
        # Read the cases
        if 'cases' not in config:
            raise ValueError("Missing 'cases' in solver configuration.")
        cases = []
        for case_name, case_config in config['cases'].items():
            if 'expression' not in case_config:
                raise ValueError("Missing 'expression' in solver configuration.")
            if 'parametric' not in case_config:
                raise ValueError("Missing 'parametric' in solver configuration.")
            if 'bounds' not in case_config:
                raise ValueError("Missing 'bounds' in solver configuration.")
            expression = case_config['expression']
            parametric = case_config['parametric']
            bounds = case_config['bounds']
            points = int(case_config['points']) if 'points' in case_config else 100
            input_data = CurveInputData(case_name, expression, parametric, bounds, points)
            cases.append(Case(input_data, {case_name: Curve()}))
        # Return the solver
        return CurveSolver(cases, parameters, output_dir, parallel=parallel, tmp_dir=tmp_dir)
