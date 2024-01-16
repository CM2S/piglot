"""Module for Dummy solver."""
from typing import Dict, Any
import time
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver, Case, CaseResult, OutputField, OutputResult
from piglot.solver.dummy.fields import dummy_fields_reader, DummyInputData


class DummySolver(Solver):
    """Dummy solver."""

    def __init__(
            self,
            cases: Dict[Case, Dict[str, OutputField]],
            parameters: ParameterSet,
            output_dir: str,
            ) -> None:
        """Constructor for the Dummy solver class.

        Parameters
        ----------
        cases : Dict[Case, Dict[str, OutputField]]
            Cases to be run and respective output fields.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        """
        super().__init__(cases, parameters, output_dir)

    def _run_case(self, values: np.ndarray, case: Case) -> CaseResult:
        """Run a single case wth Dummy.

        Parameters
        ----------
        values: np.ndarray
            Current parameter values
        case : Case
            Case to run.

        Returns
        -------
        CaseResult
            Results for this case
        """
        # Copy input file replacing parameters by passed value
        input_data = case.input_data.prepare(values, self.parameters)
        # Run dummy solver
        begin_time = time.time()
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
        def run_case(case: Case) -> CaseResult:
            return self._run_case(values, case)
        results = map(run_case, self.cases)
        # Ensure we actually resolve the map
        results = list(results)
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
        # Read the cases
        if 'cases' not in config:
            raise ValueError("Missing 'cases' in solver configuration.")
        cases = []
        for case_name, case_config in config['cases'].items():
            if 'fields' not in case_config:
                raise ValueError(f"Missing 'fields' in case '{case_name}' configuration.")
            fields = {
                field_name: dummy_fields_reader(field_config)
                for field_name, field_config in case_config['fields'].items()
            }
            cases.append(Case(DummyInputData(None, case_name), fields))
        # Return the solver
        return DummySolver(cases, parameters, output_dir)
