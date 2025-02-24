"""Module for multi-case solvers."""
from __future__ import annotations
from typing import List, Dict, Any, Type, TypeVar
from abc import ABC, abstractmethod
import os
import shutil
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver, OutputResult, CaseResult


T = TypeVar('T')


class Case(ABC):
    """Base case class for multi-case solvers."""

    @abstractmethod
    def name(self) -> str:
        """Return the name of the case.

        Returns
        -------
        str
            Name of the case.
        """

    @abstractmethod
    def get_fields(self) -> List[str]:
        """Get the fields to output for this case.

        Returns
        -------
        List[str]
            Fields to output for this case.
        """

    @abstractmethod
    def run(
        self,
        parameters: ParameterSet,
        values: np.ndarray,
        tmp_dir: str,
    ) -> CaseResult:
        """Run the case for the given set of parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        values : np.ndarray
            Current parameters to evaluate.
        tmp_dir : str
            Temporary directory to run the problem.

        Returns
        -------
        CaseResult
            Result of the case.
        """

    @classmethod
    @abstractmethod
    def read(
        cls: Type[T],
        name: str,
        config: Dict[str, Any],
    ) -> T:
        """Read the case from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the case.
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Case
            Case to use for this problem.
        """


class MultiCaseSolver(Solver, ABC):
    """Base class for solvers with multiple cases."""

    def __init__(
        self,
        cases: List[Case],
        parameters: ParameterSet,
        output_dir: str,
        tmp_dir: str,
        verbosity: str,
        parallel: int = 1,
    ) -> None:
        """Constructor for the solver class.

        Parameters
        ----------
        cases : List[Case]
            Cases to be run.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        tmp_dir : str
            Path to the temporary directory.
        parallel : int, optional
            Number of parallel processes to run, by default 1.
        """
        super().__init__(parameters, output_dir, tmp_dir, verbosity)
        self.cases = cases
        self.parallel = parallel
        self.cases_dir = os.path.join(output_dir, "cases")
        self.cases_hist = os.path.join(output_dir, "cases_hist")
        # Sanitise output fields
        output_fields = []
        for case in self.cases:
            for name in case.get_fields():
                if name in output_fields:
                    raise ValueError(f"Duplicate output field '{name}'.")
                output_fields.append(name)

    def prepare(self) -> None:
        """Prepare data for the optimisation."""
        self.verbosity_manager.prepare()
        # Create output directories
        os.makedirs(self.cases_dir, exist_ok=True)
        if os.path.isdir(self.cases_hist):
            shutil.rmtree(self.cases_hist)
        os.mkdir(self.cases_hist)
        # Build headers for case log files
        for case in self.cases:
            case_dir = os.path.join(self.cases_dir, case.name())
            with open(case_dir, 'w', encoding='utf8') as file:
                file.write(f"{'Start Time /s':>15}\t")
                file.write(f"{'Run Time /s':>15}\t")
                file.write(f"{'Success':>10}\t")
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Hash":>64}\n')

    def _write_history_entry(
        self,
        case: Case,
        result: CaseResult,
    ) -> None:
        """Write this case's history entry.

        Parameters
        ----------
        case : Case
            Case to write.
        result : CaseResult
            Result for this case.
        """
        # Write out the case file
        param_hash = self.parameters.hash(result.values)
        output_case_hist = os.path.join(self.cases_hist, f'{case.name()}-{param_hash}')
        result.write(output_case_hist, self.parameters)
        # Add record to case log file
        with open(os.path.join(self.cases_dir, case.name()), 'a', encoding='utf8') as file:
            file.write(f'{result.begin_time - self.begin_time:>15.8e}\t')
            file.write(f'{result.run_time:>15.8e}\t')
            file.write(f'{result.success:>10}\t')
            for value in result.values:
                file.write(f"{value:>15.6f}\t")
            file.write(f'{param_hash}\n')

    def get_output_fields(self) -> List[str]:
        """Get all output fields.

        Returns
        -------
        List[str]
            Output fields.
        """
        return [name for case in self.cases for name in case.get_fields()]

    def get_case_results(self, param_hash: str) -> List[CaseResult]:
        """Get the results for all cases for a given hash.

        Parameters
        ----------
        param_hash : str
            Hash of the case to load.

        Returns
        -------
        List[CaseResult]
            Results for all cases.
        """
        return [
            CaseResult.read(
                os.path.join(self.cases_hist, f'{case.name()}-{param_hash}'),
                self.parameters,
            )
            for case in self.cases
        ]

    def get_output_response(self, param_hash: str) -> Dict[str, OutputResult]:
        """Get the responses from all output fields for a given case.

        Parameters
        ----------
        param_hash : str
            Hash of the case to load.

        Returns
        -------
        Dict[str, OutputResult]
            Output responses.
        """
        results = self.get_case_results(param_hash)
        return {
            name: response
            for result in results
            for name, response in result.responses.items()
        }

    def get_case_params(self, param_hash: str) -> Dict[str, float]:
        """Get the parameters for a given hash.

        Parameters
        ----------
        param_hash : str
            Hash of the case to load.

        Returns
        -------
        Dict[str, float]
            Parameters for this hash.
        """
        # Just pick the first case to get the parameters
        result = self.get_case_results(param_hash)[0]
        return {param.name: result.values[i] for i, param in enumerate(self.parameters)}

    def solve(
        self,
        values: np.ndarray,
        concurrent: bool,
    ) -> Dict[str, OutputResult]:
        """Solve all cases for the given set of parameter values.

        Parameters
        ----------
        values : array
            Current parameters to evaluate.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        Dict[str, OutputResult]
            Evaluated results for each output field.
        """
        # Resolve tmp directory: use unique directory if concurrent
        tmp_dir = f'{self.tmp_dir}_{self.parameters.hash(values)}' if concurrent else self.tmp_dir
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

        # Evaluate all cases (in parallel if specified)
        def run_case(case: Case) -> CaseResult:
            with self.verbosity_manager:
                return case.run(self.parameters, values, tmp_dir)
        if self.parallel > 1:
            with Pool(self.parallel) as pool:
                results = pool.map(run_case, self.cases)
        else:
            results = map(run_case, self.cases)
        # Ensure we actually resolve the map and cleanup concurrent temporary directories
        results = list(results)
        if concurrent:
            shutil.rmtree(tmp_dir)

        # Post-process results: write history entries and collect outputs
        outputs: Dict[str, OutputResult] = {}
        for case, result in zip(self.cases, results):
            self._write_history_entry(case, result)
            for name in case.get_fields():
                outputs[name] = result.responses[name]
        return outputs

    @classmethod
    @abstractmethod
    def get_case_class(cls) -> Type[Case]:
        """Get the case class to use for this solver.

        Returns
        -------
        Type[Case]
            Case class to use for this solver.
        """

    @classmethod
    def read(
        cls: Type[T],
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> T:
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
        T
            Solver to use for this problem.
        """
        # Sanitise and extract configuration for the cases
        if 'cases' not in config:
            raise ValueError("Missing 'cases' in solver configuration.")
        config_cases = config.pop('cases')
        # Extract other information from the configuration
        tmp_dir = os.path.join(output_dir, config.pop('tmp_dir', 'tmp'))
        parallel = int(config.pop('parallel', 1))
        verbosity = config.pop('verbosity', None)
        # Initialise each case (and append any extra configuration)
        case_class = cls.get_case_class()
        cases = [case_class.read(name, case | config) for name, case in config_cases.items()]
        return cls(cases, parameters, output_dir, tmp_dir, verbosity, parallel=parallel)
