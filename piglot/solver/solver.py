"""Module for solvers."""
from dataclasses import dataclass
from typing import List, Dict, Any, Type, TypeVar
from abc import ABC, abstractmethod
import os
import time
import shutil
import numpy as np
from yaml import safe_dump_all, safe_load_all
from piglot.parameter import ParameterSet, ParameterValues
from piglot.utils.assorted import pretty_time
from piglot.utils.solver_utils import OutputStream, VerbosityManager


T = TypeVar('T', bound='Solver')
ResultT = TypeVar('ResultT', bound='CaseResult')


@dataclass
class OutputResult:
    """Container for output results."""
    time: np.ndarray
    data: np.ndarray

    def get_time(self) -> np.ndarray:
        """Get the time column of the result.

        Returns
        -------
        np.ndarray
            Time column.
        """
        return self.time

    def get_data(self) -> np.ndarray:
        """Get the data column of the result.

        Returns
        -------
        np.ndarray
            Data column.
        """
        return self.data


@dataclass
class CaseResult:
    """Class for case results."""
    begin_time: float
    run_time: float
    parameters: dict[str, float]
    success: bool
    param_hash: str
    responses: Dict[str, OutputResult]

    def write(self, filename: str) -> None:
        """Write out the case result.

        Parameters
        ----------
        filename : str
            Path to write the file to.
        parameters : ParameterSet
            Set of parameters for this case.
        """
        # Build case metadata
        metadata = {
            "start_time": time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(self.begin_time)),
            "begin_time": self.begin_time,
            "run_time": self.run_time,
            "run_time (pretty)": pretty_time(self.run_time),
            "parameters": self.parameters,
            "success": "true" if self.success else "false",
            "param_hash": self.param_hash,
        }
        # Build response data
        responses = {
            name: list(zip(result.get_time().tolist(), result.get_data().tolist()))
            for name, result in self.responses.items()
        }
        # Dump all data to file
        with open(filename, 'w', encoding='utf8') as file:
            safe_dump_all((metadata, responses), file)

    @classmethod
    def read(cls: type[ResultT], filename: str) -> ResultT:
        """Read a case result file.

        Parameters
        ----------
        filename : str
            Path to the case result file.

        Returns
        -------
        ResultT
            Result instance.
        """
        # Read the file
        with open(filename, 'r', encoding='utf8') as file:
            metadata, responses_raw = safe_load_all(file)
        # Parse the responses
        responses = {
            name: OutputResult(np.array([a[0] for a in data]), np.array([a[1] for a in data]))
            for name, data in responses_raw.items()
        }
        return cls(
            metadata["begin_time"],
            metadata["run_time"],
            {name: float(value) for name, value in metadata["parameters"].items()},
            metadata["success"] == "true",
            metadata["param_hash"],
            responses,
        )


class Solver(ABC):
    """Base class for solvers."""

    def __init__(
        self,
        parameters: ParameterSet,
        output_dir: str,
        tmp_dir: str,
        verbosity: str,
    ) -> None:
        self.parameters = parameters
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.verbosity_manager = VerbosityManager(verbosity, os.path.join(output_dir, 'solver'))
        self.begin_time = time.time()

    @abstractmethod
    def prepare(self) -> None:
        """Prepare data for the optimisation."""

    @abstractmethod
    def solve(self, values: ParameterValues, concurrent: bool) -> dict[str, OutputResult]:
        """Solve all cases for the given set of parameter values.

        Parameters
        ----------
        values : ParameterValues
            Named set of parameter values for this evaluation.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        dict[str, OutputResult]
            Evaluated results for each output field.
        """

    @abstractmethod
    def get_output_fields(self) -> List[str]:
        """Get all output fields.

        Returns
        -------
        List[str]
            Output fields.
        """

    @abstractmethod
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

    @abstractmethod
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

    def get_current_response(self) -> Dict[str, OutputResult]:
        """Get the responses from a given output field for all cases.

        Returns
        -------
        Dict[str, OutputResult]
            Output responses.
        """
        raise NotImplementedError("This solver does not support getting current responses.")

    @classmethod
    @abstractmethod
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
        Solver
            Solver to use for this problem.
        """


class SingleCaseSolver(Solver, ABC):
    """Generic class for solvers with a single case."""

    def __init__(
        self,
        output_fields: List[str],
        parameters: ParameterSet,
        output_dir: str,
        tmp_dir: str,
        verbosity: str,
    ) -> None:
        """Constructor for the solver class.

        Parameters
        ----------
        output_fields : List[str]
            List of output fields.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        tmp_dir : str
            Path to the temporary directory.
        """
        super().__init__(parameters, output_dir, tmp_dir, verbosity)
        self.output_fields = output_fields
        self.cases_dir = os.path.join(output_dir, "cases")
        self.cases_hist = os.path.join(output_dir, "cases_hist")

    def prepare(self) -> None:
        """Prepare data for the optimisation."""
        self.verbosity_manager.prepare()
        # Create output directories
        os.makedirs(self.cases_dir, exist_ok=True)
        if os.path.isdir(self.cases_hist):
            shutil.rmtree(self.cases_hist)
        os.mkdir(self.cases_hist)
        # Build headers for case log files
        for case in self.output_fields:
            case_dir = os.path.join(self.cases_dir, case)
            with open(case_dir, 'w', encoding='utf8') as file:
                file.write(f"{'Start Time /s':>15}\t")
                file.write(f"{'Run Time /s':>15}\t")
                file.write(f"{'Success':>10}\t")
                for name in self.parameters.get_scalar_names():
                    file.write(f"{name:>15}\t")
                file.write(f'{"Hash":>64}\n')

    def get_output_fields(self) -> List[str]:
        """Get all output fields.

        Returns
        -------
        List[str]
            Output fields.
        """
        return self.output_fields

    def get_case_result(self, param_hash: str) -> CaseResult:
        """Get the result for a given case.

        Parameters
        ----------
        param_hash : str
            Hash of the case to load.

        Returns
        -------
        CaseResult
            Result for this hash.
        """
        return CaseResult.read(os.path.join(self.cases_hist, param_hash))

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
        result = self.get_case_result(param_hash)
        return {
            param.name: result.parameters[param.name] for i, param in enumerate(self.parameters)
        }

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
        return self.get_case_result(param_hash).responses

    @abstractmethod
    def _solve(
        self, values: dict[str, float], concurrent: bool, stream: OutputStream
    ) -> dict[str, OutputResult]:
        """Internal solver for the prescribed problems.

        Parameters
        ----------
        values : dict[str, float]
            Named set of parameter values for this evaluation.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).
        stream : OutputStream
            Output stream for this call.

        Returns
        -------
        dict[str, OutputResult]
            Evaluated results for each output field.
        """

    def solve(self, values: ParameterValues, concurrent: bool) -> dict[str, OutputResult]:
        """Solve all cases for the given set of parameter values.

        Parameters
        ----------
        values : ParameterValues
            Named set of parameter values for this evaluation.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        dict[str, OutputResult]
            Evaluated results for each output field.
        """
        # Run the solver
        begin_time = time.time()
        with self.verbosity_manager as stream:
            results = self._solve(values.scalar_values, concurrent, stream)
        run_time = time.time() - begin_time
        # Post-process results: write history entries
        case_result = CaseResult(
            begin_time, run_time, values.scalar_values, True, values.param_hash, results
        )
        case_result.write(os.path.join(self.cases_hist, values.param_hash))
        return results
