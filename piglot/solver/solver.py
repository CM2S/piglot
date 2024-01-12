"""Module for solvers"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from abc import ABC, abstractmethod
import os
import time
import shutil
import numpy as np
from yaml import safe_dump_all, safe_load_all
from piglot.parameter import ParameterSet
from piglot.utils.assorted import pretty_time


class InputData(ABC):
    """Generic class for solver input data."""

    @abstractmethod
    def prepare(
            self,
            values: np.ndarray,
            parameters: ParameterSet,
            tmp_dir: str = None,
            ) -> InputData:
        """Prepare the input data for the simulation with a given set of parameters.

        Parameters
        ----------
        values : np.ndarray
            Parameters to run for.
        parameters : ParameterSet
            Parameter set for this problem.
        tmp_dir : str, optional
            Temporary directory to run the analyses, by default None

        Returns
        -------
        InputData
            Input data prepared for the simulation.
        """

    def check(self, parameters: ParameterSet) -> None:
        """Check if the input data is valid according to the given parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the name of the input data.

        Returns
        -------
        str
            Name of the input data.
        """


class OutputField(ABC):
    """Generic class for output fields.

    Methods
    -------
    check(case):
        Check for validity in the input file before reading. This needs to be called prior
        to any reading on the file.
    get(case):
        Read the input file and returns the requested fields.
    """

    @abstractmethod
    def check(self, input_data: InputData) -> None:
        """Check for validity in the input data before reading.

        Parameters
        ----------
        input_data : InputData
            Container for the solver input data.
        """

    @abstractmethod
    def get(self, input_data: InputData) -> OutputResult:
        """Read the output data from the simulation.

        Parameters
        ----------
        input_data : InputData
            Container for the solver input data.

        Returns
        -------
        OutputResult
            Output result for this field.
        """

    @staticmethod
    @abstractmethod
    def read(config: Dict[str, Any]) -> OutputField:
        """Read the output field from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        OutputField
            Output field to use for this problem.
        """


class Case:
    """Generic class for cases."""

    def __init__(self, input_data: InputData, fields: Dict[str, OutputField]) -> None:
        self.input_data = input_data
        self.fields = fields

    def check(self, parameters: ParameterSet) -> None:
        """Prepare the case for the simulation."""
        self.input_data.check(parameters)
        for field in self.fields.values():
            field.check(self.input_data)

    def name(self) -> str:
        """Return the name of the case.

        Returns
        -------
        str
            Name of the case.
        """
        return self.input_data.name()


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
    values: np.ndarray
    success: bool
    param_hash: str
    responses: Dict[str, OutputResult]

    def write(self, filename: str, parameters: ParameterSet) -> None:
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
            "parameters": {n: float(v) for n, v in parameters.to_dict(self.values).items()},
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

    @staticmethod
    def read(filename: str) -> CaseResult:
        """Read a case result file.

        Parameters
        ----------
        filename : str
            Path to the case result file.

        Returns
        -------
        CaseResult
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
        return CaseResult(
            metadata["begin_time"],
            metadata["run_time"],
            np.array(metadata["parameters"].values()),
            metadata["success"] == "true",
            metadata["param_hash"],
            responses,
        )


class Solver(ABC):
    """Base class for solvers."""

    def __init__(
            self,
            cases: List[Case],
            parameters: ParameterSet,
            output_dir: str,
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
        """
        self.cases = cases
        self.parameters = parameters
        self.output_dir = output_dir
        self.begin_time = time.time()
        self.cases_dir = os.path.join(output_dir, "cases")
        self.cases_hist = os.path.join(output_dir, "cases_hist")

    def prepare(self) -> None:
        """Prepare data for the optimisation."""
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
        # Prepare individual cases and output fields
        for case in self.cases:
            case.check(self.parameters)

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
            for i, param in enumerate(self.parameters):
                file.write(f"{param.denormalise(result.values[i]):>15.6f}\t")
            file.write(f'{param_hash}\n')

    def get_output_fields(self) -> Dict[str, Tuple[Case, OutputField]]:
        """Get all output fields.

        Returns
        -------
        Dict[str, Tuple[Case, OutputField]]
            Output fields.
        """
        output_fields = {}
        for case in self.cases:
            for name, field in case.fields.items():
                if name in output_fields:
                    raise ValueError(f"Duplicate output field '{name}'.")
                output_fields[name] = (case, field)
        return output_fields

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
        responses = {}
        for case in self.cases:
            # Read the case result
            result = CaseResult.read(os.path.join(self.cases_hist, f'{case.name()}-{param_hash}'))
            for name, response in result.responses.items():
                responses[name] = response
        return responses

    @abstractmethod
    def get_current_response(self) -> Dict[str, OutputResult]:
        """Get the responses from a given output field for all cases.

        Returns
        -------
        Dict[str, OutputResult]
            Output responses.
        """

    @abstractmethod
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
        # Evaluate all cases
        results = self._solve(values, concurrent)
        # Post-process results: write history entries and collect outputs
        outputs: Dict[str, OutputResult] = {}
        for case, result in results.items():
            self._write_history_entry(case, result)
            for name in case.fields:
                outputs[name] = result.responses[name]
        return outputs

    @staticmethod
    @abstractmethod
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
