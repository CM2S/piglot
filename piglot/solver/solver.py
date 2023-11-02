"""Module for solvers"""
from __future__ import annotations
from typing import Dict, Tuple, Any, Type, Callable
from abc import ABC, abstractmethod
import time
import numpy as np
from piglot.parameter import ParameterSet


class Case(ABC):
    """Generic class for cases."""

    @abstractmethod
    def prepare(self) -> None:
        """Prepare the case for the simulation."""

    @staticmethod
    @abstractmethod
    def read(config: Dict[str, Any]) -> Case:
        """Read the case from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Case
            Case to use for this problem.
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
    def check(self, case: Case) -> None:
        """Check for validity in the input data before reading.

        Parameters
        ----------
        case : Case
            Container for the solver input data.
        """

    @abstractmethod
    def get(self, case: Case) -> np.ndarray:
        """Read the output data from the simulation.

        Parameters
        ----------
        case : Case
            Container for the solver input data.
        """

    @abstractmethod
    def name(self, field_idx: int = None) -> str:
        """Return the name of the current field.

        Parameters
        ----------
        field_idx : int, optional
            Index of the field to output, by default None.

        Returns
        -------
        str
            Field name.
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


class OutputResult(ABC):
    """Generic class for output results."""

    @abstractmethod
    def get_time(self) -> np.ndarray:
        """Get the time column of the result.

        Returns
        -------
        np.ndarray
            Time column.
        """

    @abstractmethod
    def get_data(self, field_idx: int=0) -> np.ndarray:
        """Get the data column of the result.

        Parameters
        ----------
        field_idx : int
            Index of the field to output.

        Returns
        -------
        np.ndarray
            Data column.
        """


class Solver(ABC):
    """Base class for solvers."""

    def __init__(
            self,
            cases: Dict[Case, Dict[str, OutputField]],
            parameters: ParameterSet,
            output_dir: str,
        ) -> None:
        """Constructor for the solver class.

        Parameters
        ----------
        cases : Dict[Case, Dict[str, OutputField]]
            Cases to be run and respective output fields.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        """
        self.cases = cases
        self.parameters = parameters
        self.output_dir = output_dir
        self.begin_time = time.time()

    def prepare(self) -> None:
        """Prepare data for the optimsation."""
        for case, fields in self.cases.items():
            case.prepare()
            for field in fields:
                field.check(case)

    def get_output_fields(self) -> Dict[str, Tuple[Case, OutputField]]:
        """Get all output fields.

        Returns
        -------
        Dict[str, Tuple[Case, OutputField]]
            Output fields.
        """
        output_fields = {}
        for case, fields in self.cases.items():
            for name, field in fields.items():
                if name in output_fields:
                    raise ValueError(f"Duplicate output field '{name}'.")
                output_fields[name] = (case, field)
        return output_fields

    @abstractmethod
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


def generic_read_cases(
        config_cases: Dict[str, Any],
        case_class: Type[Case],
        fields_reader: Callable[[Dict[str, Any]], OutputField],
    ) -> Dict[Case, Dict[str, OutputField]]:
    """Read the cases from the configuration dictionary.

    Parameters
    ----------
    config_cases : Dict[str, Any]
        Configuration dictionary.
    case_class : Case
        Case class to use for this problem.
    fields_reader : Callable[[Dict[str, Any]], OutputField]
        Function to read the output fields.

    Returns
    -------
    Dict[Case, Dict[str, OutputField]]
        Cases to use for this problem.
    """
    cases = {}
    for case_name, case_config in config_cases.items():
        # Read the case
        case = case_class.read(case_config)
        # Read the output fields for this case
        if not 'fields' in case_config:
            raise ValueError(f"Missing output fields for case '{case_name}'.")
        cases[case] = {name: fields_reader(field) for name, field in case_config['fields'].items()}
    return cases
