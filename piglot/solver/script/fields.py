"""Module for output fields from Script solver."""
from __future__ import annotations
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
import os
import copy
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import InputData, OutputField, OutputResult
from piglot.utils.solver_utils import write_parameters, get_case_name


class ScriptCallable(ABC):
    """Wrapper class for the callable function to use as the solver."""

    @abstractmethod
    def __call__(self, x: np.ndarray, tmp_dir: str) -> OutputResult:
        """Evaluate the function for the given point.

        Parameters
        ----------
        x : np.ndarray
            Point to evaluate the response for.
        tmp_dir : str
            Temporary directory to run the analyses.

        Returns
        -------
        OutputResult
            Response of the solver for the given point.
        """


class ScriptInputData(InputData):
    """Container for dummy input data."""

    def __init__(
            self,
            case_name: str,
            callback: ScriptCallable,
            ) -> None:
        super().__init__()
        self.case_name = case_name
        self.callback = callback

    def prepare(
        self,
        values: np.ndarray,
        parameters: ParameterSet,
        tmp_dir: str = None,
    ) -> ScriptInputData:
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
        ScriptInputData
            Input data prepared for the simulation.
        """
        return self

    def check(self, parameters: ParameterSet) -> None:
        """Check if the input data is valid according to the given parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        """

    def name(self) -> str:
        """Return the name of the input data.

        Returns
        -------
        str
            Name of the input data.
        """
        return self.case_name

    def get_current(self, target_dir: str) -> ScriptInputData:
        """Get the current input data.

        Parameters
        ----------
        target_dir : str
            Target directory to copy the input file.

        Returns
        -------
        ScriptInputData
            Current input data.
        """
        raise NotImplementedError("Cannot get current case for script solver")


class Script(OutputField):
    """Script output reader."""

    def check(self, input_data: ScriptInputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : ScriptInputData
            Input data for this case.

        """

    def get(self, input_data: ScriptInputData) -> OutputResult:
        """Reads reactions from a Script analysis.

        Parameters
        ----------
        input_data : ScriptInputData
            Input data for this case.

        Returns
        -------
        array
            2D array with parametric value and corresponding expression value.
        """
        raise NotImplementedError("Not available for script solvers")

    @staticmethod
    def read(config: Dict[str, Any]) -> Script:
        """Read the output field from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Reaction
            Output field to use for this problem.
        """
        return Script()
