"""Module for output fields from Dummy solver."""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import InputData, OutputField, OutputResult


class DummyInputData(InputData):
    """Container for dummy input data."""

    def __init__(self, parameters: Dict[str, float], case_name: str) -> None:
        super().__init__()
        self.parameters = parameters
        self.case_name = case_name

    def prepare(
            self,
            values: np.ndarray,
            parameters: ParameterSet,
            tmp_dir: str = None,
            ) -> DummyInputData:
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
        DummyInputData
            Input data prepared for the simulation.
        """
        return DummyInputData(parameters.to_dict(values), self.case_name)

    def check(self, parameters: ParameterSet) -> None:
        """Check if the input data is valid according to the given parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        """
        # Generate a dummy set of parameters (to ensure proper handling of output parameters)
        values = np.array([parameter.inital_value for parameter in parameters])
        param_dict = parameters.to_dict(values, input_normalised=False)
        # Check if the require parameters are present in the input file
        parameters_dummy = ('m', 'c')
        if len(parameters_dummy) != len(param_dict):
            raise RuntimeError("Invalid parameters: the parameters 'm' and 'c' are required.")
        if not all(name in param_dict for name in parameters_dummy):
            raise RuntimeError("Invalid parameters: the parameters 'm' and 'c' are required.")

    def name(self) -> str:
        """Return the name of the input data.

        Returns
        -------
        str
            Name of the input data.
        """
        return self.case_name


class Modifier(OutputField):
    """Modifier outputs reader."""

    def __init__(self, y_offset: float = 0.0, y_scale: float = 1.0):
        """Constructor for reaction reader

        Parameters
        ----------
        y_offset : float, optional
            Offset to apply to the y coordinate, by default 0.0
        y_scale : float, optional
            Scale to apply to the y coordinate, by default 1.0
        """
        super().__init__()
        self.y_offset = y_offset
        self.y_scale = y_scale

    def check(self, input_data: DummyInputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : DummyInputData
            Input data for this case.

        """
        # Is macroscopic file?
        pass

    def get(self, input_data: DummyInputData) -> OutputResult:
        """Reads reactions from a Dummy analysis.

        Parameters
        ----------
        input_data : DummyInputData
            Input data for this case.

        Returns
        -------
        array
            2D array with x in the first column and modified y in the second.
        """
        x = np.linspace(0.0, 1.0, 100)
        parameters = input_data.parameters
        y = parameters['m'] * x + parameters['c']
        return OutputResult(x, y*self.y_scale + self.y_offset)

    @staticmethod
    def read(config: Dict[str, Any]) -> Modifier:
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
        # Read the y_offset (if passed)
        y_offset = int(config.get('y_offset', 0.0))
        # Read the y_scale (if passed)
        y_scale = int(config.get('y_scale', 1.0))
        return Modifier(y_offset, y_scale)


def dummy_fields_reader(config: Dict[str, Any]) -> OutputField:
    """Read the output fields for the dummy solver.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    OutputField
        Output field to use for this problem.
    """
    # Extract name of output field
    if 'name' not in config:
        raise ValueError("Missing 'name' in output field configuration.")
    field_name = config['name']
    # Delegate to the appropriate reader
    readers = {
        'Modifier': Modifier,
    }
    if field_name not in readers:
        raise ValueError(f"Unknown output field name '{field_name}'.")
    return readers[field_name].read(config)
