"""Module for output fields from Crate solver."""
from __future__ import annotations
from typing import Dict, Any, Union
import os
import numpy as np
import pandas as pd
from piglot.parameter import ParameterSet
from piglot.solver.solver import InputData, OutputField, OutputResult
from piglot.utils.solver_utils import get_case_name, write_parameters
from piglot.utils.solver_utils import has_parameter


class CrateInputData(InputData):
    """Container for CRATE input data."""

    def __init__(self, input_file: str) -> None:
        super().__init__()
        self.input_file = input_file

    def prepare(
            self,
            values: np.ndarray,
            parameters: ParameterSet,
            tmp_dir: str = None,
            ) -> CrateInputData:
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
        CrateInputData
            Input data prepared for the simulation.
        """
        # Copy file and write out the parameters
        dest_file = os.path.join(tmp_dir, os.path.basename(self.input_file))
        write_parameters(parameters.to_dict(values), self.input_file, dest_file)
        return CrateInputData(dest_file)

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
        for name in param_dict:
            if not has_parameter(self.input_file, f'<{name}>'):
                raise RuntimeError(f"Parameter '{name}' not found in input file.")

    def name(self) -> str:
        """Return the name of the input data.

        Returns
        -------
        str
            Name of the input data.
        """
        return os.path.basename(self.input_file)

    def get_current(self, target_dir: str) -> CrateInputData:
        """Get the current input data.

        Parameters
        ----------
        target_dir : str
            Target directory to copy the input file.

        Returns
        -------
        CrateInputData
            Current input data.
        """
        return CrateInputData(os.path.join(target_dir, os.path.basename(self.input_file)))


class HresFile(OutputField):
    """CRATE .hres file reader."""

    def __init__(
            self,
            y_field: Union[str, int],
            x_field: str = "LoadFactor",
            ):
        """Constructor for .out file reader

        Parameters
        ----------
        y_field : Union[str, int]
            Field to read from the output file. Can be a single column index or name.
        x_field : str, optional
            Field to use as index in the resulting DataFrame, by default "LoadFactor".

        Raises
        ------
        RuntimeError
            If element and GP numbers are not consistent.
        """
        super().__init__()
        self.y_field = y_field
        self.x_field = x_field
        self.separator = 16

    def check(self, input_data: CrateInputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : CrateInputData
            Input data for this case.
        """
        pass

    def get(self, input_data: CrateInputData) -> OutputResult:
        """Get a parameter from the .hres file.

        Parameters
        ----------
        input_data : CrateInputData
            Input data for this case.

        Returns
        -------
        DataFrame
            DataFrame with the requested fields.
        """
        input_file = input_data.input_file
        casename = get_case_name(input_file)
        output_dir = os.path.splitext(input_file)[0]
        filename = os.path.join(output_dir, f'{casename}.hres')
        # Ensure the file exists
        if not os.path.exists(filename):
            return OutputResult(np.empty(0), np.empty(0))
        # Read the first line of the file to find the total number of columns
        with open(filename, 'r', encoding='utf8') as file:
            line_len = len(file.readline())
        n_columns = int((line_len-10) / self.separator)
        # Fixed-width read
        df = pd.read_fwf(filename, widths=[10] + n_columns*[self.separator])
        # Extract indices for named columns
        columns = df.columns.tolist()
        x_column = columns.index(self.x_field) if isinstance(self.x_field, str) else self.x_field
        y_column = columns.index(self.y_field) if isinstance(self.y_field, str) else self.y_field
        # Return the given quantity as the x-variable
        return OutputResult(df.iloc[:, x_column].to_numpy(), df.iloc[:, y_column].to_numpy())

    @staticmethod
    def read(config: Dict[str, Any]) -> HresFile:
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
        # Read the field
        if 'y_field' not in config:
            raise ValueError("Missing 'y_field' in hresFile configuration.")
        y_field = config['y_field']
        # Read the x field (if passed)
        x_field = config.get('x_field', 'LoadFactor')
        return HresFile(y_field, x_field)


def crate_fields_reader(config: Dict[str, Any]) -> OutputField:
    """Read the output fields for the CRATE solver.

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
        'hresFile': HresFile,
    }
    if field_name not in readers:
        raise ValueError(f"Unknown output field name '{field_name}'.")
    return readers[field_name].read(config)
