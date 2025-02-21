"""Module for output fields from Crate solver."""
from __future__ import annotations
from typing import Dict, Any, Union
import os
import numpy as np
import pandas as pd
from piglot.solver.input_file_solver import InputData, OutputField
from piglot.solver.solver import OutputResult
from piglot.utils.solver_utils import get_case_name


class HresFile(OutputField):
    """CRATE .hres file reader."""

    def __init__(
        self,
        y_field: Union[str, int],
        x_field: str = "LoadFactor",
    ) -> None:
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

    def check(self, input_data: InputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : InputData
            Input data for this case.
        """

    def get(self, input_data: InputData) -> OutputResult:
        """Get a parameter from the .hres file.

        Parameters
        ----------
        input_data : InputData
            Input data for this case.

        Returns
        -------
        DataFrame
            DataFrame with the requested fields.
        """
        input_file = os.path.join(input_data.tmp_dir, input_data.input_file)
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

    @classmethod
    def read(cls, config: Dict[str, Any]) -> HresFile:
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
        return cls(y_field, x_field)
