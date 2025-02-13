"""Module for output fields from Links solver."""
from __future__ import annotations
from typing import Dict, Any
import os
import re
import numpy as np
import pandas as pd
from piglot.solver.input_file_solver import InputData, OutputField
from piglot.solver.solver import OutputResult
from piglot.utils.solver_utils import get_case_name


class Reaction(OutputField):
    """Reaction outputs reader."""

    def __init__(self, field: str, x_field: str, set_name: str, direction: str) -> None:
        """Constructor for reaction reader

        Parameters
        ----------
        field : str
            Field to read.
        x_field : str
            x_field to read.
        set_name : str
            Node set to read.
        direction : str
            Direction to read.
        """
        super().__init__()
        direction_dict = {'x': 1, 'y': 2, 'z': 3}
        self.field = field
        self.x_field = x_field
        self.set_name = set_name
        self.direction = direction_dict[direction]

    def check(self, input_data: InputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : AbaqusInputData
            Input data for this case.
        """
        has_space = ' ' in self.set_name
        cwd = os.getcwd()
        case_name, ext = os.path.splitext(os.path.basename(input_data.input_file))
        input_file = os.path.join(cwd, case_name + ext)
        with open(input_file, 'r', encoding='utf-8') as file:
            data = file.read()
            if has_space:
                nsets_list = re.findall(r'\*Nset, nset="?([^",]+)"?', data)
            else:
                nsets_list = re.findall(r'\*Nset, nset="?([^",\s]+)"?', data)
            if len(nsets_list) == 0:
                raise ValueError("No sets found in the file.")
            if self.set_name not in nsets_list:
                raise ValueError(f"Set name '{self.set_name}' not found in the file.")

    def get(self, input_data: InputData) -> OutputResult:
        """Reads reactions from a Abaqus analysis.

        Parameters
        ----------
        input_data : AbaqusInputData
            Input data for this case.

        Returns
        -------
        array
            2D array with load factor in the first column and reactions in the second.
        """
        cwd = os.getcwd()
        input_file = input_data.input_file
        casename = get_case_name(input_file)
        output_dir = os.path.join(
            cwd,
            os.path.splitext(input_file)[0],
        )
        output_dir = os.path.dirname(output_dir)

        reduction = {
            'RF': np.sum,
            'U': np.mean,
            'S': np.mean,
            'LE': np.mean,
            'E': np.mean,
        }
        # X field
        field_filename = os.path.join(
            output_dir,
            f'{casename}_{self.set_name}_{self.x_field}.txt',
        )
        # Ensure the file exists
        if not os.path.exists(field_filename):
            return OutputResult(np.empty(0), np.empty(0))
        data = pd.read_csv(field_filename, sep=' ', index_col=False)
        columns = [a for a in data.columns if f"{self.x_field}{self.direction}" in a]
        # Extract columns from data frame
        data_group = data[columns].to_numpy()
        x_field = reduction[self.x_field](data_group, axis=1)

        # Y field
        field_filename = os.path.join(
            output_dir,
            f'{casename}_{self.set_name}_{self.field}.txt',
        )
        # Ensure the file exists
        if not os.path.exists(field_filename):
            return OutputResult(np.empty(0), np.empty(0))
        data = pd.read_csv(field_filename, sep=' ', index_col=False)
        columns = [a for a in data.columns if f"{self.field}{self.direction}" in a]
        # Extract columns from data frame
        data_group = data[columns].to_numpy()
        y_field = reduction[self.field](data_group, axis=1)

        return OutputResult(x_field, y_field)

    @classmethod
    def read(cls, config: Dict[str, Any]) -> Reaction:
        """Read the output field from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        FieldsOutput
            Output field to use for this problem.
        """
        # Read the field
        if 'field' not in config:
            raise ValueError("Missing 'field' in reaction configuration.")
        field = config['field']
        # Read the x_field
        if 'x_field' not in config:
            raise ValueError("Missing 'x_field' in reaction configuration.")
        x_field = config['x_field']
        # Read the set_name
        if 'set_name' not in config:
            raise ValueError("Missing 'set_name' in reaction configuration.")
        set_name = config['set_name']
        # Read the direction
        if 'direction' not in config:
            raise ValueError("Missing 'direction' in reaction configuration.")
        direction = config['direction']

        return cls(field, x_field, set_name, direction)
