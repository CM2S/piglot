"""Module for output fields from Links solver."""
from __future__ import annotations
from typing import Dict, Any
import os
import re
import numpy as np
import pandas as pd
from piglot.solver.input_file_solver import InputData, OutputField
from piglot.solver.solver import OutputResult, FullFieldOutputResult
from piglot.utils.solver_utils import get_case_name


class FieldsOutput(OutputField):
    """Fields output reader."""

    def __init__(self, field: str, x_field: str, set_name: str) -> None:
        """Constructor for reaction reader

        Parameters
        ----------
        field : str
            Field to read.
        x_field : str
            x_field to read.
        set_name : str
            Node set to read.
        """
        super().__init__()
        self.field = field
        self.x_field = x_field
        self.set_name = set_name

    def check(self, input_data: InputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : AbaqusInputData
            Input data for this case.
        """
        has_space = ' ' in self.set_name
        input_file = os.path.join(input_data.tmp_dir, input_data.input_file)
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

    @staticmethod
    def get_reduction(field: str) -> np.ufunc:
        """Get the reduction function for the field.

        Parameters
        ----------
        field : str
            Field to read.

        Returns
        -------
        np.ufunc
            Reduction function.
        """
        if field.startswith('RF') or field.startswith('RM'):
            return np.sum
        return np.mean

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
        input_file = os.path.join(input_data.tmp_dir, input_data.input_file)
        casename = get_case_name(input_file)
        output_dir = input_data.tmp_dir
        # X field
        field_filename = os.path.join(
            output_dir,
            f'{casename}_{self.set_name}_{self.x_field}.txt',
        )
        # Ensure the file exists
        if not os.path.exists(field_filename):
            return OutputResult(np.empty(0), np.empty(0))
        data = pd.read_csv(field_filename, sep=' ', index_col=False)
        columns = [a for a in data.columns if self.x_field in a]
        x_reduction = self.get_reduction(self.x_field)
        # Extract columns from data frame
        data_group = data[columns].to_numpy()
        x_field = x_reduction(data_group, axis=1)

        # Y field
        field_filename = os.path.join(
            output_dir,
            f'{casename}_{self.set_name}_{self.field}.txt',
        )
        # Ensure the file exists
        if not os.path.exists(field_filename):
            return OutputResult(np.empty(0), np.empty(0))
        data = pd.read_csv(field_filename, sep=' ', index_col=False)
        columns = [a for a in data.columns if self.field in a]
        y_reduction = self.get_reduction(self.field)
        # Extract columns from data frame
        data_group = data[columns].to_numpy()
        y_field = y_reduction(data_group, axis=1)

        return OutputResult(x_field, y_field)

    @classmethod
    def read(cls, config: Dict[str, Any]) -> FieldsOutput:
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
        return cls(field, x_field, set_name)


class ODBFullField(OutputField):
    """Full-field ODB outputs reader."""

    def __init__(self, field: list[str], set_name: str = 'All', ndim: int = 3):
        """Constructor for ODB outputs reader

        Parameters
        ----------
        fields : list[str]
            List of fields to read.
        set_name : str
            Node set to read.
        ndim : int
            Number of dimensions.
        """
        super().__init__()
        self.field = field
        self.set_name = set_name
        self.ndim = ndim
        if self.ndim == 2:
            self.x_field = ['COORD1', 'COORD2']
        elif self.ndim == 3:
            self.x_field = ['COORD1', 'COORD2', 'COORD3']
        else:
            raise ValueError("ndim must be 2 or 3.")
        

    def check(self, input_data: InputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : AbaqusInputData
            Input data for this case.
        """
        has_space = ' ' in self.set_name
        input_file = os.path.join(input_data.tmp_dir, input_data.input_file)
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

    def get(self, input_data: InputData) -> FullFieldOutputResult:
        """Reads full-field data from Links.

        Parameters
        ----------
        input_data : InputData
            Input data for this case.

        Returns
        -------
        FullFieldOutputResult
            Full-field output result.
        """
        input_file = os.path.join(input_data.tmp_dir, input_data.input_file)
        casename = get_case_name(input_file)
        output_dir = input_data.tmp_dir
        coordinates = np.empty((0, self.ndim + 1))
        # X field
        x_data = []
        for x_field in self.x_field:
            field_filename = os.path.join(
                output_dir,
                f'{casename}_{self.set_name}_{x_field}.txt',
            )
            # Ensure the file exists
            if not os.path.exists(field_filename):
                return FullFieldOutputResult(coordinates, np.empty(0))
            data = pd.read_csv(field_filename, sep=' ', index_col=False)
            columns = [a for a in data.columns if x_field in a]
            data_group = data[columns].to_numpy()
            x_data.append(data_group.reshape(-1))
        x_data = np.array(x_data).T
        # Y field
        field_filename = os.path.join(
            output_dir,
            f'{casename}_{self.set_name}_{self.field}.txt',
        )
        # Ensure the file exists
        if not os.path.exists(field_filename):
            return FullFieldOutputResult(coordinates, np.empty(0))
        data = pd.read_csv(field_filename, sep=' ', index_col=False)
        columns = [a for a in data.columns if self.field in a]
        # Extract columns from data frame
        data_group = data[columns].to_numpy()
        # Only one field is supported for now
        y_data = data_group.reshape(-1, 1)
        # Create coordinates array
        time_scale = data[[a for a in data.columns if 'Frame' in a]].to_numpy()
        for time_id, timestep in enumerate(time_scale):
            n_points = x_data.shape[0] // time_scale.shape[0]
            coords = x_data[time_id*n_points:(1 + time_id)*n_points,:]
            coordinates = np.vstack((
                coordinates,
                np.hstack((np.ones((coords.shape[0], 1)) * timestep, coords)),
            ))
        return FullFieldOutputResult(coordinates, y_data)
    
    @classmethod
    def read(cls, config: Dict[str, Any]) -> ODBFullField:
        """Read the output field from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        ODBFullField
            Output field to use for this problem.
        """
        # Read the field
        if 'field' not in config:
            raise ValueError("Missing 'field' in reaction configuration.")
        field = config['field']
        ndim = int(config.get('ndim', 3))
        set_name = str(config.get('set_name', 'All'))
        return cls(field, set_name, ndim=ndim)