"""Module for output fields from Abaqus solver."""
from __future__ import annotations
from typing import Dict, Any
import os
import re
import glob
import numpy as np
import pandas as pd
from piglot.parameter import ParameterSet
from piglot.solver.solver import InputData, OutputField, OutputResult
from piglot.utils.solver_utils import write_parameters, has_parameter, get_case_name

class AbaqusInputData(InputData):
    """Container for Abaqus input data."""

    def __init__(self, input_file: str, job_name: str, step_name: str, instance_name: str) -> None:
        super().__init__()
        self.input_file = input_file
        self.job_name = job_name
        self.step_name = step_name
        self.instance_name = instance_name

    def prepare(
            self,
            values: np.ndarray,
            parameters: ParameterSet,
            tmp_dir: str=None,
        ) -> AbaqusInputData:
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
        AbaqusInputData
            Input data prepared for the simulation.
        """
        # Copy file and write out the parameters
        dest_file = os.path.join(tmp_dir, os.path.basename(self.input_file))
        write_parameters(parameters.to_dict(values), self.input_file, dest_file)
        return AbaqusInputData(dest_file, self.job_name, self.step_name, self.instance_name)

    def check(self, parameters: ParameterSet) -> None:
        """Check if the input data is valid according to the given parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        """
        for parameter in parameters:
            if not has_parameter(self.input_file, f'<{parameter.name}>'):
                raise RuntimeError(f"Parameter '{parameter.name}' not found in input file.")

        input_file, ext = os.path.splitext(os.path.basename(self.input_file))
        with open(input_file + ext, 'r', encoding='utf-8') as file:
            data = file.read()

            job_list = re.findall(r'\*\* Job name: ([^M]+)', data)
            job_list = [job.strip() for job in job_list]  # Remove trailing whitespace
            if len(job_list) == 0:
                raise ValueError("No steps found in the file.")
            if self.job_name is not None:
                if self.job_name not in job_list:
                    raise ValueError(f"Job name '{self.job_name}' not found in the file.")
            if self.job_name is None:
                if len(job_list) > 1:
                    raise ValueError("Multiple jobs found in the file. \
                                     Please specify the job name.")
                self.job_name = job_list[0]

            instance_list = re.findall(r'\*Instance, name=([^,]+)', data)
            if len(instance_list) == 0:
                raise ValueError("No instances found in the file.")
            if self.instance_name is not None:
                if self.instance_name not in instance_list:
                    raise ValueError(f"Instance name '{self.instance_name}' not found in the file.")
            if self.instance_name is None:
                if len(instance_list) > 1:
                    raise ValueError("Multiple instances found in the file. \
                                     Please specify the instance name.")
                self.instance_name = instance_list[0]

            step_list = re.findall(r'\*Step, name=([^,]+)', data)
            if len(step_list) == 0:
                raise ValueError("No steps found in the file.")
            if self.step_name is not None:
                if self.step_name not in step_list:
                    raise ValueError(f"Step name '{self.step_name}' not found in the file.")
            if self.step_name is None:
                if len(step_list) > 1:
                    raise ValueError("Multiple steps found in the file. \
                                     Please specify the step name.")
                self.step_name = step_list[0]


    def name(self) -> str:
        """Return the name of the input data.

        Returns
        -------
        str
            Name of the input data.
        """
        return os.path.basename(self.input_file)

    def get_current(self, target_dir: str) -> AbaqusInputData:
        """Get the current input data.

        Parameters
        ----------
        target_dir : str
            Target directory to copy the input file.

        Returns
        -------
        AbaqusInputData
            Current input data.
        """
        return AbaqusInputData(os.path.join(target_dir, os.path.basename(self.input_file)),
                               self.job_name, self.step_name, self.instance_name)


class FieldsOutput(OutputField):
    """Fields outputs reader."""

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

    def check(self, input_data: AbaqusInputData) -> None:
        """Sanity checks on the input file.

        Parameters
        ----------
        input_data : AbaqusInputData
            Input data for this case.
        """
        input_file, ext = os.path.splitext(os.path.basename(input_data.input_file))
        with open(input_file + ext, 'r', encoding='utf-8') as file:
            data = file.read()

            nsets_list = re.findall(r'\*Nset, nset=([^,]+)', data)
            if len(nsets_list) == 0:
                raise ValueError("No sets found in the file.")
            if self.set_name not in nsets_list:
                raise ValueError(f"Set name '{self.set_name}' not found in the file.")

    def get(self, input_data: AbaqusInputData) -> OutputResult:
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
        input_file = get_case_name(input_data.input_file) # sample.inp
        output_dir = os.path.dirname(input_data.input_file)
        reduction = {
            'RF': np.sum,
            'U': np.mean,
            'S': np.mean,
            'LE': np.mean,
            'E': np.mean,
        }

        # X field
        field_filename = os.path.join(output_dir,
                                      f'{input_file}_{self.set_name}_{self.x_field}.txt')
        # Ensure the file exists
        if not os.path.exists(field_filename):
            return OutputResult(np.empty(0), np.empty(0))
        data = pd.read_csv(field_filename, sep=' ', index_col=False)
        columns = [a for a in data.columns if f"{self.x_field}{self.direction}" in a]
        # Extract columns from data frame
        data_group = data[columns].to_numpy()
        x_field = reduction[self.x_field](data_group, axis=1)

        # Y field
        field_filename = os.path.join(output_dir, f'{input_file}_{self.set_name}_{self.field}.txt')
        # Ensure the file exists
        if not os.path.exists(field_filename):
            return OutputResult(np.empty(0), np.empty(0))
        data = pd.read_csv(field_filename, sep=' ', index_col=False)
        columns = [a for a in data.columns if f"{self.field}{self.direction}" in a]
        # Extract columns from data frame
        data_group = data[columns].to_numpy()
        y_field = reduction[self.field](data_group, axis=1)

        # Delete the extra temporary files
        files = glob.glob(output_dir + '/' + input_file + '*.txt')
        for file in files:
            if self.set_name not in file:
                os.remove(file)

        return OutputResult(x_field, y_field)

    @staticmethod
    def read(config: Dict[str, Any]) -> FieldsOutput:
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
        if not 'field' in config:
            raise ValueError("Missing 'field' in reaction configuration.")
        field = config['field']
        x_field = config['x_field']
        # Read the set_name
        set_name = config['set_name']
        # Read thhe direction
        direction = config['direction']

        return FieldsOutput(field, x_field, set_name, direction)

def abaqus_fields_reader(config: Dict[str, Any]) -> OutputField:
    """Read the output fields for the Abaqus solver.

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
    if not 'name' in config:
        raise ValueError("Missing 'name' in output field configuration.")
    field_name = config['name']
    # Delegate to the appropriate reader
    readers = {
        'FieldsOutput': FieldsOutput,
    }
    if not field_name in readers:
        raise ValueError(f"Unknown output field name '{field_name}'.")
    return readers[field_name].read(config)
