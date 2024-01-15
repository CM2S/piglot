"""Module for output fields from Links solver."""
from __future__ import annotations
from typing import Dict, Any, Union
import os
import numpy as np
import pandas as pd
from piglot.parameter import ParameterSet
from piglot.solver.solver import InputData, OutputField, OutputResult
from piglot.utils.solver_utils import get_case_name, has_keyword, find_keyword, write_parameters
from piglot.utils.solver_utils import has_parameter


class LinksInputData(InputData):
    """Container for Links input data."""

    def __init__(self, input_file: str) -> None:
        super().__init__()
        self.input_file = input_file

    def prepare(
            self,
            values: np.ndarray,
            parameters: ParameterSet,
            tmp_dir: str = None,
            ) -> LinksInputData:
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
        LinksInputData
            Input data prepared for the simulation.
        """
        # Copy file and write out the parameters
        dest_file = os.path.join(tmp_dir, os.path.basename(self.input_file))
        write_parameters(parameters.to_dict(values), self.input_file, dest_file)
        return LinksInputData(dest_file)

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

    def get_current(self, target_dir: str) -> LinksInputData:
        """Get the current input data.

        Parameters
        ----------
        target_dir : str
            Target directory to copy the input file.

        Returns
        -------
        LinksInputData
            Current input data.
        """
        return LinksInputData(os.path.join(target_dir, os.path.basename(self.input_file)))


class Reaction(OutputField):
    """Reaction outputs reader."""

    def __init__(self, field: str, group: int = 1):
        """Constructor for reaction reader

        Parameters
        ----------
        field : str
            Direction to read.
        group : int, optional
            Node group to read, by default 1.
        """
        super().__init__()
        field_dict = {'x': 1, 'y': 2, 'z': 3}
        self.field = field_dict[field]
        self.field_name = field
        self.group = group

    def check(self, input_data: LinksInputData) -> None:
        """Sanity checks on the input file.

        This checks if:
        - we are reading the reactions of a macroscopic analysis;
        - the file has NODE_GROUPS outputs.

        Parameters
        ----------
        input_data : LinksInputData
            Input data for this case.

        Raises
        ------
        RuntimeError
            If not reading a macroscopic analysis file.
        RuntimeError
            If reaction output is not requested in the input file.
        """
        # Is macroscopic file?
        input_file = input_data.input_file
        if not input_file.endswith('.dat'):
            raise RuntimeError("Reactions only available for macroscopic simulations.")
        # Has node groups keyword?
        if not has_keyword(input_file, "NODE_GROUPS"):
            raise RuntimeError("Reactions requested on an input file without NODE_GROUPS.")
        # TODO: check group number and dimensions

    def get(self, input_data: LinksInputData) -> OutputResult:
        """Reads reactions from a Links analysis.

        Parameters
        ----------
        input_data : LinksInputData
            Input data for this case.

        Returns
        -------
        array
            2D array with load factor in the first column and reactions in the second.
        """
        input_file = input_data.input_file
        casename = get_case_name(input_file)
        output_dir, _ = os.path.splitext(input_file)
        reac_filename = os.path.join(output_dir, f'{casename}.reac')
        # Ensure the file exists
        if not os.path.exists(reac_filename):
            return OutputResult(np.empty(0), np.empty(0))
        data = np.genfromtxt(reac_filename)
        # Filter-out the requested group
        data_group = data[data[:, 0] == self.group, 1:]
        return OutputResult(data_group[:, 0], data_group[:, self.field])

    @staticmethod
    def read(config: Dict[str, Any]) -> Reaction:
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
        if 'field' not in config:
            raise ValueError("Missing 'field' in reaction configuration.")
        field = config['field']
        # Read the group (if passed)
        group = int(config.get('group', 1))
        return Reaction(field, group)


class OutFile(OutputField):
    """Links .out file reader."""

    def __init__(
            self,
            field: Union[str, int],
            i_elem: int = None,
            i_gauss: int = None,
            x_field: str = "LoadFactor",
            ):
        """Constructor for .out file reader

        Parameters
        ----------
        field : Union[str, int]
            Field to read from the output file. Can be a single column index or name.
        i_elem : integer, optional
            For GP outputs, from which element to read, by default None.
        i_gauss : integer, optional
            From GP outputs, from which GP to read, by default None.
        x_field : str, optional
            Field to use as index in the resulting DataFrame, by default "LoadFactor".

        Raises
        ------
        RuntimeError
            If element and GP numbers are not consistent.
        """
        super().__init__()
        self.field = field
        # Homogenised .out or GP output?
        if i_elem is None and i_gauss is None:
            # Homogenised: no suffix
            self.suffix = ''
        else:
            # GP output: needs both element and GP numbers
            if (i_elem is None) or (i_gauss is None):
                raise RuntimeError("Need to pass both element and Gauss point numbers.")
            self.suffix = f'_ELEM_{i_elem}_GP_{i_gauss}'
            self.i_elem = i_elem
            self.i_gauss = i_gauss
        self.x_field = x_field
        self.separator = 16

    def check(self, input_data: LinksInputData) -> None:
        """Sanity checks on the input file.

        This checks for:
        - homogenised outputs only from microscopic simulations;
        - for GP outputs, ensure the input file lists the element and GP number requested;
        - whether the .out file is in single or double precision.

        Parameters
        ----------
        input_data : LinksInputData
            Input data for this case.

        Raises
        ------
        RuntimeError
            If requesting homogenised outputs from macroscopic analyses.
        RuntimeError
            If GP outputs have not been specified in the input file.
        RuntimeError
            If the requested element and GP has not been specified in the input file.
        """
        # Check if appropriate scale
        input_file = input_data.input_file
        extension = os.path.splitext(input_file)[1]
        if extension == ".dat" and self.suffix == '':
            raise RuntimeError("Cannot extract homogenised .out from macro simulations.")
        # For GP outputs, check if the output has been requsted
        if self.suffix != '':
            if not has_keyword(input_file, "GAUSS_POINTS_OUTPUT"):
                raise RuntimeError("GP outputs have not been specified in the input file.")
            # Check number of GP outputs and if ours is a valid one
            with open(input_file, 'r', encoding='utf8') as file:
                line = find_keyword(file, "GAUSS_POINTS_OUTPUT")
                n_points = int(line.split()[1])
                found = False
                for _ in range(0, n_points):
                    line_split = file.readline().split()
                    i_elem = int(line_split[0])
                    i_gaus = int(line_split[1])
                    if self.i_elem == i_elem and self.i_gauss == i_gaus:
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"The specified GP output {self.i_elem} "
                                       f"{self.i_gauss} was not found.")
        # Check if single or double precision output
        self.separator = 24 if has_keyword(input_file, "DOUBLE_PRECISION_OUTPUT") else 16

    def get(self, input_data: LinksInputData) -> OutputResult:
        """Get a parameter from the .out file.

        Parameters
        ----------
        input_data : LinksInputData
            Input data for this case.

        Returns
        -------
        DataFrame
            DataFrame with the requested fields.
        """
        input_file = input_data.input_file
        casename = get_case_name(input_file)
        output_dir = os.path.splitext(input_file)[0]
        filename = os.path.join(output_dir, f'{casename}{self.suffix}.out')
        from_gp = self.suffix != ''
        # Ensure the file exists
        if not os.path.exists(filename):
            return OutputResult(np.empty(0), np.empty(0))
        # Read the first line of the file to find the total number of columns
        with open(filename, 'r', encoding='utf8') as file:
            # Consume the first line if from a GP output (contains GP coordinates)
            if from_gp:
                file.readline()
            line_len = len(file.readline())
        n_columns = int(line_len / self.separator)
        # Fixed-width read (with a skip on the first line for GP outputs)
        header = 1 if from_gp else 0
        df = pd.read_fwf(filename, widths=n_columns*[self.separator], header=header)
        # Extract indices for named columns
        columns = df.columns.tolist()
        x_column = columns.index(self.x_field) if isinstance(self.x_field, str) else self.x_field
        y_column = columns.index(self.field) if isinstance(self.field, str) else self.field
        # Return the given quantity as the x-variable
        return OutputResult(df.iloc[:, x_column].to_numpy(), df.iloc[:, y_column].to_numpy())

    @staticmethod
    def read(config: Dict[str, Any]) -> OutFile:
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
        if 'field' not in config:
            raise ValueError("Missing 'field' in OutFile configuration.")
        field = config['field']
        # Read the element and GP numbers (if passed)
        i_elem = int(config['i_elem']) if 'i_elem' in config else None
        i_gauss = int(config['i_gauss']) if 'i_gauss' in config else None
        # Read the x field (if passed)
        x_field = config.get('x_field', 'LoadFactor')
        return OutFile(field, i_elem, i_gauss, x_field)


def links_fields_reader(config: Dict[str, Any]) -> OutputField:
    """Read the output fields for the Links solver.

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
        'Reaction': Reaction,
        'OutFile': OutFile,
    }
    if field_name not in readers:
        raise ValueError(f"Unknown output field name '{field_name}'.")
    return readers[field_name].read(config)
