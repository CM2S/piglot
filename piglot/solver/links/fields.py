"""Module for output fields from Links solver."""
from __future__ import annotations
from typing import Dict, Any, Union
import os
import numpy as np
import pandas as pd
from piglot.solver.solver import Case, OutputField
from piglot.utils.solver_utils import get_case_name, has_keyword, find_keyword


class Reaction(OutputField):
    """Reaction outputs reader."""

    def __init__(self, field: str, group: int=1):
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

    def check(self, case: Case) -> None:
        """Sanity checks on the input file.

        This checks if:
        - we are reading the reactions of a macroscopic analysis;
        - the file has NODE_GROUPS outputs.

        Parameters
        ----------
        input_file : str
            Path to the input file.

        Raises
        ------
        RuntimeError
            If not reading a macroscopic analysis file.
        RuntimeError
            If reaction output is not requested in the input file.
        """
        # Is macroscopic file?
        if not case.filename.endswith('.dat'):
            raise RuntimeError("Reactions only available for macroscopic simulations.")
        # Has node groups keyword?
        if not has_keyword(case.filename, "NODE_GROUPS"):
            raise RuntimeError("Reactions requested on an input file without NODE_GROUPS.")
        # TODO: check group number and dimensions

    def get(self, case: Case) -> np.ndarray:
        """Reads reactions from a Links analysis.

        Parameters
        ----------
        input_file : str
            Path to the input file

        Returns
        -------
        array
            2D array with load factor in the first column and reactions in the second.
        """
        casename = get_case_name(case.filename)
        output_dir, _ = os.path.splitext(case.filename)
        reac_filename = os.path.join(output_dir, f'{casename}.reac')
        # Ensure the file exists
        if not os.path.exists(reac_filename):
            return np.empty((0, 2))
        data = np.genfromtxt(reac_filename)
        return (data[data[:,0] == self.group, 1:])[:,[0, self.field]]

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
        if not 'field' in config:
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
            i_elem: int=None,
            i_gauss: int=None,
            x_field: str="LoadFactor",
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

    def check(self, case: Case) -> None:
        """Sanity checks on the input file.

        This checks for:
        - homogenised outputs only from microscopic simulations;
        - for GP outputs, ensure the input file lists the element and GP number requested;
        - whether the .out file is in single or double precision.

        Parameters
        ----------
        case : Case
            Case to get data from.

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
        input_file = case.filename
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

    def get(self, case: Case) -> np.ndarray:
        """Get a parameter from the .out file.

        Parameters
        ----------
        case : Case
            Input case to read.

        Returns
        -------
        DataFrame
            DataFrame with the requested fields.
        """
        input_file = case.filename
        casename = get_case_name(input_file)
        output_dir = os.path.splitext(input_file)[0]
        filename = os.path.join(output_dir, f'{casename}{self.suffix}.out')
        from_gp = self.suffix != ''
        df_columns = [self.x_field] + [self.field]
        # Ensure the file exists
        if not os.path.exists(filename):
            return np.empty((0, len(df_columns)))
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
        int_columns = [columns.index(a) if isinstance(a, str) else a for a in df_columns]
        # Return the given quantity as the x-variable
        return df.iloc[:,int_columns].to_numpy()

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
        if not 'field' in config:
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
    if not 'name' in config:
        raise ValueError("Missing 'name' in output field configuration.")
    field_name = config['name']
    # Delegate to the appropriate reader
    readers = {
        'Reaction': Reaction,
        'OutFile': OutFile,
    }
    if not field_name in readers:
        raise ValueError(f"Unknown output field name '{field_name}'.")
    return readers[field_name].read(config)
