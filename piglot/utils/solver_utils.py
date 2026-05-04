"""Utilities for the solver module."""
from dataclasses import dataclass
from io import TextIOWrapper
import os
import sys
import shutil
from typing import Optional, Union


def get_case_name(input_file: str) -> str:
    """Extracts the name of a given case.

    Parameters
    ----------
    input_file : str
        Path for the input file.

    Returns
    -------
    str
        Name of the case.
    """
    filename = os.path.basename(input_file)
    return os.path.splitext(filename)[0]


def has_keyword(input_file: str, keyword: str) -> bool:
    """Checks whether an input file contains a given keyword.

    Parameters
    ----------
    input_file : str
        Path for the input file.
    keyword : str
        Keyword to locate.

    Returns
    -------
    bool
        Whether the input file contains the keyword or not.
    """
    with open(input_file, 'r', encoding='utf8') as file:
        for line in file:
            if line.lstrip().startswith(keyword):
                return True
    return False


def has_parameter(input_file: str, parameter: str) -> bool:
    """Checks whether an input file contains a given parameter.

    Parameters
    ----------
    input_file : str
        Path for the input file.
    parameter : str
        parameter to locate.

    Returns
    -------
    bool
        Whether the input file contains the parameter or not.
    """
    with open(input_file, 'r', encoding='utf8') as file:
        for line in file:
            if parameter in line.lstrip():
                return True
    return False


def find_keyword(file: str, keyword: str) -> str:
    """Finds the first line where a keyword is defined.

    Parameters
    ----------
    file : str
        Path for the input file.
    keyword : str
        Keyword to locate.

    Returns
    -------
    str
        Line containing the keyword.

    Raises
    ------
    RuntimeError
        If the keyword is not found.
    """
    for line in file:
        if line.lstrip().startswith(keyword):
            return line
    raise RuntimeError(f"Keyword {keyword} not found!")


@dataclass
class OutputStream:
    """Class to manage output streams."""
    stdout: TextIOWrapper
    stderr: TextIOWrapper

    def print(
        self, *values, sep: Optional[str] = " ", end: Optional[str] = "\n", flush: bool = False
    ) -> None:
        """Wrapper for the print function in the output stream context.
        
        Parameters
        ----------
        *values : Any
            Values to print.
        sep : Optional[str], default=" "
            Separator between values.
        end : Optional[str], default="\n"
            End character.
        flush : bool, default=False
            Whether to flush the output.

        """
        print(*values, sep=sep, end=end, file=self.stdout, flush=flush)

    def print_error(
        self, *values, sep: Optional[str] = " ", end: Optional[str] = "\n", flush: bool = False
    ) -> None:
        """Wrapper for the print function in the error stream context.

        Parameters
        ----------
        *values : Any
            Values to print.
        sep : Optional[str], default=" "
            Separator between values.
        end : Optional[str], default="\n"
            End character.
        flush : bool, default=False
            Whether to flush the output.

        """
        print(*values, sep=sep, end=end, file=self.stderr, flush=flush)


class VerbosityManager:
    """Class to manage output streams based on verbosity levels."""

    DEFAULT_VERBOSITY = 'none'
    AVAILABLE_VERBOSITIES = [
        'none',
        'file',
        'error',
        'all',
    ]

    def __init__(self, verbosity: Union[str, None], output_dir: str) -> None:
        # Sanitise verbosity
        if verbosity is None:
            verbosity = self.DEFAULT_VERBOSITY
        if verbosity not in self.AVAILABLE_VERBOSITIES:
            raise ValueError(f"Invalid verbosity level: {verbosity}")
        self.verbosity = verbosity
        self.output_dir = output_dir
        self.stdout = None
        self.stderr = None
        self.__devnull = None

    def prepare(self) -> None:
        """Set up the verbosity level for the solver."""
        # Prepare output streams
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        self.__devnull = open(os.devnull, 'w', encoding='utf8')
        if self.verbosity in ('file', 'error'):
            os.mkdir(self.output_dir)
            self.stdout = open(os.path.join(self.output_dir, 'stdout'), 'w', encoding='utf8')
            self.stderr = (
                open(os.path.join(self.output_dir, 'stderr'), 'w', encoding='utf8')
                if self.verbosity == 'file' else sys.__stderr__
            )
        elif self.verbosity == 'all':
            self.stdout = sys.__stdout__
            self.stderr = sys.__stderr__
        elif self.verbosity == 'none':
            self.stdout = self.__devnull
            self.stderr = self.__devnull

    def flush(self) -> None:
        """Flush the solver outputs, if needed."""
        if self.verbosity in ('file', 'error'):
            self.stdout.flush()
            if self.stderr is not None:
                self.stderr.flush()

    def __enter__(self) -> OutputStream:
        return OutputStream(self.stdout, self.stderr)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()
