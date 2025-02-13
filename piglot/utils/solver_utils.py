"""Utilities for the solver module."""
from __future__ import annotations
import os
import re
import sys
import shutil
import importlib.util
from typing import Union
from piglot.parameter import ParameterSet


def extract_parameters(input_file: str) -> ParameterSet:
    """Extract a ParameterSet from an input file.

    Parameters
    ----------
    input_file : str
        Input file path.

    Returns
    -------
    ParameterSet
        Set of parameters discovered in the input file.

    Raises
    ------
    RuntimeError
        When a repeated pattern is found.
    RuntimeError
        When a pattern is referenced but never defined.
    """
    parameters = ParameterSet()
    full_parameters = []
    short_parameters = []
    with open(input_file, 'r', encoding='utf8') as file:
        for line in file:
            full_expression = re.findall(r"\<.*?\(.*?\>", line)
            short_expression = re.findall(r"\<\w+\>", line)
            for a in full_expression:
                full_parameters.append(a)
            for a in short_expression:
                short_parameters.append(a)

    for value in full_parameters:
        pattern = value[value.find("<")+1:value.find("(")]
        if pattern in [a.name for a in parameters]:
            raise RuntimeError(f"Repeated pattern {pattern} in file!")
        init = float(value[value.find("(")+1:value.find(",")])
        low_bound = float(value[value.find(",")+1:value.rfind(",")])
        up_bound = float(value[value.rfind(",")+1:value.rfind(")")])
        parameters.add(pattern, init, low_bound, up_bound)

    for value in short_parameters:
        pattern = value[value.find("<")+1:value.find(">")]
        if pattern not in [a.name for a in parameters]:
            raise RuntimeError(f"Pattern {pattern} referenced but not defined!")

    return parameters


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


def load_module_from_file(filename: str, attribute) -> object:
    """Loads a module from a given file.

    Parameters
    ----------
    filename : str
        Path for the file.
    attribute : str
        Attribute to load from the module.

    Returns
    -------
    object
        Module loaded from the file.
    """
    module_name = f'piglot_{os.path.basename(filename).replace(".", "_")}'
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


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

    def __enter__(self) -> VerbosityManager:
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
