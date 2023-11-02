"""Utilities for the solver module."""
import os
import re
from typing import Dict
from piglot.parameter import ParameterSet


def write_parameters(param_value: Dict[str, float], source: str, dest: str) -> None:
    """Write the set of parameters to the input file.

    Parameters
    ----------
    param_value : Dict[str, float]
        Collection of parameters and their values.
    source : str
        Source input file, to be copied to the destination.
    dest : str
        Destination input file.
    """
    with open(source, 'r', encoding='utf8') as fin:
        with open(dest, 'w', encoding='utf8') as fout:
            for line in fin:
                out = line
                for parameter, value in param_value.items():
                    # Replace full expression
                    regex = r'\<' + parameter + '\(.*?\>'
                    out = re.sub(regex, str(value), out)
                    # Replace short expression
                    regex = r'\<' + parameter + '\>'
                    out = re.sub(regex, str(value), out)
                fout.write(out)


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
