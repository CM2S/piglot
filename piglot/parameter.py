"""Optimisation parameter module."""
from typing import Iterator, Dict, Any, List, Callable
import os
from hashlib import sha256
import numpy as np
import pandas as pd
import sympy
from piglot.utils.yaml_parser import parse_config_file


class Parameter:
    """Base parameter class."""

    def __init__(self, name: str, inital_value: float, lbound: float, ubound: float):
        """Constructor for the parameter class.

        Parameters
        ----------
        name : str
            Parameter name.
        inital_value : float
            Initial value for the parameter.
        lbound : float
            Lower bound for the parameter.
        ubound : float
            Upper bound for the parameter.
        """
        self.name = name
        self.inital_value = inital_value
        self.lbound = lbound
        self.ubound = ubound
        if inital_value > ubound or inital_value < lbound:
            raise RuntimeError("Initial shot outside of bounds")

    def clip(self, value: float) -> float:
        """Clamp a value to the [lbound,ubound] interval.

        Parameters
        ----------
        value : float
            Value to clip.

        Returns
        -------
        float
            Clamped value.
        """
        return min(max(self.lbound, value), self.ubound)


class OutputParameter:
    """Base class for output parameters."""

    def __init__(self, name: str, mapping: Callable[[Dict[str, float]], float]):
        """Constructor for an output parameter

        Parameters
        ----------
        name : str
            Output parameter name
        mapping : Callable[..., float]
            Function to map from internal to output parameters.
            The internal parameters are passed in an expanded dict.
        """
        self.name = name
        self.mapping = mapping


class ParameterSet:
    """Container class for a set of parameters.

    This type should be used if the internal optimised parameters and the output parameters
    are equal. By default, the internal names are used for output name resolution.
    """

    def __init__(self):
        """Constructor for a parameter set."""
        self.parameters: List[Parameter] = []

    def __iter__(self) -> Iterator[Parameter]:
        """Iterator for a parameter set.

        Returns
        -------
        Iterator[Parameter]
            Iterator for a parameter set."""
        return iter(self.parameters)

    def __len__(self) -> int:
        """Length of the parameter set.

        Returns
        -------
        int
            Length of the parameter set."""
        return len(self.parameters)

    def __getitem__(self, key: int) -> Parameter:
        """Get a parameter by index.

        Parameters
        ----------
        key : int
            Index of the parameter.

        Returns
        -------
        Parameter
            Parameter with the given index."""
        return self.parameters[key]

    def add(self, name: str, inital_value: float, lbound: float, ubound: float) -> None:
        """Add a parameter to this set.

        Parameters
        ----------
        name : str
            Parameter name.
        inital_value : float
            Initial value for the parameter.
        lbound : float
            Lower bound of the parameter.
        ubound : float
            Upper bound of the parameter.

        Raises
        ------
        RuntimeError
            If a repeated parameter is given.
        """
        # Sanity checks
        if name in [p.name for p in self.parameters]:
            raise RuntimeError(f"Repeated parameter {name} in set!")
        self.parameters.append(Parameter(name, inital_value, lbound, ubound))

    def clip(self, values: np.ndarray) -> np.ndarray:
        """Clamp the parameter set to the [lbound,ubound] interval.

        Parameters
        ----------
        values : np.ndarray
            Values to clip. Their order is used for parameter resolution.

        Returns
        -------
        np.ndarray
            Clamped parameters.
        """
        return np.array([p.clip(values[i]) for i, p in enumerate(self.parameters)])

    def to_dict(self, values: np.ndarray) -> Dict[str, float]:
        """Build a dict with name-value pairs given a list of values.

        Parameters
        ----------
        values : np.ndarray
            Values to pack. Their order is used for parameter resolution.

        Returns
        -------
        Dict[str, float]
            Name-value pair for each parameter.
        """
        return dict(zip([p.name for p in self.parameters], values))

    @staticmethod
    def hash(values) -> str:
        """Build the hash for the current parameter values.

        Parameters
        ----------
        values : array
            Parameters to hash.

        Returns
        -------
        str
            Hex digest of the hash.
        """
        hasher = sha256()
        values = np.array(values)
        for value in values:
            hasher.update(value)
        return hasher.hexdigest()


class DualParameterSet(ParameterSet):
    """Container class for a set of parameters with distinct internal and output parameters.

    This type should be used if the internal optimised parameters and the output parameters
    are not equal. Output names are used for output name resolution. All output fields
    must have a mapping function to internal parameters.
    """

    def __init__(self):
        """Constructor for a dual parameter set."""
        super().__init__()
        self.output_parameters = []

    def add_output(self, name: str, mapping: Callable[[Dict[str, float]], float]) -> None:
        """Adds an output parameter to the set.

        Parameters
        ----------
        name : str
            Parameter name
        mapping : Callable[[Dict[str, float]], float]
            Mapping function from internal parameters to this output parameter's value.
            Internal parameters are passed as arguments for this function as an expanded
            dict.

        Raises
        ------
        RuntimeError
            If an output parameter is repeated.
        """
        # Sanity checks
        if name in [p.name for p in self.output_parameters]:
            raise RuntimeError(f"Repeated output parameter {name} in set!")
        self.output_parameters.append(OutputParameter(name, mapping))

    def clone_output(self, name: str) -> None:
        """Creates an output parameter identical to a given internal parameter.

        Parameters
        ----------
        name : str
            Name of the internal parameter to clone.
        """
        self.output_parameters.append(OutputParameter(name, lambda **vals_dict: vals_dict[name]))

    def to_output(self, values: np.ndarray) -> np.ndarray:
        """Compute the output parameters' values given an array of internal inputs.

        Parameters
        ----------
        values : np.ndarray
            Values to pack. Their order is used for parameter resolution.

        Returns
        -------
        np.ndarray
            Values of the output parameters.
        """
        vals_dict = super().to_dict(values)
        return np.array([p.mapping(**vals_dict) for p in self.output_parameters])

    def to_dict(self, values: np.ndarray) -> Dict[str, float]:
        """Build a dict with name-value pairs for output parameters given a list of values.

        Parameters
        ----------
        values : np.ndarray
            Values to pack. Their order is used for parameter resolution.

        Returns
        -------
        Dict[str, float]
            Name-value pair for each output parameter.
        """
        return dict(zip([p.name for p in self.output_parameters], self.to_output(values)))


def read_parameters(config: Dict[str, Any]) -> ParameterSet:
    """Parse the parameters from the configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    ParameterSet
        Parameter set for this problem.
    """
    # Read the parameters
    if 'parameters' not in config:
        raise ValueError("Missing parameters from configuration file.")
    parameters = DualParameterSet() if 'output_parameters' in config else ParameterSet()
    for name, spec in config['parameters'].items():
        int_spec = [float(s) for s in spec]
        parameters.add(name, *int_spec)
    if "output_parameters" in config:
        symbs = sympy.symbols(list(config['parameters'].keys()))
        for name, spec in config["output_parameters"].items():
            parameters.add_output(name, sympy.lambdify(symbs, spec))
    # Fetch initial shot from another run
    if 'init_shot_from' in config:
        source = parse_config_file(config['init_shot_from'])
        func_calls_file = os.path.join(source['output'], 'func_calls')
        df = pd.read_table(func_calls_file)
        df.columns = df.columns.str.strip()
        min_series = df.iloc[df['Objective'].idxmin()]
        for param in parameters:
            if param.name in min_series.index:
                param.inital_value = min_series[param.name]
    return parameters
