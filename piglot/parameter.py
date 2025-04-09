"""Optimisation parameter module."""
from typing import Iterator, Dict, Any, List, Callable, Tuple
import os
from hashlib import sha256
import random
from itertools import product, chain
import numpy as np
import pandas as pd
import sympy
from scipy.stats import qmc
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


class DiscreteParameter:
    """Base class for discrete parameters."""

    def __init__(self, name: str, initial_value: float, values: List[float]):
        """Constructor for a discrete parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        initial_value : float
            Initial value for the parameter.
        values : List[float]
            Possible values for the parameter.
        """
        if initial_value not in values:
            raise RuntimeError("Initial shot not in discrete values")
        self.name = name
        self.initial_value = initial_value
        self.values = values


class ParameterSet:
    """Container class for a set of parameters.

    This type should be used if the internal optimised parameters and the output parameters
    are equal. By default, the internal names are used for output name resolution.
    """

    def __init__(self):
        """Constructor for a parameter set."""
        self.parameters: List[Parameter] = []
        self.discrete_parameters: List[DiscreteParameter] = []

    def __iter__(self) -> Iterator[Parameter]:
        """Iterator for a parameter set.

        Returns
        -------
        Iterator[Parameter]
            Iterator for a parameter set."""
        return chain(iter(self.parameters), iter(self.discrete_parameters))

    def __len__(self) -> int:
        """Length of the parameter set.

        Returns
        -------
        int
            Length of the parameter set."""
        return len(self.parameters) + len(self.discrete_parameters)

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
        return (
            self.parameters[key] if key < len(self.parameters)
            else self.discrete_parameters[key - len(self.parameters)]
        )

    def num_discrete(self) -> int:
        """Get the number of discrete parameters.

        Returns
        -------
        int
            Number of discrete parameters.
        """
        return len(self.discrete_parameters)

    def num_continuous(self) -> int:
        """Get the number of continuous parameters.

        Returns
        -------
        int
            Number of continuous parameters.
        """
        return len(self.parameters)

    def names(self) -> List[str]:
        """Get the names of the parameters.

        Returns
        -------
        List[str]
            List of parameter names.
        """
        return [p.name for p in self.parameters] + [p.name for p in self.discrete_parameters]

    def initial_values(self) -> np.ndarray:
        """Get the initial values of the parameters.

        Returns
        -------
        np.ndarray
            List of initial values.
        """
        return np.array(
            [p.inital_value for p in self.parameters] +
            [p.initial_value for p in self.discrete_parameters]
        )

    def bounds(self) -> np.ndarray:
        """Get the bounds of the parameters.

        Returns
        -------
        np.ndarray
            List of bounds for each parameter.
        """
        return np.array(
            [(p.lbound, p.ubound) for p in self.parameters] + 
            [(min(p.values), max(p.values)) for p in self.discrete_parameters]
        )

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
        if name in self.names():
            raise RuntimeError(f"Repeated parameter {name} in set!")
        self.parameters.append(Parameter(name, inital_value, lbound, ubound))

    def add_discrete(self, name: str, initial_value: float, values: List[float]) -> None:
        """Add a discrete parameter to this set.

        Parameters
        ----------
        name : str
            Parameter name.
        initial_value : float
            Initial value for the parameter.
        values : List[float]
            Possible values for the parameter.

        Raises
        ------
        RuntimeError
            If a repeated parameter is given.
        """
        # Sanity checks
        if name in self.names():
            raise RuntimeError(f"Repeated parameter {name} in set!")
        self.discrete_parameters.append(DiscreteParameter(name, initial_value, values))

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
        # Sanitise discrete parameters
        disc_values = values[self.num_continuous():]
        for par, value in zip(self.discrete_parameters, disc_values):
            if value not in par.values:
                raise RuntimeError(f"Discrete parameter {par.name} not in values!")
        return dict(zip(self.names(), values))

    def discrete_combinations(self) -> List[List[float]]:
        """Get all combinations of discrete parameters.

        Returns
        -------
        List[List[float]]
            All combinations of discrete parameters.
        """
        return list(product(*[p.values for p in self.discrete_parameters]))

    def get_random(self, n_points: int, seed: int = None) -> np.ndarray:
        """Get a random set of parameters.

        Parameters
        ----------
        n_points : int
            Number of points to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Random set of parameters.
        """
        # Continuous parameters
        bounds = np.array(self.bounds()[:self.num_continuous()])
        samples = qmc.Sobol(self.num_continuous(), seed=seed).random(n_points)
        continuous = samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        if self.num_discrete() == 0:
            return continuous
        # Discrete parameters
        discrete = random.choices(self.discrete_combinations(), k=n_points)
        return np.hstack((continuous, np.array(discrete)))

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
    if 'parameters' not in config and 'discrete_parameters' not in config:
        raise ValueError("Missing parameters from configuration file.")
    parameters = DualParameterSet() if 'output_parameters' in config else ParameterSet()
    if 'parameters' in config:
        for name, spec in config['parameters'].items():
            int_spec = [float(s) for s in spec]
            parameters.add(name, *int_spec)
    if 'discrete_parameters' in config:
        for name, spec in config['discrete_parameters'].items():
            if 'initial' not in spec:
                raise ValueError(f"Missing initial value for discrete parameter '{name}'.")
            if 'values' not in spec:
                raise ValueError(f"Missing values for discrete parameter '{name}'.")
            initial = float(spec['initial'])
            values = [float(s) for s in spec['values']]
            parameters.add_discrete(name, initial, values)
    if "output_parameters" in config:
        symbs = sympy.symbols(parameters.names())
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
