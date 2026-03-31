"""Optimisation parameter module."""
import math
from typing import Iterator, Any, Union, TypeVar, Optional
from abc import ABC, abstractmethod
from hashlib import sha256
from itertools import product
import numpy as np
import torch


OptimisableT = TypeVar("OptimisableT", bound="OptimisableParameter")
ComputedT = TypeVar("ComputedT", bound="ComputedParameter")


class Parameter:
    """Base class for optimisation parameters."""

    def __init__(self, name: str, optimisable: bool, num_components: int) -> None:
        self.name = name
        self.optimisable = optimisable
        self.num_components = num_components

    def get_scalar_names(self) -> list[str]:
        """Get the names of the parameters in a scalar form.

        Returns
        -------
        list[str]
            Names of the scalar parameters.
        """
        if self.num_components == 1:
            return [self.name]
        return [f"{self.name}_{i}" for i in range(self.num_components)]

    def get_name_value_pair(self, values: np.ndarray) -> dict[str, float]:
        """Get a dictionary of parameter names and their corresponding values.

        Parameters
        ----------
        values : np.ndarray
            Array of parameter values.

        Returns
        -------
        dict[str, float]
            Dictionary of parameter names and their corresponding values.
        """
        if self.num_components == 1:
            return {self.name: float(values)}
        return {f"{self.name}_{i}": float(values[i]) for i in range(self.num_components)}

    def get_value(self, values: np.ndarray) -> Union[float, np.ndarray]:
        """Get the value of the parameter.

        Parameters
        ----------
        values : np.ndarray
            Array of parameter values.

        Returns
        -------
        Union[float, np.ndarray]
            Value of the parameter.
        """
        if self.num_components == 1:
            return float(values)
        return np.array(values)


class OptimisableParameter(Parameter, ABC):
    """Base class for optimisable parameters."""

    def __init__(self, name: str, initial_value: float, num_components: int) -> None:
        super().__init__(name, optimisable=True, num_components=num_components)
        self.initial_value = initial_value

    def get_initial_value(self) -> Union[float, np.ndarray]:
        """Get the initial value of the parameter.

        Returns
        -------
        Union[float, np.ndarray]
            Initial value of the parameter.
        """
        if self.num_components == 1:
            return self.initial_value
        return self.get_initial_vector()

    def get_initial_vector(self) -> np.ndarray:
        """Get the initial vector of optimisable parameters.

        Returns
        -------
        np.ndarray
            Initial values of the optimisable parameters.
        """
        return np.full(self.num_components, self.initial_value)

    @abstractmethod
    def get_bounds(self) -> np.ndarray:
        """Get the bounds of the optimisable parameters.

        Returns
        -------
        np.ndarray
            Bounds of the optimisable parameters.
        """

    @abstractmethod
    def get_random(self, rng: np.random.Generator) -> Union[float, np.ndarray]:
        """Get a random value for the parameter.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        Union[float, np.ndarray]
            Random value for the parameter.
        """

    @classmethod
    @abstractmethod
    def read(cls: type[OptimisableT], name: str, config: dict[str, Any]) -> OptimisableT:
        """Read a parameter from a configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the parameter.
        config : dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        OptimisableT
            An instance of the parameter.
        """


class RealParameter(OptimisableParameter):
    """Class for real-valued optimisation parameters."""

    def __init__(
        self, name: str, initial: float, lbound: float, ubound: float, num_components: int = 1
    ) -> None:
        super().__init__(name, initial, num_components=num_components)
        self.lbound = lbound
        self.ubound = ubound
        if initial > ubound or initial < lbound:
            raise RuntimeError(
                f"Initial value {initial} outside of bounds "
                f"[{lbound}, {ubound}] for parameter {name}."
            )

    def get_bounds(self) -> np.ndarray:
        """Get the bounds of the optimisable parameters.

        Returns
        -------
        np.ndarray
            Bounds of the optimisable parameters.
        """
        return np.array([[self.lbound, self.ubound]] * self.num_components)

    def get_random(self, rng: np.random.Generator) -> Union[float, np.ndarray]:
        """Get a random value for the parameter.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        Union[float, np.ndarray]
            Random value for the parameter.
        """
        return rng.uniform(self.lbound, self.ubound, size=self.num_components)

    @classmethod
    def read(cls: type[OptimisableT], name: str, config: dict[str, Any]) -> OptimisableT:
        """Read a parameter from a configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the parameter.
        config : dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        OptimisableT
            An instance of the parameter.
        """
        for key in ['initial', 'lbound', 'ubound']:
            if key not in config:
                raise RuntimeError(f"Missing '{key}' value for parameter {name}.")
        return cls(
            name,
            float(config['initial']),
            float(config['lbound']),
            float(config['ubound']),
            int(config.get('num_components', 1))
        )


class DiscreteParameter(OptimisableParameter):
    """Class for discrete-valued optimisation parameters."""

    def __init__(
        self, name: str, initial: float, values: list[float], num_components: int = 1
    ) -> None:
        super().__init__(name, initial, num_components=num_components)
        self.values = values
        if initial not in values:
            raise RuntimeError(
                f"Initial value {initial} not in allowed values {values} of parameter {name}."
            )

    def get_bounds(self) -> np.ndarray:
        """Get the bounds of the optimisable parameters.

        Returns
        -------
        np.ndarray
            Bounds of the optimisable parameters.
        """
        return np.array([[min(self.values), max(self.values)]] * self.num_components)

    def get_random(self, rng: np.random.Generator) -> Union[float, np.ndarray]:
        """Get a random value for the parameter.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        Union[float, np.ndarray]
            Random value for the parameter.
        """
        return rng.choice(self.values, size=self.num_components)

    @classmethod
    def read(cls: type[OptimisableT], name: str, config: dict[str, Any]) -> OptimisableT:
        """Read a parameter from a configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the parameter.
        config : dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        OptimisableT
            An instance of the parameter.
        """
        for key in ['initial', 'values']:
            if key not in config:
                raise RuntimeError(f"Missing '{key}' value for parameter {name}.")
        values = [float(val) for val in config['values']]
        return cls(name, float(config['initial']), values, int(config.get('num_components', 1)))


class ComputedParameter(Parameter):
    """Class for computed parameters.

    Parameter evaluation uses eval() based on the values of the optimisable parameters. The
    expression should be a valid Python expression using the names of the optimisable parameters.

    A modified environment is used for evaluation, containing only numpy and all math functions
    readily available.

    Note: Do not use this class for untrusted input, as eval() can execute arbitrary code.
    """

    def __init__(
        self, name: str, expression: str, optim_params: list[OptimisableParameter]
    ) -> None:
        # Compile the compute expression
        self.expression = expression
        self.code = compile(expression, "<string>", "eval")
        # Determine the number of components by evaluating the expression with the initial values
        values = {p.name: p.get_initial_value() for p in optim_params}
        evaluated = self.compute(values)
        if isinstance(evaluated, np.ndarray):
            if len(evaluated.shape) != 1:
                raise RuntimeError(
                    f"Computed value {evaluated} of parameter {name} has "
                    f"unsupported shape {evaluated.shape}."
                )
            num_components = evaluated.size
        elif isinstance(evaluated, (int, float)):
            num_components = 1
        else:
            raise RuntimeError(
                f"Computed value {evaluated} of parameter {name} has "
                f"unsupported type {type(evaluated)}."
            )
        super().__init__(name, optimisable=False, num_components=num_components)

    def compute(self, values: dict[str, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """Compute the value of the parameter based on other parameters.

        Parameters
        ----------
        values : dict[str, Union[float, np.ndarray]]
            Dictionary of parameter values.

        Returns
        -------
        Union[float, np.ndarray]
            Computed value of the parameter.
        """
        # Use a globals with only numpy and math functions (inject all math functions into globals)
        allowed_globals = {
            'np': np,
            'math': math,
            **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        }
        try:
            return eval(self.code, allowed_globals, values)
        except Exception as e:
            raise RuntimeError(f"Error computing parameter {self.name}: {e}")

    @classmethod
    def read(
        cls: type[ComputedT],
        name: str,
        config: dict[str, Any],
        optim_params: list[OptimisableParameter],
    ) -> ComputedT:
        """Read a parameter from a configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the parameter.
        config : dict[str, Any]
            Configuration dictionary.
        optim_params : list[OptimisableParameter]
            List of optimisable parameters.

        Returns
        -------
        ComputedT
            An instance of the parameter.
        """
        if "expression" not in config:
            raise ValueError(f"Missing 'expression' in configuration for parameter {name}.")
        expression = config["expression"]
        return cls(name, expression, optim_params)


class ParameterSet:
    """Container class for a set of parameters."""

    def __init__(
        self, optim_params: list[OptimisableParameter], computed_params: list[ComputedParameter]
    ) -> None:
        """Constructor for a parameter set."""
        self.optim_parameters = optim_params
        self.computed_parameters = computed_params
        self.indices = np.cumsum([0] + [p.num_components for p in self.optim_parameters])

    def __iter__(self) -> Iterator[OptimisableParameter]:
        """Iterator for a parameter set.

        Returns
        -------
        Iterator[OptimisableParameter]
            Iterator for a parameter set."""
        return iter(self.optim_parameters)

    def __len__(self) -> int:
        """Length of the parameter set.

        Returns
        -------
        int
            Length of the parameter set."""
        return len(self.optim_parameters)

    def __getitem__(self, index: int) -> OptimisableParameter:
        """Get an optimisable parameter by index.

        Parameters
        ----------
        index : int
            Index of the parameter.

        Returns
        -------
        OptimisableParameter
            The optimisable parameter at the given index.
        """
        return self.optim_parameters[index]

    def num_optim_parameters(self) -> int:
        """Get the number of optimisable parameters.

        Returns
        -------
        int
            Number of optimisable parameters.
        """
        return sum(p.num_components for p in self.optim_parameters)

    def num_discrete(self) -> int:
        """Get the number of discrete optimisable parameters.

        Returns
        -------
        int
            Number of discrete optimisable parameters.
        """
        return sum(
            p.num_components for p in self.optim_parameters if isinstance(p, DiscreteParameter)
        )

    def get_names(self) -> list[str]:
        """Get the names of the parameters.

        Returns
        -------
        list[str]
            Names of the parameters.
        """
        return [p.name for p in self.optim_parameters] + [p.name for p in self.computed_parameters]

    def get_scalar_names(self, include_computed: bool = True) -> list[str]:
        """Get the names of the parameters in a scalar form.

        Parameters
        ----------
        include_computed : bool
            Whether to include computed parameters.

        Returns
        -------
        list[str]
            Names of the parameters in a scalar form.
        """
        names = [name for p in self.optim_parameters for name in p.get_scalar_names()]
        if not include_computed:
            return names
        return names + [name for p in self.computed_parameters for name in p.get_scalar_names()]

    def get_initial_vector(self) -> np.ndarray:
        """Get the initial vector of optimisable parameters.

        Returns
        -------
        np.ndarray
            Initial values of the optimisable parameters.
        """
        return np.concatenate([p.get_initial_vector() for p in self.optim_parameters])

    def get_random_vector(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Get a random vector of optimisable parameters.

        Parameters
        ----------
        rng : Optional[np.random.Generator]
            Random number generator. If None, a new generator is created.

        Returns
        -------
        np.ndarray
            Random values of the optimisable parameters.
        """
        if rng is None:
            rng = np.random.default_rng()
        return np.concatenate([p.get_random(rng) for p in self.optim_parameters])

    def get_bounds(self) -> np.ndarray:
        """Get the bounds of the optimisable parameters.

        Returns
        -------
        np.ndarray
            Bounds of the optimisable parameters.
        """
        return np.concatenate([p.get_bounds() for p in self.optim_parameters])

    def get_discrete_combinations(self) -> list[dict[int, float]]:
        """Get all possible combinations of discrete optimisable parameters.

        We return a list of dictionaries, where each dictionary represents a unique combination
        of discrete parameter values. The keys in the dictionary are the indices of the discrete
        parameters, and the values are the corresponding discrete values.

        Returns
        -------
        list[dict[int, float]]
            List of dictionaries representing all possible combinations of discrete parameters.
        """
        values = {
            int(self.indices[i] + j): p.values
            for i, p in enumerate(self.optim_parameters) if isinstance(p, DiscreteParameter)
            for j in range(p.num_components)
        }
        return [dict(zip(values.keys(), combination)) for combination in product(*values.values())]

    def to_dict(
        self, values: np.ndarray, include_computed: bool
    ) -> dict[str, Union[float, np.ndarray]]:
        """Build a dict with name-value pairs given a list of values.

        Parameters
        ----------
        values : np.ndarray
            Values to pack. Their order is used for parameter resolution.
        include_computed : bool
            Whether to include computed parameters in the output.

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            Name-value pair for each parameter.
        """
        optim_params = {
            p.name: p.get_value(values[self.indices[i]:self.indices[i + 1]])
            for i, p in enumerate(self.optim_parameters)
        }

        # Nothing more to do if computed parameters are not included
        if not include_computed or len(self.computed_parameters) == 0:
            return optim_params

        # Add computed parameters
        for param in self.computed_parameters:
            optim_params[param.name] = param.compute(optim_params)
        return optim_params

    def to_scalar_dict(self, values: np.ndarray, include_computed: bool = True) -> dict[str, float]:
        """Build a dict with name-value scalar pairs given a list of values.

        Parameters
        ----------
        values : np.ndarray
            Values to pack. Their order is used for parameter resolution.
        include_computed : bool, optional
            Whether to include computed parameters in the output. Default is True.

        Returns
        -------
        dict[str, float]
            Name-value pair for each parameter.
        """
        # Build a dictionary of optimisable parameter names and their values scalar
        optim_params: dict[str, float] = {}
        for i, param in enumerate(self.optim_parameters):
            param_values = values[self.indices[i]:self.indices[i + 1]]
            for name, value in param.get_name_value_pair(param_values).items():
                optim_params[name] = value

        # Nothing more to do if computed parameters are not included
        if not include_computed or len(self.computed_parameters) == 0:
            return optim_params

        # Add computed parameters from the vector-valued optimisable parameters
        vector_optim_params = self.to_dict(values, include_computed=False)
        for param in self.computed_parameters:
            param_values = param.compute(vector_optim_params)
            for name, value in param.get_name_value_pair(param_values).items():
                optim_params[name] = value
        return optim_params

    def to_torch_dict(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build a dict with name-value pairs given a list of values (with tensors).

        Important note: this does NOT return the computed parameters.

        Parameters
        ----------
        values : torch.Tensor
            Values to pack, with shape `(batch_shape) x n_optim_param`.

        Returns
        -------
        dict[str, torch.Tensor]
            Name-value pair for each parameter with shape `(batch_shape) x n_components`.
        """
        return {
            p.name: values[..., self.indices[i]:self.indices[i + 1]]
            for i, p in enumerate(self.optim_parameters)
        }

    @staticmethod
    def hash(values: np.ndarray) -> str:
        """Build the hash for the current parameter values.

        Parameters
        ----------
        values : np.ndarray
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


def legacy_converter(name: str, param_spec: Any) -> dict[str, Any]:
    """Convert legacy real scalar parameter specifications to the new format.

    Parameters
    ----------
    name : str
        Name of the parameter.
    param_spec : Any
        Legacy parameter specification.

    Returns
    -------
    dict[str, Any]
        Converted parameter specification.
    """
    # Legacy parameters are a list with [initial, lbound, ubound]. Check if we match this format
    if isinstance(param_spec, list) and len(param_spec) == 3:
        return {
            'type': 'real',
            'initial': param_spec[0],
            'lbound': param_spec[1],
            'ubound': param_spec[2],
        }

    # Do some sanity checks on the types
    if not isinstance(param_spec, dict):
        raise TypeError(f"Parameter specification for '{name}' must be a dictionary.")
    if not all(isinstance(k, str) for k in param_spec.keys()):
        raise TypeError(f"All keys in the parameter specification for '{name}' must be strings.")
    if 'type' not in param_spec:
        raise ValueError(f"Parameter specification for '{name}' must include a 'type' key.")
    return param_spec


AVALIABLE_OPTIMISABLE_PARAMS: dict[str, type[OptimisableParameter]] = {
    'real': RealParameter,
    'discrete': DiscreteParameter,
}


AVAILABLE_COMPUTED_PARAMS: dict[str, type[ComputedParameter]] = {
    'computed': ComputedParameter,
}


def read_parameters(config: dict[str, Any]) -> ParameterSet:
    """Parse the parameters from the configuration dictionary.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    ParameterSet
        Parameter set for this problem.
    """
    # Sanitise and convert any legacy parameter specifications
    for name, spec in config.items():
        config[name] = legacy_converter(name, spec)

    # First, read optimisable parameters
    optim_params: list[OptimisableParameter] = []
    for name, spec in config.items():
        param_type = spec['type']
        if param_type in AVALIABLE_OPTIMISABLE_PARAMS:
            cls = AVALIABLE_OPTIMISABLE_PARAMS[param_type]
            optim_params.append(cls.read(name, spec))
        elif param_type not in AVAILABLE_COMPUTED_PARAMS:
            raise ValueError(f"Unknown parameter type '{param_type}' for parameter '{name}'.")

    # Then, read computed parameters
    computed_params: list[ComputedParameter] = []
    for name, spec in config.items():
        param_type = spec['type']
        if param_type in AVAILABLE_COMPUTED_PARAMS:
            cls = AVAILABLE_COMPUTED_PARAMS[param_type]
            computed_params.append(cls.read(name, spec, optim_params))

    return ParameterSet(optim_params, computed_params)
