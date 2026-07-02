"""Optimisation parameter module."""
from dataclasses import dataclass
from typing import Iterator, Any, Literal, Union, TypeVar, Optional
from abc import ABC, abstractmethod
from functools import cached_property
from hashlib import sha256
from itertools import product
import numpy as np
import torch
from torch.distributions import Distribution, Normal
from piglot.utils.distributions import (
    read_real_distribution, get_discrete_distribution, ClosedUniform
)


OptimisableT = TypeVar("OptimisableT", bound="OptimisableParameter")
ComputedT = TypeVar("ComputedT", bound="ComputedParameter")


@dataclass(frozen=True)
class ParameterValues:
    """Container for a set of parameter values for objective evaluation."""
    scalar_values: dict[str, float]
    vector_values: dict[str, np.ndarray]

    @cached_property
    def param_hash(self) -> str:
        values = np.array(list(self.scalar_values.values()))
        return sha256(values.tobytes()).hexdigest()

    @staticmethod
    def join(*args: "ParameterValues") -> "ParameterValues":
        """Join multiple ParameterValues into a single ParameterValues.

        Parameters
        ----------
        *args : ParameterValues
            ParameterValues to join.

        Returns
        -------
        ParameterValues
            Joined ParameterValues.
        """
        scalar_values: dict[str, float] = {}
        vector_values: dict[str, np.ndarray] = {}
        for pv in args:
            # Check for name collisions
            for name in pv.scalar_values:
                if name in scalar_values:
                    raise ValueError(f"Duplicate scalar parameter name: {name}")
            for name in pv.vector_values:
                if name in vector_values:
                    raise ValueError(f"Duplicate vector parameter name: {name}")
            scalar_values.update(pv.scalar_values)
            vector_values.update(pv.vector_values)
        return ParameterValues(scalar_values, vector_values)


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

    def get_values(self, values: np.ndarray) -> ParameterValues:
        """Get the value of the parameter.

        Parameters
        ----------
        values : np.ndarray
            Array of parameter values.

        Returns
        -------
        ParameterValues
            Values of the parameter.
        """
        return ParameterValues(
            {name: float(values[i]) for i, name in enumerate(Parameter.get_scalar_names(self))},
            {self.name: values}
        )


class OptimisableParameter(Parameter, ABC):
    """Base class for optimisable parameters."""

    def __init__(
        self, name: str, initial_value: float, prior: Distribution, num_components: int
    ) -> None:
        super().__init__(name, optimisable=True, num_components=num_components)
        self.initial_value = initial_value
        self.prior = prior

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

    def to_torch_dict(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert parameter values to a dictionary of torch tensors.

        Parameters
        ----------
        values : torch.Tensor
            Values of the parameter.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of parameter values as torch tensors.
        """
        return {self.name: values}

    @abstractmethod
    def get_bounds(self) -> np.ndarray:
        """Get the bounds of the optimisable parameters.

        Returns
        -------
        np.ndarray
            Bounds of the optimisable parameters.
        """

    @abstractmethod
    def get_random(self) -> Union[float, np.ndarray]:
        """Get a random value for the parameter.

        Returns
        -------
        Union[float, np.ndarray]
            Random value for the parameter.
        """

    @abstractmethod
    def log_prob(self, values: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the parameter values.

        Parameters
        ----------
        values : torch.Tensor
            Values of the parameter.

        Returns
        -------
        torch.Tensor
            Log probability of the parameter values.
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
        self,
        name: str,
        initial: float,
        lbound: float,
        ubound: float,
        prior: Distribution,
        num_components: int = 1,
    ) -> None:
        super().__init__(name, initial, prior, num_components=num_components)
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

    def get_random(self) -> Union[float, np.ndarray]:
        """Get a random value for the parameter.

        Returns
        -------
        Union[float, np.ndarray]
            Random value for the parameter.
        """
        samples = self.prior.sample().to(torch.float64).cpu().numpy()
        return np.clip(samples, self.lbound, self.ubound)

    def log_prob(self, values: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the parameter values.

        Parameters
        ----------
        values : torch.Tensor
            Values of the parameter.

        Returns
        -------
        torch.Tensor
            Log probability of the parameter values.
        """
        return self.prior.log_prob(values)

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
        num_components = int(config.get('num_components', 1))
        # Read prior distribution
        if 'prior' in config:
            prior = read_real_distribution(config['prior'])
        else:
            prior = ClosedUniform(float(config['lbound']), float(config['ubound']))
        return cls(
            name,
            float(config['initial']),
            float(config['lbound']),
            float(config['ubound']),
            prior.expand((num_components,)),
            num_components,
        )


class DiscreteParameter(OptimisableParameter):
    """Class for discrete-valued optimisation parameters."""

    def __init__(
        self,
        name: str,
        initial: float,
        values: list[float],
        prior: Distribution,
        num_components: int = 1,
    ) -> None:
        super().__init__(name, initial, prior, num_components=num_components)
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

    def get_random(self) -> Union[float, np.ndarray]:
        """Get a random value for the parameter.

        Returns
        -------
        Union[float, np.ndarray]
            Random value for the parameter.
        """
        indices = self.prior.sample().tolist()
        return np.array([self.values[int(idx)] for idx in indices])

    def log_prob(self, values: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the parameter values.

        Parameters
        ----------
        values : torch.Tensor
            Values of the parameter.

        Returns
        -------
        torch.Tensor
            Log probability of the parameter values.
        """
        indices = torch.tensor(
            [self.values.index(val.item()) for val in values.flatten()], dtype=torch.long
        ).reshape(values.shape)
        return self.prior.log_prob(indices)

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
        num_components = int(config.get('num_components', 1))
        # Read prior probabilities
        if 'prior_probs' in config:
            prior_probs = [float(prob) for prob in config['prior_probs']]
        else:
            prior_probs = [1.0] * len(values)
        prior = get_discrete_distribution(prior_probs).expand((num_components,))
        return cls(name, float(config['initial']), values, prior, num_components)


class FidelityParameter(DiscreteParameter):
    """Class for fidelity parameters."""

    def __init__(
        self,
        name: str,
        initial: float,
        values: list[float],
        prior: Distribution,
        cost: Optional[dict[float, float]],
        cost_model: Literal["fixed", "infer", "infer_full"],
    ) -> None:
        super().__init__(name, initial, values, prior, num_components=1)
        self.cost = cost
        self.cost_model = cost_model

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
        # Read fidelity values
        if 'values' not in config:
            raise RuntimeError(f"Missing 'values' value for fidelity parameter {name}.")
        values = [float(val) for val in config['values']]
        if any(val < 0 for val in values) or any(val > 1 for val in values):
            raise RuntimeError(f"Fidelity parameter {name} must be between 0 and 1.")
        if 1.0 not in values:
            raise RuntimeError(f"Fidelity parameter {name} must have a target fidelity of 1.")

        # Read cost
        cost = None
        cost_model = "infer"
        if 'cost' in config:
            cost = {val: float(c) for val, c in zip(values, config['cost'])}
            if config.get("cost_model", "fixed") != "fixed":
                raise RuntimeError("Cost model must be 'fixed' when cost is provided.")
            cost_model = "fixed"

        # Read cost model
        if 'cost_model' in config:
            cost_model = config['cost_model']
            if cost_model not in ["fixed", "infer", "infer_full"]:
                raise RuntimeError(
                    f"Invalid cost model '{cost_model}' for fidelity parameter {name}."
                )

        # Read prior
        if 'prior_probs' in config:
            prior_probs = [float(prob) for prob in config['prior_probs']]
        else:
            prior_probs = [1.0] * len(values)
        prior = get_discrete_distribution(prior_probs).expand((1,))

        return cls(name, float(config.get('initial', 1.0)), values, prior, cost, cost_model)


class LatentParameter(RealParameter, ABC):
    """Base class for optimisation in latent spaces.
    
    This class of parameters use a set of optimisable latent parameters to generate the actual
    parameter values. For compatibility with composition, we must support gradients between 
    the latent parameters and the generated parameter values.
    """

    def __init__(
        self,
        name: str,
        param_lbound: float,
        param_ubound: float,
        param_prior: Distribution,
        latent_initial: float,
        latent_lbound: float,
        latent_ubound: float,
        latent_prior: Distribution,
        num_param_components: int,
        num_latent_components: int,
    ) -> None:
        # Store a fake parameter with correct output dimensions
        initial = param_prior.icdf(torch.tensor([latent_initial]))
        self.field_param = RealParameter(
            name,
            torch.clamp(initial, param_lbound, param_ubound).item(),
            param_lbound,
            param_ubound,
            param_prior,
            num_components=num_param_components,
        )

        # Initialise the latent parameter
        super().__init__(
            f"{name}_latent",
            latent_initial,
            latent_lbound,
            latent_ubound,
            latent_prior,
            num_components=num_latent_components,
        )

    def get_scalar_names(self) -> list[str]:
        """Get the names of the parameters in a scalar form.

        Returns
        -------
        list[str]
            Names of the scalar parameters.
        """
        return super().get_scalar_names() + self.field_param.get_scalar_names()

    def get_values(self, values: np.ndarray) -> ParameterValues:
        """Get the value of the parameter.

        Parameters
        ----------
        values : np.ndarray
            Array of parameter values.

        Returns
        -------
        ParameterValues
            Values of the parameter.
        """
        field_values = self.transform(torch.from_numpy(values)).numpy()
        return ParameterValues.join(
            super().get_values(values), self.field_param.get_values(field_values)
        )

    def to_torch_dict(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert parameter values to a dictionary of torch tensors.

        Parameters
        ----------
        values : torch.Tensor
            Values of the parameter.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of parameter values as torch tensors.
        """
        return {
            self.name: values,
            self.field_param.name: self.transform(values),
        }

    @abstractmethod
    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Transform the parameters from the latent space to the output space.

        Parameters
        ----------
        values : torch.Tensor
            Tensor of shape `(batch_shape) x order` with the latent values.

        Returns
        -------
        torch.Tensor
            Transformed parameters.
        """


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
        self.name = name
        self.expression = expression
        self.code = compile(expression, "<string>", "eval")
        # Determine the number of components by evaluating the expression with the initial values
        values = {
            n: v
            for p in optim_params
            for n, v in p.get_values(p.get_initial_vector()).vector_values.items()
        }
        evaluated = self.__raw_compute(values)
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

    def get_values(self, values: np.ndarray) -> ParameterValues:
        """Get the value of the parameter.

        Parameters
        ----------
        values : np.ndarray
            Array of parameter values.

        Returns
        -------
        ParameterValues
            Values of the parameter.
        """
        raise NotImplementedError("get_values() is not supported for computed parameters.")

    def __raw_compute(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Compute the raw values of the parameter based on other parameters.

        Parameters
        ----------
        values : dict[str, np.ndarray]
            Dictionary of parameter values.

        Returns
        -------
        np.ndarray
            Raw computed values of the parameter.
        """
        # Use a globals with only numpy functions
        allowed_globals = {
            k: getattr(np, k) for k in dir(np) if not k.startswith("_") and not k.endswith("_")
        }
        try:
            result = eval(self.code, allowed_globals, values)
        except Exception as e:
            raise RuntimeError(f"Error computing parameter {self.name}: {e}")

        # Sanitise result type
        if isinstance(result, (int, float)):
            result = np.array([result])
        elif not isinstance(result, np.ndarray):
            raise RuntimeError(
                f"Computed value {result} of parameter {self.name} "
                f"has unsupported type {type(result)}."
            )
        return result

    def compute(self, values: dict[str, np.ndarray]) -> ParameterValues:
        """Compute the value of the parameter based on other parameters.

        Parameters
        ----------
        values : dict[str, np.ndarray]
            Dictionary of parameter values.

        Returns
        -------
        ParameterValues
            Computed values of the parameter.
        """
        return super().get_values(self.__raw_compute(values))

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
        expression = str(config["expression"])
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

        # Sanitise fidelity parameter
        self.fidelity_params = [
            p for p in self.optim_parameters if isinstance(p, FidelityParameter)
        ]
        if len(self.fidelity_params) > 1:
            raise RuntimeError("Only one fidelity parameter is allowed.")

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

    def is_multi_fidelity(self) -> bool:
        """Check if the parameter set contains a fidelity parameter.

        Returns
        -------
        bool
            True if the parameter set contains a fidelity parameter, False otherwise.
        """
        return len(self.fidelity_params) > 0

    def get_scalar_fidelity_index(self) -> int:
        """Get the index of the scalar fidelity parameter.

        Returns
        -------
        int
            Index of the scalar fidelity parameter.
        """
        if not self.is_multi_fidelity():
            raise RuntimeError("No fidelity parameter found.")
        return self.get_scalar_names().index(self.fidelity_params[0].name)

    def get_fidelities(self) -> list[float]:
        """Get the fidelities of the parameter set.

        Returns
        -------
        list[float]
            Fidelities of the parameter set.
        """
        if not self.is_multi_fidelity():
            raise RuntimeError("No fidelity parameter found.")
        return [p.values for p in self.fidelity_params][0]

    def get_fidelity_parameter(self) -> FidelityParameter:
        """Get the fidelity parameter.

        Returns
        -------
        FidelityParameter
            The fidelity parameter.
        """
        if not self.is_multi_fidelity():
            raise RuntimeError("No fidelity parameter found.")
        return self.fidelity_params[0]

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

    def get_random_vector(self) -> np.ndarray:
        """Get a random vector of optimisable parameters.

        Returns
        -------
        np.ndarray
            Random values of the optimisable parameters.
        """
        return np.concatenate([p.get_random() for p in self.optim_parameters])

    def get_bounds(self) -> np.ndarray:
        """Get the bounds of the optimisable parameters.

        Returns
        -------
        np.ndarray
            Bounds of the optimisable parameters.
        """
        return np.concatenate([p.get_bounds() for p in self.optim_parameters])

    def get_discrete_combinations(self, fixtures: dict[int, float]) -> list[dict[int, float]]:
        """Get all possible combinations of discrete optimisable parameters.

        We return a list of dictionaries, where each dictionary represents a unique combination
        of discrete parameter values. The keys in the dictionary are the indices of the discrete
        parameters, and the values are the corresponding discrete values.

        Parameters
        ----------
        fixtures : dict[int, float]
            Fixed values for certain discrete parameters.

        Returns
        -------
        list[dict[int, float]]
            List of dictionaries representing all possible combinations of discrete parameters.
        """
        values: dict[int, list[float]] = {}
        for i, param in enumerate(self.optim_parameters):
            if isinstance(param, DiscreteParameter):
                for j in range(param.num_components):
                    idx = int(self.indices[i] + j)
                    values[idx] = param.values if idx not in fixtures else [fixtures[idx]]
        return [dict(zip(values.keys(), combination)) for combination in product(*values.values())]

    def to_values(self, values: np.ndarray) -> ParameterValues:
        """Get the parameter values as a ParameterValues object.

        Parameters
        ----------
        values : np.ndarray
            Values to pack. Their order is used for parameter resolution.

        Returns
        -------
        ParameterValues
            Parameter values.
        """
        raw_optim_values = [
            p.get_values(values[self.indices[i]:self.indices[i + 1]])
            for i, p in enumerate(self.optim_parameters)
        ]
        optim_values = ParameterValues.join(*raw_optim_values)

        # Evaluate optimisable vector parameters
        raw_computed_values = [
            p.compute(optim_values.vector_values) for p in self.computed_parameters
        ]

        # Join sets of parameter values
        return ParameterValues.join(optim_values, *raw_computed_values)

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
            k: v
            for i, p in enumerate(self.optim_parameters)
            for k, v in p.to_torch_dict(values[..., self.indices[i]:self.indices[i + 1]]).items()
        }

    def log_prob(self, values: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the parameter values.

        Parameters
        ----------
        values : torch.Tensor
            Values to pack, with shape `(batch_shape) x n_optim_param`.

        Returns
        -------
        torch.Tensor
            Log probability of the parameter values.
        """
        log_probs = [
            p.log_prob(values[..., self.indices[i]:self.indices[i + 1]])
            for i, p in enumerate(self.optim_parameters)
        ]
        return torch.sum(torch.concatenate(log_probs, dim=-1), dim=-1)


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
    'fidelity': FidelityParameter,
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
