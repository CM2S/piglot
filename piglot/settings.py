"""Module for global settings and configuration."""
from typing import Any, Callable, TypeVar
from dataclasses import dataclass, Field, _MISSING_TYPE as MISSING_TYPE
from piglot.parameter import ParameterSet, read_parameters


T = TypeVar('T')


@dataclass
class Settings:
    """Container for global settings and configuration."""

    # Number of iterations to run
    iters: int

    # Directory to store outputs
    output_dir: str

    # Parameter set for the problem
    parameters: ParameterSet

    # Compute device
    device: str = 'cpu'

    # Global seed
    seed: int = None

    # Quiet mode (suppress output)
    quiet: bool = False

    # Stopping criteria
    conv_tol: float = None
    max_timeout: float = None
    max_func_calls: int = None
    max_iters_no_improv: int = None


def type_factory(cls: type[T]) -> Callable[[dict[str, Any]], T]:
    """Factory for type conversion functions.

    Parameters
    ----------
    cls : type[T]
        Type to convert to.

    Returns
    -------
    Callable[[dict[str, Any]], T]
        Function that converts a configuration dictionary to the given type.
    """
    if cls == ParameterSet:
        return read_parameters
    return cls


def read_settings(config: dict[str, Any]) -> Settings:
    """Read the settings from the configuration dictionary.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Settings
        Settings for this problem.
    """
    settings_entries: dict[str, Field] = Settings.__dataclass_fields__  # pylint: disable=no-member
    # Check for mandatory entries
    if 'output_dir' not in config:
        raise RuntimeError("Missing output directory from the config file")
    if 'iters' not in config:
        raise RuntimeError("Missing number of iterations from the config file")
    # Set up mandatory entries
    parsed_config = {
        'iters': int(config['iters']),
        'output_dir': str(config['output_dir']),
        'parameters': read_parameters(config),
    }
    # Read optional entries from the configuration file
    for key, field in settings_entries.items():
        if key in config and field.default is MISSING_TYPE:
            # Use the type annotation to convert the value from the configuration file
            factory = type_factory(field.type)
            parsed_config[key] = factory(config[key])
    return Settings(**parsed_config)
