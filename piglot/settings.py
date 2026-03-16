"""Module for global settings and configuration."""
from typing import Any, TypeVar
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
        'iters': int(config.pop('iters')),
        'output_dir': str(config.pop('output_dir')),
        'parameters': read_parameters(config),
    }
    # Hacky: remove the parameters from the config
    config.pop('parameters', None)
    # Read optional entries from the configuration file
    for key, field in settings_entries.items():
        if key in config and field.default is not MISSING_TYPE:
            # Use the type annotation to convert the value from the configuration file
            parsed_config[key] = field.type(config[key])
    return Settings(**parsed_config)
