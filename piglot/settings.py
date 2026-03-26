"""Module for global settings and configuration."""
from typing import Any, TypeVar, Optional
from dataclasses import dataclass, Field, _MISSING_TYPE as MISSING_TYPE
import numpy as np
from piglot.parameter import ParameterSet, read_parameters
from piglot.utils.readable import ReadableModel


T = TypeVar('T')


class ExpectedResult(ReadableModel):
    """Readable container for the expected result of the optimisation."""
    value: Optional[float] = None
    parameters: Optional[list[float]] = None
    value_atol: float = 1e-2
    value_rtol: float = 1e-2
    parameters_atol: float = 1e-2
    parameters_rtol: float = 1e-2

    def check(self, value: float, parameters: Optional[np.ndarray]) -> None:
        """Check if the given value and parameters match the expected result within tolerances.

        Parameters
        ----------
        value : float
            The value to check.
        parameters : Optional[np.ndarray]
            The parameters to check.
        """
        if self.value is not None:
            if not np.isclose(value, self.value, atol=self.value_atol, rtol=self.value_rtol):
                raise ValueError(f"Value {value} does not match expected {self.value}")
        if self.parameters is not None:
            if parameters is None:
                raise ValueError("Parameters are None but expected parameters are defined.")
            if not np.allclose(
                parameters, self.parameters, atol=self.parameters_atol, rtol=self.parameters_rtol
            ):
                raise ValueError(f"Parameters {parameters} do not match expected {self.parameters}")


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

    # Expected result
    expected: Optional[ExpectedResult] = None

    # Skip last run
    skip_last_run: bool = False


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
    # Hacky, take 2: inject expected results
    if 'expected' in config:
        parsed_config['expected'] = ExpectedResult.read(config.pop('expected'))
    # Read optional entries from the configuration file
    for key, field in settings_entries.items():
        if key in config and field.default is not MISSING_TYPE:
            # Use the type annotation to convert the value from the configuration file
            parsed_config[key] = field.type(config[key])
    return Settings(**parsed_config)
