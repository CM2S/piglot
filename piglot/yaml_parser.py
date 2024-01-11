"""Module for parsing the YAML configuration file."""
from typing import Dict, Any
import os
import os.path
from yaml import safe_load
from yaml.parser import ParserError
from yaml.scanner import ScannerError


def parse_config_file(config_file: str) -> Dict[str, Any]:
    """Parses the YAML configuration file.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the YAML data.

    Raises
    ------
    RuntimeError
        When the YAML parsing fails.
    """
    try:
        with open(config_file, 'r', encoding='utf8') as file:
            config = safe_load(file)
    except (ParserError, ScannerError) as exc:
        raise RuntimeError("Failed to parse the config file: YAML syntax seems invalid.") from exc
    # Check required terms
    if 'iters' not in config:
        raise RuntimeError("Missing number of iterations from the config file")
    if 'objective' not in config:
        raise RuntimeError("Missing objective from the config file")
    if 'optimiser' not in config:
        raise RuntimeError("Missing optimiser from the config file")
    if 'parameters' not in config:
        raise RuntimeError("Missing parameters from the config file")
    # Add missing optional items
    if 'output' not in config:
        config['output'] = os.path.splitext(config_file)[0]
    if 'quiet' not in config:
        config["quiet"] = False
    elif config['quiet']:
        config["quiet"] = True
    return config
