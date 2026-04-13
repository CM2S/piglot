"""Module for parsing the YAML configuration file."""
from typing import Any
from dataclasses import dataclass
import os
import os.path
import numpy as np
import yaml
from yaml.parser import ParserError
from yaml.scanner import ScannerError
from piglot.settings import read_settings, Settings
from piglot.objectives import read_objective, Objective
from piglot.optimisers import read_optimiser, Optimiser
from piglot.objectives.synthetic import SyntheticIndividualObjective
from piglot.utils.assorted import InlineFileManager


@dataclass
class ProblemConfig:
    """Container for the optimisation problem."""

    config_path: str
    settings: Settings
    objective: Objective
    optimiser: Optimiser


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that checks for duplicate keys in mappings.

    Adapted from https://gist.github.com/pypt/94d747fe5180851196eb.
    """

    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


def read_yaml(file_path: str) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    file_path : str
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Contents of the YAML file.
    """
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            return yaml.load(file, Loader=UniqueKeyLoader)  # nosec B506
    except (ParserError, ScannerError) as exc:
        raise RuntimeError("Failed to parse the YAML file: syntax seems invalid.") from exc


def dump_yaml(config: dict[str, Any], file_path: str) -> None:
    """Dump a dictionary to a YAML file.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to dump.
    file_path : str
        Path to the YAML file.
    """
    with open(file_path, 'w', encoding='utf8') as file:
        yaml.safe_dump(config, file)


def build_sample_problem(problem_name: str) -> dict[str, Any]:
    """Construct an optimisation problem from a synthetic test function.

    Parameters
    ----------
    problem_name : str
        Name of the synthetic test function.

    Returns
    -------
    dict[str, Any]
        New configuration of the problem.
    """
    test_functions = SyntheticIndividualObjective.get_test_functions()
    if problem_name not in test_functions:
        raise ValueError(f"Unknown synthetic test function: {problem_name}")
    test_func = test_functions[problem_name]()

    # Build parameters
    rng = np.random.default_rng()
    config = {"parameters": {}}
    for i, (lbound, ubound) in enumerate(test_func._bounds):
        config["parameters"][f"x{i}"] = {
            "type": "real",
            "initial": rng.uniform(lbound, ubound),
            "lbound": lbound,
            "ubound": ubound,
        }

    # Build objective
    config["objective"] = {
        "name": "test_function",
        "function": problem_name,
    }

    # Expected result
    config["expected"] = {
        "value": float(test_func.optimal_value),
    }
    return config


def build_problem(config_path: str) -> tuple[ProblemConfig, dict[str, Any]]:
    """Build the problem from the configuration file path.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    tuple[ProblemConfig, dict[str, Any]]
        A tuple containing the ProblemConfig and the raw configuration dictionary.
    """
    # Parse the configuration file
    config = read_yaml(config_path)

    # Check if this is a sample problem: inject the sample configuration
    if "sample_problem" in config:
        config = build_sample_problem(config["sample_problem"]) | config

    # Check required terms
    if 'objective' not in config:
        raise RuntimeError("Missing objective from the config file")
    if 'optimiser' not in config:
        raise RuntimeError("Missing optimiser from the config file")
    if 'parameters' not in config:
        raise RuntimeError("Missing parameters from the config file")

    # Inject missing optional items
    if 'output_dir' not in config:
        config['output_dir'], _ = os.path.splitext(config_path)

    # Handle inline files and keep a copy of the raw configuration to return later
    raw_config = config.copy()
    inline_manager = InlineFileManager(config.pop('inline_files', {}))

    # Build the settings, objective, and optimiser within the inline file context
    with inline_manager:
        config = inline_manager.update_config(config)
        settings = read_settings(config)
        objective = read_objective(config["objective"], settings)
        optimiser = read_optimiser(config["optimiser"], settings, objective)
    return ProblemConfig(config_path, settings, objective, optimiser), raw_config
