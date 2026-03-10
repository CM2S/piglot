"""Driver script for piglot."""
import os
import os.path
import argparse
import shutil
from yaml import safe_dump
import numpy as np
import torch
from piglot.objectives import read_objective
from piglot.optimisers import read_optimiser
from piglot.settings import read_settings
from piglot.utils.yaml_parser import parse_config_file


def parse_args():
    """Parse command line arguments of the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    # Global argument parser settings
    parser = argparse.ArgumentParser(
        prog='piglot',
        description='Parameter identification toolbox',
    )

    # Add arguments: configuration file
    parser.add_argument(
        'config',
        type=str,
        help='Configuration file to use',
    )
    # PyTorch options
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Default device to use with PyTorch',
    )
    parser.add_argument(
        '--torch_num_threads',
        type=int,
        default=1,
        help='Default number of threads to use with PyTorch',
    )

    return parser.parse_args()


def main(config_path: str = None):
    """Entry point for piglot."""
    if config_path is None:
        args = parse_args()
        config_path = args.config
        device = args.device
        torch_num_threads = args.torch_num_threads
    else:
        device = 'cpu'
        torch_num_threads = 1
    # Set up PyTorch before reading the configuration file
    torch.set_default_device(device)
    torch.set_num_threads(torch_num_threads)
    config = parse_config_file(config_path)
    # Build output directory with a copy of the configuration file
    output_dir = config["output_dir"]
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config"), 'w', encoding='utf8') as file:
        safe_dump(config, file)
    # Build piglot problem
    settings = read_settings(config)
    objective = read_objective(config["objective"], settings)
    optimiser = read_optimiser(config["optimiser"], settings, objective)
    # Run the optimisation
    result = optimiser.optimise()
    best_value = result.value
    best_params = result.params
    # Re-run the best case
    if 'skip_last_run' not in config and best_params is not None and not objective.has_variance():
        objective(best_params)
    # If we have an expected solution, compare the results
    if 'expected' in config:
        if 'value' not in config['expected']:
            raise ValueError("Missing expected value for design objective.")
        if 'parameters' not in config['expected']:
            raise ValueError("Missing expected parameters for design objective.")
        expected_value = float(config['expected']['value'])
        expected_params = np.array(config['expected']['parameters'])
        value_atol = float(config['expected'].get('value_atol', 1e-2))
        value_rtol = float(config['expected'].get('value_rtol', 1e-2))
        params_atol = float(config['expected'].get('parameters_atol', 1e-2))
        params_rtol = float(config['expected'].get('parameters_rtol', 1e-2))
        value_check = np.isclose(best_value, expected_value, atol=value_atol, rtol=value_rtol)
        params_check = np.allclose(best_params, expected_params, atol=params_atol, rtol=params_rtol)
        assert value_check, f"Failed value check: {best_value} vs {expected_value}"
        assert params_check, f"Failed parameters check: {best_params} vs {expected_params}"


if __name__ == '__main__':
    main()
