"""Driver script for piglot."""
import os
import os.path
import argparse
import shutil
from yaml import safe_dump
import torch
from piglot.objectives import read_objective
from piglot.optimisers import read_optimiser
from piglot.parameter import read_parameters
from piglot.optimiser import StoppingCriteria
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
    # PyTorch compute device
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Default device to use with PyTorch',
    )

    return parser.parse_args()


def main(config_path: str = None):
    """Entry point for piglot."""
    if config_path is None:
        args = parse_args()
        config_path = args.config
        torch.set_default_device(args.device)
    config = parse_config_file(config_path)
    # Build output directory with a copy of the configuration file
    output_dir = config["output"]
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config"), 'w', encoding='utf8') as file:
        safe_dump(config, file)
    # Build piglot problem
    parameters = read_parameters(config)
    objective = read_objective(config["objective"], parameters, output_dir)
    optimiser = read_optimiser(config["optimiser"], objective)
    stop = StoppingCriteria.read(config)
    # Run the optimisation
    _, best_params = optimiser.optimise(
        config["iters"],
        parameters,
        output_dir,
        verbose=not config["quiet"],
        stop_criteria=stop,
    )
    # Re-run the best case
    if 'skip_last_run' not in config and best_params is not None:
        objective(best_params)


if __name__ == '__main__':
    main()
