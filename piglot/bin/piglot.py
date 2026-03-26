"""Driver script for piglot."""
from typing import Optional
import os
import os.path
import argparse
import shutil
from yaml import safe_dump
import torch
from piglot.utils.yaml_parser import build_problem


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


def main(
    config_path: Optional[str] = None, device: str = 'cpu', torch_num_threads: int = 1
) -> None:
    """Entry point for piglot.

    Parameters
    ----------
    config_path : Optional[str], optional
        Path to the configuration file, by default None. If not provided, use command line arguments
    device : str, optional
        Device to use for PyTorch, by default 'cpu'
    torch_num_threads : int, optional
        Number of threads to use for PyTorch, by default 1
    """
    # When we don't have a configuration path, parse command line arguments
    if config_path is None:
        args = parse_args()
        config_path = args.config
        device = args.device
        torch_num_threads = args.torch_num_threads

    # Build piglot problem
    # This reads the configuration file and sets up the problem
    # Note: This operation should NOT write to the output directory
    problem, config = build_problem(config_path)
    output_dir = problem.settings.output_dir
    optimiser = problem.optimiser
    objective = problem.objective

    # Prepare output directory (cleanup if needed)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Create a copy of the configuration file in the output directory
    with open(os.path.join(output_dir, "config"), 'w', encoding='utf8') as file:
        safe_dump(config, file)

    # Set up PyTorch
    torch.set_default_device(device)
    torch.set_num_threads(torch_num_threads)

    # Run the optimisation
    result = optimiser.optimise()

    # Re-run the best case
    if not problem.settings.skip_last_run:
        if result.params is not None and not objective.has_variance():
            objective(result.params)

    # If we have an expected solution, compare the results
    if problem.settings.expected is not None:
        problem.settings.expected.check(result.value, result.params)


if __name__ == '__main__':
    main()
