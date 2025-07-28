"""Driver script for piglot-eval."""
import os
import os.path
import argparse
import time
import shutil
from tempfile import TemporaryDirectory
import numpy as np
from yaml import safe_dump
from piglot.objectives import read_objective
from piglot.parameter import read_parameters
from piglot.utils.yaml_parser import parse_config_file
from piglot.utils.assorted import pretty_time
from piglot.bin.piglot_plot import main as plot_main


def parse_args():
    """Parse command line arguments of the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    # Global argument parser settings
    parser = argparse.ArgumentParser(
        prog='piglot-eval',
        description='Evaluate a piglot objective function with given parameters.',
    )

    # Add arguments: configuration file
    parser.add_argument(
        'config',
        type=str,
        help='Configuration file to use',
    )
    # Parameters for evaluating the objective function
    parser.add_argument(
        '--parameters',
        type=str,
        nargs='*',
        help=(
            'Parameters to use for evaluating the objective function, <param1> <param2> ... '
            'If not specified, the initial guess from the configuration file is used.'
        ),
    )
    # Add argument for plotting results
    parser.add_argument(
        '--plot',
        type=str,
        help='Invoke piglot-plot on the results with the given parameters.',
    )

    return parser.parse_args()


def main():
    """Entry point for piglot-eval."""
    args = parse_args()
    config_path = args.config
    config = parse_config_file(config_path)

    # Generate a temporary directory for the output
    with TemporaryDirectory() as tmp_dir:
        # Setup temporary output directory
        problem_name = os.path.splitext(os.path.basename(config_path))[0]
        output_dir = os.path.join(tmp_dir, problem_name)
        new_config_path = os.path.join(tmp_dir, os.path.basename(config_path))
        config["output"] = output_dir

        # Build output directory and copy the configuration file
        with open(new_config_path, 'w', encoding='utf8') as file:
            safe_dump(config, file)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Build piglot problem
        parameters = read_parameters(config)
        objective = read_objective(config["objective"], parameters, output_dir)

        # Set up the parameters to evaluate
        if args.parameters is None:
            # Use initial guess from the configuration file
            param_values = np.array([p.inital_value for p in parameters])
        else:
            # Use the parameters specified on the command line (and sanitise them)
            param_values = np.array([float(p) for p in args.parameters])
            if len(param_values) != len(parameters):
                raise ValueError(
                    f"Expected {len(parameters)} parameters, but got {len(param_values)}."
                )
            for i, p in enumerate(parameters):
                if param_values[i] < p.lbound or param_values[i] > p.ubound:
                    raise ValueError(
                        f"Parameter {p.name} with value {param_values[i]} is out of bounds."
                    )

        print(f"Evaluating {config_path} with parameters:")
        for param, value in zip(parameters, param_values):
            print(f"  {param.name}: {value}")
        print()

        # Evaluate the objective function with the given parameters
        start_time = time.perf_counter()
        objective.prepare()
        result = objective(param_values)
        elapsed_time = time.perf_counter() - start_time
        print(f"Completed in {pretty_time(elapsed_time)}")

        # Print the results
        print("\nObjective results")
        if result.scalar_value is not None:
            print("  Scalar objective value:")
            print(
                f"    {result.scalar_value:.5e}"
                +
                "" if result.scalar_variance is None else f" (var: {result.scalar_variance:.5e})"
            )
        print("  Objective values:")
        if result.obj_variances is None:
            for i, value in enumerate(result.obj_values):
                print(f"    Objective {i + 1}: {value:.5e}")
        else:
            for i, value in enumerate(result.obj_values):
                print(f"    Objective {i + 1}: {value:.5e} (var: {result.obj_variances[i]:.5e})")
        print("  Optimiser values:")
        print(result.values)
        if result.covariances is not None:
            print(result.covariances)

        # If requested, plot the results
        if args.plot:
            print("\nPlotting results...")
            plot_main(args.plot.split() + [new_config_path])


if __name__ == '__main__':
    main()
