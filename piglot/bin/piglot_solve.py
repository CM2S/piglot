"""Driver script for piglot-solve."""
import os
import os.path
import argparse
from tempfile import TemporaryDirectory
import numpy as np
from yaml import safe_dump
from piglot.solver import read_solver
from piglot.parameter import read_parameters
from piglot.utils.assorted import change_cwd
from piglot.utils.yaml_parser import parse_solver_file


def parse_args():
    """Parse command line arguments of the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    # Global argument parser settings
    parser = argparse.ArgumentParser(
        prog='piglot-solve',
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
    parser.add_argument(
        '--wait_reply',
        action='store_true',
        help='Wait for a reply from the user before exiting.',
    )

    return parser.parse_args()


def main():
    """Entry point for piglot-solve."""
    args = parse_args()
    config_path = args.config
    config = parse_solver_file(config_path)

    with change_cwd(os.path.dirname(config_path)):

        # Generate a temporary directory for the output
        with TemporaryDirectory() as output_dir:
            # Build output directory with a copy of the configuration file
            with open(os.path.join(output_dir, "config"), 'w', encoding='utf8') as file:
                safe_dump(config, file)

            # Build piglot problem
            parameters = read_parameters(config)
            solver = read_solver(config["solver"], parameters, output_dir)

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

            # Run the solver with the given parameters
            solver.prepare()
            results = solver.solve(param_values, False)

            # Store the output results
            print(f"Output results for {config_path}: {len(results)} results found")
            result_path = os.path.join(output_dir, "results")
            os.makedirs(result_path, exist_ok=True)
            for name, result in results.items():
                output = os.path.join(result_path, f"{name}.npz")
                result.write(output)
                print(f"{name}: {output}")
            print(f"{len(results)} results dumped")

            # If requested, wait for a reply from the user before exiting
            if args.wait_reply:
                input("Press Enter to exit...")


if __name__ == '__main__':
    main()
