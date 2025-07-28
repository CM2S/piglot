"""Interface for solvers."""
from typing import Any, Dict, Type
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver


def read_solver(config: Dict[str, Any], parameters: ParameterSet, output_dir: str) -> Solver:
    """Read the solver from the configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    parameters : ParameterSet
        Parameter set for this problem.
    output_dir : str
        Path to the output directory.

    Returns
    -------
    Solver
        Solver to use for this problem.
    """

    # Import the solvers dynamically
    # We need to import them here to avoid circular imports
    from piglot.solver.links.solver import LinksSolver  # pylint: disable=C0415
    from piglot.solver.abaqus.solver import AbaqusSolver  # pylint: disable=C0415
    from piglot.solver.curve.solver import CurveSolver  # pylint: disable=C0415
    from piglot.solver.crate.solver import CrateSolver  # pylint: disable=C0415
    from piglot.solver.script_solver import ScriptSolver  # pylint: disable=C0415
    from piglot.solver.remote import RemoteSolver  # pylint: disable=C0415

    AVAILABLE_SOLVERS: Dict[str, Type[Solver]] = {
        'links': LinksSolver,
        'abaqus': AbaqusSolver,
        'curve': CurveSolver,
        'crate': CrateSolver,
        'script': ScriptSolver,
        'remote': RemoteSolver,
    }

    # Read the solver name (and pop it from the dictionary)
    if 'name' not in config:
        raise ValueError("Missing name for solver.")
    name = config.pop('name')
    # Delegate to the solver reader
    if name not in AVAILABLE_SOLVERS:
        raise ValueError(f"Unknown solver '{name}'.")
    return AVAILABLE_SOLVERS[name].read(config, parameters, output_dir)
