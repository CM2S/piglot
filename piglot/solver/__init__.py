"""Interface for solvers."""
from typing import Any
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver
from piglot.solver.links.solver import LinksSolver
from piglot.solver.abaqus.solver import AbaqusSolver
from piglot.solver.curve.solver import CurveSolver
from piglot.solver.crate.solver import CrateSolver
from piglot.solver.script_solver import ScriptSolver


AVAILABLE_SOLVERS: dict[str, type[Solver]] = {
    'links': LinksSolver,
    'abaqus': AbaqusSolver,
    'curve': CurveSolver,
    'crate': CrateSolver,
    'script': ScriptSolver,
}


def get_available_solvers() -> dict[str, type[Solver]]:
    """Get the available solvers. This includes solvers that require lazy imports.

    Returns
    -------
    dict[str, type[Solver]]
        Dictionary mapping solver names to solver classes.
    """
    # Lazy imports to avoid circular dependencies
    from piglot.solver.multi_fidelity import MultiFidelitySolver

    # Build the dictionary of available solvers
    return {
        'multi_fidelity': MultiFidelitySolver,
        **AVAILABLE_SOLVERS,
    }


def read_solver(config: dict[str, Any], parameters: ParameterSet, output_dir: str) -> Solver:
    """Read the solver from the configuration dictionary.

    Parameters
    ----------
    config : dict[str, Any]
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
    # Read the solver name (and pop it from the dictionary)
    if 'name' not in config:
        raise ValueError("Missing name for solver.")
    name = config.pop('name')
    # Delegate to the solver reader
    solvers = get_available_solvers()
    if name not in solvers:
        raise ValueError(f"Unknown solver '{name}'.")
    return solvers[name].read(config, parameters, output_dir)