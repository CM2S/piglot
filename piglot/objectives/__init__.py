"""Interface for objectives."""
from typing import Any, Dict, Type
from piglot.parameter import ParameterSet
from piglot.objective import Objective
from piglot.objectives.analytical import AnalyticalObjective, AnalyticalMultiObjective
from piglot.objectives.synthetic import SyntheticObjective
from piglot.objectives.fitting import ResponseFittingObjective
from piglot.objectives.design import ResponseDesignObjective


AVAILABLE_OBJECTIVES: Dict[str, Type[Objective]] = {
    'analytical': AnalyticalObjective,
    'analytical_mo': AnalyticalMultiObjective,
    'test_function': SyntheticObjective,
    'fitting': ResponseFittingObjective,
    'design': ResponseDesignObjective,
}


def read_objective(config: Dict[str, Any], parameters: ParameterSet, output_dir: str) -> Objective:
    """Read the objective from the configuration dictionary.

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
    Objective
        Objective to optimise for.
    """
    # Read the objective name (and pop it from the dictionary)
    if 'name' not in config:
        raise ValueError("Missing name for objective.")
    name = config.pop('name')
    # Delegate to the objective reader
    if name not in AVAILABLE_OBJECTIVES:
        raise ValueError(f"Unknown objective '{name}'.")
    return AVAILABLE_OBJECTIVES[name].read(config, parameters, output_dir)
