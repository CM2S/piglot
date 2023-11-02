"""Module for output fields from Links solver."""
from typing import Dict, Any
from piglot.solver.solver import OutputField



def links_fields_reader(config: Dict[str, Any]) -> OutputField:
    """Read the output fields for the Links solver.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    OutputField
        Output field to use for this problem.
    """
    # TODO: Implement this