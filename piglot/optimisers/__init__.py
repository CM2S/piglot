"""Module for optimisers."""
from typing import Any, Dict, Type
from piglot.objective import Objective
from piglot.optimiser import Optimiser
from piglot.optimisers.botorch import BoTorchOptimiser
from piglot.optimisers.direct import DIRECT
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.optimisers.query import QueryOptimiser
from piglot.optimisers.random_search import RandomSearchOptimiser
from piglot.optimisers.scipy_optim import ScipyOptimiser
from piglot.optimisers.spsa_adam import SPSA_Adam
from piglot.optimisers.spsa import SPSA
from piglot.settings import Settings
from piglot.utils.assorted import convert_simple_spec


AVAILABLE_OPTIMISERS: Dict[str, Type[Optimiser]] = {
    'botorch': BoTorchOptimiser,
    'direct': DIRECT,
    'generic': GenericOptimiser,
    'query': QueryOptimiser,
    'random': RandomSearchOptimiser,
    'scipy': ScipyOptimiser,
    'spsa-adam': SPSA_Adam,
    'spsa': SPSA,
}


def read_optimiser(config: Dict[str, Any], settings: Settings, objective: Objective) -> Optimiser:
    """Read the optimiser from the configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    settings : Settings
        Global settings for the problem.
    objective : Objective
        Objective to optimise.

    Returns
    -------
    Optimiser
        Optimiser to use.
    """
    # If needed, convert simple specification to detailed format
    config = convert_simple_spec(config)
    # Mandatory fields
    if 'name' not in config:
        raise RuntimeError("Missing optimiser name.")
    name = config.pop("name")
    # Build optimiser instance
    if name not in AVAILABLE_OPTIMISERS:
        raise RuntimeError(f"Unknown optimiser '{name}'.")
    return AVAILABLE_OPTIMISERS[name].read(config, settings, objective)
