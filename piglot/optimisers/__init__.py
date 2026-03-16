"""Module for optimisers."""
from typing import Any, Dict, Type
from piglot.objective import Objective
from piglot.optimiser import Optimiser
from piglot.optimisers.botorch.bayes import BayesianBoTorch
from piglot.optimisers.direct import DIRECT
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.optimisers.query import QueryOptimiser
from piglot.optimisers.random_search import PureRandomSearch
from piglot.optimisers.spsa_adam import SPSA_Adam
from piglot.optimisers.spsa import SPSA
from piglot.settings import Settings


AVAILABLE_OPTIMISERS: Dict[str, Type[Optimiser]] = {
    'bayesian': BayesianBoTorch,
    'bayes_skopt': BayesianBoTorch,
    'botorch': BayesianBoTorch,
    'direct': DIRECT,
    'generic': GenericOptimiser,
    'query': QueryOptimiser,
    'random': PureRandomSearch,
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
    if isinstance(config, str):
        # Parse the simple specification: optimiser name
        name = config
        kwargs = {}
    else:
        # Parse the detailed specification
        if 'name' not in config:
            raise RuntimeError("Missing optimiser name.")
        name = config.pop("name")
        kwargs = config
    # Build optimiser instance
    if name not in AVAILABLE_OPTIMISERS:
        raise RuntimeError(f"Unknown optimiser '{name}'.")
    return AVAILABLE_OPTIMISERS[name].read(kwargs, settings, objective)
