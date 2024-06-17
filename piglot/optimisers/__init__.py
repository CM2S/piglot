"""Module for optimisers."""
from typing import Any, Dict, Type, Union
from piglot.objective import Objective
from piglot.optimiser import Optimiser
from piglot.optimisers.aoa import AOA
from piglot.optimisers.botorch.bayes import BayesianBoTorch
from piglot.optimisers.direct import DIRECT
from piglot.optimisers.ga import GA
from piglot.optimisers.lipo_opt import LIPO
from piglot.optimisers.query import QueryOptimiser
from piglot.optimisers.pso import PSO
from piglot.optimisers.random_search import PureRandomSearch
from piglot.optimisers.spsa_adam import SPSA_Adam
from piglot.optimisers.spsa import SPSA


AVAILABLE_OPTIMISERS: Dict[str, Type[Optimiser]] = {
    'aoa': AOA,
    'bayesian': BayesianBoTorch,
    'bayes_skopt': BayesianBoTorch,
    'botorch': BayesianBoTorch,
    'direct': DIRECT,
    'ga': GA,
    'lipo': LIPO,
    'pso': PSO,
    'query': QueryOptimiser,
    'random': PureRandomSearch,
    'spsa-adam': SPSA_Adam,
    'spsa': SPSA,
}


def str_to_numeric(data: str) -> Union[int, float, str]:
    """Tries to convert a string to a numeric value.

    Parameters
    ----------
    data : str
        String to convert.

    Returns
    -------
    Union[int, float, str]
        Converted value.
    """
    try:
        data = float(data)
    except (TypeError, ValueError):
        return data
    if int(data) == data:
        return int(data)
    return data


def read_optimiser(config: Dict[str, Any], objective: Objective) -> Optimiser:
    """Read the optimiser from the configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
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
        kwargs = {n: str_to_numeric(v) for n, v in config.items()}
    # Build optimiser instance
    if name not in AVAILABLE_OPTIMISERS:
        raise RuntimeError(f"Unknown optimiser '{name}'.")
    return AVAILABLE_OPTIMISERS[name](objective, **kwargs)
