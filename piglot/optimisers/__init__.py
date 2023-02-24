"""Module for optimisers."""

from piglot.optimisers.aoa import AOA
from piglot.optimisers.bayes_skopt import BayesSkopt
from piglot.optimisers.bayesian import Bayesian
from piglot.optimisers.direct import DIRECT
from piglot.optimisers.ga import GA
from piglot.optimisers.lipo_opt import LIPO
from piglot.optimisers.prs_spsa import PRS_SPSA
from piglot.optimisers.pso import PSO
from piglot.optimisers.random_search import PureRandomSearch
from piglot.optimisers.spsa_adam import SPSA_Adam
from piglot.optimisers.spsa import SPSA


def names():
    """Return list with the names of the available optimisers.

    Returns
    -------
    list[str]
        Names of the available optimisers
    """
    return [
        'aoa',
        'bayesian',
        'bayes_skopt',
        'direct',
        'ga',
        'lipo',
        'hybrid',
        'pso',
        'random',
        'spsa-adam',
        'spsa',
    ]
