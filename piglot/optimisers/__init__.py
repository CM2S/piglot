"""Module for optimisers."""

from piglot.optimisers.aoa import AOA
from piglot.optimisers.bayes_skopt import BayesSkopt
from piglot.optimisers.bayesian import Bayesian
from piglot.optimisers.bayes_botorch import BayesianBoTorch
from piglot.optimisers.bayes_botorch_cf import BayesianBoTorchComposite
from piglot.optimisers.bayes_botorch_mf import BayesianBoTorchMF
from piglot.optimisers.bayes_botorch_stochastic import BayesianBoTorchStochastic
from piglot.optimisers.bayes_botorch_cf_stochastic import BayesianBoTorchCompositeStochastic
from piglot.optimisers.bayes_botorch_mfcf import BayesianBoTorchMultiFidelityComposite
from piglot.optimisers.direct import DIRECT
from piglot.optimisers.ga import GA
from piglot.optimisers.lipo_opt import LIPO
from piglot.optimisers.pso import PSO
from piglot.optimisers.random_search import PureRandomSearch
from piglot.optimisers.spsa_adam import SPSA_Adam
from piglot.optimisers.spsa_adam_cf import SPSA_Adam_Composite
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
        'botorch',
        'botorch_cf',
        'botorch_mf',
        'botorch_stochastic',
        'botorch_cf_stochastic',
        'botorch_mf_cf',
        'direct',
        'ga',
        'lipo',
        'hybrid',
        'pso',
        'random',
        'spsa-adam',
        'spsa-adam_cf',
        'spsa',
    ]
