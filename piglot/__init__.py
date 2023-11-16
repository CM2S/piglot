"""Main piglot module."""

import piglot.optimisers
import piglot.losses

def optimiser(name, *args, **kwargs):
    """Returns one of the optimisers in piglot.

    Parameters
    ----------
    name : name
        Name of the method to use. Options:
        - `aoa`: Arithmetic Optimisation Algorithm in `piglot.optimisers.aoa`;
        - `bayesian`: Bayesian optimisation in `piglot.optimisers.bayesian`;
        - `bayes_skopt`: Bayesian optimisation in `piglot.optimisers.bayes_skopt`;
        - `direct`: DIRECT in `piglot.optimisers.direct`;
        - `ga`: Genetic Algorithm in `piglot.optimisers.ga`;
        - `lipo`: LIPO in `piglot.optimisers.lipo_opt`;
        - `pso`: Particle swarm optimisation in `piglot.optimisers.pso`;
        - `random`: Random search in `piglot.optimisers.random_search`;
        - `spsa-adam`: SPSA with Adam gradient update in `piglot.optimisers.spsa_adam`;
        - `spsa`: SPSA method in `piglot.optimisers.spsa`;

    Returns
    -------
    Optimiser
        Optimiser instance.
    """
    if name not in piglot.optimisers.names():
        raise NameError(f"Method {name} unknown! Check available methods.")
    if name == 'aoa':
        return piglot.optimisers.AOA(*args, **kwargs)
    if name == 'bayesian':
        return piglot.optimisers.Bayesian(*args, **kwargs)
    if name == 'bayes_skopt':
        return piglot.optimisers.BayesSkopt(*args, **kwargs)
    if name == 'botorch':
        return piglot.optimisers.BayesianBoTorch(*args, **kwargs)
    if name == 'botorch_stochastic':
        return piglot.optimisers.BayesianBoTorchStochastic(*args, **kwargs)
    if name == 'botorch_cf':
        return piglot.optimisers.BayesianBoTorchComposite(*args, **kwargs)
    if name == 'botorch_mf':
        return piglot.optimisers.BayesianBoTorchMF(*args, **kwargs)
    if name == 'botorch_mf_cf':
        return piglot.optimisers.BayesianBoTorchMultiFidelityComposite(*args, **kwargs)
    if name == 'direct':
        return piglot.optimisers.DIRECT(*args, **kwargs)
    if name == 'ga':
        return piglot.optimisers.GA(*args, **kwargs)
    if name == 'lipo':
        return piglot.optimisers.LIPO(*args, **kwargs)
    if name == 'pso':
        return piglot.optimisers.PSO(*args, **kwargs)
    if name == 'random':
        return piglot.optimisers.PureRandomSearch(*args, **kwargs)
    if name == 'spsa-adam':
        return piglot.optimisers.SPSA_Adam(*args, **kwargs)
    if name == 'spsa-adam_cf':
        return piglot.optimisers.SPSA_Adam_Composite(*args, **kwargs)
    if name == 'spsa':
        return piglot.optimisers.SPSA(*args, **kwargs)
    raise NameError("Internal error: loss function not implemented!")


def loss(name, *args, **kwargs):
    """Returns one of the losses in piglot.

    Parameters
    ----------
    name : name
        Name of the loss to use. Options:
        - `mse`: Mean squared error
        - `nmse`: Normalised mean squared error
        - `rmse`: Root mean squared error
        - `rnmse` or `nrmse`: Root normalised mean squared error
        - `mae`: Mean absolute error
        - `nmae`: Normalised mean absolute error
        - `rmae`: Root mean absolute error
        - `rnmae` or `nrmae`: Root normalised mean absolute error

    Returns
    -------
    Loss
        Loss instance.
    """
    if name not in piglot.losses.names():
        raise NameError(f"Loss {name} unknown! Check available losses.")
    if name == 'mse':
        return piglot.losses.MSE(*args, **kwargs)
    if name == 'lognmse':
        return piglot.losses.LogNMSE(*args, **kwargs)
    if name == 'nmse':
        return piglot.losses.NMSE(*args, **kwargs)
    if name == 'rmse':
        return piglot.losses.RMSE(*args, **kwargs)
    if name in ('rnmse', 'nrmse'):
        return piglot.losses.RNMSE(*args, **kwargs)
    if name == 'mae':
        return piglot.losses.MAE(*args, **kwargs)
    if name == 'nmae':
        return piglot.losses.NMAE(*args, **kwargs)
    if name == 'rmae':
        return piglot.losses.RMAE(*args, **kwargs)
    if name in ('rnmae', 'nrmae'):
        return piglot.losses.RNMAE(*args, **kwargs)
    if name == 'vector':
        return piglot.losses.VectorLoss(*args, **kwargs)
    if name == 'scalar_vector':
        return piglot.losses.ScalarVectorLoss(*args, **kwargs)
    raise NameError("Internal error: loss function not implemented!")
