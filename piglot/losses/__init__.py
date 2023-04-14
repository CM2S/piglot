"""Main loss module."""

from piglot.losses.loss import MSE, MAE, NMSE, RMSE, NMAE, RNMSE, RNMAE, RMAE, MixedLoss, LogNMSE, VectorLoss
from piglot.losses.loss import Range, Maximum, Minimum, Weightings, Slope
from piglot.losses.weights import UniformWeights, MultiModalWeights


def names():
    """Return list with the names of the available loss functions.

    Returns
    -------
    list[str]
        Names of the available loss functions
    """
    return [
        'mse',
        'lognmse',
        'nmse',
        'rmse',
        'rnmse',
        'mae',
        'nmae',
        'rmae',
        'rnmae',
        'vector',
    ]
