"""Module for surrogate models."""
from typing import Union
import numpy as np
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize


def get_model(
        x_data: np.ndarray,
        y_data: np.ndarray,
        var_data: np.ndarray = None,
        noisy: bool = False,
        ) -> Union[SingleTaskGP, FixedNoiseGP]:
    """Get a GP regression model for the current data.

    Parameters
    ----------
    x_data : np.ndarray
        Input features.
    y_data : np.ndarray
        Output outcomes.
    var_data : np.ndarray, optional
        Observation variance, by default None
    noisy : bool, optional
        Whether to use a noise-infering GPs or fixed noise ones, by default False

    Returns
    -------
    Union[SingleTaskGP, FixedNoiseGP]
        GP regression model.
    """
    x = torch.tensor(x_data, dtype=torch.float64)
    y = torch.tensor(y_data, dtype=torch.float64).unsqueeze(1)
    var = (
        torch.ones_like(y) * 1e-6 * torch.std(y, dim=0) if var_data is None else
        torch.tensor(var_data, dtype=torch.float64).unsqueeze(1)
    )
    gp = SingleTaskGP(
        x,
        y,
        train_Yvar=None if noisy else var,
        input_transform=Normalize(d=x.shape[-1]),
        outcome_transform=Standardize(m=y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp
