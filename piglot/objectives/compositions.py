"""Module for compositions of objectives."""
import numpy as np
import torch
from piglot.objective import Composition


class MSEComposition(Composition):
    """Mean squared error outer composite function with gradients."""

    def composition(self, inner: np.ndarray) -> float:
        """Compute the MSE outer function of the composition.

        Parameters
        ----------
        inner : np.ndarray
            Return value from the inner function.

        Returns
        -------
        float
            Composition result.
        """
        return np.mean(np.square(inner), axis=-1)

    def composition_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Compute the MSE outer function of the composition with gradients.

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function.

        Returns
        -------
        torch.Tensor
            Composition result.
        """
        return torch.mean(torch.square(inner), dim=-1)


class MAEComposition(Composition):
    """Mean absolute error outer composite function with gradients."""

    def composition(self, inner: np.ndarray) -> float:
        """Compute the CAE outer function of the composition.

        Parameters
        ----------
        inner : np.ndarray
            Return value from the inner function.

        Returns
        -------
        float
            Composition result.
        """
        return np.mean(np.abs(inner), axis=-1)

    def composition_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Compute the CAE outer function of the composition with gradients.

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function.

        Returns
        -------
        torch.Tensor
            Composition result.
        """
        return torch.mean(torch.abs(inner), dim=-1)


AVAILABLE_COMPOSITIONS = {
    'mse': MSEComposition,
    'mae': MAEComposition,
}
