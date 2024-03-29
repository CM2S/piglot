"""Module for compositions of objectives."""
from typing import List, Tuple
import numpy as np
import torch
from piglot.objective import Composition


class MSEComposition(Composition):
    """Mean squared error outer composite function with gradients."""

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

    def composition_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Compute the MAE outer function of the composition with gradients.

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


class UnflattenUtility:
    """Utility for unflattening a set of responses (with gradients)."""

    def __init__(self, lengths: List[int]):
        self.lengths = lengths
        self.indices = np.cumsum([0] + lengths)

    def unflatten(self, data: np.ndarray) -> List[np.ndarray]:
        """Unflatten a vector containing a set of responses.

        Parameters
        ----------
        data : np.ndarray
            Flattened data.

        Returns
        -------
        List[np.ndarray]
            List of unflatten responses.
        """
        return [data[..., self.indices[i]:self.indices[i+1]] for i in range(len(self.lengths))]

    def unflatten_torch(self, data: torch.Tensor) -> List[torch.Tensor]:
        """Unflatten a vector containing a set of responses with gradients.

        Parameters
        ----------
        data : torch.Tensor
            Flattened data.

        Returns
        -------
        List[torch.Tensor]
            List of unflatten responses (with gradients).
        """
        return [data[..., self.indices[i]:self.indices[i+1]] for i in range(len(self.lengths))]


class FixedLengthTransformer:
    """Utility for transforming a response into a fixed-length format and back."""

    def __init__(self, n_points: int):
        self.n_points = n_points

    def length(self) -> int:
        """Return the length of the fixed-length response.

        Returns
        -------
        int
            Length of the fixed-length response.
        """
        return self.n_points + 2

    def transform(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Transform a response into a fixed-length format.

        Parameters
        ----------
        time : np.ndarray
            Time points to transform.
        data : np.ndarray
            Data points to transform.

        Returns
        -------
        np.ndarray
            Fixed-length response.
        """
        bounds = np.array([np.min(time), np.max(time)])
        grid = np.linspace(bounds[0], bounds[1], self.n_points)
        response = np.interp(grid, time, data)
        return np.concatenate([response, bounds], axis=-1)

    def untransform(self, response: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Untransform a fixed-length response into a response.

        Parameters
        ----------
        response : np.ndarray
            Fixed-length response.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time and data points.
        """
        grid, data = self.untransform_torch(torch.from_numpy(response))
        return grid.numpy(force=True), data.numpy(force=True)

    def untransform_torch(self, response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Untransform a fixed-length response into a response (with gradients).

        Parameters
        ----------
        response : torch.Tensor
            Fixed-length response.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Time and data points.
        """
        response_gridless = response[..., :-2]
        lbounds = response[..., -2].unsqueeze(-1).expand_as(response_gridless)
        ubounds = response[..., -1].unsqueeze(-1).expand_as(response_gridless)
        reg_grid = torch.linspace(0, 1, self.n_points).expand_as(response_gridless)
        grid = lbounds + reg_grid * (ubounds - lbounds)
        return grid, response_gridless
