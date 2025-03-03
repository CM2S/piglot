"""Module with utilities for transforming responses under compositions."""
from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import torch


class FlattenUtility(ABC):
    """Utility for flattening a response into a fixed-size vector (with gradients)."""

    @abstractmethod
    def length(self) -> int:
        """Return the length of the flattened vector.

        Returns
        -------
        int
            Length of the flattened vector.
        """

    @abstractmethod
    def flatten_torch(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Flatten a response into a single vector (with gradients).

        Parameters
        ----------
        time : torch.Tensor
            Time points of the response.
        data : torch.Tensor
            Data points of the response.

        Returns
        -------
        torch.Tensor
            Flattened responses.
        """

    @abstractmethod
    def unflatten_torch(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unflatten a vector containing a response (with gradients).

        Parameters
        ----------
        data : torch.Tensor
            Flattened responses.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            List of responses.
        """

    def flatten(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Flatten a response into a single vector.

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.

        Returns
        -------
        np.ndarray
            Flattened responses.
        """
        return self.flatten_torch(torch.from_numpy(time), torch.from_numpy(data)).numpy(force=True)

    def unflatten(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unflatten a vector containing a response.

        Parameters
        ----------
        data : np.ndarray
            Flattened responses.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            List of responses.
        """
        time, values = self.unflatten_torch(torch.from_numpy(data))
        return time.numpy(force=True), values.numpy(force=True)


class FixedFlatteningUtility(FlattenUtility):
    """Response flattening utility for fixed time grids."""

    def __init__(self, time_grid: np.ndarray):
        self.time_grid = time_grid

    def length(self) -> int:
        """Return the length of the flattened vector.

        Returns
        -------
        int
            Length of the flattened vector.
        """
        return len(self.time_grid)

    def flatten_torch(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Flatten a response into a single vector (with gradients).

        Parameters
        ----------
        time : torch.Tensor
            Time points of the response.
        data : torch.Tensor
            Data points of the response.

        Returns
        -------
        torch.Tensor
            Flattened responses.
        """
        if time.shape[-1] != len(self.time_grid):
            raise ValueError("Time grid does not match the expected length.")
        if time.shape != data.shape:
            raise ValueError("Mismatched time and data shapes are not supported.")
        return data

    def unflatten_torch(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unflatten a vector containing a response (with gradients).

        Parameters
        ----------
        data : torch.Tensor
            Flattened responses.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            List of responses.
        """
        return torch.from_numpy(self.time_grid).expand_as(data), data


class EndpointFlattenUtility(FlattenUtility):
    """Response flattening utility based on the time endpoints of the response."""

    def __init__(self, n_points: int):
        self.n_points = n_points

    def length(self) -> int:
        """Return the length of the flattened vector.

        Returns
        -------
        int
            Length of the flattened vector.
        """
        return self.n_points + 2

    def flatten(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Flatten a response into a single vector.

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.

        Returns
        -------
        np.ndarray
            Flattened responses.
        """
        # Sanitise input shape
        if time.shape != data.shape:
            raise ValueError("Mismatched time and data shapes are not supported.")
        bounds = np.array([np.min(time), np.max(time)])
        grid = np.linspace(bounds[0], bounds[1], self.n_points)
        response = np.interp(grid, time, data)
        return np.concatenate([response, bounds], axis=-1)

    def flatten_torch(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Flatten a response into a single vector (with gradients).

        Parameters
        ----------
        time : torch.Tensor
            Time points of the response.
        data : torch.Tensor
            Data points of the response.

        Returns
        -------
        torch.Tensor
            Flattened responses.
        """
        raise NotImplementedError("Flattening with gradients is not yet supported.")

    def unflatten_torch(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unflatten a vector containing a response (with gradients).

        Parameters
        ----------
        data : torch.Tensor
            Flattened responses.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            List of responses.
        """
        data_gridless = data[..., :-2]
        lbounds = data[..., -2].unsqueeze(-1).expand_as(data_gridless)
        ubounds = data[..., -1].unsqueeze(-1).expand_as(data_gridless)
        reg_grid = torch.linspace(0, 1, self.n_points, device=data.device).expand_as(data_gridless)
        grid = lbounds + reg_grid * (ubounds - lbounds)
        return grid, data_gridless


class ConcatUtility:
    """Utility for concatenating a set of responses (with gradients)."""

    def __init__(self, lengths: List[int]):
        self.lengths = lengths
        self.indices = np.cumsum([0] + lengths)

    def concat_torch(self, responses: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate a list of responses (with gradients).

        Parameters
        ----------
        responses : List[torch.Tensor]
            List of responses.

        Returns
        -------
        torch.Tensor
            Flattened responses.
        """
        return torch.cat(responses, dim=-1)

    def concat_covar_torch(self, covars: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate a list of covariance matrices (with gradients).

        Parameters
        ----------
        covars : List[torch.Tensor]
            List of covariance matrices.

        Returns
        -------
        torch.Tensor
            Flattened covariance matrices.
        """
        return torch.block_diag(*covars)

    def split_torch(self, data: torch.Tensor) -> List[torch.Tensor]:
        """Split a vector containing a set of responses with gradients.

        Parameters
        ----------
        data : torch.Tensor
            Flattened data.

        Returns
        -------
        List[torch.Tensor]
            List of split responses (with gradients).
        """
        return [data[..., self.indices[i]:self.indices[i+1]] for i in range(len(self.lengths))]

    def concat(self, responses: List[np.ndarray]) -> np.ndarray:
        """Concatenate a list of responses.

        Parameters
        ----------
        responses : List[np.ndarray]
            List of responses.

        Returns
        -------
        np.ndarray
            Flattened responses.
        """
        return self.concat_torch([torch.from_numpy(res) for res in responses]).numpy(force=True)

    def concat_covar(self, covars: List[np.ndarray]) -> np.ndarray:
        """Concatenate a list of covariance matrices.

        Parameters
        ----------
        covars : List[np.ndarray]
            List of covariance matrices.

        Returns
        -------
        np.ndarray
            Flattened covariance matrices.
        """
        return self.concat_covar_torch([torch.from_numpy(cov) for cov in covars]).numpy(force=True)

    def split(self, data: np.ndarray) -> List[np.ndarray]:
        """Split a vector containing a set of responses.

        Parameters
        ----------
        data : np.ndarray
            Flattened data.

        Returns
        -------
        List[np.ndarray]
            List of split responses.
        """
        return [res.numpy(force=True) for res in self.split_torch(torch.from_numpy(data))]
