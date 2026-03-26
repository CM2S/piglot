"""Module for utilities related to composition in piglot."""
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import block_diag
import torch


class ConcatUtility:
    """Utility for concatenating a set of latent spaces."""

    def __init__(self, lengths: list[int]):
        self.lengths = lengths
        self.indices = np.cumsum([0] + lengths)

    def length(self) -> int:
        """Return the total length of the concatenated latent spaces.

        Returns
        -------
        int
            Total length of the concatenated latent spaces.
        """
        return self.indices[-1]

    def concat(self, latent_spaces: list[np.ndarray]) -> np.ndarray:
        """Concatenate a list of latent spaces.

        Parameters
        ----------
        latent_spaces : list[np.ndarray]
            List of latent spaces.

        Returns
        -------
        np.ndarray
            Flattened latent spaces.
        """
        return np.concatenate(latent_spaces, axis=-1)

    def concat_covar(self, covars: list[np.ndarray]) -> np.ndarray:
        """Concatenate a list of covariance matrices for the latent spaces.

        Parameters
        ----------
        covars : list[np.ndarray]
            List of covariance matrices.

        Returns
        -------
        np.ndarray
            Flattened covariance matrices.
        """
        return block_diag(*covars)

    def split(self, data: torch.Tensor) -> list[torch.Tensor]:
        """Split a vector containing a set of latent spaces.

        Parameters
        ----------
        data : torch.Tensor
            Flattened data.

        Returns
        -------
        list[torch.Tensor]
            List of split latent spaces.
        """
        return [
            data[..., self.indices[i]:self.indices[i + 1]]
            for i in range(len(self.lengths))
        ]


class LatentTransformer(ABC):
    """Base class for latent space transformations."""

    @abstractmethod
    def length(self) -> int:
        """Return the length of the latent space vector.

        Returns
        -------
        int
            The length of the latent space vector.
        """

    @abstractmethod
    def latent_space(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Transform the data into the latent space.

        Parameters
        ----------
        time : np.ndarray
            Time points corresponding to the data.
        data : np.ndarray
            The data to be transformed.

        Returns
        -------
        np.ndarray
            The transformed latent space.
        """

    @abstractmethod
    def inverse_transform(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse transform from the latent space back to the original space.

        Parameters
        ----------
        latent : torch.Tensor
            The latent space vector to be inverse transformed.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The original time points and data corresponding to the latent vector.
        """


class FixedTimeLatentTransformer(LatentTransformer):
    """Latent transformer that assumes a fixed time grid."""

    def __init__(self, time: np.ndarray) -> None:
        self.time = time

    def length(self) -> int:
        """Return the length of the latent space vector.

        Returns
        -------
        int
            The length of the latent space vector.
        """
        return len(self.time)

    def latent_space(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Transform the data into the latent space.

        Parameters
        ----------
        time : np.ndarray
            Time points corresponding to the data.
        data : np.ndarray
            The data to be transformed.

        Returns
        -------
        np.ndarray
            The transformed latent space.
        """
        if time.shape[-1] != len(self.time):
            raise ValueError("Time grid does not match the expected length.")
        if time.shape != data.shape:
            raise ValueError("Mismatched time and data shapes are not supported.")
        return data

    def inverse_transform(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse transform from the latent space back to the original space.

        Parameters
        ----------
        latent : torch.Tensor
            The latent space vector to be inverse transformed.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The original time points and data corresponding to the latent vector.
        """
        return torch.from_numpy(self.time).expand_as(latent), latent


class EndpointLatentTransformer(LatentTransformer):
    """Latent transformer based on the time endpoints of the response."""

    def __init__(self, n_points: int) -> None:
        self.n_points = n_points

    def length(self) -> int:
        """Return the length of the latent space vector.

        Returns
        -------
        int
            The length of the latent space vector.
        """
        return self.n_points + 2

    def latent_space(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Transform the data into the latent space.

        Parameters
        ----------
        time : np.ndarray
            Time points corresponding to the data.
        data : np.ndarray
            The data to be transformed.

        Returns
        -------
        np.ndarray
            The transformed latent space.
        """
        # Sanitise input shape
        if time.shape != data.shape:
            raise ValueError("Mismatched time and data shapes are not supported.")
        bounds = np.array([np.min(time), np.max(time)])
        grid = np.linspace(bounds[0], bounds[1], self.n_points)
        response = np.interp(grid, time, data)
        return np.concatenate([response, bounds], axis=-1)

    def inverse_transform(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse transform from the latent space back to the original space.

        Parameters
        ----------
        latent : torch.Tensor
            The latent space vector to be inverse transformed.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The original time points and data corresponding to the latent vector.
        """
        data_gridless = latent[..., :-2]
        lbounds = latent[..., -2].unsqueeze(-1).expand_as(data_gridless)
        ubounds = latent[..., -1].unsqueeze(-1).expand_as(data_gridless)
        reg_grid = torch.linspace(
            0, 1, self.n_points, device=latent.device
        ).expand_as(data_gridless)
        grid = lbounds + reg_grid * (ubounds - lbounds)
        return grid, data_gridless
