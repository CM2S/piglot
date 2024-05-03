"""Dataset classes for optimising with BoTorch."""
from __future__ import annotations
from typing import Tuple, Callable
import copy
import numpy as np
import torch

class Standardiser:
    """Standardisation transformation."""

    def __init__(self, std_tol: float = 1e-6) -> None:
        self.mean: torch.Tensor = None
        self.stds: torch.Tensor = None
        self.mask: torch.Tensor = None
        self.std_tol = std_tol

    def fit(self, data: torch.Tensor) -> None:
        """Fit the standardisation transformation to the given data.

        Parameters
        ----------
        data : torch.Tensor
            Data to fit the standardisation transformation to.
        """
        self.mean = torch.mean(data, dim=-2)
        self.stds = torch.std(data, dim=-2)
        y_abs_avg = torch.mean(torch.abs(data), dim=-2)
        self.mask = torch.abs(self.stds / y_abs_avg) > self.std_tol

    def transform(
        self,
        values: torch.Tensor,
        variances: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Standardise data.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        variances : torch.Tensor | None
            Variances to transform, if any.

        Returns
        -------
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            Transformed values and variances (if any).
        """
        means = (values[:, self.mask] - self.mean[self.mask]) / self.stds[self.mask]
        if variances is None:
            return means
        stds = variances[:, self.mask] / self.stds[self.mask]
        return means, stds

    def untransform(self, data: torch.Tensor) -> torch.Tensor:
        """Unstandardise data.

        Parameters
        ----------
        data : torch.Tensor
            Data to unstandardise.

        Returns
        -------
        torch.Tensor
            Unstandardised data.
        """
        values = data * self.stds[self.mask] + self.mean[self.mask]
        # Nothing more to do if no outputs are suppressed
        if torch.all(self.mask):
            return values
        # Infer the shape of the expanded tensor: only modify the last dimension
        new_shape = list(values.shape)
        new_shape[-1] = self.mean.numel()
        expanded = torch.empty(new_shape, dtype=values.dtype, device=values.device)
        # Fill the tensor using a 2D view:
        # i) modelled outcomes are directly inserted
        # ii) missing outcomes are filled with the average of the observed ones
        expanded_flat = expanded.view(-1, new_shape[-1])
        expanded_flat[:, self.mask] = values.view(-1, values.shape[-1])
        expanded_flat[:, ~self.mask] = self.mean[~self.mask]
        # Note: we are using a view, so the expanded tensor is already modified
        return expanded


class PCA:
    """Principal Component Analysis transformation."""

    def __init__(self, variance: float) -> None:
        self.variance = variance
        self.standardiser = Standardiser()
        self.num_components: int = None
        self.transformation: torch.Tensor = None

    def fit(self, data: torch.Tensor) -> None:
        """Fit the PCA transformation to the given data.

        Parameters
        ----------
        data : torch.Tensor
            Data to fit the PCA transformation to.
        """
        # Standardise data
        self.standardiser.fit(data)
        data_std: torch.Tensor = self.standardiser.transform(data)
        # Compute eigenvalues and vectors of the covariance matrix and sort by decreasing variance
        vals, vecs = torch.linalg.eigh(torch.cov(data_std.T))
        idx = torch.argsort(vals, descending=True)
        vals = vals[idx]
        vecs = vecs[:, idx]
        # Select the number of components and update the transformation matrix
        vals_norm = vals / vals.sum()
        cumsum = torch.cumsum(vals_norm, dim=0)
        # self.num_components = torch.count_nonzero(vals_norm[vals_norm > 0])
        self.num_components = torch.searchsorted(cumsum, 1.0 - self.variance) + 1
        self.transformation = vecs[:, :self.num_components]

    def transform(
        self,
        values: torch.Tensor,
        variances: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Transform data to the latent space.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        variances : torch.Tensor | None
            Variances to transform, if any.

        Returns
        -------
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            Transformed values and variances (if any).
        """
        if variances is None:
            return torch.matmul(self.standardiser.transform(values), self.transformation)
        values, variances = self.standardiser.transform(values, variances)
        return (
            torch.matmul(values, self.transformation),
            torch.matmul(variances, self.transformation),
        )

    def untransform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform data back to the original space.

        Parameters
        ----------
        data : torch.Tensor
            Data to untransform.

        Returns
        -------
        torch.Tensor
            Untransformed data.
        """
        return self.standardiser.untransform(torch.matmul(data, self.transformation.T))


class BayesDataset:
    """Dataset class for multi-outcome data."""

    def __init__(
            self,
            n_dim: int,
            n_outputs: int,
            export: str = None,
            dtype: torch.dtype = torch.float64,
            std_tol: float = 1e-6,
            pca_variance: float = None,
            device: str = "cpu",
            ) -> None:
        self.dtype = dtype
        self.n_points = 0
        self.n_dim = n_dim
        self.n_outputs = n_outputs
        self.params = torch.empty((0, n_dim), dtype=dtype, device=device)
        self.values = torch.empty((0, n_outputs), dtype=dtype, device=device)
        self.variances = torch.empty((0, n_outputs), dtype=dtype, device=device)
        self.export = export
        self.std_tol = std_tol
        self.pca_variance = pca_variance
        self.standardiser = Standardiser(std_tol)
        self.pca = PCA(pca_variance) if pca_variance is not None else None
        self.device = device

    def __update_stats(self) -> None:
        """Update the statistics of the dataset."""
        values = torch.clone(self.values)
        # Update PCA transformation
        if self.pca_variance is not None and values.shape[0] > 1:
            self.pca.fit(values)
            values = self.pca.transform(self.values)
        # Update standardisation transformation
        self.standardiser.fit(values)

    def load(self, filename: str) -> None:
        """Load data from a given input file.

        Parameters
        ----------
        filename : str
            Path to the file to read from.
        """
        joint = torch.load(filename, map_location=self.device)
        idx1 = self.n_dim
        idx2 = self.n_dim + self.n_outputs
        for point in joint:
            point_np = point.numpy()
            self.push(point_np[:idx1], point_np[idx1:idx2], point_np[idx2:])
        self.__update_stats()

    def save(self, filename: str) -> None:
        """Save all dataset data to a file.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        # Build a joint tensor with all data
        joint = torch.cat([self.params, self.values, self.variances], dim=1)
        torch.save(joint, filename)

    def push(
        self,
        params: np.ndarray,
        results: np.ndarray,
        variance: np.ndarray,
    ) -> None:
        """Add a point to the dataset.

        Parameters
        ----------
        params : np.ndarray
            Parameter values for this observation.
        result : ObjectiveResult
            Result for this observation.
        """
        torch_params = torch.tensor(params, dtype=self.dtype, device=self.device)
        torch_value = torch.tensor(results, dtype=self.dtype, device=self.device)
        torch_variance = torch.tensor(variance, dtype=self.dtype, device=self.device)
        self.params = torch.cat([self.params, torch_params.unsqueeze(0)], dim=0)
        self.values = torch.cat([self.values, torch_value.unsqueeze(0)], dim=0)
        self.variances = torch.cat([self.variances, torch_variance.unsqueeze(0)], dim=0)
        self.n_points += 1
        self.__update_stats()
        # Update the dataset file after every push
        if self.export is not None:
            self.save(self.export)

    def transform_outcomes(
        self,
        values: torch.Tensor,
        variances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform outcomes to the latent standardised space.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        variances : torch.Tensor
            Variances to transform.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed values and variances.
        """
        if self.pca is not None:
            values, variances = self.pca.transform(values, variances)
        return self.standardiser.transform(values, variances)

    def untransform_outcomes(self, values: torch.Tensor) -> torch.Tensor:
        """Transform outcomes back to the original space.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.

        Returns
        -------
        torch.Tensor
            Transformed values.
        """
        values = self.standardiser.untransform(values)
        if self.pca is not None:
            values = self.pca.untransform(values)
        return values

    def min(
        self,
        transformer: Callable[[torch.Tensor, torch.Tensor], float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the minimum value of the dataset, according to a given transformation.

        Parameters
        ----------
        transformer : Callable[[torch.Tensor], float]
            Transformation to apply to the dataset.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Parameters and values for the minimum point.
        """
        values = [transformer(value, params) for params, value in zip(self.params, self.values)]
        idx = np.argmin(values)
        return self.params[idx, :].cpu().numpy(), self.values[idx, :].cpu().numpy()

    def to(self, device: str) -> BayesDataset:
        """Move the dataset to a given device.

        Parameters
        ----------
        device : str
            Device to move the dataset to.

        Returns
        -------
        BayesDataset
            The dataset in the new device.
        """
        new_dataset = copy.deepcopy(self)
        new_dataset.params = self.params.to(device)
        new_dataset.values = self.values.to(device)
        new_dataset.variances = self.variances.to(device)
        new_dataset.device = device
        return new_dataset
