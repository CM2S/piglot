"""Dataset classes for optimising with BoTorch."""
from __future__ import annotations
from typing import Tuple, Callable
import copy
import numpy as np
import torch


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
        self.outcome_mask = torch.empty(n_outputs, dtype=torch.bool, device=device)
        self.outcome_means = torch.empty(n_outputs, dtype=dtype, device=device)
        self.outcome_stds = torch.empty(n_outputs, dtype=dtype, device=device)
        self.export = export
        self.std_tol = std_tol
        self.pca_variance = pca_variance
        self.pca_transformation = None
        self.pca_components = None
        self.pca_means = None
        self.device = device

    def __update_stats(self) -> None:
        """Update the statistics of the dataset."""
        values = torch.clone(self.values)
        # Update PCA transformation
        if self.pca_variance is not None and values.shape[0] > 1:
            # De-mean data and compute eigenvalues and vectors of the covariance matrix
            data = values - torch.mean(values, dim=-2)
            vals, vecs = torch.linalg.eigh(torch.cov(data.T))
            # Sory by decreasing variance
            idx = torch.argsort(vals, descending=True)
            vals = vals[idx]
            vecs = vecs[:, idx]
            # Compute cummulative variances and select the number of components
            variances = torch.cumsum(vals / vals.sum(), 0)
            n_components = torch.searchsorted(variances, self.pca_variance) + 2
            # Update the transformation matrix and compute transformed data
            self.pca_components = min(n_components, data.shape[1])
            self.pca_transformation = vecs[:, :self.pca_components]
            values = torch.matmul(values, self.pca_transformation)
        # Get observation statistics
        self.outcome_means = torch.mean(values, dim=-2)
        self.outcome_stds = torch.std(values, dim=-2)
        # Build mask of points with near-null variance
        y_abs_avg = torch.mean(torch.abs(values), dim=-2)
        self.outcome_mask = torch.abs(self.outcome_stds / y_abs_avg) > self.std_tol

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
        if not torch.any(self.outcome_mask):
            raise RuntimeError("All observed points are equal: add more initial samples")
        if self.pca_transformation is not None:
            values = torch.matmul(values, self.pca_transformation)
            variances = torch.matmul(variances, self.pca_transformation)
        means = self.outcome_means[self.outcome_mask]
        stds = self.outcome_stds[self.outcome_mask]
        std_values = (values[:, self.outcome_mask] - means) / stds
        std_variances = variances[:, self.outcome_mask] / stds
        return std_values, std_variances

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
        # Destandardise the values
        means = self.outcome_means[self.outcome_mask]
        stds = self.outcome_stds[self.outcome_mask]
        values = values * stds + means
        # Infer the shape of the expanded tensor: only modify the last dimension
        new_shape = list(values.shape)
        new_shape[-1] = self.outcome_means.numel()
        expanded = torch.empty(new_shape, dtype=self.dtype, device=self.device)
        # Fill the tensor using a 2D view:
        # i) modelled outcomes are directly inserted
        # ii) missing outcomes are filled with the average of the observed ones
        expanded_flat = expanded.view(-1, new_shape[-1])
        expanded_flat[:, self.outcome_mask] = values.view(-1, values.shape[-1])
        expanded_flat[:, ~self.outcome_mask] = self.outcome_means[~self.outcome_mask]
        # Note: we are using a view, so the expanded tensor is already modified
        if self.pca_transformation is None:
            return expanded
        # De-PCA values
        return torch.matmul(expanded, self.pca_transformation.T)

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
