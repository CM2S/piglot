"""Dataset classes for optimising with BoTorch."""
from __future__ import annotations
from typing import Tuple, Type, TypeVar, Union
import copy
import numpy as np
import torch


T = TypeVar('T', bound='BayesDataset')


class Standardiser:
    """Standardisation transformation."""

    def __init__(self, std_tol: float = 1e-6) -> None:
        self.mean: torch.Tensor = None
        self.stds: torch.Tensor = None
        self.mask: torch.Tensor = None
        self.inv_mask: torch.Tensor = None
        self.num_components: int = None
        self.std_tol = std_tol

    def fit(self, data: torch.Tensor) -> None:
        """Fit the standardisation transformation to the given data.

        Parameters
        ----------
        data : torch.Tensor
            Data to fit the standardisation transformation to.
        """
        self.mean = torch.mean(data, dim=-2)
        self.stds = torch.std(data, dim=-2) if data.shape[-2] > 1 else torch.zeros_like(self.mean)
        y_abs_avg = torch.mean(torch.abs(data), dim=-2)
        self.mask = torch.abs(self.stds / y_abs_avg) > self.std_tol
        self.inv_mask = ~self.mask  # pylint: disable=invalid-unary-operand-type
        self.num_components = torch.count_nonzero(self.mask)

    def transform(
        self,
        values: torch.Tensor,
        covariances: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Standardise data.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        covariances : torch.Tensor | None
            Variances to transform, if any.

        Returns
        -------
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            Transformed values and covariances (if any).
        """
        # Are all observed points equal?
        if torch.all(self.inv_mask):
            raise ValueError("All observed points are equal!.")
        means = (values[:, self.mask] - self.mean[self.mask]) / self.stds[self.mask]
        if covariances is None:
            return means
        covariances = covariances[:, :, self.mask][:, self.mask, :]
        scale_matrix = torch.diag(1 / self.stds[self.mask])
        return means, scale_matrix @ covariances @ scale_matrix.T

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
        expanded_flat[:, self.inv_mask] = self.mean[self.inv_mask]
        # Note: we are using a view, so the expanded tensor is already modified
        return expanded

    def to(self, device: str) -> Standardiser:
        """Move the standardiser to a given device.

        Parameters
        ----------
        device : str
            Device to move the standardiser to.

        Returns
        -------
        Standardiser
            The standardiser in the new device.
        """
        new_standardiser = copy.deepcopy(self)
        if self.mean is not None:
            new_standardiser.mean = self.mean.to(device)
            new_standardiser.stds = self.stds.to(device)
            new_standardiser.mask = self.mask.to(device)
        return new_standardiser


class PCA:
    """Principal Component Analysis transformation."""

    def __init__(self, variance: float) -> None:
        self.variance = variance
        self.standardiser = Standardiser()
        self.num_components: int = None
        self.transformation: torch.Tensor = None

    def fit(self, data: torch.Tensor, covariances: torch.Tensor) -> None:
        """Fit the PCA transformation to the given data.

        Parameters
        ----------
        data : torch.Tensor
            Data to fit the PCA transformation to.
        covariances : torch.Tensor
            Covariances of the data to fit the PCA transformation to.
        """
        # Standardise data
        self.standardiser.fit(data)
        data_std, covariances_std = self.standardiser.transform(data, covariances)
        # Compute the joint covariance matrix
        # Refer to Eq.(4) of https://doi.org/10.1109/TVCG.2019.2934812 for this
        cov = torch.cov(data_std.T) + torch.mean(covariances_std, dim=0)
        # Compute eigenvalues and vectors of the covariance matrix and sort by decreasing variance
        vals, vecs = torch.linalg.eigh(cov)  # pylint: disable=not-callable
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
        covariances: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Transform data to the latent space.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        covariances : torch.Tensor | None
            Variances to transform, if any.

        Returns
        -------
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            Transformed values and covariances (if any).
        """
        if covariances is None:
            return self.standardiser.transform(values) @ self.transformation
        values, covariances = self.standardiser.transform(values, covariances)
        return (
            values @ self.transformation,
            self.transformation.T @ covariances @ self.transformation,
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
        return self.standardiser.untransform(data @ self.transformation.T)

    def to(self, device: str) -> PCA:
        """Move the PCA to a given device.

        Parameters
        ----------
        device : str
            Device to move the PCA to.

        Returns
        -------
        PCA
            The PCA in the new device.
        """
        new_pca = copy.deepcopy(self)
        new_pca.standardiser = self.standardiser.to(device)
        if self.transformation is not None:
            new_pca.transformation = self.transformation.to(device)
        return new_pca


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
        self.covariances = torch.empty((0, n_outputs, n_outputs), dtype=dtype, device=device)
        self.objectives = torch.empty((0,), dtype=dtype, device=device)
        self.export = export
        self.std_tol = std_tol
        self.pca_variance = pca_variance
        self.standardiser = Standardiser(std_tol)
        self.pca = PCA(pca_variance) if pca_variance is not None else None
        self.device = device

    def update_stats(self) -> None:
        """Update the statistics of the dataset."""
        values = torch.clone(self.values)
        # Update PCA transformation
        if self.pca_variance is not None and values.shape[0] > 1:
            self.pca.fit(values, self.covariances)
            values = self.pca.transform(self.values)
        # Update standardisation transformation
        self.standardiser.fit(values)

    def numel_latent_space(self) -> int:
        """Return the number of components of the latent space.

        Returns
        -------
        int
            Number of components of the latent space.
        """
        if self.pca_variance is not None and self.values.shape[0] > 1:
            return self.pca.num_components
        return self.standardiser.num_components

    @classmethod
    def load(cls: Type[T], filename: str) -> T:
        """Load data from a given input file.

        Parameters
        ----------
        filename : str
            Path to the file to read from.

        Returns
        -------
        BayesDataset
            Dataset loaded from the file.
        """
        data: T = torch.load(filename, weights_only=False)
        # Rebuild standardiser and PCA
        data.standardiser = Standardiser(data.std_tol)
        data.pca = PCA(data.pca_variance) if data.pca_variance is not None else None
        data.update_stats()
        # Send to the correct device before returning
        return data.to(data.device)

    def save(self, filename: str) -> None:
        """Save all dataset data to a file.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        torch.save(self, filename)

    def push(
        self,
        params: np.ndarray,
        results: np.ndarray,
        covariance: np.ndarray,
        objective: Union[float, None],
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
        torch_covariance = torch.tensor(covariance, dtype=self.dtype, device=self.device)
        torch_variance = torch.diagonal(torch_covariance)
        self.params = torch.cat([self.params, torch_params.unsqueeze(0)], dim=0)
        self.values = torch.cat([self.values, torch_value.unsqueeze(0)], dim=0)
        self.variances = torch.cat([self.variances, torch_variance.unsqueeze(0)], dim=0)
        self.covariances = torch.cat([self.covariances, torch_covariance.unsqueeze(0)], dim=0)
        if objective is not None:
            torch_objective = torch.tensor([objective], dtype=self.dtype, device=self.device)
            self.objectives = torch.cat([self.objectives, torch_objective], dim=0)
        self.n_points += 1
        self.update_stats()
        # Update the dataset file after every push
        if self.export is not None:
            self.save(self.export)

    def transform_outcomes(
        self,
        values: torch.Tensor = None,
        covariances: torch.Tensor = None,
        diagonalise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform outcomes to the latent standardised space.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        covariances : torch.Tensor
            Variances to transform.
        diagonalise : bool
            Whether to diagonalise the covariance matrix (default: True).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed values and variances.
        """
        if (values is None) != (covariances is None):
            raise ValueError("Values and covariances must be provided together.")
        # When computing the outcomes from the dataset, use the stored values and covariances
        if values is None:
            values = self.values
            covariances = self.covariances
        # Transform outcomes
        if self.pca is not None:
            values, covariances = self.pca.transform(values, covariances)
        values, covariances = self.standardiser.transform(values, covariances)
        # Diagonalise the covariance matrix
        return values, torch.diagonal(covariances, dim1=-2, dim2=-1) if diagonalise else covariances

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

    def min(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the minimum objective value of the dataset.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Parameters and objective value for the minimum point.
        """
        idx = torch.argmin(self.objectives)
        return self.params[idx, :].cpu().numpy(), self.objectives[idx].cpu().numpy()

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
        new_dataset.covariances = self.covariances.to(device)
        new_dataset.objectives = self.objectives.to(device)
        new_dataset.standardiser = self.standardiser.to(device)
        if self.pca is not None:
            new_dataset.pca = self.pca.to(device)
        new_dataset.device = device
        return new_dataset
