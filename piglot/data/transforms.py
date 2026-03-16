"""Module for data transformations in surrogate models."""
from typing import Any, Tuple, Union, Optional, TypeVar, Type
from abc import ABC, abstractmethod
import torch
from piglot.utils.assorted import TorchContainer


T = TypeVar('T', bound='Transform')


class Transform(ABC, TorchContainer):
    """Abstract base class for data transformations."""

    def __init__(self, data: torch.Tensor, covariances: torch.Tensor) -> None:
        self.data = data
        self.covariances = covariances

    @abstractmethod
    def transform(
        self,
        values: torch.Tensor,
        covariances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform data.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        covariances : Optional[torch.Tensor]
            Variances to transform, if any.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed values and covariances (if any).
        """

    @abstractmethod
    def untransform(self, data: torch.Tensor) -> torch.Tensor:
        """Untransform data.

        Parameters
        ----------
        data : torch.Tensor
            Data to untransform.

        Returns
        -------
        torch.Tensor
            Untransformed data.
        """

    @abstractmethod
    def output_dim(self) -> int:
        """Get the output dimension after transformation.

        Returns
        -------
        int
            Output dimension.
        """


class Standardiser(Transform):
    """Standardisation transformation."""

    def __init__(
        self,
        data: torch.Tensor,
        covariances: torch.Tensor,
        std_tol: float = 1e-6,
        inject_noise: bool = False,
    ) -> None:
        super().__init__(data, covariances)
        self.mean = torch.mean(data, dim=-2)
        self.stds = torch.std(data, dim=-2) if data.shape[-2] > 1 else torch.zeros_like(self.mean)
        y_abs_avg = torch.mean(torch.abs(data), dim=-2)
        self.mask = torch.abs(self.stds / y_abs_avg) > std_tol
        self.inv_mask = ~self.mask  # pylint: disable=invalid-unary-operand-type
        self.num_components = torch.count_nonzero(self.mask)
        self.inject_noise = inject_noise
        if self.num_components == 0:
            raise ValueError("All observed points are equal!.")

    def transform(
        self,
        values: torch.Tensor,
        covariances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform data.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        covariances : Optional[torch.Tensor]
            Variances to transform, if any.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed values and covariances (if any).
        """
        means = (values[..., self.mask] - self.mean[self.mask]) / self.stds[self.mask]
        if covariances is None:
            return means, None
        covariances = covariances[..., :, self.mask][..., self.mask, :]
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
        expanded_flat = expanded.view(-1, new_shape[-1])
        # i) modelled outcomes are directly inserted
        expanded_flat[..., self.mask] = values.view(-1, values.shape[-1])
        # ii) missing outcomes are filled with either their average or samples with noise
        if self.inject_noise:
            expanded_flat[..., self.inv_mask] = (
                self.mean[self.inv_mask] + self.stds[self.inv_mask] * torch.randn(
                    expanded_flat[..., self.inv_mask].shape,
                    dtype=values.dtype,
                    device=values.device,
                )
            )
        else:
            expanded_flat[..., self.inv_mask] = self.mean[self.inv_mask]
        # Note: we are using a view, so the expanded tensor is already modified
        return expanded

    def output_dim(self) -> int:
        """Get the output dimension after transformation.

        Returns
        -------
        int
            Output dimension.
        """
        return self.num_components.item()


class PCA(Transform):
    """Principal Component Analysis transformation."""

    def __init__(
        self,
        data: torch.Tensor,
        covariances: torch.Tensor,
        variance: float = 1e-6,
        std_tol: float = 1e-6,
        inject_noise: bool = False,
    ) -> None:
        super().__init__(data, covariances)
        # Standardise data
        self.standardiser = Standardiser(
            data, covariances, std_tol=std_tol, inject_noise=inject_noise
        )
        if covariances is None:
            data_std, _ = self.standardiser.transform(data)
        else:
            data_std, covariances_std = self.standardiser.transform(data, covariances)
        # Compute the joint covariance matrix
        if covariances is None:
            cov = torch.cov(data_std.T)
        else:
            # Refer to Eq.(4) of https://doi.org/10.1109/TVCG.2019.2934812 for this
            cov = torch.cov(data_std.T) + torch.mean(covariances_std, dim=0)
        # Compute eigenvalues and vectors of the covariance matrix and sort by decreasing variance
        vals, vecs = torch.linalg.eigh(cov)  # pylint: disable=not-callable
        idx = torch.argsort(vals, descending=True)
        self.vals = vals[idx]
        self.vecs = vecs[:, idx]
        # Select the number of components and update the transformation matrix
        vals_norm = self.vals / self.vals.sum()
        cumsum = torch.cumsum(vals_norm, dim=0)
        self.num_components = torch.searchsorted(cumsum, 1.0 - variance) + 1
        self.transformation = self.vecs[:, :self.num_components]
        self.inject_noise = inject_noise

    def transform(
        self,
        values: torch.Tensor,
        covariances: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Transform data.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        covariances : Optional[torch.Tensor]
            Variances to transform, if any.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed values and covariances (if any).
        """
        values, covariances = self.standardiser.transform(values, covariances)
        if covariances is None:
            return values @ self.transformation, None
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
        if self.inject_noise:
            noise_shape = data.shape[:-1] + (self.transformation.shape[0],)
            samples = torch.randn(noise_shape, dtype=data.dtype, device=data.device)
            samples = samples * torch.sqrt(self.vals[self.num_components:])
            data = torch.cat([data, samples], dim=-1)
            return self.standardiser.untransform(data @ self.vecs.T)
        return self.standardiser.untransform(data @ self.transformation.T)

    def output_dim(self) -> int:
        """Get the output dimension after transformation.

        Returns
        -------
        int
            Output dimension.
        """
        return self.num_components.item()


class ChainTransform(Transform):
    """Chain multiple transformations together."""

    def __init__(
        self,
        data: torch.Tensor,
        covariances: torch.Tensor,
        transforms: list[Tuple[Type[Transform], dict[str, Any]]],
    ) -> None:
        super().__init__(data, covariances)
        self.transforms: list[Transform] = []
        for transform_cls, params in transforms:
            transform = transform_cls(data, covariances, **params)
            self.transforms.append(transform)
            # Update data and covariances for the next transform in the chain
            data, covariances = transform.transform(data, covariances)

    def transform(
        self,
        values: torch.Tensor,
        covariances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform data.

        Parameters
        ----------
        values : torch.Tensor
            Values to transform.
        covariances : Optional[torch.Tensor]
            Variances to transform, if any.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed values and covariances (if any).
        """
        for transform in self.transforms:
            values, covariances = transform.transform(values, covariances)
        return values, covariances

    def untransform(self, data: torch.Tensor) -> torch.Tensor:
        """Untransform data through the chain of transformations.

        Parameters
        ----------
        data : torch.Tensor
            Data to untransform.

        Returns
        -------
        torch.Tensor
            Untransformed data.
        """
        for transform in reversed(self.transforms):
            data = transform.untransform(data)
        return data

    def output_dim(self) -> int:
        """Get the output dimension after transformation.

        Returns
        -------
        int
            Output dimension.
        """
        return self.transforms[-1].output_dim()
