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
            def_variance: float = 1e-6,
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
        self.def_variance = def_variance
        self.device = device

    def __update_stats(self) -> None:
        """Update the statistics of the dataset."""

        # Get observation statistics
        y_avg = torch.mean(self.values, dim=-2)
        y_std = torch.std(self.values, dim=-2)
        # Build mask of points with near-null variance
        y_abs_avg = torch.mean(torch.abs(self.values), dim=-2)
        mask = torch.abs(y_std / y_abs_avg) > self.std_tol
        # Update the dataset statistics
        self.outcome_mask = mask
        self.outcome_means = y_avg
        self.outcome_stds = y_std

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
        new_shape[-1] = self.n_outputs
        expanded = torch.empty(new_shape, dtype=self.dtype, device=self.device)
        # Fill the tensor using a 2D view:
        # i) modelled outcomes are directly inserted
        # ii) missing outcomes are filled with the average of the observed ones
        expanded_flat = expanded.view(-1, self.n_outputs)
        expanded_flat[:, self.outcome_mask] = values.view(-1, values.shape[-1])
        expanded_flat[:, ~self.outcome_mask] = self.outcome_means[~self.outcome_mask]
        # Note: we are using a view, so the expanded tensor is already modified
        return expanded

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
        new_dataset.lbounds = self.lbounds.to(device)
        new_dataset.ubounds = self.ubounds.to(device)
        new_dataset.device = device
        return new_dataset
