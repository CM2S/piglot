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
            bounds: np.ndarray,
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
        self.lbounds = torch.tensor(bounds[:, 0], dtype=dtype, device=device)
        self.ubounds = torch.tensor(bounds[:, 1], dtype=dtype, device=device)
        self.export = export
        self.std_tol = std_tol
        self.def_variance = def_variance
        self.device = device

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
        # Update the dataset file after every push
        if self.export is not None:
            self.save(self.export)

    def get_obervation_stats(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return statistics of the observations.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Mask, average and standard deviation of the observations.

        Raises
        ------
        RuntimeError
            When all observed points are equal.
        """
        # Get observation statistics
        y_avg = torch.mean(self.values, dim=-2)
        y_std = torch.std(self.values, dim=-2)
        # Build mask of points with near-null variance
        y_abs_avg = torch.mean(torch.abs(self.values), dim=-2)
        mask = torch.abs(y_std / y_abs_avg) > self.std_tol
        if not torch.any(mask):
            raise RuntimeError("All observed points are equal: add more initial samples")
        # Remove points that have near-null variance: not relevant to the model
        y_avg = y_avg[mask]
        y_std = y_std[mask]
        return mask, y_avg, y_std

    def standardised(self) -> BayesDataset:
        """Return a dataset with unit-cube parameters and standardised outputs.

        Returns
        -------
        BayesDataset
            The resulting dataset.
        """
        # Build unit cube space and standardised dataset
        std_dataset = copy.deepcopy(self)
        std_dataset.params = self.normalise(self.params)
        std_dataset.values, std_dataset.variances = self.standardise(self.values, self.variances)
        std_dataset.lbounds = torch.zeros_like(self.lbounds)
        std_dataset.ubounds = torch.ones_like(self.ubounds)
        return std_dataset

    def normalise(self, params: torch.Tensor) -> torch.Tensor:
        """Convert parameters to unit-cube.

        Parameters
        ----------
        params : torch.Tensor
            Parameters to convert.

        Returns
        -------
        torch.Tensor
            Unit-cube parameters.
        """
        return (params - self.lbounds) / (self.ubounds - self.lbounds)

    def standardise(
        self,
        values: torch.Tensor,
        variances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standardise outcomes.

        Parameters
        ----------
        values : torch.Tensor
            Values to standardise.
        variances : torch.Tensor
            Variances to standardise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Standardised values and variances.
        """
        mask, y_avg, y_std = self.get_obervation_stats()
        std_values = (values[:, mask] - y_avg) / y_std
        std_variances = variances[:, mask] / y_std
        return std_values, std_variances

    def denormalise(self, std_params: torch.Tensor) -> torch.Tensor:
        """Convert parameters from unit-cube to initial bounds.

        Parameters
        ----------
        std_params : torch.Tensor
            Parameters to convert.

        Returns
        -------
        torch.Tensor
            Original bound parameters.
        """
        return std_params * (self.ubounds - self.lbounds) + self.lbounds

    def destandardise(
        self,
        std_values: torch.Tensor,
        std_variances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """De-standardise outcomes.

        Parameters
        ----------
        std_values : torch.Tensor
            Values to de-standardise.
        std_variances : torch.Tensor
            Variances to de-standardise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            De-standardised values and variances.
        """
        _, y_avg, y_std = self.get_obervation_stats()
        values = std_values * y_std + y_avg
        variances = std_variances * y_std
        return values, variances

    def min(
            self,
            transformer: Callable[[torch.Tensor], float],
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
        idx = np.argmin([transformer(value) for value in self.values])
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
