"""Module with utilities for transforming responses under compositions."""
from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import torch
from piglot.objective import Composition, ObjectiveResult
from piglot.solver.solver import OutputResult
from piglot.utils.reductions import Reduction


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


class ResponseComposition(Composition):
    """Composition for transforming responses."""

    def __init__(
        self,
        scalarise: bool,
        stochastic: bool,
        weights: List[float],
        reductions: List[Reduction],
        flatten_list: List[FlattenUtility],
        scalarisation: str,
        bounds: np.ndarray = None,
        types: List[bool] = None,
    ) -> None:
        if len(flatten_list) != len(reductions):
            raise ValueError("Mismatched number of reductions and responses.")
        self.scalarise = scalarise
        self.stochastic = stochastic
        self.weights = weights
        self.reductions = reductions
        self.flatten_list = flatten_list
        self.lenghts = [flatten.length() for flatten in self.flatten_list]
        self.concat = ConcatUtility(self.lenghts)
        self.bounds = bounds
        self.scalarisation = scalarisation
        self.types = types

    def transform(self, params: np.ndarray, responses: List[List[OutputResult]]) -> ObjectiveResult:
        """Transform a set of responses into a fixed-size ObjectiveResult for the optimiser.

        Parameters
        ----------
        params : np.ndarray
            Parameters for the given responses.
        responses : List[List[OutputResult]]
            List of responses.

        Returns
        -------
        ObjectiveResult
            Transformed responses.
        """
        # Sanitise input shape
        if len(responses) != len(self.flatten_list):
            raise ValueError("Mismatched number of objectives.")
        # Build set of responses to concatenate
        means = []
        variances = []
        for i, flatten in enumerate(self.flatten_list):
            flat_responses = np.array([
                flatten.flatten(response.get_time(), response.get_data())
                for response in responses[i]
            ])
            # Build stochastic model for this set of responses
            means.append(np.mean(flat_responses, axis=0))
            variances.append(np.var(flat_responses, axis=0) / flat_responses.shape[0])
        # Concatenate the transformed responses
        return ObjectiveResult(
            params,
            self.concat.concat(means),
            self.concat.concat(variances) if self.stochastic else None,
        )

    @staticmethod
    def _expand_params(time: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Expand the set of parameters to match the time grid.

        Parameters
        ----------
        time : torch.Tensor
            Time grid for the responses.
        params : torch.Tensor
            Parameters for the given responses.

        Returns
        -------
        torch.Tensor
            Expanded parameter values.
        """
        # Nothing to do when shapes are consistent
        if len(params.shape) == len(time.shape):
            return params
        # Expand the parameters along the first dimensions
        return params.expand(*([a for a in time.shape[:-1]] + [params.shape[-1]]))

    @staticmethod
    def normalise_objective(objective: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """Normalise the objective.

        Parameters
        ----------
        objective : torch.Tensor
            Objective to normalise.
        bounds : torch.Tensor
            Bounds for the objectives.
        types : torch.Tensor
            Types for the objectives.

        Returns
        -------
        torch.Tensor
            Normalised objective.
        """
        # Normalise the objective
        return (objective - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    def composition_torch(self, inner: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute the outer function of the composition with gradients.

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function.
        params : torch.Tensor
            Parameters for the given responses.

        Returns
        -------
        torch.Tensor
            Composition result.
        """
        # Sanitise the scalarisation method
        if self.scalarise and self.scalarisation not in ('mean', 'stch', 'linear'):
            raise ValueError(
                f"Invalid scalarisation '{self.scalarisation}'. Use 'mean', 'stch' or 'linear'."
            )
        # Split the inner responses
        responses = self.concat.split_torch(inner)
        # Unflatten each response
        unflattened = [
            flatten.unflatten_torch(response)
            for response, flatten in zip(responses, self.flatten_list)
        ]
        # Evaluate and stack the objective for each response
        objective = torch.stack([
            reduction.reduce_torch(time, data, self._expand_params(time, params))
            for (time, data), reduction in zip(unflattened, self.reductions)
        ], dim=-1)
        # Mean scalarisation if requested
        if self.scalarise:
            # Sanitise the weights
            weights = torch.tensor(self.weights).to(inner.device)
            # Smooth TCHebycheff scalarisation (STCH) or linear if requested
            if self.scalarisation in ('stch', 'linear'):
                # Sanitise the weights
                if torch.sum(weights) != 1.0:
                    raise ValueError(f'Weights must sum to 1.0, got {torch.sum(weights)}.')
                # Set all the objectives to be positive
                objective = objective.abs()
                # Set the bounds and types
                bounds = torch.tensor(self.bounds).to(inner.device)
                types = torch.tensor(self.types).to(inner.device)
                # Calculate the costs
                costs = torch.where(types, torch.tensor(-1.0), torch.tensor(1.0))
                # Calculate the normalised objective values
                norm_objective = self.normalise_objective(objective, bounds)
                if self.scalarisation == 'stch':
                    # Calculate the ideal point
                    ideal_point = torch.where(types, torch.tensor(1.0), torch.tensor(0.0))
                    # Smoothing parameter for STCH
                    u = 0.006
                    # Precompute the numerator of Tchebycheff function value
                    tch_numerator = torch.abs((norm_objective - ideal_point) * costs) * weights
                    # Calculate the initial Tchebycheff function value
                    tch_values = tch_numerator / u
                    # Parameters to ensure numerical stability
                    u_increment = 0.001  # Increment value for u
                    max_u = 0.2  # Maximum value for u
                    max_iterations = max_u/u  # Limit to prevent infinite loop
                    for iteration in range(int(max_iterations)):
                        if not torch.isinf(torch.sum(torch.exp(tch_values))):
                            break  # Early stopping if condition is met
                        u += u_increment
                        tch_values = tch_numerator / u  # Recalculate tch_values with updated u
                    # Return the final scalarisation value
                    return torch.log(torch.sum(torch.exp(tch_values), dim=-1)) * u
                return torch.sum((norm_objective*costs) * weights, dim=-1)
            # Mean scalarisation otherwise
            # Apply the weights
            objective = objective * weights
            return torch.mean(objective, dim=-1)
        return objective * torch.tensor(self.weights).to(inner.device)
