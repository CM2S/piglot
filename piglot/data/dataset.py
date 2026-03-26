"""Module for data generation and handling in piglot."""
import os
from typing import Optional
from dataclasses import dataclass
from threading import Lock
import time
import pickle
import numpy as np
import torch
from piglot.settings import Settings
from piglot.objective import ObjectiveResult, Objective


@dataclass
class Observation:
    """Container for a single observation from an objective evaluation."""
    call_number: int
    elapsed_time: float
    params: np.ndarray
    params_dict: dict[str, float]
    result: ObjectiveResult


class RawDataset:
    """Container for raw data to be used in surrogate modelling."""

    def __init__(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        covariances: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.inputs = torch.tensor(inputs, dtype=dtype)
        self.outputs = torch.tensor(outputs, dtype=dtype)
        self.covariances = (
            torch.tensor(covariances, dtype=dtype) if covariances is not None else None
        )


class ObjectiveDataset:
    """Container for data generated from objective evaluations."""

    def __init__(self, settings: Settings, objective: Objective) -> None:
        self.settings = settings
        self.objective = objective
        self.num_evaluations: int = 0
        self.data: list[Observation] = []
        self.lock = Lock()

    def evaluate(self, params: np.ndarray, *args, **kwargs) -> ObjectiveResult:
        """Evaluate the objective for the given parameters and store the observation.

        Parameters
        ----------
        params : np.ndarray
            The parameters at which to evaluate the objective.
        *args, **kwargs
            Additional arguments to pass to the objective.

        Returns
        -------
        ObjectiveResult
            The result of the objective evaluation.
        """
        # Evaluate the objective
        start_time = time.perf_counter()
        result = self.objective(params, *args, **kwargs)
        elapsed_time = time.perf_counter() - start_time

        # Store the observation
        with self.lock:
            observation = Observation(
                call_number=self.num_evaluations,
                elapsed_time=elapsed_time,
                params=params,
                params_dict=self.settings.parameters.to_dict(params),
                result=result,
            )
            self.num_evaluations += 1
            self.data.append(observation)
            with open(os.path.join(self.settings.output_dir, "dataset.bin"), "wb") as f:
                pickle.dump(self.data, f)

        return result

    def export_raw(self) -> RawDataset:
        """Export the dataset in raw format for surrogate modelling.

        Returns
        -------
        RawDataset
            The raw dataset containing inputs, outputs, and covariances.
        """
        with self.lock:
            inputs = np.array([obs.params for obs in self.data])
            # Under composition, always use the latent space representation
            if self.objective.is_composite():
                return RawDataset(
                    inputs=inputs,
                    outputs=np.stack([obs.result.latent_values for obs in self.data], axis=0),
                    covariances=(
                        np.stack([obs.result.latent_covariances for obs in self.data], axis=0)
                        if self.objective.has_variance() else None
                    ),
                )

            # Multi-objective problems: use the individual objectives (with independent covariances)
            if self.objective.is_multi_objective():
                return RawDataset(
                    inputs=inputs,
                    outputs=np.stack([obs.result.obj_values for obs in self.data], axis=0),
                    covariances=(
                        np.stack([np.diag(obs.result.obj_variances) for obs in self.data], axis=0)
                        if self.objective.has_variance() else None
                    ),
                )

            # Non-composite scalarised single-objective problems: use the scalarised objective
            if self.objective.scalarisation is not None:
                return RawDataset(
                    inputs=inputs,
                    outputs=np.array([[obs.result.scalar_value] for obs in self.data]),
                    covariances=(
                        np.array([[[obs.result.scalar_variance]] for obs in self.data])
                        if self.objective.has_variance() else None
                    ),
                )

            # Non-composite single-objective problems: use the raw objective
            return RawDataset(
                inputs=inputs,
                outputs=np.array([[obs.result.obj_values.item()] for obs in self.data]),
                covariances=(
                    np.array([[[obs.result.obj_variances.item()]] for obs in self.data])
                    if self.objective.has_variance() else None
                ),
            )

    def load(self) -> None:
        """Load the dataset from a file."""
        with self.lock, open(os.path.join(self.settings.output_dir, "dataset.bin"), "rb") as f:
            self.data = pickle.load(f)
            self.num_evaluations = len(self.data)
