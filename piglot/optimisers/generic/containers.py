"""Module for generic data-driven optimisation containers in piglot."""
from typing import Optional, TypeVar
from dataclasses import dataclass
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import FastNondominatedPartitioning
from piglot.parameter import ParameterSet
from piglot.objective import ObjectiveResult
from piglot.optimiser import OptimisationResult
from piglot.data.dataset import ObjectiveDataset
from piglot.data.surrogate import SurrogateSettings
from piglot.utils.readable import ReadableModel


MultiObjectiveStateDataT = TypeVar('MultiObjectiveStateDataT', bound='MultiObjectiveStateData')


T = TypeVar('T', bound='OptimisationSettings')


class OptimisationSettings(ReadableModel):
    """Container for settings related to optimisation."""
    noisy: bool = False
    num_workers: int = 1
    nadir_scale: float = 0.1
    ref_point: Optional[list[float]] = None
    surrogate_settings: SurrogateSettings = SurrogateSettings()


@dataclass
class MultiObjectiveStateData:
    """Container for data used during a multi-objective optimisation run."""
    pareto_x: torch.Tensor = None
    pareto_y: torch.Tensor = None
    pareto_yvar: torch.Tensor = None
    partitioning: FastNondominatedPartitioning = None
    hypervolume: float = None

    @staticmethod
    def adjust_ref_point(y_points: torch.Tensor, nadir_scale: float) -> torch.Tensor:
        """Adjust the reference point based on current observations.

        Parameters
        ----------
        y_points : torch.Tensor
            Objective points.
        nadir_scale : float
            Scale factor for nadir adjustment.

        Returns
        -------
        torch.Tensor
            Adjusted reference point.
        """
        nadir = torch.min(y_points, dim=0).values
        ideal = torch.max(y_points, dim=0).values
        return nadir - nadir_scale * (ideal - nadir)

    @classmethod
    def from_dataset(
        cls: type[MultiObjectiveStateDataT],
        dataset: ObjectiveDataset,
        optim_settings: OptimisationSettings,
    ) -> MultiObjectiveStateDataT:
        """Create a multi-objective state instance from the given dataset and optimisation settings.

        Parameters
        ----------
        dataset : ObjectiveDataset
            Dataset containing the observations from the optimisation campaign.
        optim_settings : OptimisationSettings
            Settings for the optimisation campaign.

        Returns
        -------
        MultiObjectiveStateDataT
            Multi-objective state data instance.
        """
        # Extract objective values and variances
        x_points = torch.tensor([obs.params for obs in dataset.data])
        y_points = torch.tensor([obs.result.obj_values for obs in dataset.data])
        yvar_points = None
        if dataset.objective.has_variance():
            yvar_points = torch.tensor([obs.result.obj_variances for obs in dataset.data])

        # Check if we need to update the reference point
        if optim_settings.ref_point is None:
            ref_point = cls.adjust_ref_point(
                y_points=y_points, nadir_scale=optim_settings.nadir_scale
            )
        else:
            ref_point = torch.tensor(optim_settings.ref_point)

        # Update partitioning and Pareto front
        partitioning = FastNondominatedPartitioning(ref_point, Y=y_points)
        hypervolume = partitioning.compute_hypervolume().item()
        pareto_y = partitioning.pareto_Y

        # Map each Pareto point to the original parameter space
        param_indices = [
            torch.argmin((y_points - pareto_y[i, :]).norm(dim=1)).item()
            for i in range(pareto_y.shape[0])
        ]
        pareto_x = x_points[param_indices, :]
        pareto_yvar = yvar_points[param_indices, :] if yvar_points is not None else None

        # Return the multi-objective state data instance
        return cls(
            pareto_x=pareto_x,
            pareto_y=pareto_y,
            pareto_yvar=pareto_yvar,
            partitioning=partitioning,
            hypervolume=hypervolume,
        )

    def dump(self, output_file: str, parameters: ParameterSet) -> None:
        """Dump the Pareto front to a file.

        Parameters
        ----------
        output_file : str
            File to write the Pareto front to.
        parameters : ParameterSet
            Parameter set for the problem.
        """
        with open(output_file, 'w', encoding='utf8') as file:
            # Write header
            num_obj = self.pareto_y.shape[1]
            file.write('\t'.join([f'{"Objective_" + str(i + 1):>15}' for i in range(num_obj)]))
            file.write('\t' + '\t'.join([f'{param.name:>15}' for param in parameters]) + '\n')
            # Write each point
            for i in range(self.pareto_y.shape[0]):
                file.write('\t'.join([f'{-x.item():>15.8f}' for x in self.pareto_y[i, :]]) + '\t')
                file.write('\t'.join([f'{x.item():>15.8f}' for x in self.pareto_x[i, :]]) + '\n')


@dataclass
class OptimisationState:
    """Container for the state of an optimisation campaign."""
    num_evaluations: int
    best_value: Optional[float] = None
    best_params: Optional[np.ndarray] = None
    best_result: Optional[ObjectiveResult] = None
    mo_state: Optional[MultiObjectiveStateData] = None
    conf_interval: Optional[tuple[float, float]] = None

    def get_result(self) -> OptimisationResult:
        """Get the current best result as an optimisation result instance.

        Returns
        -------
        OptimisationResult
            Current best result.
        """
        # Multi-objective results
        if self.mo_state is not None:
            return OptimisationResult(
                value=self.mo_state.hypervolume,
                params=None,
                conf_interval=None,
                pareto_params=self.mo_state.pareto_x.numpy(),
                pareto_values=self.mo_state.pareto_y.numpy(),
                ref_point=self.mo_state.partitioning.ref_point.numpy(),
            )

        # Single-objective results
        return OptimisationResult(
            value=self.best_value,
            params=self.best_params,
            conf_interval=self.conf_interval,
        )
