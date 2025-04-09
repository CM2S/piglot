"""Module for a random search optimiser."""
from typing import Tuple
import os
import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from piglot.optimiser import Optimiser, InvalidOptimiserException
from piglot.objective import Objective, GenericObjective


class RandomSearch(Optimiser):
    """Optimiser based on random searches."""

    def __init__(
        self,
        objective: Objective,
        reference_point: list[float] = None,
        nadir_scale: float = 0.1,
        seed: int = None,
    ) -> None:
        """Constructor for the random search optimiser class.

        Parameters
        ----------
        objective : Objective
            Objective to optimise.
        reference_point : list[float], optional
            Reference point for multi-objective optimisation, by default None.
        nadir_scale : float, optional
            Scale factor for the nadir point, by default 0.1.
        seed : int, optional
            Random seed for candidate generation, by default None.
        """
        super().__init__('Random', objective)
        self.reference_point = reference_point
        self.nadir_scale = nadir_scale
        self.seed = seed

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective.

        Parameters
        ----------
        objective : Objective
            Objective to optimise.
        """
        if not isinstance(objective, GenericObjective):
            raise InvalidOptimiserException('Generic objective required for this optimiser')
        if objective.composition is not None:
            raise InvalidOptimiserException('This optimiser does not support composition')
        if objective.stochastic:
            raise InvalidOptimiserException('This optimiser does not support stochasticity')

    def update_mo_data(self, parameters: np.ndarray, observations: np.ndarray) -> Tuple[float, int]:
        """Get the partitioning of the observations in multi-objective optimisation.

        Parameters
        ----------
        parameters : np.ndarray
            Array of parameters.
        observations : np.ndarray
            Array of observations.

        Returns
        -------
        Tuple[float, int]
            Hypervolume and number of non-dominated points.
        """
        if self.reference_point is not None:
            ref_point = np.array(self.reference_point)
        else:
            nadir = np.max(observations, axis=0)
            ref_point = nadir + self.nadir_scale * (nadir - np.min(observations, axis=0))
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.from_numpy(-ref_point),
            Y=torch.from_numpy(-observations),
        )
        hypervolume = partitioning.compute_hypervolume().item()
        pareto = -partitioning.pareto_Y
        # Map each Pareto point to the original parameter space
        param_indices = [
            torch.argmin((torch.from_numpy(observations) - pareto[i, :]).norm(dim=1)).item()
            for i in range(pareto.shape[0])
        ]
        # Dump the Pareto front to a file
        with open(os.path.join(self.output_dir, "pareto_front"), 'w', encoding='utf8') as file:
            # Write header
            num_obj = pareto.shape[1]
            file.write('\t'.join([f'{"Objective_" + str(i + 1):>15}' for i in range(num_obj)]))
            file.write('\t' + '\t'.join([f'{param.name:>15}' for param in self.parameters]) + '\n')
            # Write each point
            for i, idx in enumerate(param_indices):
                file.write('\t'.join([f'{x.item():>15.8f}' for x in pareto[i, :]]) + '\t')
                file.write('\t'.join([f'{x.item():>15.8f}' for x in parameters[idx, :]]) + '\n')
        return -np.log(hypervolume)

    def _optimise(
        self,
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Optimise the objective.

        Parameters
        ----------
        n_dim : int
            Number of parameters to optimise.
        n_iter : int
            Maximum number of iterations.
        bound : np.ndarray
            Array where first and second columns correspond to lower and upper bounds, respectively.
        init_shot : np.ndarray
            Initial shot for the optimisation problem.

        Returns
        -------
        float
            Best observed objective value.
        np.ndarray
            Observed optimum of the objective.
        """

        # Initial shot
        result = self.objective(init_shot)
        best_value = result.values if self.objective.multi_objective else result.scalar_value
        best_solution = init_shot

        # Build observation datasets
        param_dataset = np.array([best_solution])
        objective_dataset = np.array([best_value])
        if len(objective_dataset.shape) == 1:
            objective_dataset = objective_dataset.reshape(-1, 1)

        # Update progress
        if self.objective.multi_objective:
            best_value = self.update_mo_data(param_dataset, objective_dataset)
            best_solution = None
        self._progress_check(0, best_value, best_solution)

        # Generate random parameters
        random_params = self.parameters.get_random(n_iter, seed=self.seed)

        # Iterate over all parameter sets
        for i in range(n_iter):
            # Evaluate objective and add to dataset
            param_set = random_params[i, :]
            result = self.objective(param_set)
            value = result.values if self.objective.multi_objective else result.scalar_value
            param_dataset = np.vstack((param_dataset, param_set))
            objective_dataset = np.vstack((objective_dataset, value))

            # Update best-observed value
            if self.objective.multi_objective:
                best_value = self.update_mo_data(param_dataset, objective_dataset)
                best_solution = None
            elif value < best_value:
                best_value = value
                best_solution = param_set
            self._progress_check(i + 1, best_value, best_solution)

        return best_value, best_solution
