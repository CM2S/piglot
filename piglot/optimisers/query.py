"""Module for a simple query optimiser."""
from typing import Tuple
import numpy as np
from piglot.optimiser import Optimiser, InvalidOptimiserException
from piglot.objective import Objective, GenericObjective


class QueryOptimiser(Optimiser):
    """Query optimiser."""

    def __init__(
        self,
        objective: Objective,
        param_list_file: str,
    ) -> None:
        """Constructor for the Query optimiser class.

        Parameters
        ----------
        objective : Objective
            Objective to optimise.
        param_list_file : str
            File containing the list of parameters.
        """
        super().__init__('Query', objective)
        self.param_list = np.genfromtxt(param_list_file)
        if len(self.param_list.shape) == 1:
            self.param_list = self.param_list.reshape(-1, 1)

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

        # Sanitise input
        if n_dim != self.param_list.shape[1]:
            raise ValueError('Number of parameters does not match the number of columns.')
        if n_iter != self.param_list.shape[0]:
            raise ValueError('Number of iterations does not match the number of rows.')
        for i in range(n_dim):
            if np.any(self.param_list[:, i] < bound[i, 0]):
                raise ValueError('Parameter values outside lower bounds.')
            if np.any(self.param_list[:, i] > bound[i, 1]):
                raise ValueError('Parameter values outside upper bounds.')

        # Initial shot
        best_value = self.objective(init_shot).scalarise()
        best_solution = init_shot
        self._progress_check(0, best_value, best_solution)

        # Iterate over all parameter sets
        for i, param_set in enumerate(self.param_list):
            value = self.objective(param_set).scalarise()
            if value < best_value:
                best_value = value
                best_solution = param_set
            self._progress_check(i + 1, best_value, best_solution)

        return best_value, best_solution
