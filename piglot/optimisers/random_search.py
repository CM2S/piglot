"""Pure random search optimiser module."""
from typing import Tuple, Callable, Optional
import numpy as np
from scipy.stats import norm, qmc
from piglot.objective import Objective
from piglot.optimiser import ScalarOptimiser


class PureRandomSearch(ScalarOptimiser):
    """
    Pure Random Search optimiser.

    Three sampling methods for generating random numbers are available:
    - Uniform distribution.
    - Normal distribution, centered around the best value, and with decreasing standard
    deviation throughout the iterative process.
    - Sampling based on the Sobol sequence (requires scipy >= 1.7).

    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, objective: Objective, sampling='uniform', seed=1):
        """
        Constructs all the necessary attributes for the PRS optimiser.

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
        sampling : str
            Sampling method to use for the random values. Valid options are:
            - 'uniform': Uniform distribution.
            - 'normal': Normal distribution, centered around the best value, and with
            decreasing standard deviation throughout the iterative process.
            - 'sobol': Sampling based on the Sobol sequence (requires scipy >= 1.7).
        """
        super().__init__('PRS', objective)
        # Check if sampling method is valid
        valid_samples = ['uniform', 'normal', 'sobol']
        if sampling not in valid_samples:
            raise ValueError(f"Invalid sampling {sampling}!")
        # Store parameters
        self.sampling = sampling
        self.seed = seed

    def _scalar_optimise(
        self,
        objective: Callable[[np.ndarray, Optional[bool]], float],
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Abstract method for optimising the objective.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to optimise.
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
        # Define lower and upper bounds
        lower_bounds = bound[:, 0]
        upper_bounds = bound[:, 1]

        # Compute loss function value given the initial shot
        best_solution = init_shot
        best_value = objective(best_solution)

        # Check convergence for the initial shot
        if self._progress_check(0, best_value, best_solution):
            return best_solution, best_value

        # Prepare random number generators
        if self.sampling == 'uniform':
            sampler = np.random.default_rng(self.seed)
        elif self.sampling == 'sobol':
            sampler = qmc.Sobol(n_dim, seed=self.seed)

        # Optimization problem iterative procedure
        for i in range(n_iter):
            # Evaluate next random point
            if self.sampling == 'uniform':
                new_solution = lower_bounds + sampler.random(n_dim) \
                                            * (upper_bounds - lower_bounds)
            elif self.sampling == 'normal':
                sigma = (upper_bounds - lower_bounds) / (2 + (100.0 * (i + 1)) / n_iter)
                new_solution = norm.rvs(loc=best_solution, scale=sigma, size=n_dim,
                                        random_state=i)
                new_solution = np.where(new_solution > bound[:, 1], bound[:, 1], new_solution)
                new_solution = np.where(new_solution < bound[:, 0], bound[:, 0], new_solution)
            elif self.sampling == 'sobol':
                new_solution = lower_bounds + np.squeeze(sampler.random()) \
                                            * (upper_bounds - lower_bounds)

            # Compute function value and update best solution
            new_value = objective(new_solution)
            if new_value < best_value:
                best_solution = new_solution
                best_value = new_value

            # Update progress and check convergence
            if self._progress_check(i+1, new_value, new_solution):
                break

        return best_solution, best_value
