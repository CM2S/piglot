"""AOA optimiser module."""
from typing import Tuple, Callable, Optional
import numpy as np
from piglot.objective import Objective
from piglot.optimiser import ScalarOptimiser


class AOA(ScalarOptimiser):
    """
    AOA optimiser.
    Documentation: https://www.sciencedirect.com/science/article/pii/S0045782520307945

    Attributes
    ----------
    n_solutions : integer
        population size (number of candidate solutions)
    alpha : float
        non-negative sensitive parameter used to define the accuracy of the exploitation
        over the iterations
    mu : float
        non-negative control parameter to adjust the search process
    epsilon : float
        small number to avoid division by zero
    seed : int
        random state seed
    MOA_start : float
        Math Optimizer Accelerated function initial value
    MOA_end : float
        Math Optimizer Accelerated function end value


    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """
    def __init__(self, objective: Objective, n_solutions=10, alpha=5.0, mu=0.5, epsilon=1e-12,
                 seed=1, MOA_start=0.2, MOA_end=1.0):
        """
        Constructs all the necessary attributes for the AOA optimiser

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
        n_solutions : integer
            population size (number of candidate solutions)
        alpha : float
            non-negative sensitive parameter used to define the accuracy of the exploitation
            over the iterations
        mu : float
            non-negative control parameter to adjust the search process
        epsilon : float
            small number to avoid division by zero
        seed : int
            random state seed
        MOA_start : float
            Math Optimizer Accelerated function initial value
        MOA_end : float
            Math Optimizer Accelerated function end value
        """
        super().__init__('AOA', objective)
        self.n_solutions = n_solutions
        self.alpha = alpha
        self.mu = mu
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.MOA_start = MOA_start
        self.MOA_end = MOA_end

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

        # Initiate candidate solutions randomly
        solutions = ((bound[:, 1] - bound[:, 0]) *
                     self.rng.random(size=(self.n_solutions, n_dim)) + bound[:, 0])
        # Replace first candidate solution by the initial shot
        solutions[0, :] = init_shot
        #
        beta = (bound[:, 1] - bound[:, 0])*self.mu
        # Evaluate fitness
        fitness = np.zeros(self.n_solutions)
        for i in range(0, self.n_solutions):
            fitness[i] = objective(solutions[i, :])
        # Update best solution
        pos_best_solution = np.argmin(fitness)
        best_value = fitness[pos_best_solution]
        best_solution = solutions[pos_best_solution, :]
        # Check solution progress
        if self._progress_check(0, best_value, best_solution):
            return best_value, best_solution
        #
        # Start iterative cycle
        for i in range(0, n_iter):
            # Update MOA and MOP
            MOA = self.MOA_start + i * (self.MOA_end - self.MOA_start) / n_iter
            MOP = 1.0 - (i ** (1 / self.alpha)) / (n_iter ** (1 / self.alpha))
            # Update solutions
            for i_solution in range(0, self.n_solutions):
                new_solution = np.zeros(n_dim)
                for i_pos in range(0, n_dim):
                    # Random values
                    r1, r2, r3 = self.rng.random(size=3)
                    if r1 > MOA:
                        # Exploration phase
                        if r2 > 0.5:
                            # Division operator
                            new_solution[i_pos] = (best_solution[i_pos] / (MOP + self.epsilon) *
                                                   beta[i_pos])
                        else:
                            # Multiplication operator
                            new_solution[i_pos] = best_solution[i_pos] * MOP * beta[i_pos]
                    else:
                        # Exploitation phase
                        if r3 > 0.5:
                            # Subtraction operator
                            new_solution[i_pos] = best_solution[i_pos] - MOP * beta[i_pos]
                        else:
                            # Addition operator
                            new_solution[i_pos] = best_solution[i_pos] + MOP * beta[i_pos]
                # Boundary check
                new_solution = np.where(new_solution > bound[:, 1], bound[:, 1], new_solution)
                new_solution = np.where(new_solution < bound[:, 0], bound[:, 0], new_solution)
                # Update fitness function
                new_fitness = objective(new_solution)
                if new_fitness < fitness[i_solution]:
                    fitness[i_solution] = new_fitness
                    solutions[i_solution, :] = new_solution
                if fitness[i_solution] < best_value:
                    best_value = fitness[i_solution]
                    best_solution = solutions[i_solution, :]
            # Update progress and check convergence
            if self._progress_check(i + 1, best_value, best_solution):
                break

        return best_value, best_solution
