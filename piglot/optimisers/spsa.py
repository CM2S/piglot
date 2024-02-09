"""SPSA optimiser module."""
from typing import Tuple, Callable, Optional
import numpy as np
from scipy.stats import bernoulli
from piglot.objective import Objective
from piglot.optimiser import ScalarOptimiser, boundary_check


class SPSA(ScalarOptimiser):
    """
    Simultaneous Perturbation Stochastic Approximation method for optimisation.

    Reference:
    https://ieeexplore.ieee.org/document/705889

    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, objective: Objective, alpha=0.602, gamma=0.101, prob=0.5, seed=1, A=None,
                 a=None, c=None):
        """Constructs all necessary attributes for the SPSA optimiser.

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
        alpha : float, optional
            Model parameter, refer to documentation, by default 0.602
        gamma : float, optional
            Model parameter, refer to documentation, by default 0.101
        prob : float, optional
            Model parameter, refer to documentation, by default 0.5
        seed : int, optional
            Random number generator seed, by default 1
        A : float, optional
            Model parameter, refer to documentation, by default None.
            If None, this parameter is defined according to internal heuristics.
        a : float, optional
            Model parameter, refer to documentation, by default None
            If None, this parameter is defined according to internal heuristics.
        c : float, optional
            Model parameter, refer to documentation, by default None
            If None, this parameter is defined according to internal heuristics.
        """
        super().__init__('SPSA', objective)
        self.alpha = alpha
        self.gamma = gamma
        self.prob = prob
        self.seed = seed
        self.A = A
        self.a = a
        self.c = 1e-6 if c is None else c

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

        if self.A is None:
            self.A = n_iter / 20
        if self.a is None:
            self.a = 2 * (self.A + 1)**self.alpha

        x = init_shot
        new_value = objective(x)
        if self._progress_check(0, new_value, x):
            return x, new_value

        for i in range(0, n_iter):
            a_k = self.a / (self.A + i + 1) ** self.alpha
            c_k = self.c / (i + 1) ** self.gamma
            # [-1,1] Bernoulli distribution
            delta = 2 * bernoulli.rvs(self.prob, size=n_dim, random_state=self.seed + i) - 1
            # Bound check
            up = boundary_check(x + c_k * delta, bound)
            low = boundary_check(x - c_k * delta, bound)
            pos_loss = objective(up)
            neg_loss = objective(low)
            gradient = (pos_loss - neg_loss) / (up - low)
            # Update solution
            x = x - a_k * gradient
            # Bound check
            x = boundary_check(x, bound)
            new_value = objective(x)
            # Update progress and check convergence
            if self._progress_check(i+1, new_value, x):
                break

        return x, new_value
