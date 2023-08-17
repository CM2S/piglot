"""SPSA optimiser module."""
import numpy as np
from scipy.stats import bernoulli
from piglot.objective import SingleObjective
from piglot.optimisers.optimiser import ScalarOptimiser, boundary_check


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

    def __init__(self, alpha=0.602, gamma=0.101, prob=0.5, seed=1, A=None, a=None, c=None):
        """Constructs all necessary attributes for the SPSA optimiser.

        Parameters
        ----------
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
        super().__init__('SPSA')
        self.alpha = alpha
        self.gamma = gamma
        self.prob = prob
        self.seed = seed
        self.A = A
        self.a = a
        self.c = 1e-6 if c is None else c


    def _optimise(
        self,
        objective: SingleObjective,
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ):
        """Solves the optimisation problem.

        Parameters
        ----------
        func : callable
            Function to optimise
        n_dim : integer
            Dimension, i.e., number of parameter to optimise
        n_iter : integer
            Maximum number of iterations
        bound : array
            2D array with upper and lower bounds. First column refers to lower bounds,
            whilst the second refers to the upper bounds.
        init_shot : array
            Initial shot for the optimisation problem.

        Returns
        -------
        best_value : float
            Best loss function value
        best_solution : array
            Best parameters

        Raises
        ------
        RuntimeError
            If an initial shot is not passed.
        """
        if init_shot is None:
            raise RuntimeError('Need to pass an initial shot for SPSA!')

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
  