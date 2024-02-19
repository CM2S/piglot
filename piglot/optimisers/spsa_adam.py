"""Hybrid SPSA-Adam optimiser module."""
from typing import Tuple, Callable, Optional
import numpy as np
from scipy.stats import bernoulli
from piglot.objective import Objective
from piglot.optimiser import ScalarOptimiser, boundary_check


class SPSA_Adam(ScalarOptimiser):
    """
    Hybrid Simultaneous Perturbation Stochastic Approximation-Adam method for optimisation.

    References:
    https://ieeexplore.ieee.org/document/705889
    https://arxiv.org/abs/1412.6980

    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, objective: Objective, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 gamma=0.101, prob=0.5, c=None, seed=1):
        """Constructs all necessary attributes for the SPSA-Adam optimiser.

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
        alpha : float, optional
            Model parameter, refer to documentation, by default 0.01
        beta1 : float, optional
            Model parameter, refer to documentation, by default 0.9
        beta2 : float, optional
            Model parameter, refer to documentation, by default 0.999
        epsilon : float, optional
            Model parameter, refer to documentation, by default 1e-8
        gamma : float, optional
            Model parameter, refer to documentation, by default 0.101
        prob : float, optional
            Model parameter, refer to documentation, by default 0.5
        c : float, optional
            Model parameter, refer to documentation, by default None
            If None, this parameter is defined according to internal heuristics.
        """
        super().__init__('AdamSPSA', objective)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gamma = gamma
        self.prob = prob
        self.c = 1e-6 if c is None else c
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

        x = init_shot
        new_value = objective(x)
        if self._progress_check(0, new_value, x):
            return x, new_value

        # First and second moments for Adam
        m = np.zeros(n_dim)
        v = np.zeros(n_dim)

        for i in range(0, n_iter):
            c_k = self.c / (i + 1) ** self.gamma
            # [-1,1] Bernoulli distribution
            delta = 2 * bernoulli.rvs(self.prob, size=n_dim, random_state=self.seed + i) - 1
            # Bound check
            up = boundary_check(x + c_k * delta, bound)
            low = boundary_check(x - c_k * delta, bound)
            pos_loss = objective(up)
            neg_loss = objective(low)
            gradient = (pos_loss - neg_loss) / (up - low)
            # Update solution with Adam
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * np.square(gradient)
            mhat = m / (1 - self.beta1**(i+1))
            vhat = v / (1 - self.beta2**(i+1))
            x = x - self.alpha * mhat / (np.sqrt(vhat) + self.epsilon)
            # Bound check
            x = boundary_check(x, bound)
            new_value = objective(x)
            # Update progress and check convergence
            if self._progress_check(i+1, new_value, x):
                break

        return x, new_value
