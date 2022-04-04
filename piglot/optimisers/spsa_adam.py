"""Hybrid SPSA-Adam optimiser module."""
import numpy as np
from scipy.stats import bernoulli
from piglot.optimisers.optimiser import Optimiser, boundary_check


class SPSA_Adam(Optimiser):
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

    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, gamma=0.101,
                 prob=0.5, c=None, seed=1):
        """Constructs all necessary attributes for the SPSA-Adam optimiser.

        Parameters
        ----------
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
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gamma = gamma
        self.prob = prob
        self.c = 1e-6 if c is None else c
        self.seed = seed
        self.name = 'AdamSPSA'


    def _optimise(self, func, n_dim, n_iter, bound, init_shot):
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
        Exception
            If an initial shot is not passed.
        """
        if init_shot is None:
            raise Exception('Need to pass an initial shot for SPSA!')

        x = init_shot
        new_value = func(x)
        if self._progress_check(0, new_value, x):
            return x, new_value

        # First and second moments for Adam
        m = np.zeros(n_dim)
        v = np.zeros(n_dim)

        for i in range(0, n_iter):
            c_k = self.c / (i + 1) ** self.gamma
            # [-1,1] Bernoulli distribution 
            delta = 2 * bernoulli.rvs(self.prob, size=n_dim, random_state=self.seed) - 1
            # Bound check
            up = boundary_check(x + c_k * delta, bound)
            low = boundary_check(x - c_k * delta, bound)
            pos_loss = func(up)
            neg_loss = func(low)
            gradient = (pos_loss - neg_loss) / (up - low)
            # Update solution with Adam
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * np.square(gradient)
            mhat = m / (1 - self.beta1**(i+1))
            vhat = v / (1 - self.beta2**(i+1))
            x = x - self.alpha * mhat / (np.sqrt(vhat) + self.epsilon)
            # Bound check
            x = boundary_check(x, bound)
            new_value = func(x)
            # Update progress and check convergence
            if self._progress_check(i+1, new_value, x):
                break

        return x, new_value
