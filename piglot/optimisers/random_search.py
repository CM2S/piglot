"""Pure random search optimiser module."""
import numpy as np
from scipy.stats import norm
try:
    from scipy.stats import qmc
except ImportError:
    qmc = None
from piglot.optimisers.optimiser import Optimiser


class PureRandomSearch(Optimiser):
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

    def __init__(self, sampling='uniform'):
        """
        Constructs all the necessary attributes for the PRS optimiser.

        Parameters
        ----------
        sampling : str
            Sampling method to use for the random values. Valid options are:
            - 'uniform': Uniform distribution.
            - 'normal': Normal distribution, centered around the best value, and with
            decreasing standard deviation throughout the iterative process.
            - 'sobol': Sampling based on the Sobol sequence (requires scipy >= 1.7).
        """
        # Check if sampling method is valid
        valid_samples = ['uniform', 'normal', 'sobol']
        if sampling not in valid_samples:
            raise ValueError("Invalid sampling {0}!".format(sampling))
        # Check if Quasi-Monte Carlo is avaliable
        if sampling == 'sobol' and qmc is None:
            raise ValueError("Sobol sequence requires qmc module in scipy >= 1.7!")
        # Store parameters
        self.name = 'PRS'
        self.sampling = sampling

    def _optimise(self, func, n_dim, n_iter, bound, init_shot):
        """
        Parameters
        ----------
        func : callable
            function to optimize
        n_dim : integer
            dimension, i.e., number of parameters to optimize
        n_iter : integer
            maximum number of iterations
        bound : array
            first column corresponding to the lower bound, and second column to the
            upper bound
        init_shot : list
            initial shot for the optimization problem

        Returns
        -------
        best_value : float
            best loss function value
        best_solution : list
            best parameter solution
        """
        # Define lower and upper bounds
        lower_bounds = bound[:,0]
        upper_bounds = bound[:,1]

        # Compute loss function value given the initial shot
        best_solution = init_shot
        best_value = func(best_solution)

        # Check convergence for the initial shot
        if self._progress_check(0, best_value, best_solution):
            return best_solution, best_value

        # Prepare random number generators
        if self.sampling == 'uniform':
            sampler = np.random.default_rng(1)
        elif self.sampling == 'sobol':
            sampler = qmc.Sobol(n_dim, seed=1)

        # Optimization problem iterative procedure
        for i in range(n_iter):
            # Evaluate next random point
            if self.sampling == 'uniform':
                new_solution = lower_bounds + sampler.random(n_dim) \
                                            * (upper_bounds - lower_bounds)
            elif self.sampling == 'normal':
                sigma = (upper_bounds - lower_bounds) / (2 + (100.0 * (i+1)) / n_iter)
                new_solution = norm.rvs(loc=best_solution, scale=sigma, size=n_dim,
                                        random_state=i)
                new_solution = np.where(new_solution > bound[:,1], bound[:,1], new_solution)
                new_solution = np.where(new_solution < bound[:,0], bound[:,0], new_solution)
            elif self.sampling == 'sobol':
                new_solution = lower_bounds + np.squeeze(sampler.random()) \
                                            * (upper_bounds - lower_bounds)

            # Compute function value and update best solution
            new_value = func(new_solution)
            if new_value < best_value:
                best_solution = new_solution
                best_value = new_value

            # Update progress and check convergence
            if self._progress_check(i+1, new_value, new_solution):
                break

        return best_solution, best_value
