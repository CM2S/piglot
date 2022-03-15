"""Hybrid PRS-SPSA optimiser module."""
import numpy as np
from piglot.optimiser import Optimiser, StoppingCriteria
from piglot.optimisers.spsa_adam import SPSA_Adam


class PRS_SPSA(Optimiser):
    """
    Hybrid PRS-Simultaneous Perturbation Stochastic Approximation method for optimisation.

    Reference:
    https://ieeexplore.ieee.org/document/705889
    https://arxiv.org/abs/1412.6980

    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, n_cycles, exploit_iter_no_improv=None, exploration_ratio=0.2, seed=1,
                 local_optimiser=SPSA_Adam()):
        """Constructs all necessary attributes for the SPSA-Adam optimiser.

        Parameters
        ----------
        n_cycles : integer
            Number of optimiser cycles to use.
        exploit_iter_no_improv : integer, optional
            Number of local iterations without improvement, by default None
        exploration_ratio : float, optional
            Ratio of exploration/total iterations, by default 0.2
        seed : integer, optional
            Seed for random number generator, by default 1e-8
        local_optimiser : Optimiser, optional
            Optimiser to use in the local exploitation stage, by default SPSA_Adam()
        """
        self.n_cycles = n_cycles
        self.exploration_ratio = exploration_ratio
        self.exploit_iter_no_improv = exploit_iter_no_improv
        self.rng = np.random.default_rng(seed)
        self.local_optimiser = local_optimiser
        self.name = 'PRS+SPSA'

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
            best parameter <solution
        """
        # Define some values
        n_iter_per_cycle = int(n_iter / self.n_cycles)
        n_iter_exploration = int(n_iter_per_cycle * self.exploration_ratio)
        n_iter_exploitation = n_iter_per_cycle - n_iter_exploration
        local_stop_criteria = StoppingCriteria(self.stop_criteria.conv_tol,
                                               self.exploit_iter_no_improv,
                                               self.stop_criteria.max_func_calls)
        x_trial = init_shot
        best_solution = x_trial
        best_value = func(best_solution)
        i_iter = 0
        lower_bounds = bound[:,0]
        upper_bounds = bound[:,1]
        i_cycle = 0
        while i_cycle < self.n_cycles:
            # Exploitation phase
            self.local_optimiser._init_optimiser(n_iter_exploitation, self.parameters,
                                                 self.pbar, self.loss, local_stop_criteria,
                                                 self.output)
            # Local optimisation
            self.local_optimiser.best_value = best_value
            self.local_optimiser.best_solution = best_solution
            self.local_optimiser._optimise(func, n_dim, n_iter_exploitation, bound, x_trial)
            value = self.local_optimiser.best_value
            solution = self.local_optimiser.best_solution

            # Update our histories
            i_done = self.local_optimiser.iiter
            self.value_history[i_iter:i_done+i_iter+1] = \
                self.local_optimiser.value_history[0:i_done+1]
            self.best_value_history[i_iter:i_done+i_iter+1] = \
                self.local_optimiser.best_value_history[0:i_done+1]
            self.solution_history[i_iter:i_done+i_iter+1,:] = \
                self.local_optimiser.solution_history[0:i_done+1,:]
            i_iter += self.local_optimiser.iiter

            # Update best solution
            if value < best_value:
                best_value = value
                best_solution = solution
            self.best_value = best_value
            self.best_solution = best_solution

            # Full convergence of the method
            if value < self.stop_criteria.conv_tol:
                break

            # Exploration phase
            exp_value = np.inf
            for _ in range(0, n_iter_exploration):
                sample = lower_bounds + self.rng.random(n_dim) \
                                      * (upper_bounds - lower_bounds)
                sample_value = func(sample)
                # Report progress
                i_iter += 1
                if self._progress_check(i_iter, sample_value, sample):
                    return best_solution, best_value
                # Update best exploration value
                if sample_value < exp_value:
                    exp_value = sample_value
                    x_trial = sample
                    # Update best solution
                    if sample_value < best_value:
                        best_value = sample_value
                        best_solution = sample
            if (i_cycle == self.n_cycles - 1) and ((n_iter - i_iter) > 0):
                self.n_cycles += 1
                if (n_iter - i_iter) < n_iter_per_cycle:
                    n_iter_per_cycle = n_iter - i_iter
                    n_iter_exploration = int(n_iter_per_cycle * self.exploration_ratio)
                    n_iter_exploitation = n_iter_per_cycle - n_iter_exploration
            i_cycle += 1
        return best_solution, best_value
