"""Hybrid SPSA-Adam optimiser module."""
from typing import Any, Callable, Optional
from functools import partial
import warnings
import numpy as np
from scipy.stats import bernoulli
from piglot.objective import Objective
from piglot.optimiser import SimpleOptimiser
from piglot.settings import Settings
from piglot.utils.assorted import parallel_map


class SPSA_Adam(SimpleOptimiser):
    """Hybrid Simultaneous Perturbation Stochastic Approximation-Adam method for optimisation.

    References:
    https://ieeexplore.ieee.org/document/705889
    https://arxiv.org/abs/1412.6980
    """

    def __init__(
        self,
        settings: Settings,
        objective: Objective,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        gamma: float = 0.101,
        c: float = 1e-6,
        num_workers: int = 1,
    ) -> None:
        super().__init__(settings, objective, normalise_params=True)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gamma = gamma
        self.c = c
        self.num_workers = num_workers
        if num_workers > 2:
            warnings.warn(
                f"A total of {num_workers} workers have been specified, "
                "but the optimiser cannot use more than 2 workers."
            )

    def name(self) -> str:
        """Name of the optimiser.

        Returns
        -------
        str
            Name of the optimiser.
        """
        return "SPSA-Adam"

    def _simple_optimise(
        self,
        num_iters: int,
        initial_guess: np.ndarray,
        bounds: list[tuple[float, float]],
        objective: Callable[[np.ndarray, Optional[bool]], float],
        callback: Callable[[Any], None],
    ) -> None:
        """Optimise the objective function.

        Parameters
        ----------
        num_iters : int
            Number of iterations for the optimisation.
        initial_guess : np.ndarray
            Initial guess for the optimisation.
        bounds : list[tuple[float, float]]
            Bounds for the optimisation variables.
        objective : Callable[[np.ndarray, Optional[bool]], float]
            Objective function to be minimised.
        callback : Callable[[Any], None]
            Callback function for reporting the optimiser progress and checking for termination.
            This function is called at the end of each iteration and will raise StopIteration if
            the optimisation should be stopped. Keyword arguments are reported from the optimiser.
        """
        # Set up helpers
        x = initial_guess
        n_dim = len(initial_guess)
        lbounds = np.array([b[0] for b in bounds])
        ubounds = np.array([b[1] for b in bounds])
        parallel_objective = partial(objective, concurrent=self.num_workers > 1)

        # Initial evaluation
        objective(x)

        # First and second moments for Adam
        m = np.zeros(n_dim)
        v = np.zeros(n_dim)

        for i in range(0, num_iters):
            # Search direction using a [-1,1] Bernoulli distribution
            c_k = self.c / (i + 1) ** self.gamma
            seed = i + (0 if self.settings.seed is None else self.settings.seed)
            delta = 2 * bernoulli.rvs(0.5, size=n_dim, random_state=seed) - 1

            # Estimate gradient
            up = np.clip(x + c_k * delta, lbounds, ubounds)
            low = np.clip(x - c_k * delta, lbounds, ubounds)
            pos_loss, neg_loss = parallel_map(parallel_objective, [up, low], self.num_workers)
            gradient = (pos_loss - neg_loss) / (up - low)

            # Update solution with Adam
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * np.square(gradient)
            mhat = m / (1 - self.beta1**(i+1))
            vhat = v / (1 - self.beta2**(i+1))
            x = np.clip(x - self.lr * mhat / (np.sqrt(vhat) + self.epsilon), lbounds, ubounds)

            # Evaluate solution
            objective(x)
            callback()
