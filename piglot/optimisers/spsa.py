"""SPSA optimiser module."""
from typing import Any, Callable, Optional
from functools import partial
import warnings
import numpy as np
from scipy.stats import bernoulli
from piglot.objective import Objective
from piglot.optimiser import SimpleOptimiser
from piglot.settings import Settings
from piglot.utils.assorted import parallel_map


class SPSA(SimpleOptimiser):
    """Simultaneous Perturbation Stochastic Approximation method for optimisation.

    Reference:
    https://ieeexplore.ieee.org/document/705889
    """

    def __init__(
        self,
        settings: Settings,
        objective: Objective,
        alpha: float = 0.602,
        gamma: float = 0.101,
        c: float = 1e-6,
        lr: float = 0.01,
        A: Optional[float] = None,
        a: Optional[float] = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(settings, objective, normalise_params=True)
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.A = A
        self.a = a
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
        return "SPSA"

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
        # Initialise learning rates
        A_val = num_iters / 20 if self.A is None else self.A
        a_val = (A_val + 1) ** self.alpha if self.a is None else self.a

        # Set up helpers
        x = initial_guess
        n_dim = len(initial_guess)
        lbounds = np.array([b[0] for b in bounds])
        ubounds = np.array([b[1] for b in bounds])
        parallel_objective = partial(objective, concurrent=self.num_workers > 1)

        # Initial evaluation
        objective(x)

        for i in range(0, num_iters):
            # This iteration's learning rates
            a_k = self.lr * a_val / (A_val + i + 1) ** self.alpha
            c_k = self.c / (i + 1) ** self.gamma

            # Search direction using a [-1,1] Bernoulli distribution
            seed = i + (0 if self.settings.seed is None else self.settings.seed)
            delta = 2 * bernoulli.rvs(0.5, size=n_dim, random_state=seed) - 1

            # Estimate gradient
            up = np.clip(x + c_k * delta, lbounds, ubounds)
            low = np.clip(x - c_k * delta, lbounds, ubounds)
            pos_loss, neg_loss = parallel_map(parallel_objective, [up, low], self.num_workers)
            gradient = (pos_loss - neg_loss) / (up - low)

            # Update and evaluate solution
            x = np.clip(x - a_k * gradient, lbounds, ubounds)
            objective(x)
            callback()
