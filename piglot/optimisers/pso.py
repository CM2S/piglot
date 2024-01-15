"""PSO optimiser module."""
from typing import Tuple, Callable, Optional
import logging
from collections import deque
import numpy as np
try:
    from pyswarms.single.global_best import GlobalBestPSO
    from pyswarms.backend.operators import compute_pbest
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimiser import missing_method
    GlobalBestPSO = missing_method("PSO", "pyswarms")
from piglot.objective import Objective
from piglot.optimiser import ScalarOptimiser


class GlobalBestPSOMod(GlobalBestPSO):

    def optimize(self, optimiser, objective_func, iters, n_processes=None, verbose=False, **kwargs):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation
            (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """

        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            # fmt: off
            #print(self.swarm)
            self.swarm.current_cost = self.compute_objective_function(self.swarm,
                                                                      objective_func,
                                                                      pool=pool, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
            # fmt: on
            if verbose:
                self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform options update
            self.swarm.options = self.oh(
                self.options, iternow=i, itermax=iters
            )
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )
            if optimiser._progress_check(i+1, self.swarm.best_cost,
                                         self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()]):
                break

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
        return (final_best_cost, final_best_pos)

    def compute_objective_function(self, swarm, objective_func, pool=None, **kwargs):
        """Evaluate particles using the objective function

        This method evaluates each particle in the swarm according to the objective
        function passed.

        If a pool is passed, then the evaluation of the particles is done in
        parallel using multiple processes.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        objective_func : function
            objective function to be evaluated
        pool: multiprocessing.Pool
            multiprocessing.Pool to be used for parallel particle evaluation
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        numpy.ndarray
            Cost-matrix for the given swarm
        """
        func_value = []
        if pool is None:
            for i in swarm.position:
                func_value.append(objective_func(i, **kwargs))
            return func_value
        else:
            raise NotImplementedError("Parallel solving not implemented!")
            # results = pool.map(
            #     partial(objective_func, **kwargs),
            #     np.array_split(swarm.position, pool._processes),
            # )
            # return np.concatenate(results)


class PSO(ScalarOptimiser):
    """
    PSO optimiser.
    Documentation:
    https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/single/global_best.html#GlobalBestPSO

    Attributes
    ----------
    n_part : int
        number of particles in the swarm.
    options : dict with keys :code:`{'c1', 'c2', 'w'}`
    a dictionary containing the parameters for the specific
    optimization technique.
        * c1 : float
            cognitive parameter
        * c2 : float
            social parameter
        * w : float
            inertia parameter
    oh_strategy : dict, optional, default=None(constant options)
        a dict of update strategies for each option.
    bh_strategy : str
        a strategy for the handling of out-of-bounds particles.
    velocity_clamp : tuple, optional
        a tuple of size 2 where the first entry is the minimum velocity and
        the second entry is the maximum velocity. It sets the limits for
        velocity clamping.
    vh_strategy : str
        a strategy for the handling of the velocity of out-of-bounds particles.
    center : list (default is :code:`None`)
        an array of size :code:`dimensions`
    ftol_iter : int
        number of iterations over which the relative error in objective_func(best_pos)
        is acceptable for convergence. Default is :code:`1`
    n_processes : int
        number of processes to use for parallel particle evaluation (default: None =
        no parallelization)


    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, objective: Objective, n_part, options, oh_strategy=None,
                 bh_strategy='periodic', velocity_clamp=None, vh_strategy='unmodified', center=1.0,
                 ftol_iter=1, n_processes=None):
        """
        Constructs all the necessary attributes for the PSO optimiser

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
        n_part : int
            number of particles in the swarm.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
             a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        oh_strategy : dict, optional, default=None(constant options)
            a dict of update strategies for each option.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol_iter : int
            number of iterations over which the relative error in objective_func(best_pos)
            is acceptable for convergence. Default is :code:`1`
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None =
            no parallelization)
        """
        super().__init__('PSO', objective)
        self.n_part = n_part
        self.options = options
        self.oh_strategy = oh_strategy
        self.bh_strategy = bh_strategy
        self.velocity_clamp = velocity_clamp
        self.vh_strategy = vh_strategy
        self.center = center
        self.ftol_iter = ftol_iter
        self.n_processes = n_processes
        self.rng = np.random.default_rng(1)

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
        if bound is not None:
            new_bound = tuple(map(tuple, np.stack((bound[:, 0], bound[:, 1]))))
        population = ((bound[:, 1] - bound[:, 0]) *
                      self.rng.random(size=(self.n_part, n_dim)) + bound[:, 0])
        population[0, :] = init_shot
        model = GlobalBestPSOMod(self.n_part, n_dim, self.options, new_bound,
                                 self.oh_strategy, self.bh_strategy, self.velocity_clamp,
                                 self.vh_strategy, self.center, -np.inf, self.ftol_iter,
                                 population)

        x, new_value = model.optimize(self, objective, n_iter, self.n_processes, False)
        return x, new_value
