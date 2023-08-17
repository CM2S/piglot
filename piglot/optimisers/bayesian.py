"""Bayesian optimiser module."""
import numpy as np
try:
    from bayes_opt import BayesianOptimization
    from bayes_opt.util import UtilityFunction
    from bayes_opt.event import Events
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimisers.optimiser import missing_method
    BayesianOptimization = missing_method("Bayesian optimisation", "bayes_opt")
from piglot.objective import SingleObjective
from piglot.optimisers.optimiser import ScalarOptimiser


class BayesianOptimizationMod(BayesianOptimization):

    def maximize(self, optimiser, init_points=5, n_iter=25, acq='ucb', kappa=2.576,
                 kappa_decay=1, kappa_decay_delay=0, xi=0.0, log_space=False, **gp_params):
        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.
        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.
        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.
        acq: {'ucb', 'ei', 'poi'}
            The acquisition method used.
                * 'ucb' stands for the Upper Confidence Bounds method
                * 'ei' is the Expected Improvement method
                * 'poi' is the Probability Of Improvement criterion.
        kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
                Higher value = favors spaces that are least explored.
                Lower value = favors spaces where the regression function is the
                highest.
        kappa_decay: float, optional(default=1)
            `kappa` is multiplied by this factor every iteration.
        kappa_decay_delay: int, optional(default=0)
            Number of iterations that must have passed before applying the decay
            to `kappa`.
        xi: float, optional(default=0.0)
            [unused]
        log_space : bool
            Whether to optimise the loss in a log space (requires non-negative losses)
        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        loss_transformer = lambda loss: np.exp(-loss) if log_space else -loss

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1

            if iteration == 0 and self._queue.empty:
                best_value = loss_transformer(self.max["target"])
                best_solution = list(self.max["params"].values())
                if optimiser._progress_check(iteration, best_value, best_solution):
                    break

            if (iteration != 0 and iteration <= len(self.res)):
                solution = self.res[iteration]
                current_value = solution.get('params').values()
                if optimiser._progress_check(iteration, loss_transformer(solution.get('target')),
                                             list(current_value)):
                    break
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)


class Bayesian(ScalarOptimiser):
    """
    Bayesian optimiser.
    Documentation:
        https://github.com/fmfn/BayesianOptimization/blob/91441fe4002fb6ebdb4aa5e33826230d8df560d0/bayes_opt/bayesian_optimization.py#L65

    Attributes
    ----------
    random_state : int or numpy.random.RandomState
        if the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provided is used.
        When set to None, an unseeded random state is generated
        (default = None).
    verbose : int
        The level of verbosity (default = 2).
    bounds_transformer : DomainTransformer
        If provided, the transformation is applied to the bounds (default = None).
    init_points : int
        Number of iterations before the explorations starts the exploration
        for the maximum (default = 5).
    acq : {'ucb', 'ei', 'poi'}
        The acquisition function used (default = 'ucb').
            - 'ucb' stands for the Upper Confidence Bounds method
            - 'ei' is the Expected Improvement method
            - 'poi' is the Probability Of Improvement criterion.
    kappa : float
        Parameter to indicate how closed are the next parameters sampled
        (default = 2.576):
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is the
            highest.
    kappa_decay : float
        `kappa` is multiplied by this factor every iteration (default = 1).
    kappa_decay_delay : int
        Number of iterations that must have passed before applying the decay
        to `kappa` (default = 0).
    xi : flaot
        [unused] (default = 0.0)
    **gp_params : arbitrary keyword arguments
        Set parameters to the internal Gaussian Process Regressor.


    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, random_state=None, verbose=0, bounds_transformer=None,
                 init_points=5, acq='ucb', kappa=2.576, kappa_decay=1,
                 kappa_decay_delay=0, xi=0.0, log_space=False, **gp_params):
        """
        Constructs all the necessary attributes for the Bayesian optimiser

        Parameters
        ----------
        random_state : int or numpy.random.RandomState
            if the value is an integer, it is used as the seed for creating a
            numpy.random.RandomState. Otherwise the random state provided is used.
            When set to None, an unseeded random state is generated
            (default = None).
        verbose : int
            The level of verbosity (default = 2).
        bounds_transformer : DomainTransformer
            If provided, the transformation is applied to the bounds (default = None).
        init_points : int
            Number of iterations before the explorations starts the exploration
            for the maximum (default = 5).
        acq : {'ucb', 'ei', 'poi'}
            The acquisition function used (default = 'ucb').
                - 'ucb' stands for the Upper Confidence Bounds method
                - 'ei' is the Expected Improvement method
                - 'poi' is the Probability Of Improvement criterion.
        kappa : float
            Parameter to indicate how closed are the next parameters sampled
            (default = 2.576):
                Higher value = favors spaces that are least explored.
                Lower value = favors spaces where the regression function is the
                highest.
        kappa_decay : float
            `kappa` is multiplied by this factor every iteration (default = 1).
        kappa_decay_delay : int
            Number of iterations that must have passed before applying the decay
            to `kappa` (default = 0).
        xi : float
            [unused] (default = 0.0)
        log_space : bool
            Whether to optimise the loss in a log space (requires non-negative losses)
        **gp_params : arbitrary keyword arguments
            Set parameters to the internal Gaussian Process Regressor.

        """
        if BayesianOptimization is None:
            raise ImportError("bayes_opt failed to load. Check your installation.")
        super().__init__('BO')
        self.random_state = random_state
        self.verbose = verbose
        self.bounds_transformer = bounds_transformer
        self.init_points = init_points
        self.acq = acq
        self.kappa = kappa
        self.kappa_decay = kappa_decay
        self.kappa_decay_delay = kappa_decay_delay
        self.xi = xi
        self.gp_params = gp_params
        self.log_space = log_space

    def _optimise(
        self,
        objective: SingleObjective,
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ):
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
        # Convert the optimization problem to a minimization problem by negating the
        # optimization function
        loss_transformer = lambda loss: np.log(loss) if self.log_space else loss
        def negate(**kwargs):
            return -loss_transformer(objective(list(kwargs.values())))
        # Convert the bounds in array type to dicitionary type, as required in the
        # BayesianOptimization documentation
        bound = tuple(map(tuple, bound))
        patterns = [par.name for par in self.parameters]
        pbounds = dict(zip(patterns, bound))
        # Run optimization problem
        model = BayesianOptimizationMod(negate, pbounds, self.random_state,
                                        self.verbose, self.bounds_transformer)
        model.probe(init_shot)
        model.maximize(self, self.init_points, n_iter, self.acq, self.kappa,
                       self.kappa_decay, self.kappa_decay_delay, self.xi, self.log_space,
                       **self.gp_params)
        # Best solution
        x = model.max.get('params').values()
        return list(x), model.max.get('target')
