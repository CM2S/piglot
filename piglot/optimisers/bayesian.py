"""Bayesian optimiser module."""
import numpy as np
import warnings
from multiprocessing.pool import ThreadPool as Pool
try:
    from bayes_opt import BayesianOptimization
    from bayes_opt.util import UtilityFunction
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimisers.optimiser import missing_method
    BayesianOptimization = missing_method("Bayesian optimisation", "bayes_opt")
from piglot.optimisers.optimiser import Optimiser



class Bayesian(Optimiser):
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

    def __init__(self, random_state=1, verbose=0, bounds_transformer=None,
                 init_points=5, acq='ucb', kappas=[2.576], kappa_decay=1,
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
        self.random_state = random_state
        self.verbose = verbose
        self.bounds_transformer = bounds_transformer
        self.init_points = init_points
        self.acq = acq
        self.kappas = kappas
        self.kappa_decay = kappa_decay
        self.kappa_decay_delay = kappa_decay_delay
        self.xi = xi
        self.gp_params = gp_params
        self.log_space = log_space
        self.name = 'BO'

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
        # Convert the optimization problem to a minimization problem by negating the
        # optimization function
        loss_transformer = lambda loss: -np.log(-loss) if self.log_space else loss
        def negate(x):
            return -func(x, unique=True)
        # Convert the bounds in array type to dicitionary type, as required in the
        # BayesianOptimization documentation
        bound = tuple(map(tuple, bound))
        patterns = [par.name for par in self.parameters]
        pbounds = dict(zip(patterns, bound))
        # Run optimization problem
        optimiser = BayesianOptimization(None, pbounds, self.random_state,
                                         self.verbose, self.bounds_transformer)
        utilities = [UtilityFunction(kind=self.acq, kappa=kappa, xi=self.xi, kappa_decay_delay=self.kappa_decay_delay) for kappa in self.kappas]
        optimiser.set_gp_params(**self.gp_params)
        optimiser.probe(init_shot)
        pool = Pool(len(utilities))
        for iiter in range(n_iter):
            # utility = utilities[iiter % len(utilities)]
            # utility.update_params()
            # next_points = [optimiser.suggest(utility)]
            for utility in utilities:
                utility.update_params()
            next_points = [optimiser.suggest(utility) for utility in utilities]
            x = [list(next_point.values()) for next_point in next_points]
            targets = pool.map(negate, x)
            min_target = 0
            for i, next_point in enumerate(next_points):
                optimiser.register(next_point, loss_transformer(targets[i]))
                if targets[i] == min(targets):
                    min_target = i
            if self._progress_check(iiter, -targets[min_target], x[min_target]):
                break
        # Best solution
        x = optimiser.max.get('params').values()
        return list(x), -optimiser.max.get('target')
