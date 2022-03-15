"""BayesSkopt optimiser module."""
try:
    from skopt.utils import create_result
    from bask import Optimizer
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimiser import missing_method
    Optimizer = missing_method("SK Bayesian optimisation", "bask")
from piglot.optimiser import Optimiser


class OptimizerMod(Optimizer):

    def run(self, optimiser, func, n_iter=1, replace=False, n_samples=5,
            gp_samples=100, gp_burnin=10):
        """Execute the ask/tell-loop on a given objective function.
        Parameters
        ----------
        func : function
            The objective function to minimize. Should either return a scalar value,
            or a tuple (value, noise) where the noise should be a variance.
        n_iter : int, optional (default: 1)
            Number of iterations to perform.
        replace : bool, optional (default: False)
            If True, the existing data points will be replaced with the ones collected
            from now on. The existing model will be used as initialization.
        n_samples : int, optional (default: 5)
            Number of hyperposterior samples over which to average the acquisition
            function.
        gp_samples : int, optional (default: 100)
            Number of hyperposterior samples to collect during inference. More samples
            result in a more accurate representation of the hyperposterior, but
            increase the running time.
            Has to be a multiple of 100.
        gp_burnin : int, optional (default: 10)
            Number of inference iterations to discard before beginning collecting
            hyperposterior samples. Only needs to be increased, if the hyperposterior
            after burnin has not settled on the typical set. Drastically increases
            running time.
        Returns
        -------
        scipy.optimize.OptimizeResult object
            Contains the points, the values of the objective function, the search space,
            the random state and the list of models.
        """
        for iiter in range(n_iter):
            x = self.ask()
            out = func(x)
            if hasattr(out, "__len__"):
                val, noise = out
            else:
                val = out
                noise = 0.0
            self.tell(
                x,
                val,
                noise_vector=noise,
                n_samples=n_samples,
                gp_samples=gp_samples,
                gp_burnin=gp_burnin,
                replace=replace,
            )
            replace = False
            # Check history of the optimization problem
            if optimiser._progress_check(iiter, self.yi[iiter], self.Xi[iiter]):
                break

        return create_result(self.Xi, self.yi, self.space, self.rng, models=[self.gp])


class BayesSkopt(Optimiser):
    """
    BayesSkopt optimiser.
    Documentation:
        https://github.com/kiudee/bayes-skopt/blob/master/bask/optimizer.py

    Attributes
    ----------
    priors : list of callables, optional
        List of prior distributions for the kernel hyperparameters of the GP.
        Each callable returns the logpdf of the prior distribution.
        Remember that a WhiteKernel is added to the ``gp_kernel``, which is why
        you need to include a prior distribution for that as well.
        If None, will try to guess suitable prior distributions.
    n_initial_points : int, default=10
        Number of initial points to sample before fitting the GP.
    init_strategy : string or None, default="sb"
        Type of initialization strategy to use for the initial
        ``n_initial_points``. Should be one of:
        - "sb": The Steinberger low-discrepancy sequence
        - "r2": The R2 sequence (works well for up to two parameters)
        - "random" or None: Uniform random sampling
    acq_func : string or Acquisition object, default="pvrs"
        Acquisition function to use as a criterion to select new points to test.
        By default we use "pvrs", which is a very robust criterion with fast
        convergence.
        Should be one of
            - 'pvrs' Predictive variance reductions search
            - 'mes' Max-value entropy search
            - 'ei' Expected improvement
            - 'ttei' Top-two expected improvement
            - 'lcb' Lower confidence bound
            - 'mean' Expected value of the GP
            - 'ts' Thompson sampling
            - 'vr' Global variance reduction
        Can also be a custom :class:`Acquisition` object.
    n_points : int
        Number of points to return. This is currently not implemented and will raise
        a NotImplementedError.
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, a suitable default kernel is constructed.
        Note that the kernel’s hyperparameters are estimated using MCMC during
        fitting.


    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, priors=None, n_initial_points=10, init_strategy='sb',
                 acq_func='pvrs', n_points=500, kernel=None):
        """
        Constructs all the necessary attributes for the BayesSkopt optimiser

        Parameters
        ----------
        priors : list of callables, optional
            List of prior distributions for the kernel hyperparameters of the GP.
            Each callable returns the logpdf of the prior distribution.
            Remember that a WhiteKernel is added to the ``gp_kernel``, which is why
            you need to include a prior distribution for that as well.
            If None, will try to guess suitable prior distributions.
        n_initial_points : int, default=10
            Number of initial points to sample before fitting the GP.
        init_strategy : string or None, default="sb"
            Type of initialization strategy to use for the initial
            ``n_initial_points``. Should be one of:
            - "sb": The Steinberger low-discrepancy sequence
            - "r2": The R2 sequence (works well for up to two parameters)
            - "random" or None: Uniform random sampling
        acq_func : string or Acquisition object, default="pvrs"
            Acquisition function to use as a criterion to select new points to test.
            By default we use "pvrs", which is a very robust criterion with fast
            convergence.
            Should be one of
                - 'pvrs' Predictive variance reductions search
                - 'mes' Max-value entropy search
                - 'ei' Expected improvement
                - 'ttei' Top-two expected improvement
                - 'lcb' Lower confidence bound
                - 'mean' Expected value of the GP
                - 'ts' Thompson sampling
                - 'vr' Global variance reduction
            Can also be a custom :class:`Acquisition` object.
        n_points : int
            Number of points to return. This is currently not implemented and will raise
            a NotImplementedError.
        kernel : kernel object
            The kernel specifying the covariance function of the GP. If None is
            passed, a suitable default kernel is constructed.
            Note that the kernel's hyperparameters are estimated using MCMC during
            fitting.
        """
        if Optimizer is None:
            raise ImportError("BayesSkopt failed to load. Check your installation.")
        self.priors = priors
        self.n_initial_points = n_initial_points
        self.init_strategy = init_strategy
        self.acq_func = acq_func
        self.n_points = n_points
        self.kernel = kernel
        self.name = 'BO skopt'

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
        # Run optimization problem
        opt = OptimizerMod(list(map(tuple, bound)),
                           n_points=self.n_points,
                           n_initial_points=self.n_initial_points,
                           init_strategy=self.init_strategy,
                           gp_kernel=self.kernel,
                           gp_kwargs=dict(normalize_y=False),
                           gp_priors=self.priors,
                           acq_func=self.acq_func,
                           acq_func_kwargs=dict(n_thompson=3),
                           random_state=None)

        model = opt.run(self, func, n_iter, replace=False, n_samples=5,
                        gp_samples=100, gp_burnin=10)
        return model.x, model.fun
