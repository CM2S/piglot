"""LIPO optimiser module."""
import numpy as np
try:
    from lipo import GlobalOptimizer
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimisers.optimiser import missing_method
    GlobalOptimizer = missing_method("LIPO", "lipo")
from piglot.objective import SingleObjective
from piglot.optimisers.optimiser import ScalarOptimiser


class GlobalOptimizerMod(GlobalOptimizer):

    def run(self, optimiser, num_function_calls: int = 1):
        """
        run optimization
        Args:
            num_function_calls (int): number of function calls
        """
        for i in range(num_function_calls):
            candidate = self.get_candidate()
            candidate.set(self.function(**candidate.x))
            solution = self.evaluations[i]
            if optimiser._progress_check(i+1, -solution[1], list(solution[0].values())):
                break


class LIPO(ScalarOptimiser):
    """
    LIPO optimiser.
    Documentation:
        https://github.com/jdb78/lipo
        http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html

    Attributes
    ----------
    log_args : list[str]
        list of arguments to treat in log space, if "auto", then a variable is
        optimized in log space if (default = 'auto'):
            - The lower bound on the variable is > 0
            - The ratio of the upper bound to lower bound is > 1000
            - The variable is not an integer variable
    flexible_bounds : dict[str, list[bool]]
        dictionary of parameters and list of booleans indicating if parameters are
        deemed flexible or not. By default all parameters are deemed flexible
        but only if `flexible_bound_threshold > 0` (default = {}).
    flexible_bound_threshold : float
        enlarge bounds if optimum is top or bottom flexible_bound_threshold quantile
        (default = -1.0)
    epsilon : float
        accuracy below which exploration will be priorities vs exploitation
        (default = 0)
    random_state : int
        random state


    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, log_args='auto', flexible_bounds={}, flexible_bound_threshold=-1.0,
                 epsilon=0.0, random_state=None):
        """
        Constructs all the necessary attributes for the LIPO optimiser

        Parameters
        ----------
        log_args : list[str]
            list of arguments to treat in log space, if "auto", then a variable is
            optimized in log space if (default = 'auto'):
                - The lower bound on the variable is > 0
                - The ratio of the upper bound to lower bound is > 1000
                - The variable is not an integer variable
        flexible_bounds : dict[str, list[bool]]
            dictionary of parameters and list of booleans indicating if parameters are
            deemed flexible or not. By default all parameters are deemed flexible
            but only if `flexible_bound_threshold > 0` (default = {}).
        flexible_bound_threshold : float
            enlarge bounds if optimum is top or bottom flexible_bound_threshold quantile
            (default = -1.0)
        epsilon : float
            accuracy below which exploration will be prioritized vs exploitation
            (default = 0)
        random_state : int
            random state
        """
        super().__init__('LIPO')
        self.log_args = log_args
        self.flexible_bounds = flexible_bounds
        self.flexible_bound_threshold = flexible_bound_threshold
        self.flexible_bound_threshold = flexible_bound_threshold
        self.epsilon = epsilon
        self.random_state = random_state

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
        # Set optimization problem as a maximization one
        maximize = True
        # Convert the optimization problem to a minimization problem by negating the
        # optimization function
        def negate(**kwargs):
            return -objective(list(kwargs.values()))
        #def negate(**kwargs):
        #    return func(list(kwargs.values()))
        # Convert the bounds in array type to dicitionary type, as required in the
        # GlobalOptimizer documentation
        lower_bounds = {}
        upper_bounds = {}
        patterns = [par.name for par in self.parameters]
        if bound is not None:
            lower_bounds = dict(zip(patterns, bound[:,0]))
            upper_bounds = dict(zip(patterns, bound[:,1]))
        # Compute loss function value given the initial shot
        evaluations = []
        if init_shot is not None:
            pre_eval_x = dict(zip(patterns, init_shot))
            evaluations = [(pre_eval_x, negate(**pre_eval_x))]
        # Run optimization problem
        categories = {}
        model = GlobalOptimizerMod(negate, lower_bounds, upper_bounds, categories,
                                   self.log_args, self.flexible_bounds,
                                   self.flexible_bound_threshold, evaluations, maximize,
                                   self.epsilon, self.random_state)
        model.run(self, n_iter)
        return list(model.optimum[0].values()), model.optimum[1]
