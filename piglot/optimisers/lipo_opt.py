"""LIPO optimiser module."""
from typing import Tuple, Callable, Optional
import numpy as np
try:
    from lipo import GlobalOptimizer
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimiser import missing_method
    GlobalOptimizer = missing_method("LIPO", "lipo")
from piglot.objective import Objective
from piglot.optimiser import ScalarOptimiser


class GlobalOptimizerMod(GlobalOptimizer):

    def run(self, optimiser, num_function_calls: int = 1):
        """Run optimization.

        Parameters
        ----------
        num_function_calls : int
            Number of function calls
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

    def __init__(self, objective: Objective, log_args='auto', flexible_bounds={},
                 flexible_bound_threshold=-1.0, epsilon=0.0, seed=None):
        """
        Constructs all the necessary attributes for the LIPO optimiser

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
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
        seed : int
            random state
        """
        super().__init__('LIPO', objective)
        self.log_args = log_args
        self.flexible_bounds = flexible_bounds
        self.flexible_bound_threshold = flexible_bound_threshold
        self.flexible_bound_threshold = flexible_bound_threshold
        self.epsilon = epsilon
        self.seed = seed

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
        # Set optimization problem as a maximization one
        maximize = True

        # Convert the optimization problem to a minimization problem by negating the
        # optimization function
        def negate(**kwargs):
            return -objective(list(kwargs.values()))
        # Convert the bounds in array type to dicitionary type, as required in the
        # GlobalOptimizer documentation
        lower_bounds = {}
        upper_bounds = {}
        patterns = [par.name for par in self.parameters]
        if bound is not None:
            lower_bounds = dict(zip(patterns, bound[:, 0]))
            upper_bounds = dict(zip(patterns, bound[:, 1]))
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
                                   self.epsilon, self.seed)
        model.run(self, n_iter)
        return list(model.optimum[0].values()), model.optimum[1]
