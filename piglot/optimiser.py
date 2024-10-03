"""Main optimiser module"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Callable, Optional
import os
import time
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from piglot.parameter import ParameterSet
from piglot.utils.assorted import pretty_time
from piglot.objective import Objective, GenericObjective


def boundary_check(arg, bounds):
    """Check if the values are within the bounds and correct them if not.

    Parameters
    ----------
    arg : array
        Values to check.
    bounds : array
        Lower and upper bounds.

    Returns
    -------
    array
        Corrected values.
    """
    arg = np.where(arg > bounds[:, 1], bounds[:, 1], arg)
    arg = np.where(arg < bounds[:, 0], bounds[:, 0], arg)
    return arg


def missing_method(name, package):
    """Class generator for missing packages.

    Parameters
    ----------
    name : str
        Name of the missing method.
    package : str
        Name of the package to install.
    """
    def err_func(name, package):
        """Raise an error for this missing method.

        Parameters
        ----------
        name : str
            Name of the missing method.
        package : str
            Name of the package to install.

        Raises
        ------
        ImportError
            Every time it is called.
        """
        raise ImportError(f"{name} is not available. You need to install package {package}!")

    return type(f'Missing_{package}', (),
                {
                    'name': name,
                    'package': package,
                    '__init__': (lambda *args, **kwargs: err_func(name, package))
                })


class StoppingCriteria:
    """
    Implements different stopping criteria.

    Attributes
    ----------
    conv_tol : float
        Stop the optimiser if the loss becomes small than this value.
    max_iters_no_improv : int
        Stop the optimiser if the loss does not improve after this number of iterations in a row.
    max_func_calls : int
        Stop the optimiser after this number of function calls.
    max_timeout : float
        Stop the optimiser after this elapsed time (in seconds).

    Methods
    -------
    check_criteria(loss_value, iters_no_improv, func_calls):
        check the status of the stopping criteria.
    """
    def __init__(
            self,
            conv_tol: float = None,
            max_iters_no_improv: int = None,
            max_func_calls: int = None,
            max_timeout: float = None,
            ):
        """
        Constructs all the necessary attributes for the stopping criteria.

        Parameters
        ----------
        conv_tol : float
            Stop the optimiser if the loss becomes small than this value.
        max_iters_no_improv : int
            Stop the optimiser if the loss does not improve after this number of iterations.
        max_func_calls : int
            Stop the optimiser after this number of function calls.
        max_timeout : float
            Stop the optimiser after this elapsed time (in seconds).
        """
        self.conv_tol = conv_tol
        self.max_iters_no_improv = max_iters_no_improv
        self.max_func_calls = max_func_calls
        self.max_timeout = max_timeout

    def check_criteria(
            self,
            loss_value: float,
            iters_no_improv: int,
            func_calls: int,
            elapsed: float,
            ) -> bool:
        """
        Check the status of the stopping criteria.

        Parameters
        ----------
        loss_value : float
            Current loss value.
        iters_no_improv : int
            Current number of iterations without improvement.
        func_calls : int
            Current number of function calls.
        elapsed : float
            Elapsed time in seconds.

        Returns
        -------
        bool
            Whether any of the criteria are satisfied.
        """
        conv = self.conv_tol is not None and loss_value < self.conv_tol
        func = self.max_func_calls is not None and func_calls > self.max_func_calls
        impr = self.max_iters_no_improv is not None and iters_no_improv > self.max_iters_no_improv
        timr = self.max_timeout is not None and elapsed > self.max_timeout
        return conv or impr or func or timr

    @staticmethod
    def read(config: Dict[str, Any]) -> StoppingCriteria:
        """Read the stopping criteria from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        StoppingCriteria
            Stopping criteria.
        """
        def optional_get(key, convert):
            return convert(config[key]) if key in config else None
        return StoppingCriteria(
            conv_tol=optional_get('conv_tol', float),
            max_iters_no_improv=optional_get('max_iters_no_improv', int),
            max_func_calls=optional_get('max_func_calls', int),
            max_timeout=optional_get('max_timeout', float),
        )


class InvalidOptimiserException(Exception):
    """Exception signaling invalid combination of optimiser and objective function."""


class Optimiser(ABC):
    """
    Interface for implementing different optimization algorithms.

    Methods
    -------
    _init_optimiser(n_iter, parameters, pbar, loss, stop_criteria):
        constructs the attributes for the optimiser.
    optimise(loss, n_iter, parameters, stop_criteria = StoppingCriteria()):
        initiates optimiser.
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        performs the optimization.
    _progress_check(self, i_iter, curr_value, curr_solution):
        evaluates the optimiser progress.
    """

    def __init__(self, name: str, objective: Objective) -> None:
        self.name = name
        self.objective = objective
        self.parameters = None
        self.i_iter = None
        self.n_iter = None
        self.iters_no_improv = None
        self.pbar = None
        self.stop_criteria = None
        self.output_dir = None
        self.best_value = None
        self.best_solution = None
        self.begin_time = None

    @abstractmethod
    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective.

        Parameters
        ----------
        objective : Objective
            Objective to optimise.
        """

    def optimise(
            self,
            n_iter: int,
            parameters: ParameterSet,
            output_dir: str,
            stop_criteria: StoppingCriteria = StoppingCriteria(),
            verbose: bool = True,
            ) -> Tuple[float, np.ndarray]:
        """
        Optimiser for the outside world.

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
        n_iter : int
            Maximum number of iterations.
        parameters : ParameterSet
            Set of parameters to optimise.
        output_dir : str
            Whether to write output to the output directory, by default None.
        stop_criteria : StoppingCriteria
            List of stopping criteria, by default none attributed.
        verbose : bool
            Whether to output progress status, by default True.

        Returns
        -------
        float
            Best observed objective value.
        np.ndarray
            Observed optimum of the objective.
        """
        # Sanity check
        self._validate_problem(self.objective)
        # Initialise optimiser
        self.n_iter = n_iter
        self.parameters = parameters
        self.stop_criteria = stop_criteria
        self.output_dir = output_dir
        self.iters_no_improv = 0
        # Build initial shot and bounds
        n_dim = len(self.parameters)
        init_shot = np.array([par.inital_value for par in self.parameters])
        bounds = np.array([[par.lbound, par.ubound] for par in self.parameters])
        # Build best solution
        self.best_value = np.nan
        self.best_solution = None
        # Prepare history output files
        with open(os.path.join(self.output_dir, "history"), 'w', encoding='utf8') as file:
            file.write(f'{"Iteration":>10}\t')
            file.write(f'{"Time /s":>15}\t')
            file.write(f'{"Best Loss":>15}\t')
            file.write(f'{"Current Loss":>15}\t')
            for par in self.parameters:
                file.write(f'{par.name:>15}\t')
            file.write('\tOptimiser info')
            file.write('\n')
        # Prepare optimiser
        self.objective.prepare()
        self.pbar = tqdm(total=n_iter, desc=self.name) if verbose else None
        # Optimise
        self.begin_time = time.perf_counter()
        self._optimise(n_dim, n_iter, bounds, init_shot)
        elapsed = time.perf_counter() - self.begin_time
        # Output progress
        if verbose:
            self.pbar.close()
            print(f'Completed {self.i_iter} iterations in {pretty_time(elapsed)}')
            print(f'Best loss: {self.best_value:15.8e}')
            if self.best_solution is not None:
                print('Best parameters')
                max_width = max(len(par.name) for par in self.parameters)
                for i, par in enumerate(self.parameters):
                    print(f'- {par.name.rjust(max_width)}: {self.best_solution[i]:>12.6f}')
        # Return the best value
        return self.best_value, self.best_solution

    @abstractmethod
    def _optimise(
        self,
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Abstract method for optimising the objective.

        Parameters
        ----------
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

    def __update_progress_files(
            self,
            i_iter: int,
            curr_solution: np.ndarray,
            curr_value: float,
            extra_info: str,
            ) -> None:
        """Update progress on output files.

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        curr_solution : np.ndarray
            Current objective minimiser.
        curr_value : float
            Current objective value.
        extra_info : str
            Additional information to pass to user.
        """
        elapsed = time.perf_counter() - self.begin_time
        skip_pars = curr_solution is None
        # Update progress file
        with open(os.path.join(self.output_dir, "progress"), 'w', encoding='utf8') as file:
            file.write(f'Iteration: {i_iter}\n')
            file.write(f'Function calls: {self.objective.func_calls}\n')
            file.write(f'Best loss: {self.best_value}\n')
            if extra_info is not None:
                file.write(f'Optimiser info: {extra_info}\n')
            if not skip_pars:
                file.write('Best parameters:\n')
                for i, par in enumerate(self.parameters):
                    file.write(f'\t{par.name}: {self.best_solution[i]}\n')
            file.write(f'\nElapsed time: {pretty_time(elapsed)}\n')
        # Update history file
        with open(os.path.join(self.output_dir, "history"), 'a', encoding='utf8') as file:
            file.write(f'{i_iter:>10}\t')
            file.write(f'{elapsed:>15.8e}\t')
            file.write(f'{self.best_value:>15.8e}\t')
            file.write(f'{curr_value:>15.8e}\t')
            for i, par in enumerate(self.parameters):
                file.write('None\t'.rjust(16) if skip_pars else f'{curr_solution[i]:>15.8f}\t')
            file.write(f"\t{'-' if extra_info is None else extra_info}")
            file.write('\n')

    def _progress_check(
            self,
            i_iter: int,
            curr_value: float,
            curr_solution: np.ndarray,
            extra_info: str = None,
            ) -> bool:
        """
        Report the optimiser progress and check for termination.

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        curr_value : float
            Current objective value.
        curr_solution : np.ndarray
            Current objective minimiser.
        extra_info : str
            Additional information to pass to user.

        Returns
        -------
        bool
            Whether any of the stopping criteria is satisfied.
        """
        # Update new value to best value
        self.i_iter = i_iter
        if curr_value >= self.best_value:
            self.iters_no_improv += 1
        else:
            self.best_value = curr_value
            self.best_solution = curr_solution
            self.iters_no_improv = 0
        # Update progress bar
        if self.pbar is not None:
            info = f'Loss: {self.best_value:6.4e}' + (f' ({extra_info})' if extra_info else '')
            self.pbar.set_postfix_str(info)
            if i_iter > 0:
                self.pbar.update()
        # Update progress in output files
        self.__update_progress_files(i_iter, curr_solution, curr_value, extra_info)
        # Convergence criterion
        return i_iter > self.n_iter or self.stop_criteria.check_criteria(
            curr_value,
            self.iters_no_improv,
            self.objective.func_calls,
            time.perf_counter() - self.begin_time,
        )


class ScalarOptimiser(Optimiser):
    """Base class for scalar optimisers."""

    def __init__(self, name: str, objective: Objective) -> None:
        super().__init__(name, objective)
        self.bounds = None

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective.

        Parameters
        ----------
        objective : Objective
            Objective to optimise.

        Raises
        ------
        InvalidOptimiserException
            With an invalid combination of optimiser and objective function.
        """
        if not isinstance(objective, GenericObjective):
            raise InvalidOptimiserException('Generic objective required for this optimiser')
        if objective.composition is not None:
            raise InvalidOptimiserException('This optimiser does not support composition')
        if objective.stochastic:
            raise InvalidOptimiserException('This optimiser does not support stochasticity')

    @abstractmethod
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

    def _norm_params(self, params: np.ndarray) -> np.ndarray:
        """Normalise the parameters.

        Parameters
        ----------
        params : np.ndarray
            Denormalised parameters.

        Returns
        -------
        np.ndarray
            Normalised parameters.
        """
        return 2.0 * (params - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0]) - 1.0

    def _denorm_params(self, params: np.ndarray) -> np.ndarray:
        """Denormalise the parameters.

        Parameters
        ----------
        params : np.ndarray
            Normalised parameters.

        Returns
        -------
        np.ndarray
            Denormalised parameters.
        """
        return self.bounds[:, 0] + (1.0 + params) * (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0

    def _optimise(
        self,
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Abstract method for optimising the objective.

        Parameters
        ----------
        objective : Objective
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
        self.bounds = bound
        # Optimise the scalarised objective
        return self._scalar_optimise(
            lambda x, concurrent=False: self.objective(
                self._denorm_params(x),
                concurrent=concurrent
            ).values.item(),
            n_dim,
            n_iter,
            np.array([[-1.0, 1.0]]).repeat(n_dim, axis=0),
            self._norm_params(init_shot),
        )

    def _progress_check(
        self,
        i_iter: int,
        curr_value: float,
        curr_solution: np.ndarray,
        extra_info: str = None,
    ) -> bool:
        """
        Report the optimiser progress and check for termination (with parameter denormalisation).

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        curr_value : float
            Current objective value.
        curr_solution : np.ndarray
            Current objective minimiser.
        extra_info : str
            Additional information to pass to user.

        Returns
        -------
        bool
            Whether any of the stopping criteria is satisfied.
        """
        denorm_solution = None if curr_solution is None else self._denorm_params(curr_solution)
        super()._progress_check(i_iter, curr_value, denorm_solution, extra_info)
