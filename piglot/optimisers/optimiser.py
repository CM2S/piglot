"""Main optimiser module"""
import os
import time
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from tqdm import tqdm
from piglot.parameter import ParameterSet
from piglot.objective import Objective, SingleObjective, SingleCompositeObjective
from piglot.objective import MultiFidelitySingleObjective, MultiFidelityCompositeObjective


def pretty_time(elapsed_sec):
    """Return a human-readable representation of a given elapsed time

    Parameters
    ----------
    elapsed_sec : float
        Elapsed time, in seconds

    Returns
    -------
    str
        Pretty elapsed time string
    """
    mults = {
        'y': 60*60*24*365,
        'd': 60*60*24,
        'h': 60*60,
        'm': 60,
        's': 1,
    }
    time_str = ''
    for suffix, factor in mults.items():
        count = elapsed_sec // factor
        if count > 0:
            time_str += str(int(elapsed_sec / factor)) + suffix
        elapsed_sec %= factor
    if time_str == '':
        time_str = f'{elapsed_sec:.5f}s'
    return time_str



def boundary_check(arg, bounds):
    """Check if the values are within the bounds and correct them if not

    Parameters
    ----------
    arg : array
        Values to check
    bounds : array
        Lower and upper bounds

    Returns
    -------
    array
        Corrected values
    """
    arg = np.where(arg > bounds[:,1], bounds[:,1], arg)
    arg = np.where(arg < bounds[:,0], bounds[:,0], arg)
    return arg


def missing_method(name, package):
    """Class generator for missing packages

    Parameters
    ----------
    name : str
        Name of the missing method
    package : str
        Name of the package to install
    """
    def err_func(name, package):
        """Raise an error for this missing method

        Parameters
        ----------
        name : str
            Name of the missing method
        package : str
            Name of the package to install

        Raises
        ------
        ImportError
            Every time it is called
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
    Implements different stopping criteria

    Attributes
    ----------
    conv_tol : float
        Stop the optimiser if the loss becomes small than this value
    max_iters_no_improv : int
        Stop the optimiser if the loss does not improve after this number of iterations in a row
    max_func_calls : int
        Stop the optimiser after this number of function calls
    max_timeout : float
        Stop the optimiser after this elapsed time (in seconds)

    Methods
    -------
    check_criteria(loss_value, iters_no_improv, func_calls):
        check the status of the stopping criteria
    """
    def __init__(
            self,
            conv_tol: float=None,
            max_iters_no_improv: int=None,
            max_func_calls: int=None,
            max_timeout: float=None,
        ):
        """
        Constructs all the necessary attributes for the stopping criteria

        Parameters
        ----------
        conv_tol : float
            Stop the optimiser if the loss becomes small than this value
        max_iters_no_improv : int
            Stop the optimiser if the loss does not improve after this number of iterations in a row
        max_func_calls : int
            Stop the optimiser after this number of function calls
        max_timeout : float
            Stop the optimiser after this elapsed time (in seconds)
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
        Check the status of the stopping criteria

        Parameters
        ----------
        loss_value : float
            Current loss value
        iters_no_improv : int
            Current number of iterations without improvement
        func_calls : int
            Current number of function calls
        elapsed : float
            Elapsed time in seconds

        Returns
        -------
        bool
            Whether any of the criteria are satisfied
        """
        conv = self.conv_tol is not None and loss_value < self.conv_tol
        func = self.max_func_calls is not None and func_calls > self.max_func_calls
        impr = self.max_iters_no_improv is not None and iters_no_improv > self.max_iters_no_improv
        timr = self.max_timeout is not None and elapsed > self.max_timeout
        return conv or impr or func or timr



class InvalidOptimiserException(Exception):
    """Exception signaling invalid combination of optimiser and objective function"""



class Optimiser(ABC):
    """
    Interface for implementing different optimization algorithms

    Methods
    -------
    _init_optimiser(n_iter, parameters, pbar, loss, stop_criteria):
        constructs the attributes for the optimiser
    optimise(loss, n_iter, parameters, stop_criteria = StoppingCriteria()):
        initiates optimiser
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        performs the optimization
    _progress_check(self, i_iter, curr_value, curr_solution):
        evaluates the optimiser progress
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.parameters = None
        self.i_iter = None
        self.n_iter = None
        self.iters_no_improv = None
        self.pbar = None
        self.objective = None
        self.stop_criteria = None
        self.output = None
        self.best_value = None
        self.best_solution = None
        self.begin_time = None


    @abstractmethod
    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective

        Parameters
        ----------
        objective : Objective
            Objective to optimise
        """


    def optimise(
            self,
            objective: Objective,
            n_iter: int,
            parameters: ParameterSet,
            stop_criteria: StoppingCriteria=StoppingCriteria(),
            output: str=None,
            verbose: bool=True,
        ) -> Tuple[float, np.ndarray]:
        """
        Optimiser for the outside world

        Parameters
        ----------
        objective : Objective
            Objective function to optimise
        n_iter : int
            Maximum number of iterations
        parameters : ParameterSet
            Set of parameters to optimise
        stop_criteria : StoppingCriteria
            List of stopping criteria, by default none attributed
        output : bool
            Whether to write output to the output directory, by default None
        verbose : bool
            Whether to output progress status, by default True

        Returns
        -------
        float
            Best observed objective value
        np.ndarray
            Observed optimum of the objective
        """
        # Sanity check
        self._validate_problem(objective)
        # Initialise optimiser
        self.pbar = tqdm(total=n_iter, desc=self.name) if verbose else None
        self.n_iter = n_iter
        self.parameters = parameters
        self.objective = objective
        self.stop_criteria = stop_criteria
        self.output = output
        self.iters_no_improv = 0
        # Build initial shot and bounds
        n_dim = len(self.parameters)
        init_shot = [par.normalise(par.inital_value) for par in self.parameters]
        bounds = np.ones((n_dim, 2))
        bounds[:,0] = -1
        # Build best solution
        self.best_value = np.inf
        self.best_solution = None
        # Prepare history output files
        if output is not None:
            with open(os.path.join(self.output, "history"), 'w', encoding='utf8') as file:
                file.write(f'{"Iteration":>10}\t')
                file.write(f'{"Time /s":>15}\t')
                file.write(f'{"Best Loss":>15}\t')
                file.write(f'{"Current Loss":>15}\t')
                for par in self.parameters:
                    file.write(f'{par.name:>15}\t')
                file.write('\tOptimiser info')
                file.write('\n')
        # Optimise
        self.begin_time = time.perf_counter()
        self._optimise(objective, n_dim, n_iter, bounds, init_shot)
        elapsed = time.perf_counter() - self.begin_time
        # Denormalise best solution
        new_solution = self.parameters.denormalise(self.best_solution)
        if verbose:
            self.pbar.close()
            print(f'Completed {self.i_iter} iterations in {pretty_time(elapsed)}')
            print(f'Best loss: {self.best_value:15.8e}')
            print('Best parameters')
            max_width = max(len(par.name) for par in self.parameters)
            for i, par in enumerate(self.parameters):
                print(f'- {par.name.rjust(max_width)}: {new_solution[i]:>12.6f}')
        # Return the best value
        return self.best_value, np.array(new_solution)


    @abstractmethod
    def _optimise(
        self,
        objective: Objective,
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Abstract method for optimising the objective

        Parameters
        ----------
        objective : Objective
            Objective function to optimise
        n_dim : int
            Number of parameters to optimise
        n_iter : int
            Maximum number of iterations
        bound : np.ndarray
            Array where first and second columns correspond to lower and upper bounds, respectively
        init_shot : np.ndarray
            Initial shot for the optimisation problem

        Returns
        -------
        float
            Best observed objective value
        np.ndarray
            Observed optimum of the objective
        """


    def __update_progress_files(
            self,
            i_iter: int,
            curr_solution: float,
            curr_value: np.ndarray,
            extra_info: str,
        ) -> None:
        """Update progress on output files

        Parameters
        ----------
        i_iter : int
            Current iteration number
        curr_value : float
            Current objective value
        curr_solution : float
            Current objective minimiser
        extra_info : str
            Additional information to pass to user
        """
        elapsed = time.perf_counter() - self.begin_time
        denorm_curr = self.parameters.denormalise(curr_solution)
        denorm_best = self.parameters.denormalise(self.best_solution)
        # Update progress file
        with open(os.path.join(self.output, "progress"), 'w', encoding='utf8') as file:
            file.write(f'Iteration: {i_iter}\n')
            file.write(f'Function calls: {self.objective.func_calls}\n')
            file.write(f'Best loss: {self.best_value}\n')
            if extra_info is not None:
                file.write(f'Optimiser info: {extra_info}\n')
            file.write('Best parameters:\n')
            for i, par in enumerate(self.parameters):
                file.write(f'\t{par.name}: {denorm_best[i]}\n')
            file.write(f'\nElapsed time: {pretty_time(elapsed)}\n')
        # Update history file
        with open(os.path.join(self.output, "history"), 'a', encoding='utf8') as file:
            file.write(f'{i_iter:>10}\t')
            file.write(f'{elapsed:>15.8e}\t')
            file.write(f'{self.best_value:>15.8e}\t')
            file.write(f'{curr_value:>15.8e}\t')
            for i, par in enumerate(self.parameters):
                file.write(f'{denorm_curr[i]:>15.8f}\t')
            file.write(f"\t{'-' if extra_info is None else extra_info}")
            file.write('\n')


    def _progress_check(
            self,
            i_iter: int,
            curr_value: float,
            curr_solution: np.ndarray,
            extra_info: str=None,
        ) -> bool:
        """
        Report the optimiser progress and check for termination

        Parameters
        ----------
        i_iter : int
            Current iteration number
        curr_value : float
            Current objective value
        curr_solution : float
            Current objective minimiser
        extra_info : str
            Additional information to pass to user

        Returns
        -------
        bool
            Whether any of the stopping criteria is satisfied
        """
        # Update new value to best value
        self.i_iter = i_iter
        if curr_value < self.best_value:
            self.best_value = curr_value
            self.best_solution = curr_solution
            self.iters_no_improv = 0
        else:
            self.iters_no_improv += 1
        # Update progress bar
        if self.pbar is not None:
            info = f'Loss: {self.best_value:6.4e}' + (f' ({extra_info})' if extra_info else '')
            self.pbar.set_postfix_str(info)
            if i_iter > 0:
                self.pbar.update()
        # Update progress in output files
        if self.output:
            self.__update_progress_files(i_iter, curr_solution, curr_value, extra_info)
        # Convergence criterion
        return i_iter > self.n_iter or self.stop_criteria.check_criteria(
            curr_value,
            self.iters_no_improv,
            self.objective.func_calls,
            time.perf_counter() - self.begin_time,
        )



class ScalarOptimiser(Optimiser):
    """Base class for scalar single-objective optimisers"""

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective

        Parameters
        ----------
        objective : Objective
            Objective to optimise

        Raises
        ------
        InvalidOptimiserException
            With an invalid combination of optimiser and objective function
        """
        if not isinstance(objective, SingleObjective):
            raise InvalidOptimiserException('Scalar objective required for this optimiser')



class CompositeOptimiser(Optimiser):
    """Base class for composite single-objective optimisers"""

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective

        Parameters
        ----------
        objective : Objective
            Objective to optimise

        Raises
        ------
        InvalidOptimiserException
            With an invalid combination of optimiser and objective function
        """
        if not isinstance(objective, SingleCompositeObjective):
            raise InvalidOptimiserException('Composite objective required for this optimiser')



class ScalarMultiFidelityOptimiser(Optimiser):
    """Base class for scalar single-objective multi-fidelity optimisers"""

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective

        Parameters
        ----------
        objective : Objective
            Objective to optimise

        Raises
        ------
        InvalidOptimiserException
            With an invalid combination of optimiser and objective function
        """
        if not isinstance(objective, MultiFidelitySingleObjective):
            raise InvalidOptimiserException('Multi-fidelity objective required for this optimiser')



class CompositeMultiFidelityOptimiser(Optimiser):
    """Base class for composite single-objective multi-fidelity optimisers"""

    def _validate_problem(self, objective: Objective) -> None:
        """Validate the combination of optimiser and objective

        Parameters
        ----------
        objective : Objective
            Objective to optimise

        Raises
        ------
        InvalidOptimiserException
            With an invalid combination of optimiser and objective function
        """
        if not isinstance(objective, MultiFidelityCompositeObjective):
            raise InvalidOptimiserException('Multi-fidelity objective required for this optimiser')
