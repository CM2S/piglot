"""Main optimizer module."""
import os
import time 
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm


def pretty_time(elapsed_sec):
    """Return a human-readable representation of a given elapsed time.

    Parameters
    ----------
    elapsed_sec : float
        Elapsed time, in seconds.

    Returns
    -------
    str
        Pretty elapsed time string.
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
    arg = np.where(arg > bounds[:,1], bounds[:,1], arg)
    arg = np.where(arg < bounds[:,0], bounds[:,0], arg)
    return arg


def missing_method(name, package):
    """Class generator for missing packages.

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
        raise ImportError("{0} is not available. You need to install package {1}!"\
                          .format(name, package))

    return type('Missing_{0}'.format(package), (),
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
        stop the optimizer if the loss reaches the conv_tol value
    max_iters_no_improv : integer
        stop the optimizer if the loss does not improve in max_iters_no_improv iterations
    max_func_calls : integer
        stop the optimizer if the optimizer called max_func_calls times the loss function

    Methods
    -------
    check_criteria(loss_value, iters_no_improv, func_calls):
        check the status of the stopping criteria
    """
    def __init__(self, conv_tol=None, max_iters_no_improv=None, max_func_calls=None):
        """
        Constructs all the necessary attributes for the stopping criteria

        Parameters
        ----------
        conv_tol : float
            stop the optimizer if the loss reaches the conv_tol value
        max_iters_no_improv : integer
            stop the optimizer if the loss does not improve in max_iters_no_improv
            iterations
        max_func_calls : integer
            stop the optimizer if the optimizer called max_func_calls times the loss
            function

        """
        self.conv_tol = conv_tol
        self.max_iters_no_improv = max_iters_no_improv
        self.max_func_calls = max_func_calls

    def check_criteria(self, loss_value, iters_no_improv, func_calls):
        """
        Check the status of the stopping criteria

        Parameters
        ----------
        loss_value : float
            current loss value
        iters_no_improv : integer
            current number of iterations without loss improvement
        func_calls : integer
            current number of loss function calls

        Returns
        -------
        True if any of the criteria are satisfied
        """
        conv = False if self.conv_tol is None else loss_value < self.conv_tol
        func = False if self.max_func_calls is None else func_calls > self.max_func_calls
        impr = False if self.max_iters_no_improv is None \
                     else iters_no_improv > self.max_iters_no_improv
        return conv or impr or func


class Optimiser(ABC):
    """
    Interface for implementing different optimization algorithms

    Methods
    -------
    _init_optimiser(n_iter, parameters, pbar, loss, stop_criteria):
        constructs the attributes for the optimizer
    optimise(loss, n_iter, parameters, stop_criteria = StoppingCriteria()):
        initiates optimizer
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        performs the optimization
    _progress_check(self, iiter, curr_value, curr_solution):
        evaluates the optimizer progress
    """
    def _init_optimiser(self, n_iter, parameters, pbar, loss, stop_criteria, output, verbose):
        """
        Constructs the attributes for the optimizer

        Parameters
        ----------
        n_iter : integer
            maximum number of iterations
        parameters : Parameter()
            parameters class
        pbar : tqdm
            progress bar
        loss : Loss()
            loss class
        stop_criteria : StoppingCriteria()
            stopping criteria
        output : bool
            Whether to write output to a file
        """
        self.parameters = parameters
        self.n_iter = n_iter
        self.iters_no_improv = 0
        n_dim = len(self.parameters)
        # Prepare history arrays
        self.value_history = np.empty(n_iter + 1)
        self.best_value_history = np.empty(n_iter + 1)
        self.solution_history = np.empty((n_iter + 1, n_dim))
        self.value_history[:] = np.inf
        self.best_value_history[:] = np.inf
        self.solution_history[:,:] = np.nan
        self.pbar = pbar
        self.loss = loss
        self.stop_criteria = stop_criteria
        self.output = output
        self._verbose = verbose


    def optimise(self, loss, n_iter, parameters, stop_criteria=StoppingCriteria(), output=None, verbose=True):
        """
        Initiates optimizer

        Parameters
        ----------
        loss : Loss()
            loss class
        n_iter : integer
            maximum number of iterations
        parameters : Parameter()
            parameters class
        stop_criteria : StoppingCriteria()
            stopping criteria
        output : bool
            Whether to write output to a file

        Returns
        -------
        best_value : float
            best loss function value
        best_solution : list
            best parameter solution
        """
        # Initialise optimiser
        pbar = tqdm(total=n_iter, desc=self.name) if verbose else None
        self._init_optimiser(n_iter, parameters, pbar, loss, stop_criteria, output, verbose)
        # Build initial shot and bounds
        n_dim = len(self.parameters)
        init_shot = [par.normalise(par.inital_value) for par in self.parameters]
        new_bound = np.ones((n_dim, 2))
        new_bound[:,0] = -1
        # Build best solution
        self.best_value = np.inf
        self.best_solution = None
        # Prepare history output files
        if output:
            with open(os.path.join(self.output, "history"), 'w') as file:
                file.write(f'{"Iteration":>10}\t{"Time /s":>15}\t{"Best Loss":>15}\t{"Current Loss":>15}')
                for par in self.parameters:
                    file.write(f'\t{par.name:>15}')
                file.write('\n')
        # Optimise
        self.begin = time.perf_counter()
        self._optimise(self.loss, n_dim, n_iter, new_bound, init_shot)
        elapsed = time.perf_counter() - self.begin
        # Denormalise best solution
        new_solution = [par.denormalise(self.best_solution[j])
                        for j, par in enumerate(self.parameters)]
        if verbose:
            self.pbar.close()
            print(f'Completed {self.iiter} iterations in {pretty_time(elapsed)}')
            print(f'Best loss: {self.best_value:15.8e}')
            print(f'Best parameters')
            largest = max([len(par.name) for par in self.parameters])
            for i, par in enumerate(self.parameters):
                print(f'- {par.name.rjust(largest)}: {new_solution[i]:>12.6f}')
        # Return the best value
        return self.best_value, new_solution

    @abstractmethod
    def _optimise(self, func, n_dim, n_iter, bound, init_shot):
        """
        Performs the optimization

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

    def __update_progress_files(self, iiter, curr_solution, curr_value):
        elapsed = time.perf_counter() - self.begin
        denorm_best = [par.denormalise(self.best_solution[i]) for i, par in enumerate(self.parameters)]
        denorm_curr = [par.denormalise(curr_solution[i]) for i, par in enumerate(self.parameters)]
        # Update progress file
        with open(os.path.join(self.output, "progress"), 'w') as file:
            file.write(f'Iteration: {iiter}\n')
            file.write(f'Function calls: {self.loss.func_calls}\n')
            file.write(f'Best loss: {self.best_value}\n')
            file.write(f'Best parameters:\n')
            for i, par in enumerate(self.parameters):
                file.write(f'\t{par.name}: {denorm_best[i]}\n')
            file.write(f'\nElapsed time: {pretty_time(elapsed)}\n')
        # Update history file
        with open(os.path.join(self.output, "history"), 'a') as file:
            file.write(f'{iiter:>10}\t{elapsed:>15.8e}\t{self.best_value:>15.8e}\t{curr_value:>15.8e}')
            for i, par in enumerate(self.parameters):
                file.write(f'\t{denorm_curr[i]:>15.8f}')
            file.write('\n')


    def _progress_check(self, iiter, curr_value, curr_solution, extra_info=None):
        """
        Check the optimizer progress

        Parameters
        ----------
        iiter : integer
            current iteration
        curr_value : float
            current loss value
        curr_solution : float
            current parameter solution value

        Returns
        -------
        True if any of the stopping criteria are satisfied
        """
        # Update new value to best value
        self.iiter = iiter
        if curr_value < self.best_value:
            self.best_value = curr_value
            self.best_solution = curr_solution
        if iiter > 0:
            if self._verbose:
                if extra_info:
                    self.pbar.set_postfix_str(f'Loss: {self.best_value:6.4e} ({extra_info})')
                else:
                    self.pbar.set_postfix_str(f'Loss: {self.best_value:6.4e}')
                self.pbar.update()
            # CHANGE TO BEST_HISTORY
            if curr_value < self.value_history[iiter-1]:
                self.iters_no_improv = 0
            else:
                self.iters_no_improv += 1
        else:
            self.iters_no_improv = 0
        # Update histories
        self.value_history[iiter] = curr_value
        self.best_value_history[iiter] = self.best_value
        self.solution_history[iiter,:] = self.best_solution
        # Output progress
        if self.output:
            self.__update_progress_files(iiter, curr_solution, curr_value)
        # Convergence criterion
        iter = False if self.n_iter is None else iiter > self.n_iter
        return iter or self.stop_criteria.check_criteria(curr_value, self.iters_no_improv,
                                                         self.loss.func_calls)
