"""Main optimizer module."""
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm


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
    def _init_optimiser(self, n_iter, parameters, pbar, loss, stop_criteria, output):
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


    def optimise(self, loss, n_iter, parameters, stop_criteria=StoppingCriteria(), output=False):
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
        pbar = tqdm(total=n_iter, desc=self.name)
        self._init_optimiser(n_iter, parameters, pbar, loss, stop_criteria, output)
        # Build initial shot and bounds
        n_dim = len(self.parameters)
        init_shot = [par.normalise(par.inital_value) for par in self.parameters]
        new_bound = np.ones((n_dim, 2))
        new_bound[:,0] = -1
        # Build best solution
        self.best_value = np.inf
        self.best_solution = None
        # Optimise
        x, new_value = self._optimise(self.loss.loss, n_dim, n_iter, new_bound, init_shot)
        self.pbar.close()
        # Denormalise best solution
        new_solution = [par.denormalise(self.best_solution[j])
                        for j, par in enumerate(self.parameters)]
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

    def _progress_check(self, iiter, curr_value, curr_solution):
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
        True if any of the stoppign criteria are satisfied
        """
        # Update new value to best value
        self.iiter = iiter
        if curr_value < self.best_value:
            self.best_value = curr_value
            self.best_solution = curr_solution
        if iiter > 0:
            self.pbar.set_postfix_str('Loss: {0:6.4e}'.format(self.best_value))
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
            with open('progress.{0}'.format(self.name), 'w') as file:
                best_solution = [par.denormalise(self.best_solution[i]) 
                                 for i, par in enumerate(self.parameters)]
                file.write('Iteration: {0}\t Best Loss: {1}\t Best Parameters: {2}\n'\
                        .format(iiter, self.best_value, best_solution))
        # Convergence criterion
        iter = False if self.n_iter is None else iiter > self.n_iter
        return iter or self.stop_criteria.check_criteria(curr_value, self.iters_no_improv,
                                                         self.loss.func_calls)
