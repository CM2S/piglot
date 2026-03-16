"""Main optimiser module"""
from __future__ import annotations
from typing import Tuple, Callable, Optional, TypeVar, Any
import os
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from piglot.settings import Settings
from piglot.utils.assorted import pretty_time, str_to_numeric
from piglot.objective import Objective


T = TypeVar('T', bound='Optimiser')


@dataclass
class OptimisationResult:
    """Container for the result of the optimisation."""

    value: float
    params: Optional[np.ndarray] = None
    conf_interval: Tuple[float, float] = (None, None)


@dataclass
class OptimiserState:
    """Container for the state of the optimiser."""

    i_iter: int = 0
    iters_without_improvement: int = 0
    best_result: OptimisationResult = None

    def update(self, i_iter: int, result: OptimisationResult, stochastic: bool) -> None:
        """Update the optimiser state with a new optimisation result.

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        result : OptimisationResult
            Result of the current iteration.
        stochastic : bool
            Whether the objective is stochastic (i.e., has variance).
        """
        # Update iteration counts first
        self.i_iter = i_iter
        self.iters_without_improvement += 1
        # Update best result
        if stochastic or self.best_result is None or result.value < self.best_result.value:
            self.best_result = result
            self.iters_without_improvement = 0


class InvalidOptimiserException(Exception):
    """Exception signaling invalid combination of optimiser and objective function."""


class Optimiser(ABC):
    """Optimiser interface."""

    def __init__(self, settings: Settings, objective: Objective) -> None:
        # Sanity check on the objective
        self.validate_problem(objective)

        self.settings = settings
        self.objective = objective
        self.parameters = settings.parameters
        self.state = OptimiserState()
        self.begin_time = time.time()

        # Lazy initialisation of the progress bar
        self.pbar: tqdm = None

    def optimise(self) -> OptimisationResult:
        """Optimiser for the outside world.

        Returns
        -------
        OptimisationResult
            Result of the optimisation.
        """
        # Reset state and time
        self.state = OptimiserState()
        self.begin_time = time.time()
        # Prepare history output files
        with open(os.path.join(self.settings.output_dir, "history"), 'w', encoding='utf8') as file:
            file.write(f'{"Iteration":>10}\t')
            file.write(f'{"Time /s":>15}\t')
            if self.objective.has_variance():
                file.write(f'{"Current Loss":>15}\t')
                file.write(f'{"Lower CI":>15}\t')
                file.write(f'{"Upper CI":>15}\t')
            else:
                file.write(f'{"Best Loss":>15}\t')
                file.write(f'{"Current Loss":>15}\t')
            for par in self.parameters:
                file.write(f'{par.name:>15}\t')
            file.write('\tOptimiser info')
            file.write('\n')
        # Prepare optimiser
        self.objective.prepare()
        self._progress_report_prepare()
        # Optimise
        result = self._optimise(self.__update_progress)
        # Output progress
        self._progress_report_close()
        # Return the best value
        return result

    def __update_progress_files(
        self,
        i_iter: int,
        result: OptimisationResult,
        extra_info: str,
    ) -> None:
        """Update progress on output files.

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        result : OptimisationResult
            Current optimisation result.
        extra_info : str
            Additional information to pass to user.
        """
        elapsed = time.perf_counter() - self.begin_time
        skip_pars = result.params is None
        # Update progress file
        with open(os.path.join(self.settings.output_dir, "progress"), 'w', encoding='utf8') as file:
            file.write(f'Iteration: {i_iter}\n')
            file.write(f'Function calls: {self.objective.num_calls}\n')
            file.write(f'Best loss: {self.state.best_result.value}\n')
            if self.objective.has_variance() and result.conf_interval and all(result.conf_interval):
                file.write(
                    'Confidence interval (95%): '
                    f'[{result.conf_interval[0]}, {result.conf_interval[1]}]\n'
                )
            if extra_info is not None:
                file.write(f'Optimiser info: {extra_info}\n')
            if not skip_pars:
                file.write('Best parameters:\n')
                for i, par in enumerate(self.settings.parameters):
                    file.write(f'\t{par.name}: {self.state.best_result.params[i]}\n')
            file.write(f'\nElapsed time: {pretty_time(elapsed)}\n')
        # Update history file
        with open(os.path.join(self.settings.output_dir, "history"), 'a', encoding='utf8') as file:
            file.write(f'{i_iter:>10}\t')
            file.write(f'{elapsed:>15.8e}\t')
            if self.objective.has_variance():
                file.write(f'{result.value:>15.8e}\t')
                if result.conf_interval and all(result.conf_interval):
                    file.write(f'{result.conf_interval[0]:>15.8e}\t')
                    file.write(f'{result.conf_interval[1]:>15.8e}\t')
                else:
                    file.write(''.rjust(15) + '\t')
                    file.write(''.rjust(15) + '\t')
            else:
                file.write(f'{self.state.best_result.value:>15.8e}\t')
                file.write(f'{result.value:>15.8e}\t')
            for i, par in enumerate(self.settings.parameters):
                file.write('None\t'.rjust(16) if skip_pars else f'{result.params[i]:>15.8f}\t')
            file.write(f"\t{'-' if extra_info is None else extra_info}")
            file.write('\n')

    def __convergence_check(self, i_iter: int) -> bool:
        """Check the convergence criteria.

        Parameters
        ----------
        i_iter : int
            Current iteration number.

        Returns
        -------
        bool
            Whether any of the stopping criteria is satisfied.
        """
        # Iteration number
        if i_iter > self.settings.iters:
            return True
        # Time
        if self.settings.max_timeout is not None:
            elapsed = time.time() - self.begin_time
            if elapsed > self.settings.max_timeout:
                return True
        # Function calls
        if self.settings.max_func_calls is not None:
            if self.objective.num_calls > self.settings.max_func_calls:
                return True
        # Improvement
        if self.settings.max_iters_no_improv is not None:
            if self.state.iters_without_improvement > self.settings.max_iters_no_improv:
                return True
        # Value
        if self.settings.conv_tol is not None:
            if self.state.best_result.value < self.settings.conv_tol:
                return True
        return False

    def __update_progress(
        self, i_iter: int, result: OptimisationResult, extra_info: dict[str, str] = None
    ) -> bool:
        """Update the optimiser progress and check for termination.

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        result : OptimisationResult
            Result of the current iteration.
        extra_info : dict[str, str]
            Additional information to pass to user.

        Returns
        -------
        bool
            Whether any of the stopping criteria is satisfied.
        """
        # Parse extra info
        if extra_info is not None and len(extra_info) > 0:
            extra_info = ', '.join(f'{key}: {value}' for key, value in extra_info.items())

        # Update optimiser state
        self.state.update(i_iter, result, self.objective.has_variance())

        # Update progress bar
        self._progress_report_update(i_iter, extra_info)

        # Update progress in output files
        self.__update_progress_files(i_iter, result, extra_info)

        # Check convergence criteria
        return self.__convergence_check(i_iter)

    @abstractmethod
    def name(self) -> str:
        """Name of the optimiser.

        Returns
        -------
        str
            Name of the optimiser.
        """

    @classmethod
    @abstractmethod
    def validate_problem(cls, objective: Objective) -> None:
        """Validate the combination of optimiser and objective.

        Parameters
        ----------
        objective : Objective
            Objective to optimise.
        """

    @abstractmethod
    def _optimise(
        self, callback: Callable[[int, OptimisationResult, dict[str, str]], bool]
    ) -> OptimisationResult:
        """Abstract method for optimising the objective.

        Parameters
        ----------
        callback : Callable[[OptimisationResult, dict[str, str]], bool]
            Callback function for reporting the optimiser progress and checking for termination.
            The first argument is the current optimisation result, while the second argument is a
            dictionary with additional information to pass to the user. Call this function at the
            end of each iteration, and if it returns True, stop the optimisation.

        Returns
        -------
        OptimisationResult
            Result of the optimisation.
        """

    def _progress_report_prepare(self) -> None:
        """Initialising the progress bar."""
        if not self.settings.quiet:
            self.pbar = tqdm(total=self.settings.iters, desc=self.name())

    def _progress_report_update(self, i_iter: int, extra_info: dict[str, str]) -> None:
        """Update the progress bar.

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        result : OptimisationResult
            Result of the current iteration.
        extra_info : dict[str, str]
            Additional information to pass to user.
        """
        if self.pbar is not None:
            info = f'Loss: {self.state.best_result.value:6.3e}'
            if (
                self.objective.has_variance()
                and self.state.best_result.conf_interval
                and all(self.state.best_result.conf_interval)
            ):
                delta = (
                    self.state.best_result.conf_interval[1]
                    - self.state.best_result.conf_interval[0]
                ) / 2
                info += f' ± {delta:6.3e}'
            self.pbar.set_postfix_str(info + (f' ({extra_info})' if extra_info else ''))
            if i_iter > 0:
                self.pbar.update()

    def _progress_report_close(self) -> None:
        """Close the progress bar."""
        if self.pbar is not None:
            self.pbar.close()

    @classmethod
    def read(cls: type[T], config: dict[str, Any], settings: Settings, objective: Objective) -> T:
        """Read an optimiser from the given configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the optimiser.
        settings : Settings
            Settings for the optimiser.
        objective : Objective
            Objective to optimise.

        Returns
        -------
        T
            The created optimiser instance.
        """
        config = {k: str_to_numeric(v) for k, v in config.items()}
        return cls(settings, objective, **config)


class ScalarOptimiser(Optimiser):
    """Base class for scalar optimisers."""

    def __init__(self, name: str, settings: Settings, objective: Objective) -> None:
        super().__init__(settings, objective)
        self.__name = name
        self.bounds = np.array([[par.lbound, par.ubound] for par in self.parameters])
        # Lazy callback for the scalar optimiser (TODO: refactor to avoid this)
        self.__callback: Callable[[int, OptimisationResult, dict[str, str]], bool] = None

    def name(self) -> str:
        """Name of the optimiser.

        Returns
        -------
        str
            Name of the optimiser.
        """
        return self.__name

    @classmethod
    def validate_problem(cls, objective: Objective) -> None:
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
        if objective.is_composite():
            raise InvalidOptimiserException('This optimiser does not support composition')
        if objective.has_variance():
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
        """Abstract method for optimising the objective.

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
        self, callback: Callable[[int, OptimisationResult, dict[str, str]], bool]
    ) -> OptimisationResult:
        """Abstract method for optimising the objective.

        Parameters
        ----------
        callback : Callable[[OptimisationResult, dict[str, str]], bool]
            Callback function for reporting the optimiser progress and checking for termination.
            The first argument is the current optimisation result, while the second argument is a
            dictionary with additional information to pass to the user. Call this function at the
            end of each iteration, and if it returns True, stop the optimisation.

        Returns
        -------
        OptimisationResult
            Result of the optimisation.
        """
        # Set up problem
        n_dim = len(self.parameters)
        init_shot = np.array([par.inital_value for par in self.parameters])
        n_iter = self.settings.iters
        self.__callback = callback   # TODO: refactor to avoid this
        # Optimise the scalarised objective
        best_value, best_params = self._scalar_optimise(
            lambda x, concurrent=False: self.objective(
                self._denorm_params(x),
                concurrent=concurrent
            ).obj_values.item(),
            n_dim,
            n_iter,
            np.array([[-1.0, 1.0]]).repeat(n_dim, axis=0),
            self._norm_params(init_shot),
        )
        # Return the best value
        return OptimisationResult(
            value=best_value,
            params=None if best_params is None else self._denorm_params(best_params),
        )

    def _progress_check(
        self,
        i_iter: int,
        curr_value: float,
        curr_solution: np.ndarray,
        extra_info: str = None,
    ) -> bool:
        """Report the optimiser progress and check for termination (with parameter denormalisation).

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
        solution = None if curr_solution is None else self._denorm_params(curr_solution)
        result = OptimisationResult(value=curr_value, params=solution)
        info = {'info': extra_info} if extra_info is not None else None
        return self.__callback(i_iter, result, info)
