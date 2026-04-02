"""Main optimiser module"""
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
            for name in self.parameters.get_scalar_names():
                file.write(f'{name:>15}\t')
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
        if not skip_pars:
            param_dict = self.settings.parameters.to_values(self.state.best_result.params)
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
                for name, value in param_dict.scalar_values.items():
                    file.write(f'\t{name}: {value}\n')
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
            if skip_pars:
                file.write('None\t'.rjust(16) * len(self.settings.parameters.get_scalar_names()))
            else:
                for value in param_dict.scalar_values.values():
                    file.write(f'{value:>15.8f}\t')
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
        callback : Callable[[int, OptimisationResult, dict[str, str]], bool]
            Callback function for reporting the optimiser progress and checking for termination.
            The first argument is the iteration number, the second argument is the current
            optimisation result, and the third argument is a dictionary with additional information
            to pass to the user. Call this function at the end of each iteration, and if it returns
            True, stop the optimisation.

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


class SimpleOptimiser(Optimiser):
    """Simple optimiser for single-objective, non-composite and noiseless problems."""

    def __init__(
        self, settings: Settings, objective: Objective, normalise_params: bool = False
    ) -> None:
        super().__init__(settings, objective)
        self.normalise_params = normalise_params

    @classmethod
    def validate_problem(cls, objective: Objective) -> None:
        """Validate the combination of optimiser and objective.

        Parameters
        ----------
        objective : Objective
            Objective to optimise.
        """
        if objective.is_multi_objective():
            raise ValueError("This optimiser does not support multi-objective optimisation.")
        if objective.is_composite():
            raise ValueError("This optimiser does not support composite objectives.")
        if objective.has_variance():
            raise ValueError("This optimiser does not support objectives with variance.")

    @staticmethod
    def __norm_params(values: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
        """Normalise parameters to the range [-1, 1] based on the given bounds.

        Parameters
        ----------
        values : np.ndarray
            Parameters to normalize.
        bounds : list[tuple[float, float]]
            Bounds for each parameter.

        Returns
        -------
        np.ndarray
            Normalized parameters.
        """
        return np.array([
            2 * (value - bound[0]) / (bound[1] - bound[0]) - 1
            for value, bound in zip(values, bounds)
        ])

    @staticmethod
    def __denorm_params(values: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
        """Denormalise parameters from the range [-1, 1] to the original bounds.

        Parameters
        ----------
        values : np.ndarray
            Parameters to denormalise.
        bounds : list[tuple[float, float]]
            Bounds for each parameter.

        Returns
        -------
        np.ndarray
            Denormalised parameters.
        """
        return np.array([
            0.5 * (value + 1) * (bound[1] - bound[0]) + bound[0]
            for value, bound in zip(values, bounds)
        ])

    def _optimise(
        self, callback: Callable[[int, OptimisationResult, dict[str, str]], bool]
    ) -> OptimisationResult:
        """Abstract method for optimising the objective.

        Parameters
        ----------
        callback : Callable[[int, OptimisationResult, dict[str, str]], bool]
            Callback function for reporting the optimiser progress and checking for termination.
            The first argument is the iteration number, the second argument is the current
            optimisation result, and the third argument is a dictionary with additional information
            to pass to the user. Call this function at the end of each iteration, and if it returns
            True, stop the optimisation.

        Returns
        -------
        OptimisationResult
            Result of the optimisation.
        """
        # Set up initial guess and bounds
        true_x0 = self.settings.parameters.get_initial_vector()
        true_bounds = [(p[0], p[1]) for p in self.settings.parameters.get_bounds()]

        # Handle normalisation
        if self.normalise_params:
            x0 = self.__norm_params(true_x0, true_bounds)
            bounds = [(-1.0, 1.0) for _ in true_bounds]
        else:
            x0 = true_x0
            bounds = true_bounds

        # Create storage for number of iterations and objective evaluations
        num_iters = 0
        evaluations: list[tuple[np.ndarray, float]] = []
        curr_best: Optional[tuple[np.ndarray, float]] = None

        # Set up function to update state
        def update_state() -> OptimisationResult:
            # Fetch new evaluations and update best result
            nonlocal num_iters, curr_best
            if len(evaluations) > 0:
                best_evaluation = min(evaluations, key=lambda x: x[1])
                evaluations.clear()
                if curr_best is None or best_evaluation[1] < curr_best[1]:
                    curr_best = best_evaluation

            # Create the optimisation result
            if curr_best is None:
                raise RuntimeError("No evaluations available to determine the best result.")
            return OptimisationResult(curr_best[1], params=curr_best[0])

        # Set up inner callback that updates result back and checks for termination
        def inner_callback(**kwargs) -> None:
            result = update_state()
            if callback(num_iters + 1, result, kwargs):
                raise StopIteration

        # Set up the objective function wrapper
        def objective_wrapper(x: np.ndarray, concurrent: bool = False) -> float:
            # Handle denormalisation if required
            if self.normalise_params:
                x = self.__denorm_params(x, true_bounds)

            # Clip the parameters to the bounds
            lbounds = np.array([bound[0] for bound in true_bounds])
            ubounds = np.array([bound[1] for bound in true_bounds])
            x = np.clip(x, lbounds, ubounds)

            # Evaluate and store the observation
            value = self.objective.get_objective_value(self.objective(x, concurrent=concurrent))
            evaluations.append((x, value))
            return value

        # Run the optimisation
        try:
            self._simple_optimise(
                self.settings.iters, x0, bounds, objective_wrapper, inner_callback
            )
        except StopIteration:
            pass

        # Update state before returning
        return update_state()

    @abstractmethod
    def _simple_optimise(
        self,
        num_iters: int,
        initial_guess: np.ndarray,
        bounds: list[tuple[float, float]],
        objective: Callable[[np.ndarray, Optional[bool]], float],
        callback: Callable[[Any], None],
    ) -> None:
        """Optimise the objective function.

        Parameters
        ----------
        num_iters : int
            Number of iterations for the optimisation.
        initial_guess : np.ndarray
            Initial guess for the optimisation.
        bounds : list[tuple[float, float]]
            Bounds for the optimisation variables.
        objective : Callable[[np.ndarray, Optional[bool]], float]
            Objective function to be minimised.
        callback : Callable[[Any], None]
            Callback function for reporting the optimiser progress and checking for termination.
            This function is called at the end of each iteration and will raise StopIteration if
            the optimisation should be stopped. Keyword arguments are reported from the optimiser.
        """
