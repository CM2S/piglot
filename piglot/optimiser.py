"""Main optimiser module"""
from functools import partial
from typing import Tuple, Callable, Optional, TypeVar, Any
import os
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from piglot.settings import Settings
from piglot.objective import Objective
from piglot.utils.assorted import pretty_time, str_to_numeric
from piglot.utils.tabular import (
    TabularFile,
    TabularStringColumn,
    TabularFloatColumn,
    TabularIntColumn,
)


T = TypeVar('T', bound='Optimiser')


@dataclass
class OptimisationResult:
    """Container for the result of the optimisation."""

    value: float
    params: Optional[np.ndarray] = None
    conf_interval: Tuple[float, float] = (None, None)
    pareto_params: Optional[list[np.ndarray]] = None
    pareto_values: Optional[list[np.ndarray]] = None
    ref_point: Optional[np.ndarray] = None


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


@dataclass
class HistoryFileData:
    """Container for history file data."""
    iteration: list[int]
    elapsed_time: np.ndarray
    best_value: np.ndarray
    best_lbound: Optional[np.ndarray]
    best_ubound: Optional[np.ndarray]
    best_params: Optional[np.ndarray]
    ref_point: Optional[np.ndarray]
    info: list[str]


class HistoryFileManager:
    """Manager for the optimiser's history file."""

    def __init__(self, file_path: str, settings: Settings, objective: Objective) -> None:
        self.file_path = file_path
        self.settings = settings
        self.objective = objective

        # State
        self.num_iters = 0
        self.start_time = time.time()

        # Base output formats
        iter_spec = partial(TabularIntColumn, width=11)
        pareto_spec = partial(TabularIntColumn, width=15)
        time_spec = partial(TabularFloatColumn, width=15, notation='e', precision=8)
        obj_spec = partial(TabularFloatColumn, width=15, notation='e', precision=8)
        param_spec = partial(TabularFloatColumn, width=15, notation='f', precision=6)
        info_spec = partial(TabularStringColumn, width=64, align='<')

        # Set up objective and parameter columns based on the objective type
        if objective.is_multi_objective():
            param_columns = []
            obj_columns = [
                obj_spec("Hypervolume"),
                *[obj_spec(f"Ref. point {i + 1}") for i in range(objective.num_objectives())],
                pareto_spec("Num. Pareto"),
            ]
        else:
            param_columns = self.settings.parameters.get_scalar_names()
            obj_columns = [obj_spec('Best objective')]
            if objective.has_variance():
                obj_columns.extend([obj_spec('Lower CI'), obj_spec('Upper CI')])

        # Build the full column list
        self.file = TabularFile(
            file_path,
            columns=[
                iter_spec("Iteration"),
                time_spec("Time /s"),
                *obj_columns,
                *map(param_spec, param_columns),
                info_spec("Optimiser info"),
            ],
        )

    def prepare(self) -> None:
        """Prepare the optimisation history file."""
        self.file.prepare()

    def write(self, result: OptimisationResult, extra_info: str) -> None:
        """Write history data to the file.

        Parameters
        ----------
        result : OptimisationResult
            Data to write to the history file.
        extra_info : str
            Additional information to write to the history file.
        """
        # Populate output fields
        if self.objective.is_multi_objective():
            obj_values = [result.value, *result.ref_point, len(result.pareto_params)]
            param_values = []
        else:
            obj_values = [result.value]
            param_values = self.settings.parameters.to_values(result.params).scalar_values.values()
            if self.objective.has_variance():
                obj_values.extend(result.conf_interval)

        # Write row
        self.file.write_row([
            self.num_iters,
            time.time() - self.start_time,
            *obj_values,
            *param_values,
            extra_info,
        ])

        # Update state
        self.num_iters += 1

    def read(self) -> HistoryFileData:
        """Read history data from the file.

        Returns
        -------
        HistoryFileData
            Data read from the history file.
        """
        data = self.file.read()

        # Parse data
        num_entries = len(data["Iteration"])
        if self.objective.is_multi_objective():
            value_name = "Hypervolume"
            param_names = []
            ref_point = np.array([
                [data[f"Ref. point {i + 1}"][j] for i in range(self.objective.num_objectives())]
                for j in range(num_entries)
            ])
        else:
            value_name = "Best objective"
            param_names = self.settings.parameters.get_scalar_names(include_computed=False)
            ref_point = None

        return HistoryFileData(
            iteration=np.array(data['Iteration']),
            elapsed_time=np.array(data['Time /s']),
            best_value=np.array(data[value_name]),
            best_lbound=np.array(data['Lower CI']) if self.objective.has_variance() else None,
            best_ubound=np.array(data['Upper CI']) if self.objective.has_variance() else None,
            best_params=np.array([
                [data[name][i] for name in param_names] for i in range(num_entries)
            ]),
            ref_point=ref_point,
            info=data['Optimiser info']
        )


class ProgressFileManager:
    """Manager for the progress file."""

    def __init__(self, file_path: str, settings: Settings, objective: Objective) -> None:
        self.file_path = file_path
        self.settings = settings
        self.objective = objective

        # State
        self.num_iters = 0
        self.start_time = time.time()

    def prepare(self) -> None:
        """Prepare the progress file."""
        with open(self.file_path, 'w', encoding='utf8') as file:
            file.write("Starting...\n")

    def write(self, result: OptimisationResult, extra_info: str) -> None:
        """Write progress data to the file.

        Parameters
        ----------
        result : OptimisationResult
            Current optimisation result.
        extra_info : str
            Additional information to pass to user.
        """
        elapsed = time.time() - self.start_time
        # Update progress file
        with open(self.file_path, 'w', encoding='utf8') as file:
            file.write(f'Iteration: {self.num_iters}\n')
            file.write(f'Function calls: {self.objective.num_calls}\n')

            # Single- or multi-objective data
            if self.objective.is_multi_objective():
                file.write(f'Hypervolume: {result.value}\n')
                file.write(f'Reference point: {result.ref_point.tolist()}\n')
            else:
                # Scalar objective value
                file.write(f'Best objective: {result.value}\n')
                if self.objective.has_variance():
                    file.write(
                        'Confidence interval (95%): '
                        f'[{result.conf_interval[0]}, {result.conf_interval[1]}]\n'
                    )
                # Parameters
                file.write('Best parameters:\n')
                param_dict = self.settings.parameters.to_values(result.params)
                for name, value in param_dict.scalar_values.items():
                    file.write(f'\t{name}: {value}\n')

            # Timing and extra info
            file.write(f'\nElapsed time: {pretty_time(elapsed)}\n')
            file.write(f'Optimiser info: {extra_info}\n')

        # Update state
        self.num_iters += 1


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
        self.history_file = HistoryFileManager(
            os.path.join(self.settings.output_dir, "history"), settings, objective
        )
        self.progress_file = ProgressFileManager(
            os.path.join(self.settings.output_dir, "progress"), settings, objective
        )

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
        self.history_file.prepare()
        self.progress_file.prepare()
        # Prepare optimiser
        self.objective.prepare()
        self._progress_report_prepare()
        # Optimise
        result = self._optimise(self.__update_progress)
        # Output progress
        self._progress_report_close()
        # Return the best value
        return result

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
        extra_info_str = extra_info if extra_info is not None else '-'
        self.progress_file.write(result, extra_info_str)
        self.history_file.write(result, extra_info_str)

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

    def _progress_report_update(self, i_iter: int, extra_info: Optional[str]) -> None:
        """Update the progress bar.

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        result : OptimisationResult
            Result of the current iteration.
        extra_info : Optional[str]
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
