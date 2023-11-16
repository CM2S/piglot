"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.yaml_parser import parse_loss
from piglot.losses.loss import Loss
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import Solver, OutputResult
from piglot.utils.assorted import stats_interp_to_common_grid
from piglot.utils.reduce_response import reduce_response
from piglot.objective import DynamicPlotter, SingleObjective
from piglot.objective import SingleCompositeObjective, MSEComposition
from piglot.objective import StochasticSingleObjective


class Reference:
    """Container for reference solutions."""

    def __init__(
            self,
            filename: str,
            prediction: Union[str, List[str]],
            x_col: int=1,
            y_col: int=2,
            skip_header: int=0,
            x_scale: float=1.0,
            y_scale: float=1.0,
            x_offset: float=0.0,
            y_offset: float=0.0,
            filter_tol: float=0.0,
            show: bool=False,
            loss: Loss=None,
            weight: float=1.0,
        ):
        # Sanitise prediction field
        if isinstance(prediction, str):
            prediction = [prediction]
        elif not isinstance(prediction, list):
            raise ValueError(f"Invalid prediction '{prediction}' for reference '{filename}'.")
        self.filename = filename
        self.prediction = prediction
        self.data = np.genfromtxt(filename, skip_header=skip_header)[:,[x_col - 1, y_col - 1]]
        self.data[:,0] = x_offset + x_scale * self.data[:,0]
        self.data[:,1] = y_offset + y_scale * self.data[:,1]
        self.orig_data = np.copy(self.data)
        self.filter_tol = filter_tol
        self.show = show
        self.loss = loss
        self.weight = weight

    def prepare(self) -> None:
        """Prepare the reference data."""
        if self.has_filtering():
            print(f"Filtering reference {self.filename} ...", end='')
            num, error, (x, y) = reduce_response(self.data[:,0], self.data[:,1], self.filter_tol)
            self.data = np.array([x, y]).T
            print(f" done (from {self.orig_data.shape[0]} to {num} points, error = {error:.2e})")
            if self.show:
                _, ax = plt.subplots()
                ax.plot(self.orig_data[:,0], self.orig_data[:,1], label="Reference")
                ax.plot(self.data[:,0], self.data[:,1], c='r', ls='dashed')
                ax.scatter(self.data[:,0], self.data[:,1], c='r', label="Resampled")
                ax.legend()
                plt.show()

    def num_fields(self) -> int:
        """Get the number of reference fields.

        Returns
        -------
        int
            Number of reference fields.
        """
        return self.data.shape[1] - 1

    def has_filtering(self) -> bool:
        """Check if the reference has filtering.

        Returns
        -------
        bool
            Whether the reference has filtering.
        """
        return self.filter_tol > 0.0

    def get_time(self) -> np.ndarray:
        """Get the time column of the reference.

        Returns
        -------
        np.ndarray
            Time column.
        """
        return self.data[:, 0]

    def get_data(self, field_idx: int=0) -> np.ndarray:
        """Get the data column of the reference.

        Parameters
        ----------
        field_idx : int
            Index of the field to output.

        Returns
        -------
        np.ndarray
            Data column.
        """
        return self.data[:, field_idx + 1]

    def get_orig_time(self) -> np.ndarray:
        """Get the original time column of the reference.

        Returns
        -------
        np.ndarray
            Original time column.
        """
        return self.orig_data[:, 0]

    def get_orig_data(self, field_idx: int=0) -> np.ndarray:
        """Get the original data column of the reference.

        Parameters
        ----------
        field_idx : int
            Index of the field to output.

        Returns
        -------
        np.ndarray
            Original data column.
        """
        return self.orig_data[:, field_idx + 1]

    def compute_loss(self, results: OutputResult) -> Any:
        """Compute the loss for the given results.

        Parameters
        ----------
        results : OutputResult
            Results to compute the loss for.

        Returns
        -------
        Any
            Loss value.
        """
        return self.loss(
            self.get_time(),
            results.get_time(),
            self.get_data(),
            results.get_data(),
        )

    @staticmethod
    def read(filename: str, config: Dict[str, Any]) -> Reference:
        """Read the reference from the configuration dictionary.

        Parameters
        ----------
        filename : str
            Path to the reference file.
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Reference
            Reference to use for this problem.
        """
        if 'prediction' not in config:
            raise ValueError(f"Missing prediction for reference '{filename}'.")
        return Reference(
            filename,
            config['prediction'],
            x_col=int(config.get('x_col', 1)),
            y_col=int(config.get('y_col', 2)),
            skip_header=int(config.get('skip_header', 0)),
            x_scale=float(config.get('x_scale', 1.0)),
            y_scale=float(config.get('y_scale', 1.0)),
            x_offset=float(config.get('x_offset', 0.0)),
            y_offset=float(config.get('y_offset', 0.0)),
            filter_tol=float(config.get('filter_tol', 0.0)),
            show=bool(config.get('show', False)),
            loss=parse_loss(filename, config['loss']) if 'loss' in config else None,
            weight=float(config.get('weight', 1.0)),
        )


class CurrentPlot(DynamicPlotter):
    """Container for dynamically-updating plots."""

    def __init__(self, solver: Solver, references: Dict[Reference, List[str]]) -> None:
        """Constructor for dynamically-updating plots.

        Parameters
        ----------
        solver : Solver
            Solver to build the plot for.
        references : Dict[Reference, List[str]]
            References to plot.
        """
        self.solver = solver
        self.references = references
        self.figs = {}
        self.preds = {}
        # Get current solver data
        result = solver.get_current_response()
        # Plot each reference
        for reference, names in references.items():
            fig, axis = plt.subplots()
            axis.plot(reference.get_time(), reference.get_data(),
                      label='Reference', ls='dashed', c='black', marker='x')
            # Plot predictions
            for name in names:
                self.preds[name], = axis.plot(result[name].get_time(), result[name].get_data(),
                                              label=f'Prediction ({name})', c='red')
            axis.set_title(reference.filename)
            axis.grid()
            axis.legend()
            self.figs[reference] = (fig, axis)
        plt.show()
        for fig, _ in self.figs.values():
            fig.canvas.draw()
            fig.canvas.flush_events()

    def update(self) -> None:
        """Update the plot with the most recent data"""
        result = self.solver.get_current_response()
        for reference, names in self.references.items():
            for name in names:
                try:
                    self.preds[name].set_xdata(result[name].get_time())
                    self.preds[name].set_ydata(result[name].get_data())
                except (FileNotFoundError, IndexError):
                    pass
            fig, axis = self.figs[reference]
            axis.relim()
            axis.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()


class FittingSolver:
    """Interface class between fitting objectives and solvers."""

    def __init__(
            self,
            solver: Solver,
            references: Dict[Reference, List[str]],
        ) -> None:
        self.solver = solver
        self.references = references

    def prepare(self) -> None:
        """Prepare the solver for optimisation."""
        for reference in self.references.keys():
            reference.prepare()
        self.solver.prepare()

    def solve(self, values: np.ndarray, concurrent: bool) -> Dict[Reference, List[OutputResult]]:
        """Evaluate the solver for the given set of parameter values and get the output results.

        Parameters
        ----------
        values : np.ndarray
            Parameter values to evaluate.
        concurrent : bool
            Whether this call may be concurrent to others.

        Returns
        -------
        Dict[Reference, List[OutputResult]]
            Output results.
        """
        result = self.solver.solve(values, concurrent)
        # Populate output results
        output = {}
        for reference, cases in self.references.items():
            output[reference] = [result[case] for case in cases]
        return output

    def plot_case(self, case_hash: str, options: Dict[str, Any]=None) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        options : Dict[str, Any], optional
            Options for the plot, by default None

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        # Check if we need to preserve the reference axis limits
        reference_limits = options is not None and options.get('reference_limits', False)
        append_title = ''
        if options is not None and 'append_title' in options:
            append_title = f' ({options["append_title"]})'
        figures = []
        # Load all responses
        responses = self.solver.get_output_response(case_hash)
        # Plot each reference
        for reference, names in self.references.items():
            # Build figure, index axes and plot response
            fig, axis = plt.subplots()
            axis.plot(reference.get_time(), reference.get_data(),
                      label='Reference', ls='dashed', c='black', marker='x')
            xlim, ylim = axis.get_xlim(), axis.get_ylim()
            # Plot predictions
            if options is not None and 'mean_std' in options and options['mean_std']:
                # Plot the average and standard deviation of the results
                stats = stats_interp_to_common_grid([
                    (responses[name].get_time(), responses[name].get_data())
                    for name in names
                ])
                axis.plot(stats['grid'], stats['average'], label='Average', c='red')
                axis.plot(stats['grid'], stats['median'], label='Median', c='red', ls='dashed')
                axis.fill_between(
                    stats['grid'],
                    stats['average'] - stats['std'],
                    stats['average'] + stats['std'],
                    label='Standard deviation',
                    color='red',
                    alpha=0.2,
                )
            elif options is not None and 'confidence' in options and options['confidence']:
                # Plot the average and standard deviation of the results
                stats = stats_interp_to_common_grid([
                    (responses[name].get_time(), responses[name].get_data())
                    for name in names
                ])
                axis.plot(stats['grid'], stats['average'], label='Average', c='red')
                axis.fill_between(
                    stats['grid'],
                    stats['average'] - stats['confidence'],
                    stats['average'] + stats['confidence'],
                    label='Confidence interval',
                    color='red',
                    alpha=0.2,
                )
            else:
                # Plot the individual responses
                for name in names:
                    axis.plot(responses[name].get_time(), responses[name].get_data(),
                            label=f'{name}')
            if reference_limits:
                axis.set_xlim(xlim)
                axis.set_ylim(ylim)
            axis.set_title(reference.filename + append_title)
            axis.grid()
            axis.legend()
            figures.append(fig)
        return figures

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of a updatable plots
        """
        return [CurrentPlot(self.solver, self.references)]

    @staticmethod
    def read(config: Dict[str, Any], parameters: ParameterSet, output_dir: str) -> FittingSolver:
        """Read a solver for fitting objectives from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.
        parameters : ParameterSet
            Set of parameters for this problem.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        FittingSolver
            Solver for fitting objectives.
        """
        # Read the solver
        if not 'solver' in config:
            raise ValueError("Missing solver for fitting objective.")
        solver = read_solver(config['solver'], parameters, output_dir)
        # Read the references
        if not 'references' in config:
            raise ValueError("Missing references for fitting objective.")
        references: Dict[Reference, List[str]] = {}
        for reference_name, reference_config in config.pop('references').items():
            reference = Reference.read(reference_name, reference_config)
            # Map the reference to the solver cases
            references[reference] = []
            for field_name in solver.get_output_fields():
                if field_name in reference.prediction:
                    references[reference].append(field_name)
            # Sanitise reference: check if it is associated to at least one case
            if len(references[reference]) == 0:
                raise ValueError(f"Reference '{reference_name}' is not associated to any case.")
        # Read the (optional) loss
        loss = None
        if 'loss' in config:
            loss = parse_loss(output_dir, config['loss'])
            # Assign the default loss to references without a specific loss
            for reference in references:
                if reference.loss is None:
                    reference.loss = loss
        # Ensure all references have a loss
        for reference in references:
            if reference.loss is None:
                raise ValueError(f"Missing loss for reference '{reference.filename}'")
        # Return the solver
        return FittingSolver(solver, references)


class FittingSingleObjective(SingleObjective):
    """Scalar fitting objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: FittingSolver,
            output_dir: str,
        ) -> None:
        super().__init__(parameters, output_dir)
        self.solver = solver

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def _objective(self, values: np.ndarray, concurrent: bool=False) -> float:
        """Objective computation for fitting objectives.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        float
            Objective value.
        """
        # Run the solver
        output = self.solver.solve(values, concurrent)
        # Compute the loss
        loss = 0.0
        for reference, results in output.items():
            losses = [reference.compute_loss(result) for result in results]
            loss += reference.weight * np.mean(losses)
        return loss

    def plot_case(self, case_hash: str, options: Dict[str, Any]=None) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        options : Dict[str, Any], optional
            Options for the plot, by default None

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        return self.solver.plot_case(case_hash, options)

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of updatable plots
        """
        return self.solver.plot_current()

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
        ) -> FittingSingleObjective:
        """Read a fitting objective from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.
        parameters : ParameterSet
            Set of parameters for this problem.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        Objective
            Objective function to optimise.
        """
        # Inject default loss for scalar objectives
        if not 'loss' in config:
            config['loss'] = 'scalar_vector'
        return FittingSingleObjective(
            parameters,
            FittingSolver.read(config, parameters, output_dir),
            output_dir,
        )


class FittingCompositeSingleObjective(SingleCompositeObjective):
    """Composite fitting objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: FittingSolver,
            output_dir: str,
        ) -> None:
        super().__init__(parameters, MSEComposition(), output_dir)
        self.solver = solver

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def _inner_objective(self, values: np.ndarray, concurrent: bool=False) -> np.ndarray:
        """Inner objective computation for composite fitting objectives.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        np.ndarray
            Objective value.
        """
        # Run the solver
        output = self.solver.solve(values, concurrent)
        # Compute the loss
        loss = np.array([])
        for reference, results in output.items():
            losses = np.array([reference.compute_loss(result) for result in results])
            loss = np.append(loss, reference.weight * np.mean(losses, axis=0))
        return loss

    def plot_case(self, case_hash: str, options: Dict[str, Any]=None) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        options : Dict[str, Any], optional
            Options for the plot, by default None

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        return self.solver.plot_case(case_hash, options)

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of updatable plots
        """
        return self.solver.plot_current()

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
        ) -> FittingCompositeSingleObjective:
        """Read a fitting objective from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.
        parameters : ParameterSet
            Set of parameters for this problem.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        Objective
            Objective function to optimise.
        """
        # Inject default loss for composite scalar objectives
        if not 'loss' in config:
            config['loss'] = 'vector'
        return FittingCompositeSingleObjective(
            parameters,
            FittingSolver.read(config, parameters, output_dir),
            output_dir,
        )


class StochasticFittingSingleObjective(StochasticSingleObjective):
    """Scalar fitting objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: FittingSolver,
            output_dir: str,
        ) -> None:
        super().__init__(parameters, output_dir)
        self.solver = solver

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def _objective(self, values: np.ndarray, concurrent: bool=False) -> Tuple[float, float]:
        """Objective computation for fitting objectives.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        Tuple[float, float]
            Objective value and variance.
        """
        # Run the solver
        output = self.solver.solve(values, concurrent)
        # Compute the loss
        loss = 0.0
        variance = 0.0
        for reference, results in output.items():
            losses = [reference.compute_loss(result) for result in results]
            loss += reference.weight * np.mean(losses)
            variance += reference.weight * np.var(losses)
        return loss, variance

    def plot_case(self, case_hash: str, options: Dict[str, Any]=None) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        options : Dict[str, Any], optional
            Options for the plot, by default None

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        figs = [
            self.solver.plot_case(case_hash, options),
            self.solver.plot_case(case_hash, {
                'mean_std': True,
                'append_title': 'Mean, median and standard deviation',
            }),
            self.solver.plot_case(case_hash, {
                'confidence': True,
                'append_title': 'Mean confidence interval 95%',
            }),
        ]
        return [fig for fig_list in figs for fig in fig_list]

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of updatable plots
        """
        return self.solver.plot_current()

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
        ) -> StochasticFittingSingleObjective:
        """Read a fitting objective from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.
        parameters : ParameterSet
            Set of parameters for this problem.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        Objective
            Objective function to optimise.
        """
        # Inject default loss for scalar objectives
        if not 'loss' in config:
            config['loss'] = 'scalar_vector'
        return StochasticFittingSingleObjective(
            parameters,
            FittingSolver.read(config, parameters, output_dir),
            output_dir,
        )
