"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Union
from abc import abstractmethod, ABC
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import Solver, OutputResult
from piglot.utils.assorted import stats_interp_to_common_grid
from piglot.utils.responses import Transformer, reduce_response, interpolate_response
from piglot.objective import Composition, DynamicPlotter, GenericObjective, ObjectiveResult


class Reduction(ABC):
    """Abstract class for reduction functions with gradients."""

    @abstractmethod
    def reduce(self, inner: np.ndarray) -> np.ndarray:
        """Abstract method for computing the reduction of numpy arrays.

        Parameters
        ----------
        inner : np.ndarray
            Reduction input.

        Returns
        -------
        np.ndarray
            Reduction result.
        """

    @abstractmethod
    def reduce_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Abstract method for computing the reduction of torch tensors with gradients.

        Parameters
        ----------
        inner : torch.Tensor
            Reduction input.

        Returns
        -------
        torch.Tensor
            Reduction result (with gradients).
        """


class MSE(Reduction):
    """Mean squared error reduction function."""

    def reduce(self, inner: np.ndarray) -> np.ndarray:
        """Compute the mean squared error.

        Parameters
        ----------
        inner : np.ndarray
            Reduction input.

        Returns
        -------
        np.ndarray
            Reduction result.
        """
        return np.mean(np.square(inner), axis=-1)

    def reduce_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Compute the mean squared error.

        Parameters
        ----------
        inner : torch.Tensor
            Reduction input.

        Returns
        -------
        torch.Tensor
            Reduction result (with gradients).
        """
        return torch.mean(torch.square(inner), dim=-1)


AVAILABLE_REDUCTIONS = {
    'mse': MSE,
}


class CompositionFromReduction(Composition):
    """Optimisation composition function from a reduction function."""

    def __init__(self, reduction: Reduction):
        self.reduction = reduction

    def composition(self, inner: np.ndarray) -> float:
        """Compute the MSE outer function of the composition

        Parameters
        ----------
        inner : np.ndarray
            Return value from the inner function

        Returns
        -------
        float
            Scalar composition result
        """
        return self.reduction.reduce(inner)

    def composition_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Compute the MSE outer function of the composition with gradients

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function

        Returns
        -------
        torch.Tensor
            Scalar composition result
        """
        return self.reduction.reduce_torch(inner)


class Reference:
    """Container for reference solutions."""

    def __init__(
            self,
            filename: str,
            prediction: Union[str, List[str]],
            output_dir: str,
            x_col: int = 1,
            y_col: int = 2,
            skip_header: int = 0,
            transformer: Transformer = None,
            filter_tol: float = 0.0,
            show: bool = False,
            weight: float = 1.0,
            ):
        # Sanitise prediction field
        if isinstance(prediction, str):
            prediction = [prediction]
        elif not isinstance(prediction, list):
            raise ValueError(f"Invalid prediction '{prediction}' for reference '{filename}'.")
        self.filename = filename
        self.prediction = prediction
        self.output_dir = output_dir
        self.transformer = transformer
        self.filter_tol = filter_tol
        self.show = show
        self.weight = weight
        # Load the data right away
        data = np.genfromtxt(filename, skip_header=skip_header)[:, [x_col - 1, y_col - 1]]
        self.x_data = data[:, 0]
        self.y_data = data[:, 1]
        # Apply the transformer
        if self.transformer is not None:
            self.x_data, self.y_data = self.transformer(self.x_data, self.y_data)
        self.x_orig = np.copy(self.x_data)
        self.y_orig = np.copy(self.y_data)

    def prepare(self) -> None:
        """Prepare the reference data."""
        if self.has_filtering():
            # Little progress report: ensure we flush after the initial message
            print(f"Filtering reference {self.filename} ...", end='')
            sys.stdout.flush()
            num, error, (self.x_data, self.y_data) = reduce_response(
                self.x_data,
                self.y_data,
                self.filter_tol,
            )
            print(f" done (from {len(self.x_orig)} to {num} points, error = {error:.2e})")
            if self.show:
                _, ax = plt.subplots()
                ax.plot(self.x_orig, self.y_orig, label="Reference")
                ax.plot(self.x_data, self.y_data, c='r', ls='dashed')
                ax.scatter(self.x_data, self.y_data, c='r', label="Resampled")
                ax.legend()
                plt.show()
            # Write the filtered reference
            os.makedirs(os.path.join(self.output_dir, 'filtered_references'), exist_ok=True)
            np.savetxt(
                os.path.join(
                    self.output_dir,
                    'filtered_references',
                    os.path.basename(self.filename),
                ),
                np.stack((self.x_data, self.y_data), axis=1),
            )

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
        return self.x_data

    def get_data(self) -> np.ndarray:
        """Get the data column of the reference.

        Returns
        -------
        np.ndarray
            Data column.
        """
        return self.y_data

    def get_orig_time(self) -> np.ndarray:
        """Get the original time column of the reference.

        Returns
        -------
        np.ndarray
            Original time column.
        """
        return self.x_orig

    def get_orig_data(self) -> np.ndarray:
        """Get the original data column of the reference.

        Returns
        -------
        np.ndarray
            Original data column.
        """
        return self.y_orig

    def compute_errors(self, results: OutputResult) -> np.ndarray:
        """Compute the pointwise normalised errors for the given results.

        Parameters
        ----------
        results : OutputResult
            Results to compute the errors for.

        Returns
        -------
        np.ndarray
            Error for each reference point
        """
        # Interpolate response to the reference grid
        resp_interp = interpolate_response(
            results.get_time(),
            results.get_data(),
            self.get_time(),
        )
        # Compute normalised error
        factor = np.mean(np.abs(self.get_data()))
        return (resp_interp - self.get_data()) / factor

    @staticmethod
    def read(filename: str, config: Dict[str, Any], output_dir: str) -> Reference:
        """Read the reference from the configuration dictionary.

        Parameters
        ----------
        filename : str
            Path to the reference file.
        config : Dict[str, Any]
            Configuration dictionary.
        output_dir: str
            Output directory.

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
            output_dir,
            x_col=int(config.get('x_col', 1)),
            y_col=int(config.get('y_col', 2)),
            skip_header=int(config.get('skip_header', 0)),
            transformer=Transformer.read(config.get('transformer', {})),
            filter_tol=float(config.get('filter_tol', 0.0)),
            show=bool(config.get('show', False)),
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

    def plot_case(self, case_hash: str, options: Dict[str, Any] = None) -> List[Figure]:
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
        if 'solver' not in config:
            raise ValueError("Missing solver for fitting objective.")
        solver = read_solver(config['solver'], parameters, output_dir)
        # Read the references
        if 'references' not in config:
            raise ValueError("Missing references for fitting objective.")
        references: Dict[Reference, List[str]] = {}
        for reference_name, reference_config in config.pop('references').items():
            reference = Reference.read(reference_name, reference_config, output_dir)
            # Map the reference to the solver cases
            references[reference] = []
            for field_name in solver.get_output_fields():
                if field_name in reference.prediction:
                    references[reference].append(field_name)
            # Sanitise reference: check if it is associated to at least one case
            if len(references[reference]) == 0:
                raise ValueError(f"Reference '{reference_name}' is not associated to any case.")
        # Return the solver
        return FittingSolver(solver, references)


class FittingObjective(GenericObjective):
    """Scalar fitting objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: FittingSolver,
            output_dir: str,
            reduction: Reduction,
            stochastic: bool = False,
            composite: bool = False,
            ) -> None:
        super().__init__(
            parameters,
            stochastic,
            composition=CompositionFromReduction(reduction) if composite else None,
            output_dir=output_dir,
        )
        self.reduction = reduction
        self.composite = composite
        self.solver = solver

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def _objective(self, values: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Abstract method for objective computation.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        ObjectiveResult
            Objective result.
        """
        # Run the solver
        output = self.solver.solve(values, concurrent)
        # Compute the loss
        losses = []
        variances = []
        for reference, results in output.items():
            # Compute the errors for each result and apply reduction according to the objective type
            errors = np.array([reference.compute_errors(result) for result in results])
            if not self.composite:
                errors = self.reduction.reduce(errors)
            loss = np.mean(errors, axis=0)
            variance = np.var(errors, axis=0) / errors.shape[0]
            # Collect the weighted losses and variances
            losses.append(reference.weight * loss)
            variances.append(reference.weight * variance)
        return ObjectiveResult(losses, variances if self.stochastic else None)

    def plot_case(self, case_hash: str, options: Dict[str, Any] = None) -> List[Figure]:
        """Plot a given function call given the parameter hash.

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot.
        options : Dict[str, Any], optional
            Options for the plot, by default None.

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        if not self.stochastic:
            return self.solver.plot_case(case_hash, options)
        if options is None:
            options = {}
        options_mean = options.copy()
        options_mean['mean_std'] = True
        options_mean['append_title'] = 'Mean, median and standard deviation'
        options_conf = options.copy()
        options_conf['confidence'] = True
        options_conf['append_title'] = 'Mean confidence interval 95%'
        figs = [
            self.solver.plot_case(case_hash, options),
            self.solver.plot_case(case_hash, options_mean),
            self.solver.plot_case(case_hash, options_conf),
        ]
        return [fig for fig_list in figs for fig in fig_list]

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call.

        Returns
        -------
        List[DynamicPlotter]
            List of instances of updatable plots.
        """
        return self.solver.plot_current()

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
            ) -> FittingObjective:
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
        # Parse the reduction
        if 'reduction' not in config:
            config['reduction'] = 'mse'
        elif config['reduction'] not in AVAILABLE_REDUCTIONS:
            raise ValueError(f"Invalid reduction '{config['reduction']}' for fitting objective.")
        return FittingObjective(
            parameters,
            FittingSolver.read(config, parameters, output_dir),
            output_dir,
            AVAILABLE_REDUCTIONS[config['reduction']](),
            stochastic=bool(config.get('stochastic', False)),
            composite=bool(config.get('composite', False)),
        )
