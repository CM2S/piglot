"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Union
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import Solver, OutputResult
from piglot.utils.assorted import stats_interp_to_common_grid
from piglot.utils.reductions import Reduction, read_reduction
from piglot.utils.responses import Transformer, reduce_response, interpolate_response
from piglot.utils.response_transformer import ResponseTransformer, read_response_transformer
from piglot.utils.composition.responses import ResponseComposition, FixedFlatteningUtility
from piglot.objective import Composition, DynamicPlotter, GenericObjective, ObjectiveResult


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
            reduction: Reduction = None,
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
        self.reduction = reduction
        # Load the data right away
        data = np.genfromtxt(filename, skip_header=skip_header)
        # Sanitise to ensure it is a 2D array
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        self.x_data = data[:, x_col - 1]
        self.y_data = data[:, y_col - 1]
        # Apply the transformer
        if self.transformer is not None:
            self.x_data, self.y_data = self.transformer(self.x_data, self.y_data)
        self.x_orig = np.copy(self.x_data)
        self.y_orig = np.copy(self.y_data)
        self.flatten_utility = FixedFlatteningUtility(self.x_data)

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
            # Reset the flattening utility
            self.flatten_utility = FixedFlatteningUtility(self.x_data)

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

    def compute_errors(self, results: OutputResult) -> OutputResult:
        """Compute the pointwise normalised errors for the given results.

        Parameters
        ----------
        results : OutputResult
            Results to compute the errors for.

        Returns
        -------
        OutputResult
            Pointwise normalised errors.
        """
        # Interpolate response to the reference grid
        resp_interp = interpolate_response(
            results.get_time(),
            results.get_data(),
            self.get_time(),
        )
        # Compute normalised error
        factor = np.mean(np.abs(self.get_data()))
        return OutputResult(self.get_time(), (resp_interp - self.get_data()) / factor)

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
            reduction=read_reduction(config['reduction']) if 'reduction' in config else None,
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
            reduction: Reduction,
            transformers: Dict[str, ResponseTransformer] = None,
            ) -> None:
        self.solver = solver
        self.references = references
        self.reduction = reduction
        self.transformers = transformers if transformers is not None else {}
        # Assign the reduction to non-defined references
        for reference in self.references:
            if reference.reduction is None:
                reference.reduction = self.reduction

    def prepare(self) -> None:
        """Prepare the solver for optimisation."""
        for reference in self.references.keys():
            reference.prepare()
        self.solver.prepare()

    def composition(
        self,
        multi_objective: bool,
        stochastic: bool,
    ) -> Composition:
        """Build the composition utility for the design objective.

        Parameters
        ----------
        multi_objective : bool
            Whether this is a multi-objective problem.
        stochastic : bool
            Whether the objective is stochastic.
        targets : List[DesignTarget]
            List of design targets to consider.

        Returns
        -------
        Composition
            Composition utility for the design objective.
        """
        return ResponseComposition(
            scalarise=not multi_objective,
            stochastic=stochastic,
            weights=[t.weight for t in self.references],
            reductions=[t.reduction for t in self.references],
            flatten_list=[t.flatten_utility for t in self.references],
        )

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
        # Transform responses
        for name, transformer in self.transformers.items():
            if name in result:
                result[name] = transformer.transform(result[name])
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
        # Transform responses if necessary
        for name, transformer in self.transformers.items():
            if name in responses:
                responses[name] = transformer.transform(responses[name])
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
                    marker = 'o' if len(responses[name].get_time()) < 2 else None
                    axis.plot(responses[name].get_time(), responses[name].get_data(),
                              label=f'{name}', marker=marker)
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
        # Read the optional reduction
        reduction = read_reduction(config.get('reduction', 'mse'))
        # Read the optional transformers
        transformers: Dict[str, ResponseTransformer] = {}
        if 'transformers' in config:
            for name, transformer_config in config['transformers'].items():
                transformers[name] = read_response_transformer(transformer_config)
        # Return the solver
        return FittingSolver(solver, references, reduction, transformers=transformers)


class FittingObjective(GenericObjective):
    """Scalar fitting objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: FittingSolver,
            output_dir: str,
            stochastic: bool = False,
            composite: bool = False,
            multi_objective: bool = False,
            ) -> None:
        super().__init__(
            parameters,
            stochastic,
            composition=solver.composition(multi_objective, stochastic) if composite else None,
            num_objectives=len(solver.references) if multi_objective else 1,
            output_dir=output_dir,
        )
        self.composite = composite
        self.solver = solver

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()
        # Reset the composition utility
        if self.composite:
            self.composition = self.solver.composition(self.multi_objective, self.stochastic)

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
        # Compute pointwise errors for each reference
        errors = {
            reference: [reference.compute_errors(result) for result in results]
            for reference, results in output.items()
        }
        # Under composition, we delegate the computation to the composition utility
        if self.composition is not None:
            return self.composition.transform(values, list(errors.values()))
        # Otherwise, compute the objective directly
        losses = []
        variances = []
        for reference, responses in errors.items():
            targets = [
                reference.reduction.reduce(response.get_time(), response.get_data(), values)
                for response in responses
            ]
            # Build statistical model for the target
            losses.append(reference.weight * np.mean(targets))
            variances.append(reference.weight * np.var(targets) / len(targets))
        return ObjectiveResult(values, losses, variances if self.stochastic else None)

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
        return FittingObjective(
            parameters,
            FittingSolver.read(config, parameters, output_dir),
            output_dir,
            stochastic=bool(config.get('stochastic', False)),
            composite=bool(config.get('composite', False)),
            multi_objective=bool(config.get('multi_objective', False)),
        )
