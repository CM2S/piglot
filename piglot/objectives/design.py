"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Union, Type
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import Solver, OutputResult
from piglot.objective import DynamicPlotter, GenericObjective, ObjectiveResult
from piglot.utils.assorted import stats_interp_to_common_grid
from piglot.utils.solver_utils import load_module_from_file


class Quantity(ABC):
    """Abstract class for quantities to be computed from a response."""

    @abstractmethod
    def compute(self, result: OutputResult) -> float:
        """Compute this quantity for a given response.

        Parameters
        ----------
        result : OutputResult
            Output result to compute the quantity for.

        Returns
        -------
        float
            Quantity value.
        """


class MaxQuantity(Quantity):
    """Maximum value of a response."""

    def compute(self, result: OutputResult) -> float:
        """Get the maximum of a given response.

        Parameters
        ----------
        result : OutputResult
            Output result to compute the quantity for.

        Returns
        -------
        float
            Quantity value.
        """
        return np.max(result.get_data())


class IntegralQuantity(Quantity):
    """Integral of a response."""

    def compute(self, result: OutputResult) -> float:
        """Get the integral of a given response.

        Parameters
        ----------
        result : OutputResult
            Output result to compute the quantity for.

        Returns
        -------
        float
            Quantity value.
        """
        return np.trapz(result.get_data(), result.get_time())


AVAILABLE_QUANTITIES: Dict[str, Type[Quantity]] = {
    'max': MaxQuantity,
    'integral': IntegralQuantity,
}


class DesignTarget:
    """Container for a design objective target."""

    def __init__(
            self,
            name: str,
            prediction: Union[str, List[str]],
            quantity: Quantity,
            negate: bool = False,
            weight: float = 1.0,
            ) -> None:
        # Sanitise prediction field
        if isinstance(prediction, str):
            prediction = [prediction]
        elif not isinstance(prediction, list):
            raise ValueError(f"Invalid prediction '{prediction}' for design target '{name}'.")
        self.name = name
        self.prediction = prediction
        self.quantity = quantity
        self.negate = negate
        self.weight = weight

    def compute_quantity(self, results: OutputResult) -> float:
        """Compute the design quantity for the given results.

        Parameters
        ----------
        results : OutputResult
            Results to compute the quantity for.

        Returns
        -------
        float
            Design quantity for this response.
        """
        value = self.quantity.compute(results)
        return -value if self.negate else value

    @staticmethod
    def read(name: str, config: Dict[str, Any], output_dir: str) -> DesignTarget:
        """Read the design target from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the design target.
        config : Dict[str, Any]
            Configuration dictionary.
        output_dir: str
            Output directory.

        Returns
        -------
        DesignTarget
            Design target to use for this problem.
        """
        # Prediction parsing
        if 'prediction' not in config:
            raise ValueError(f"Missing prediction for design target '{name}'.")
        # Read the quantity
        if 'quantity' not in config:
            raise ValueError("Missing quantity for fitting objective.")
        quantity = config['quantity']
        # Parse specification: simple or complete
        if isinstance(quantity, str):
            quantity = {'name': quantity}
        elif 'name' not in quantity:
            raise ValueError("Missing name in quantity specification.")
        # Parse script arguments, if passed
        if quantity['name'] == 'script':
            if 'script' not in quantity:
                raise ValueError("Missing script in quantity specification.")
            if 'class' not in quantity:
                raise ValueError("Missing class in quantity specification.")
            quantity_class: Type[Quantity] = load_module_from_file(quantity['script'],
                                                                   quantity['class'])
        else:
            quantity_class: Type[Quantity] = AVAILABLE_QUANTITIES[quantity['name']]
        return DesignTarget(
            name,
            config['prediction'],
            quantity_class(),
            negate=bool(config.get('negate', False)),
        )


class CurrentPlot(DynamicPlotter):
    """Container for dynamically-updating plots."""

    def __init__(self, solver: Solver) -> None:
        """Constructor for dynamically-updating plots.

        Parameters
        ----------
        solver : Solver
            Solver to build the plot for.
        """
        self.solver = solver
        self.figs = {}
        self.preds = {}
        # Get current solver data
        result = solver.get_current_response()
        # Plot each response
        for name, response in result.items():
            fig, axis = plt.subplots()
            self.preds[name], = axis.plot(response.get_time(), response.get_data(), c='red')
            axis.set_title(name)
            axis.grid()
            self.figs[name] = (fig, axis)
        plt.show()
        for fig, _ in self.figs.values():
            fig.canvas.draw()
            fig.canvas.flush_events()

    def update(self) -> None:
        """Update the plot with the most recent data"""
        try:
            result = self.solver.get_current_response()
        except (FileNotFoundError, IndexError):
            pass
        for name, response in result.items():
            self.preds[name].set_xdata(response.get_time())
            self.preds[name].set_ydata(response.get_data())
            fig, axis = self.figs[name]
            axis.relim()
            axis.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()


class DesignObjective(GenericObjective):
    """Scalar design objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: Solver,
            targets: List[DesignTarget],
            output_dir: str,
            stochastic: bool = False,
            ) -> None:
        super().__init__(
            parameters,
            stochastic,
            composition=None,
            output_dir=output_dir,
        )
        self.solver = solver
        self.targets = targets

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def _objective(self, values: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for design objectives.

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
        responses = self.solver.solve(values, concurrent)
        results = []
        variances = []
        for target in self.targets:
            targets = [target.compute_quantity(responses[pred]) for pred in target.prediction]
            results.append(target.weight * np.mean(targets))
            variances.append(target.weight * np.var(targets) / len(targets))
        return ObjectiveResult(results, variances if self.stochastic else None)

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
        append_title = ''
        if options is not None and 'append_title' in options:
            append_title = f' ({options["append_title"]})'
        figures = []
        # Load all responses
        responses = self.solver.get_output_response(case_hash)
        # Plot each target
        for target in self.targets:
            # Build figure with individual responses for this target
            fig, axis = plt.subplots()
            for pred in target.prediction:
                axis.plot(responses[pred].get_time(), responses[pred].get_data(), label=f'{pred}')
            # Under stochasticity, plot response stats and mean confidence interval
            if self.stochastic:
                stats = stats_interp_to_common_grid([
                    (responses[pred].get_time(), responses[pred].get_data())
                    for pred in target.prediction
                ])
                # Plot average and standard deviation
                axis.plot(stats['grid'], stats['average'], label='Average', c='black', ls='dashed')
                axis.fill_between(
                    stats['grid'],
                    stats['average'] - stats['std'],
                    stats['average'] + stats['std'],
                    label='Standard deviation',
                    color='black',
                    alpha=0.2,
                    ls='dashed',
                )
                axis.fill_between(
                    stats['grid'],
                    stats['average'] - stats['confidence'],
                    stats['average'] + stats['confidence'],
                    label='Confidence interval',
                    color='red',
                    alpha=0.2,
                )
            axis.set_title(target.name + append_title)
            axis.grid()
            axis.legend()
            figures.append(fig)
        return figures

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call.

        Returns
        -------
        List[DynamicPlotter]
            List of instances of updatable plots.
        """
        return [CurrentPlot(self.solver)]

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
            ) -> DesignObjective:
        """Read a design objective from a configuration dictionary.

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
        DesignObjective
            Objective function to optimise.
        """
        # Read the solver
        if 'solver' not in config:
            raise ValueError("Missing solver for fitting objective.")
        solver = read_solver(config['solver'], parameters, output_dir)
        # Read the targets
        if 'targets' not in config:
            raise ValueError("Missing targets for fitting objective.")
        targets: Dict[DesignTarget, List[str]] = {}
        for target_name, target_config in config.pop('targets').items():
            target = DesignTarget.read(target_name, target_config, output_dir)
            # Map the target to the solver cases
            targets[target] = []
            for field_name in solver.get_output_fields():
                if field_name in target.prediction:
                    targets[target].append(field_name)
            # Sanitise target: check if it is associated to at least one case
            if len(targets[target]) == 0:
                raise ValueError(f"Design target '{target_name}' is not associated to any case.")
        return DesignObjective(
            parameters,
            solver,
            list(targets.keys()),
            output_dir,
            stochastic=bool(config.get('stochastic', False)),
        )
