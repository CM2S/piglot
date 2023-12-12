"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import Solver, OutputResult
from piglot.objective import DynamicPlotter, SingleObjective
from piglot.utils.solver_utils import load_module_from_file


class Quantity(ABC):
    """Abstract class for quantities to be computed from a response."""

    @abstractmethod
    def compute(self, result: OutputResult) -> Any:
        """Compute this quantity for a given response.

        Parameters
        ----------
        result : OutputResult
            Output result to compute the quantity for.

        Returns
        -------
        Any
            Quantity value.
        """


class MaxQuantity(Quantity):
    """Maximum value of a response."""

    def compute(self, result: OutputResult) -> Any:
        """Get the maximum of a given response.

        Parameters
        ----------
        result : OutputResult
            Output result to compute the quantity for.

        Returns
        -------
        Any
            Quantity value.
        """
        return np.max(result.get_data())


class IntegralQuantity(Quantity):
    """Integral of a response."""

    def compute(self, result: OutputResult) -> Any:
        """Get the integral of a given response.

        Parameters
        ----------
        result : OutputResult
            Output result to compute the quantity for.

        Returns
        -------
        Any
            Quantity value.
        """
        return np.trapz(result.get_data(), result.get_time())


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
        # Plot each reference
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


class DesignSolver:
    """Interface class between design objectives and solvers."""

    def __init__(
            self,
            solver: Solver,
            quantity: Quantity,
            negate: bool=False,
        ) -> None:
        self.solver = solver
        self.quantity = quantity
        self.negate = negate

    def prepare(self) -> None:
        """Prepare the solver for optimisation."""
        self.solver.prepare()

    def solve(self, values: np.ndarray, concurrent: bool) -> Dict[str, Any]:
        """Evaluate the solver for the given set of parameter values and get the output results.

        Parameters
        ----------
        values : np.ndarray
            Parameter values to evaluate.
        concurrent : bool
            Whether this call may be concurrent to others.

        Returns
        -------
        Dict[str, Any]
            Output results.
        """
        return {
            name: -self.quantity.compute(value) if self.negate else self.quantity.compute(value)
            for name, value in self.solver.solve(values, concurrent).items()
        }

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
        append_title = ''
        if options is not None and 'append_title' in options:
            append_title = f' ({options["append_title"]})'
        figures = []
        # Load all responses
        responses = self.solver.get_output_response(case_hash)
        # Plot each reference
        for name, response in responses.items():
            # Build figure, index axes and plot response
            fig, axis = plt.subplots()
            axis.plot(response.get_time(), response.get_data(), c='red')
            axis.set_title(name + append_title)
            axis.grid()
            figures.append(fig)
        return figures

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of a updatable plots
        """
        return [CurrentPlot(self.solver)]

    @staticmethod
    def read(config: Dict[str, Any], parameters: ParameterSet, output_dir: str) -> DesignSolver:
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
        DesignSolver
            Solver for fitting objectives.
        """
        # Read the solver
        if not 'solver' in config:
            raise ValueError("Missing solver for fitting objective.")
        solver = read_solver(config['solver'], parameters, output_dir)
        # Read the quantity
        if not 'quantity' in config:
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
            quantity_class = load_module_from_file(quantity['script'], quantity['class'])
        else:
            quantity_class = {
                'max': MaxQuantity(),
                'integral': IntegralQuantity(),
            }[quantity['name']]
        # Parse the negation keyword, if present
        negate = bool(config.get('negate', False))
        # Return the solver
        return DesignSolver(solver, quantity_class, negate)


class DesignSingleObjective(SingleObjective):
    """Scalar design objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: DesignSolver,
            output_dir: str,
        ) -> None:
        super().__init__(parameters, output_dir)
        self.solver = solver

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def _objective(self, values: np.ndarray, concurrent: bool=False) -> float:
        """Objective computation for analytical functions

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """
        result = self.solver.solve(values, concurrent)
        return np.mean([value for value in result.values()])

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
        ) -> DesignSingleObjective:
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
        Objective
            Objective function to optimise.
        """
        return DesignSingleObjective(
            parameters,
            DesignSolver.read(config, parameters, output_dir),
            output_dir,
        )
