"""Provide analytical functions for optimisation."""
from __future__ import annotations
from typing import Dict, Any, List
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.objective import GenericObjective, ObjectiveResult


class AnalyticalObjective(GenericObjective):
    """Objective function derived from an analytical expression."""

    def __init__(self, parameters: ParameterSet, expression: str, output_dir: str = None) -> None:
        super().__init__(
            parameters,
            stochastic=False,
            composition=None,
            output_dir=output_dir,
        )
        # Generate a dummy set of parameters (to ensure proper handling of output parameters)
        values = np.array([parameter.inital_value for parameter in parameters])
        symbs = sympy.symbols(list(parameters.to_dict(values, input_normalised=False).keys()))
        self.parameters = parameters
        self.expression = sympy.lambdify(symbs, expression)

    def _objective(self, values: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for analytical functions.

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
        value = self.expression(**self.parameters.to_dict(values))
        return ObjectiveResult([np.array([value])])

    def _objective_denorm(self, values: np.ndarray) -> float:
        """Objective computation for analytical functions (denormalised parameters).

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for (denormalised).

        Returns
        -------
        float
            Objective value.
        """
        return self.expression(**self.parameters.to_dict(values, input_normalised=False))

    def _plot_1d(self, values: np.ndarray, append_title: str) -> Figure:
        """Plot the objective in 1D.

        Parameters
        ----------
        values : np.ndarray
            Parameter values to plot for.
        append_title : str
            String to append to the title.

        Returns
        -------
        Figure
            Figure with the plot.
        """
        fig, axis = plt.subplots()
        x = np.linspace(self.parameters[0].lbound, self.parameters[0].ubound, 1000)
        y = np.array([self._objective_denorm(np.array([x_i])) for x_i in x])
        axis.plot(x, y, c="black", label="Analytical Objective")
        axis.scatter(values[0], self._objective_denorm(values), c="red", label="Case")
        axis.set_xlabel(self.parameters[0].name)
        axis.set_ylabel("Analytical Objective")
        axis.set_xlim(self.parameters[0].lbound, self.parameters[0].ubound)
        axis.legend()
        axis.grid()
        axis.set_title(append_title)
        return fig

    def _plot_2d(self, values: np.ndarray, append_title: str) -> Figure:
        """Plot the objective in 2D.

        Parameters
        ----------
        values : np.ndarray
            Parameter values to plot for.
        append_title : str
            String to append to the title.

        Returns
        -------
        Figure
            Figure with the plot
        """
        fig, axis = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.linspace(self.parameters[0].lbound, self.parameters[0].ubound, 100)
        y = np.linspace(self.parameters[1].lbound, self.parameters[1].ubound, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self._objective_denorm(np.array([x_i, y_i])) for x_i in x] for y_i in y])
        axis.scatter(
            values[0],
            values[1],
            self._objective_denorm(values),
            c="r",
            label="Case",
            s=50,
        )
        axis.plot_surface(X, Y, Z, alpha=0.7, label="Analytical Objective")
        axis.set_xlabel(self.parameters[0].name)
        axis.set_ylabel(self.parameters[1].name)
        axis.set_zlabel("Analytical Objective")
        axis.set_xlim(self.parameters[0].lbound, self.parameters[0].ubound)
        axis.set_ylim(self.parameters[1].lbound, self.parameters[1].ubound)
        axis.legend()
        axis.grid()
        axis.set_title(append_title)
        fig.tight_layout()
        return fig

    def plot_case(self, case_hash: str, options: Dict[str, Any] = None) -> List[Figure]:
        """Plot a given function call given the parameter hash.

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot.
        options : Dict[str, Any], optional
            Options to pass to the plotting function, by default None.

        Returns
        -------
        List[Figure]
            List of figures with the plot.
        """
        # Find parameters associated with the hash
        df = pd.read_table(self.func_calls_file)
        df.columns = df.columns.str.strip()
        df = df[df["Hash"] == case_hash]
        values = df[[param.name for param in self.parameters]].to_numpy()[0, :]
        # Build title
        append_title = ''
        if options is not None and 'append_title' in options:
            append_title = f'{options["append_title"]}'
        # Plot depending on the dimensions
        if len(self.parameters) <= 0:
            raise RuntimeError("Missing dimensions.")
        if len(self.parameters) == 1:
            return [self._plot_1d(values, append_title)]
        if len(self.parameters) == 2:
            return [self._plot_2d(values, append_title)]
        raise RuntimeError("Plotting not supported for 3 or more parameters.")

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
            ) -> AnalyticalObjective:
        """Read the objective from a configuration dictionary.

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
        SyntheticObjective
            Objective function to optimise.
        """
        # Check for mandatory arguments
        if 'expression' not in config:
            raise RuntimeError("Missing analytical expression to minimise")
        return AnalyticalObjective(parameters, config['expression'], output_dir=output_dir)
