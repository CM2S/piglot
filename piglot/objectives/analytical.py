"""Provide analytical functions for optimisation."""
from typing import Any, Optional, TypeVar
import sympy
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.settings import Settings
from piglot.objective import IndividualObjectiveResult
from piglot.objectives.simple_objective import SimpleObjective, SimpleIndividualObjective


IndividualT = TypeVar('IndividualT', bound='AnalyticalIndividualObjective')


class AnalyticalIndividualObjective(SimpleIndividualObjective):
    """Objective function derived from an analytical expression."""

    def __init__(
        self,
        name: str,
        settings: Settings,
        expression: str,
        weight: float = 1.0,
        maximise: bool = False,
        variance: bool = False,
        bounds: tuple[float, float] = None,
        variance_expr: Optional[str] = None,
        use_random: bool = True,
        random_evals: int = 0,
    ) -> None:
        super().__init__(
            name,
            weight=weight,
            maximise=maximise,
            variance=variance,
            composite=False,
            bounds=bounds,
        )
        # Sanitise the stochastic and random_evals combination
        if random_evals > 0 and variance_expr is None:
            raise ValueError("Random evaluations require variance.")
        # Generate a dummy set of parameters (to ensure proper handling of output parameters)
        self.parameters = settings.parameters
        values = np.array([parameter.inital_value for parameter in self.parameters])
        symbs = sympy.symbols(list(self.parameters.to_dict(values).keys()))
        self.expression = sympy.lambdify(symbs, expression)
        self.variance_expr = None if variance_expr is None else sympy.lambdify(symbs, variance_expr)
        self.use_random = use_random
        self.random_evals = random_evals

    def evaluate(self, params: np.ndarray, concurrent: bool) -> IndividualObjectiveResult:
        """Evaluate objective value for the given results.

        Parameters
        ----------
        params : np.ndarray
            Parameter values for this evaluation.
        concurrent : bool
            Whether this call may be concurrent to others.

        Returns
        -------
        IndividualObjectiveResult
            Objective value and variance for the given parameters.
        """
        value = self.expression(**self.parameters.to_dict(params))
        variance = 0
        if self.variance_expr is not None:
            variance = self.variance_expr(**self.parameters.to_dict(params))
            if variance < 0:
                raise RuntimeError("Negative variance not allowed.")
        # When random evaluations are requested, replace the data from sample evaluations
        if self.random_evals > 0 and self.use_random:
            evals = np.random.normal(value, np.sqrt(variance), size=(self.random_evals,))
            value = np.mean(evals)
            if self.random_evals > 1:
                variance = np.var(evals)  # / self.random_evals
        if self.maximise:
            value = -value
        return IndividualObjectiveResult(value=value, variance=variance if self.variance else None)

    def plot_1d(self, values: np.ndarray, append_title: str) -> Figure:
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
        # TODO: fixme
        raise NotImplementedError()
        fig, axis = plt.subplots()
        x = np.linspace(self.parameters[0].lbound, self.parameters[0].ubound, 1000)
        evals = np.array([self.evaluate(np.array([x_i]), use_random=False) for x_i in x])
        curr_eval, curr_var = self.evaluate(values)
        axis.plot(x, evals[:, 0], c="black", label="Analytical Objective")
        if self.variance is not None:
            axis.fill_between(
                x,
                evals[:, 0] - 2 * np.sqrt(evals[:, 1]),
                evals[:, 0] + 2 * np.sqrt(evals[:, 1]),
                color="black",
                alpha=0.2,
                label="Analytical Variance",
            )
            axis.errorbar(
                values[0],
                curr_eval,
                yerr=2 * np.sqrt(curr_var),
                label="Case",
                fmt="o",
            )
        else:
            axis.scatter(values[0], curr_eval, label="Case")
        axis.set_xlabel(self.parameters[0].name)
        axis.set_ylabel("Analytical Objective")
        axis.set_xlim(self.parameters[0].lbound, self.parameters[0].ubound)
        axis.legend()
        axis.grid()
        axis.set_title(append_title)
        return fig

    def plot_2d(self, values: np.ndarray, append_title: str) -> Figure:
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
        # TODO: fixme
        raise NotImplementedError()
        fig, axis = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.linspace(self.parameters[0].lbound, self.parameters[0].ubound, 100)
        y = np.linspace(self.parameters[1].lbound, self.parameters[1].ubound, 100)
        X, Y = np.meshgrid(x, y)
        evals = np.array(
            [[self.evaluate(np.array([x_i, y_i]), use_random=False) for x_i in x] for y_i in y]
        )
        curr_eval, _ = self.evaluate(values)
        axis.scatter(
            values[0],
            values[1],
            curr_eval,
            c="r",
            label="Case",
            s=50,
        )
        axis.plot_surface(X, Y, evals[:, :, 0], alpha=0.7, label="Analytical Objective")
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

    @classmethod
    def read(
        cls: type[IndividualT], config: dict[str, Any], settings: Settings
    ) -> IndividualT:
        """Read the objective from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Terms from the configuration dictionary.
        settings : Settings
            Settings for this problem.

        Returns
        -------
        IndividualT
            Objective function to optimise.
        """
        # Check for mandatory arguments
        if 'expression' not in config:
            raise RuntimeError("Missing analytical expression to minimise")
        return cls(
            config.get('name', 'Analytical Objective'),
            settings,
            config['expression'],
            weight=float(config.get('weight', 1.0)),
            maximise=bool(config.get('maximise', False)),
            variance=config.get('variance', None),
            bounds=config.get('bounds', None),
            variance_expr=config.get('variance_expr', None),
            use_random=config.get('use_random', True),
            random_evals=config.get('random_evals', 0),
        )


class AnalyticalObjective(SimpleObjective[AnalyticalIndividualObjective]):
    """Objective function derived from an analytical expression."""

    # def plot_case(self, case_hash: str, options: dict[str, Any] = None) -> list[Figure]:
    #     """Plot a given function call given the parameter hash.

    #     Parameters
    #     ----------
    #     case_hash : str, optional
    #         Parameter hash for the case to plot.
    #     options : dict[str, Any], optional
    #         Options to pass to the plotting function, by default None.

    #     Returns
    #     -------
    #     list[Figure]
    #         list of figures with the plot.
    #     """
    #     # Find parameters associated with the hash
    #     df = pd.read_table(self.func_calls_file)
    #     df.columns = df.columns.str.strip()
    #     df = df[df["Hash"] == case_hash]
    #     values = df[[param.name for param in self.parameters]].to_numpy()[0, :]
    #     # Build title
    #     append_title = ''
    #     if options is not None and 'append_title' in options:
    #         append_title = f'{options["append_title"]}'
    #     # Plot depending on the dimensions
    #     if len(self.parameters) not in (1, 2):
    #         raise RuntimeError("Plotting only supported for one or two dimensions.")
    #     if len(self.parameters) == 1:
    #         return [self.expression.plot_1d(values, append_title)]
    #     return [self.expression.plot_2d(values, append_title)]

    @classmethod
    def individual_objective_type(cls) -> type[AnalyticalIndividualObjective]:
        """Get the type of the individual objective for this simple objective.

        Returns
        -------
        type[AnalyticalIndividualObjective]
            Individual objective type for this simple objective.
        """
        return AnalyticalIndividualObjective
