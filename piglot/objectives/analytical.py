"""Provide analytical functions for optimisation."""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Tuple
import sympy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.objective import (
    GenericObjective,
    ObjectiveResult,
    Scalarisation,
    Composition,
    IndividualObjective,
)
from piglot.utils.scalarisations import read_scalarisation, SumScalarisation


class AnalyticalSingleObjective(IndividualObjective):
    """Objective function derived from an analytical expression."""

    def __init__(
        self,
        parameters: ParameterSet,
        expression: str,
        variance: Optional[str] = None,
        random_evals: int = 0,
        maximise: bool = False,
        weight: float = 1.0,
        bounds: Tuple[float, float] = None,
    ) -> None:
        super().__init__(maximise, weight, bounds)
        # Sanitise the stochastic and random_evals combination
        if random_evals > 0 and variance is None:
            raise ValueError("Random evaluations require variance.")
        # Generate a dummy set of parameters (to ensure proper handling of output parameters)
        values = np.array([parameter.inital_value for parameter in parameters])
        symbs = sympy.symbols(list(parameters.to_dict(values).keys()))
        self.parameters = parameters
        self.expression = sympy.lambdify(symbs, expression)
        self.variance = None if variance is None else sympy.lambdify(symbs, variance)
        self.random_evals = random_evals

    def evaluate(self, params: np.ndarray) -> Tuple[float, float]:
        """Evaluate objective value for the given results.

        Parameters
        ----------
        params : np.ndarray
            Parameter values for this evaluation.
        raw_results : Dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        Tuple[float, float]
            Mean and variance of the objective.
        """
        value = self.expression(**self.parameters.to_dict(params))
        variance = 0 if self.variance is None else self.variance(**self.parameters.to_dict(params))
        if self.maximise:
            value = -value
        # When random evaluations are requested, replace the data from sample evaluations
        if self.random_evals > 0:
            evals = np.random.normal(value, np.sqrt(variance), size=(self.random_evals,))
            value = np.mean(evals)
            if self.random_evals > 1:
                variance = np.var(evals)  # / self.random_evals
        return value, variance

    @classmethod
    def read(
        cls,
        config: Dict[str, Any],
        parameters: ParameterSet,
    ) -> AnalyticalSingleObjective:
        """Read the objective from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.
        parameters : ParameterSet
            Set of parameters for this problem.

        Returns
        -------
        AnalyticalSingleObjective
            Objective function to optimise.
        """
        # Check for mandatory arguments
        if 'expression' not in config:
            raise RuntimeError("Missing analytical expression to minimise")
        return AnalyticalSingleObjective(
            parameters,
            config['expression'],
            variance=config.get('variance', None),
            random_evals=config.get('random_evals', 0),
            maximise=config.get('maximise', False),
            weight=config.get('weight', 1.0),
            bounds=config.get('bounds', None),
        )


class AnalyticalObjective(GenericObjective):
    """Objective function derived from an analytical expression."""

    def __init__(
        self,
        parameters: ParameterSet,
        expression: str,
        variance: Optional[str] = None,
        infer_noise: bool = False,
        stochastic: bool = False,
        random_evals: int = 0,
        output_dir: str = None,
    ) -> None:
        # Sanitise the stochastic and random_evals combination
        if random_evals > 0 and variance is None:
            raise ValueError("Random evaluations require variance.")
        super().__init__(
            parameters,
            infer_noise=infer_noise,
            stochastic=stochastic,
            composition=None,
            output_dir=output_dir,
        )
        self.parameters = parameters
        self.expression = AnalyticalSingleObjective(parameters, expression, variance, random_evals)

    def _objective(self, params: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for analytical functions.

        Parameters
        ----------
        params : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        ObjectiveResult
            Objective result.
        """
        value, variance = self.expression.evaluate(params)
        return ObjectiveResult(
            params,
            np.array([value]),
            np.array([value]),
            scalar_value=value,
            covariances=np.array([[variance]]) if self.stochastic else None,
            obj_variances=np.array([variance]) if self.stochastic else None,
            scalar_variance=variance if self.stochastic else None,
        )

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
        # TODO: fixme
        raise NotImplementedError()
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
        # TODO: fixme
        raise NotImplementedError()
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

    @classmethod
    def read(
        cls,
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
        return AnalyticalObjective(
            parameters,
            config['expression'],
            infer_noise=config.get('infer_noise', False),
            variance=config.get('variance', None),
            stochastic=config.get('stochastic', False),
            random_evals=config.get('random_evals', 0),
            output_dir=output_dir,
        )


class ScalarisationComposition(Composition):
    """Composition for scalarisation of multiple objectives."""

    def __init__(self, scalarisation: Scalarisation) -> None:
        super().__init__()
        self.scalarisation = scalarisation

    def composition_torch(self, inner: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute the composition for all objectives.

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function.
        params : torch.Tensor
            Paratemers for the given responses.

        Returns
        -------
        torch.Tensor
            Composition results.
        """
        return self.scalarisation.scalarise_torch(inner)[0]


class AnalyticalMultiObjective(GenericObjective):
    """Multi-objective problem derived from a set of analytical expressions."""

    def __init__(
        self,
        parameters: ParameterSet,
        objectives: List[AnalyticalSingleObjective],
        infer_noise: bool = False,
        stochastic: bool = False,
        scalarisation: Optional[Scalarisation] = None,
        composite: bool = False,
        output_dir: str = None,
    ) -> None:
        # Sanitise scalarisation-related stuff
        if scalarisation is None:
            if composite:
                raise ValueError("Composite objectives require scalarisation.")
            if len(objectives) == 1:
                scalarisation = SumScalarisation(objectives)
        super().__init__(
            parameters,
            infer_noise=infer_noise,
            stochastic=stochastic,
            composition=ScalarisationComposition(scalarisation) if composite else None,
            scalarisation=None if composite else scalarisation,
            num_objectives=len(objectives),
            multi_objective=len(objectives) > 1 and scalarisation is None,
            output_dir=output_dir,
        )
        self.parameters = parameters
        self.expressions = objectives

    def _objective(self, params: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for analytical functions.

        Parameters
        ----------
        params : np.ndarray
            Set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        ObjectiveResult
            Objective result.
        """
        # Compute values and variances for each objective
        results = [obj.evaluate(params) for obj in self.expressions]
        obj_values = np.array([value for value, _ in results])
        obj_variances = np.array([var for _, var in results])
        # Under single-objective, compute the scalar objective value
        scalar_value, scalar_variance = None, None
        if not self.multi_objective:
            scalarisation = self.scalarisation or self.composition.scalarisation
            scalar_value, scalar_variance = scalarisation.scalarise(obj_values, obj_variances)
        # Get the values to return to the optimiser
        if self.composition is not None or self.multi_objective:
            optim_values, optim_covar = obj_values, np.diag(obj_variances)
        else:
            optim_values, optim_covar = np.array([scalar_value]), np.array([[scalar_variance]])
        return ObjectiveResult(
            params,
            optim_values,
            obj_values,
            scalar_value=scalar_value,
            covariances=optim_covar if self.stochastic else None,
            obj_variances=obj_variances if self.stochastic else None,
            scalar_variance=scalar_variance if self.stochastic else None,
        )

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
        # TODO: fixme
        raise NotImplementedError()
        fig, axis = plt.subplots()
        x = np.linspace(self.parameters[0].lbound, self.parameters[0].ubound, 1000)
        y = np.array([self._objective_denorm(np.array([x_i])) for x_i in x])
        axis.plot(x, y, c="black", label="Analytical Objective")
        if self.variance is None:
            axis.scatter(values[0], self._objective_denorm(values), c="red", label="Case")
        else:
            y_std = np.array(
                [np.sqrt(self.variance(**self.parameters.to_dict(np.array([x_i])))) for x_i in x]
            )
            axis.fill_between(x, y - 2 * y_std, y + 2 * y_std, color="gray", alpha=0.5)

            point_value = self._objective_denorm(values)
            point_std = np.sqrt(self.variance(**self.parameters.to_dict(values)))
            axis.errorbar(
                values[0],
                point_value,
                yerr=2 * point_std,
                fmt="o",
                c="red",
                label="Case",
            )
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
        # TODO: fixme
        raise NotImplementedError()
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

    @classmethod
    def read(
        cls,
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> AnalyticalMultiObjective:
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
        # Read objectives
        if 'objectives' not in config:
            raise RuntimeError("Missing analytical objectives to optimise for")
        objectives = [
            AnalyticalSingleObjective.read(target_config, parameters)
            for _, target_config in config.pop('objectives').items()
        ]
        return AnalyticalMultiObjective(
            parameters,
            objectives,
            scalarisation=(
                read_scalarisation(config['scalarisation'], objectives)
                if 'scalarisation' in config else None
            ),
            infer_noise=bool(config.get('infer_noise', False)),
            stochastic=bool(config.get('stochastic', False)),
            composite=bool(config.get('composite', False)),
            output_dir=output_dir,
        )
