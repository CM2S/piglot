"""Module for generic response-based objectives."""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Type, TypeVar
from abc import ABC, abstractmethod
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.solver.solver import Solver, OutputResult
from piglot.objective import (
    Composition,
    GenericObjective,
    ObjectiveResult,
    DynamicPlotter,
    IndividualObjective,
)
from piglot.utils.reductions import Reduction, NegateReduction
from piglot.utils.response_transformer import ResponseTransformer
from piglot.utils.composition.responses import FlattenUtility, ConcatUtility
from piglot.utils.scalarisations import Scalarisation, SumScalarisation


T = TypeVar('T', bound='ResponseSingleObjective')


class DynamicResponsePlotter(DynamicPlotter):
    """Dynamic plotter for response-based objectives."""

    def __init__(
        self,
        figures: List[Figure],
        solver: Solver,
        mapping: Dict[Line2D, str],
        transformers: Dict[str, ResponseTransformer],
    ) -> None:
        self.figures = figures
        self.solver = solver
        self.mapping = mapping
        self.transformers = transformers

    def update(self) -> None:
        """Update the plot with new results."""
        try:
            result = self.solver.get_current_response()
        except (FileNotFoundError, IndexError):
            return
        # Update the lines
        for line, name in self.mapping.items():
            if name not in result:
                continue
            response = result[name]
            if name in self.transformers:
                response = self.transformers[name].transform(response)
            line.set_xdata(response.get_time())
            line.set_ydata(response.get_data())
        # Redraw the plot
        for fig in self.figures:
            for ax in fig.axes:
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()


class ResponseSingleObjective(IndividualObjective, ABC):
    """Base class for generic response-based objectives."""

    def __init__(
        self,
        name: str,
        prediction: List[str],
        quantity: Reduction,
        maximise: bool = False,
        weight: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None,
        flatten_utility: Optional[FlattenUtility] = None,
        prediction_transform: Optional[ResponseTransformer] = None,
    ) -> None:
        super().__init__(maximise=maximise, weight=weight, bounds=bounds)
        self.name = name
        self.prediction = prediction
        self.quantity = NegateReduction(quantity) if maximise else quantity
        self.flatten_utility = flatten_utility
        self.prediction_transform = prediction_transform

    def _extract_responses(self, raw_results: Dict[str, OutputResult]) -> List[OutputResult]:
        """Extract responses of interest from the results and compute any required transformation.

        Parameters
        ----------
        raw_results : Dict[str, OutputResult]
            Raw responses from the solver

        Returns
        -------
        List[OutputResult]
            List of transformed results.
        """
        results = [raw_results[name] for name in self.prediction]
        if self.prediction_transform is None:
            return results
        return [self.prediction_transform.transform(result) for result in results]

    @staticmethod
    def _expand_params(time: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Expand the set of parameters to match the time grid.

        Parameters
        ----------
        time : torch.Tensor
            Time grid for the responses.
        params : torch.Tensor
            Parameters for the given responses.

        Returns
        -------
        torch.Tensor
            Expanded parameter values.
        """
        # Nothing to do when shapes are consistent
        if len(params.shape) == len(time.shape):
            return params
        # Expand the parameters along the first dimensions
        return params.expand(*(list(time.shape[:-1]) + [params.shape[-1]]))

    def evaluate(
        self,
        params: np.ndarray,
        raw_results: Dict[str, OutputResult],
    ) -> Tuple[float, float]:
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
        values = [
            self.quantity.reduce(result.time, result.data, params)
            for result in self._extract_responses(raw_results)
        ]
        # Only compute the variance if we have more than one response
        # TODO: add different stochastic models
        return (
            np.mean(values),
            np.var(values) / len(values) if len(values) > 1 else 0.0,
        )

    def latent_space(self, raw_results: Dict[str, OutputResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute latent space representation of the given results (for composite objectives).

        Parameters
        ----------
        raw_results : Dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Mean and variance of the latent space representation.
        """
        latent_space = np.array([
            self.flatten_utility.flatten(result.time, result.data)
            for result in self._extract_responses(raw_results)
        ])
        # Only compute the covariance if we have more than one response
        # TODO: add different stochastic models
        covariance = (
            np.cov(latent_space.T) / latent_space.shape[0]
            if latent_space.shape[0] > 1
            else np.zeros((latent_space.shape[1], latent_space.shape[1]))
        )
        return np.mean(latent_space, axis=0), covariance

    def evaluate_from_latent_space(
        self,
        params: torch.Tensor,
        latent_responses: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the objective value(s) from the latent space representation.

        Parameters
        ----------
        params : torch.Tensor
            Parameter values for these results.
        latent_responses : torch.Tensor
            Response(s) in the latent space representation.

        Returns
        -------
        torch.Tensor
            Objective value(s) for the response(s).
        """
        time, data = self.flatten_utility.unflatten_torch(latent_responses)
        return self.quantity.reduce_torch(time, data, self._expand_params(time, params))

    @abstractmethod
    def plot(self, axis: plt.Axes, raw_results: Dict[str, OutputResult]) -> Dict[Line2D, str]:
        """Plot the response for this objective.

        Parameters
        ----------
        axis : plt.Axes
            Axis to plot the response on.
        raw_results : Dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        Dict[Line2D, str]
            Mapping of lines to response names (for dynamically updating plots).
        """

    @classmethod
    @abstractmethod
    def read(cls: Type[T], name: str, config: Dict[str, Any], output_dir: str) -> T:
        """Read the objective spec from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the objective.
        config : Dict[str, Any]
            Configuration dictionary.
        output_dir: str
            Output directory.

        Returns
        -------
        ResponseSingleObjective
            Single objective to use.
        """


class ResponseComposition(Composition, ABC):
    """Generic class for compositions to use for response-based objectives."""

    def __init__(
        self,
        objectives: List[ResponseSingleObjective],
        scalarisation: Optional[Scalarisation] = None,
    ) -> None:
        super().__init__()
        self.objectives = objectives
        self.scalarisation = scalarisation
        self.concat = ConcatUtility([obj.flatten_utility.length() for obj in self.objectives])

    @abstractmethod
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

    @abstractmethod
    def get_latent_space(
        self,
        params: np.ndarray,
        raw_responses: Dict[str, OutputResult],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the latent space representation of the given results.

        Parameters
        ----------
        params : np.ndarray
            Parameter values for these results.
        raw_responses : Dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Latent space representation of the results: mean and covariance.
        """


class FullComposition(ResponseComposition):
    """Container for the outer composition of composite response-based objectives."""

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
        # Split the inner responses and compute the objective values
        latent_responses = self.concat.split_torch(inner)
        objectives = torch.stack([
            objective.evaluate_from_latent_space(params, latent_response)
            for latent_response, objective in zip(latent_responses, self.objectives)
        ], dim=-1)
        # Scalarise the objectives if necessary
        if self.scalarisation is not None:
            objectives, _ = self.scalarisation.scalarise_torch(objectives)
        return objectives

    def get_latent_space(
        self,
        params: np.ndarray,
        raw_responses: Dict[str, OutputResult],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the latent space representation of the given results.

        Parameters
        ----------
        params : np.ndarray
            Parameter values for these results.
        raw_responses : Dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Latent space representation of the results: mean and covariance.
        """
        latent_space = [
            objective.latent_space(raw_responses)
            for objective in self.objectives
        ]
        return (
            self.concat.concat([mean for mean, _ in latent_space]),
            self.concat.concat_covar([var for _, var in latent_space]),
        )


class ScalarisationComposition(ResponseComposition):
    """Composition for scalarisation of non-composite response objectives."""

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

    def get_latent_space(
        self,
        params: np.ndarray,
        raw_responses: Dict[str, OutputResult],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the latent space representation of the given results.

        Parameters
        ----------
        params : np.ndarray
            Parameter values for these results.
        raw_responses : Dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Latent space representation of the results: mean and covariance.
        """
        objectives = [
            objective.evaluate(params, raw_responses)
            for objective in self.objectives
        ]
        return (
            self.concat.concat([np.array([mean]) for mean, _ in objectives]),
            self.concat.concat_covar([np.array([var]) for _, var in objectives]),
        )


class ResponseObjective(GenericObjective):
    """Objective for generic response-based objectives."""

    def __init__(
        self,
        parameters: ParameterSet,
        solver: Solver,
        objectives: List[ResponseSingleObjective],
        output_dir: str,
        scalarisation: Scalarisation = None,
        stochastic: bool = False,
        composite: bool = False,
        full_composite: bool = True,
        transformers: Dict[str, ResponseTransformer] = None,
    ) -> None:
        # Sanitise the scalarisation
        if scalarisation is None:
            # Everything fine if we just have a single objective: use a sum scalarisation
            if len(objectives) == 1:
                scalarisation = SumScalarisation(objectives)
            elif composite and not full_composite:
                raise ValueError('Multi-objective composite problems require full composition')
        # Get the type of composition to use
        composite_type = FullComposition if full_composite else ScalarisationComposition
        super().__init__(
            parameters,
            stochastic=stochastic,
            composition=composite_type(objectives, scalarisation) if composite else None,
            scalarisation=None if composite else scalarisation,
            num_objectives=len(objectives),
            multi_objective=len(objectives) > 1 and scalarisation is None,
            output_dir=output_dir,
        )
        self.composition: Optional[ResponseComposition] = self.composition
        self.solver = solver
        self.objectives = objectives
        self.transformers = transformers if transformers is not None else {}
        # Sanitise predictions
        for objective in self.objectives:
            for name in objective.prediction:
                if name not in self.solver.get_output_fields():
                    raise ValueError(f'Undefined prediction {name}')

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def postproc_responses(self, responses: Dict[str, OutputResult]) -> Dict[str, OutputResult]:
        """Post-process the responses from the solver.

        Parameters
        ----------
        responses : Dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        Dict[str, OutputResult]
            Post-processed responses.
        """
        # Sanitise responses
        empty_responses = [name for name, result in responses.items() if len(result.time) == 0]
        if len(empty_responses) > 0:
            warnings.warn(
                f'Solver call returned empty responses for the output fields {empty_responses}. '
                'Please validate the solver output. Sanitising to zero responses.',
                RuntimeWarning,
            )
            for name in empty_responses:
                responses[name] = OutputResult(np.zeros(1), np.zeros(1))
        # Transform responses
        for name, transformer in self.transformers.items():
            if name in responses:
                responses[name] = transformer.transform(responses[name])
        return responses

    def _objective(self, params: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for design objectives.

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
        raw_responses = self.solver.solve(params, concurrent)
        # Sanitise and post-process the responses
        raw_responses = self.postproc_responses(raw_responses)
        # Compute the objective value and variance from each response objective
        results = [objective.evaluate(params, raw_responses) for objective in self.objectives]
        obj_values = np.array([mean for mean, _ in results])
        obj_variances = np.array([var for _, var in results])
        # Under single-objective, compute the scalar objective value
        scalar_value, scalar_variance = None, None
        if not self.multi_objective:
            scalarisation = self.scalarisation or self.composition.scalarisation
            scalar_value, scalar_variance = scalarisation.scalarise(obj_values, obj_variances)
        # Get the values to return to the optimiser. Three scenarios:
        # (i) under composition, return the latent space
        # (ii) non-composite multi-objective, return the objective values
        # (iii) non-composite single-objective, return the scalarised objective
        if self.composition is not None:
            optim_values, optim_covar = self.composition.get_latent_space(params, raw_responses)
        elif self.multi_objective:
            optim_values, optim_covar = obj_values, np.diag(obj_variances)
        else:
            optim_values, optim_covar = np.array([scalar_value]), np.array([[scalar_variance]])
        # Return the objective result: only return the variances if we are stochastic
        return ObjectiveResult(
            params,
            optim_values,
            obj_values,
            scalar_value=scalar_value,
            covariances=optim_covar if self.stochastic else None,
            obj_variances=obj_variances if self.stochastic else None,
            scalar_variance=scalar_variance if self.stochastic else None,
        )

    def plot_case(self, case_hash: str, options: Dict[str, Any] = None) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        options : Dict[str, Any], optional
            Options to pass to the plotting function, by default None

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        append_title = ''
        if options is not None and 'append_title' in options:
            append_title = f' ({options["append_title"]})'
        # Load all responses and post-process them
        responses = self.postproc_responses(self.solver.get_output_response(case_hash))
        # Extract the parameters
        params = self.solver.get_case_params(case_hash)
        if options is not None and 'params' in options:
            append_title += f' - {params}'
        # Plot each target
        figures = []
        for objective in self.objectives:
            fig, axis = plt.subplots()
            objective.plot(axis, responses)
            axis.set_title(objective.name + append_title)
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
        # Get current solver data
        responses = self.postproc_responses(self.solver.get_current_response())
        # Plot each objective
        figures: List[Figure] = []
        mapping: Dict[Line2D, str] = {}
        for objective in self.objectives:
            fig, axis = plt.subplots()
            line, = objective.plot(axis, responses)
            axis.set_title(objective.name)
            axis.legend()
            # Store the line and figure
            mapping[line] = objective.name
            figures.append(fig)
        # Show the plot
        plt.show()
        for fig in figures:
            fig.canvas.draw()
            fig.canvas.flush_events()
        return [DynamicResponsePlotter(figures, self.solver, mapping, self.transformers)]
