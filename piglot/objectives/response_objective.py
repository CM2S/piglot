"""Module for generic response-based objectives."""
from typing import Any, Optional, TypeVar
from abc import ABC, abstractmethod
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from piglot.parameter import ParameterValues
from piglot.settings import Settings
from piglot.solver.solver import Solver, OutputResult
from piglot.objective import (
    Objective,
    IndividualObjective,
    IndividualObjectiveResult,
)
from piglot.utils.composition import LatentTransformer
from piglot.utils.reductions import Reduction
from piglot.utils.response_transformer import ResponseTransformer
from piglot.utils.scalarisations import Scalarisation


T = TypeVar('T', bound='ResponseSingleObjective')


class ResponseSingleObjective(IndividualObjective, ABC):
    """Base class for generic response-based objectives."""

    def __init__(
        self,
        name: str,
        prediction: list[str],
        quantity: Reduction,
        mean_dist: bool = False,
        weight: float = 1.0,
        maximise: bool = False,
        variance: bool = False,
        composite: bool = False,
        noisy: Optional[bool] = None,
        bounds: Optional[tuple[float, float]] = None,
        latent_transformer: Optional[LatentTransformer] = None,
        prediction_transform: Optional[ResponseTransformer] = None,
    ) -> None:
        super().__init__(
            name,
            weight=weight,
            maximise=maximise,
            variance=variance,
            composite=composite,
            noisy=len(prediction) > 1 if noisy is None else noisy,
            bounds=bounds,
        )
        self.name = name
        self.mean_dist = mean_dist
        self.prediction = prediction
        self.quantity = quantity
        self.latent_transformer = latent_transformer
        self.prediction_transform = prediction_transform

    def _extract_responses(self, raw_results: dict[str, OutputResult]) -> list[OutputResult]:
        """Extract responses of interest from the results and compute any required transformation.

        Parameters
        ----------
        raw_results : dict[str, OutputResult]
            Raw responses from the solver

        Returns
        -------
        list[OutputResult]
            list of transformed results.
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
        self, params: ParameterValues, raw_results: dict[str, OutputResult]
    ) -> IndividualObjectiveResult:
        """Evaluate objective value for the given results.

        Parameters
        ----------
        params : ParameterValues
            Named set of parameter values for this evaluation.
        raw_results : dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        IndividualObjectiveResult
            Result of the objective evaluation.
        """
        # Extract the responses of interest and compute the objective value and variance
        results = self._extract_responses(raw_results)
        obj_values = [
            self.quantity.reduce(result.time, result.data, params.vector_values)
            for result in results
        ]

        # Mean objective value
        value = np.mean(obj_values).item()

        # Variance when requested
        variance = None
        if self.has_variance():
            # Only compute the variance if we have more than one response
            numel = len(obj_values) if self.mean_dist else 1
            variance = np.var(obj_values, ddof=1) / numel if len(obj_values) > 1 else 0.0

        # Composition: evaluate the latent space representation
        latent_values, latent_covar = None, None
        if self.is_composite():
            # Latent space representation for all responses
            latent_space = np.array([
                self.latent_transformer.latent_space(result.time, result.data) for result in results
            ])

            # Mean of the latent space representation
            latent_values = np.mean(latent_space, axis=0)

            # Covariance of the latent space representation when requested
            if self.has_variance():
                latent_covar = (
                    np.cov(latent_space.T, ddof=1) / len(latent_space)
                    if latent_space.shape[0] > 1
                    else np.zeros((latent_space.shape[1], latent_space.shape[1]))
                )

        return IndividualObjectiveResult(
            value=value,
            variance=variance,
            latent_values=latent_values,
            latent_covariances=latent_covar,
        )

    def composition(self, latent: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Composition function for this objective, if supported.

        Parameters
        ----------
        latent : torch.Tensor
            Latent space values from the inner function.
        params : torch.Tensor
            Named parameters for the given result.

        Returns
        -------
        torch.Tensor
            Composition result.
        """
        if not self.is_composite():
            raise ValueError("Composition function is not supported for non-composite objectives.")

        # We need to reconstruct the time and data from the latent space representation, and then
        # compute the quantity reduction to get the objective value
        time, data = self.latent_transformer.inverse_transform(latent)
        expanded_params = {k: self._expand_params(time, v) for k, v in params.items()}
        return self.quantity.reduce_torch(time, data, expanded_params)

    def latent_size(self) -> int:
        """Return the size of the latent space for this objective.

        Returns
        -------
        int
            Size of the latent space.
        """
        # Under non-composite objectives, assume a size of 1 for the scalar value of the objective
        return self.latent_transformer.length() if self.is_composite() else 1

    @abstractmethod
    def plot_response(
        self, axis: plt.Axes, raw_results: dict[str, OutputResult]
    ) -> dict[Line2D, str]:
        """Plot the response for this objective.

        Parameters
        ----------
        axis : plt.Axes
            Axis to plot the response on.
        raw_results : dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        dict[Line2D, str]
            Mapping of lines to response names (for dynamically updating plots).
        """

    @classmethod
    @abstractmethod
    def read(cls: type[T], name: str, config: dict[str, Any], settings: Settings) -> T:
        """Read the objective spec from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the objective.
        config : dict[str, Any]
            Configuration dictionary.
        settings : Settings
            Global settings for the optimisation.

        Returns
        -------
        ResponseSingleObjective
            Single objective to use.
        """


class ResponseObjective(Objective):
    """Objective for generic response-based objectives."""

    def __init__(
        self,
        settings: Settings,
        solver: Solver,
        objectives: list[ResponseSingleObjective],
        scalarisation: Scalarisation = None,
        composite: bool = False,
        transformers: dict[str, ResponseTransformer] = None,
    ) -> None:
        super().__init__(
            settings,
            objectives=objectives,
            scalarisation=scalarisation,
            composite=composite,
        )
        self.solver = solver
        self.transformers = transformers if transformers is not None else {}
        # Update type hint for the objectives
        self.objectives: list[ResponseSingleObjective]
        # Sanitise predictions
        for objective in self.objectives:
            for name in objective.prediction:
                if name not in self.solver.get_output_fields():
                    raise ValueError(f'Undefined prediction {name}')

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def postproc_responses(self, responses: dict[str, OutputResult]) -> dict[str, OutputResult]:
        """Post-process the responses from the solver.

        Parameters
        ----------
        responses : dict[str, OutputResult]
            Raw responses from the solver.

        Returns
        -------
        dict[str, OutputResult]
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

    def _objective(
        self, params: ParameterValues, concurrent: bool = False
    ) -> list[IndividualObjectiveResult]:
        """Method for objective computation.

        Parameters
        ----------
        params : ParameterValues
            Named set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        list[IndividualObjectiveResult]
            List of individual objective results.
        """
        raw_responses = self.solver.solve(params, concurrent)

        # Sanitise and post-process the responses
        raw_responses = self.postproc_responses(raw_responses)

        # Compute the individual objective results
        return [objective.evaluate(params, raw_responses) for objective in self.objectives]

    def plot_case(self, case_hash: str, **kwargs) -> list[Figure]:
        """Plot a given function call given the parameter hash.

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting function

        Returns
        -------
        list[Figure]
            List of figures with the plot
        """
        append_title = ''
        if kwargs is not None and 'append_title' in kwargs:
            append_title = f' ({kwargs["append_title"]})'
        # Load all responses and post-process them
        responses = self.postproc_responses(self.solver.get_output_response(case_hash))
        # Extract the parameters
        params = self.solver.get_case_params(case_hash)
        if kwargs is not None and 'params' in kwargs:
            append_title += f' - {params}'
        # Plot each target
        figures = []
        for objective in self.objectives:
            fig, axis = plt.subplots()
            objective.plot_response(axis, responses)
            axis.set_title(objective.name + append_title)
            axis.grid()
            axis.legend()
            figures.append(fig)
        return figures

    # def plot_current(self) -> list[DynamicPlotter]:
    #     """Plot the currently running function call

    #     Returns
    #     -------
    #     list[DynamicPlotter]
    #         list of instances of a updatable plots
    #     """
    #     # Get current solver data
    #     responses = self.postproc_responses(self.solver.get_current_response())
    #     # Plot each objective
    #     figures: list[Figure] = []
    #     mapping: dict[Line2D, str] = {}
    #     for objective in self.objectives:
    #         fig, axis = plt.subplots()
    #         line, = objective.plot_response(axis, responses)
    #         axis.set_title(objective.name)
    #         axis.legend()
    #         # Store the line and figure
    #         mapping[line] = objective.name
    #         figures.append(fig)
    #     # Show the plot
    #     plt.show()
    #     for fig in figures:
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()
    #     return [DynamicResponsePlotter(figures, self.solver, mapping, self.transformers)]
