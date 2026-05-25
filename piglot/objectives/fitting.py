"""Module for curve fitting objectives"""
from typing import Any, Optional, TypeVar
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from piglot.settings import Settings
from piglot.solver import read_solver
from piglot.solver.solver import OutputResult
from piglot.utils.reductions import Reduction, read_reduction
from piglot.utils.responses import reduce_response
from piglot.utils.response_transformer import (
    ResponseTransformer,
    PointwiseErrors,
    read_response_transformer,
)
from piglot.utils.scalarisations import read_scalarisation
from piglot.utils.composition import FixedTimeLatentTransformer
from piglot.objectives.response_objective import ResponseSingleObjective, ResponseObjective


ReferenceT = TypeVar('ReferenceT', bound='Reference')
SingleObjT = TypeVar('SingleObjT', bound='FittingSingleObjective')
ObjectiveT = TypeVar('ObjectiveT', bound='ResponseFittingObjective')


class Reference:
    """Container for reference solutions."""

    def __init__(
        self,
        filename: str,
        output_dir: str,
        x_col: int = 1,
        y_col: int = 2,
        skip_header: int = 0,
        transformer: ResponseTransformer = None,
        filter_tol: float = 0.0,
        show: bool = False,
    ) -> None:
        self.filename = filename
        self.output_dir = output_dir
        self.transformer = transformer
        self.filter_tol = filter_tol
        self.show = show
        # Load the data right away
        data = np.genfromtxt(filename, skip_header=skip_header)
        # Sanitise to ensure it is a 2D array
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        self.x_data = data[:, x_col - 1]
        self.y_data = data[:, y_col - 1]
        # Check if all data is valid
        if np.any(np.isnan(self.x_data)) or np.any(np.isnan(self.y_data)):
            raise ValueError(f"Reference data in {filename} contains NaN values.")
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

    @classmethod
    def read(
        cls: type[ReferenceT], filename: str, config: dict[str, Any], output_dir: str
    ) -> ReferenceT:
        """Read the reference from the configuration dictionary.

        Parameters
        ----------
        filename : str
            Path to the reference file.
        config : dict[str, Any]
            Configuration dictionary.
        output_dir: str
            Output directory.

        Returns
        -------
        ReferenceT
            Reference to use for this problem.
        """
        return cls(
            filename,
            output_dir,
            x_col=int(config.get('x_col', 1)),
            y_col=int(config.get('y_col', 2)),
            skip_header=int(config.get('skip_header', 0)),
            transformer=(
                read_response_transformer(config['transformer'])
                if 'transformer' in config else None
            ),
            filter_tol=float(config.get('filter_tol', 0.0)),
            show=bool(config.get('show', False)),
        )


class FittingSingleObjective(ResponseSingleObjective):
    """Single objective for curve fitting optimisation objectives."""

    def __init__(
        self,
        name: str,
        reference: Reference,
        prediction: list[str],
        reduction: Reduction,
        mean_dist: bool = False,
        weight: float = 1.0,
        variance: bool = False,
        composite: bool = False,
        noisy: Optional[bool] = None,
        bounds: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            name,
            prediction,
            reduction,
            weight=weight,
            maximise=False,
            variance=variance,
            composite=composite,
            bounds=bounds,
            noisy=noisy,
            mean_dist=mean_dist,
            latent_transformer=FixedTimeLatentTransformer(reference.get_time()),
            prediction_transform=PointwiseErrors(reference.get_time(), reference.get_data()),
        )
        self.reference = reference

    def plot_raw_responses(self, axis: plt.Axes, responses: list[OutputResult]) -> None:
        """Plot the raw responses for this objective.

        Parameters
        ----------
        axis : plt.Axes
            Axis to plot the raw responses on.
        responses : list[OutputResult]
            Raw responses from the solver.
        """
        # Plot the reference
        axis.plot(
            self.reference.get_time(),
            self.reference.get_data(),
            label='Reference',
            ls='dashed',
            marker='x',
            c='k',
        )
        super().plot_raw_responses(axis, responses)

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
        # Plot the reference
        axis.plot(
            self.reference.get_time(),
            self.reference.get_data(),
            label='Reference',
            ls='dashed',
            marker='x',
            c='k',
        )
        # Plot the response
        lines: dict[Line2D, str] = {}
        for prediction in self.prediction:
            response = raw_results[prediction]
            line, = axis.plot(response.get_time(), response.get_data(), label=prediction)
            lines[line] = prediction
        return lines

    @classmethod
    def read(
        cls: type[SingleObjT], name: str, config: dict[str, Any], settings: Settings
    ) -> SingleObjT:
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
        SingleObjT
            Single objective to use.
        """
        # Prediction parsing
        if 'prediction' not in config:
            raise ValueError(f"Missing prediction for fitting target '{name}'.")
        # Sanitise prediction field
        prediction = config.pop('prediction')
        if isinstance(prediction, str):
            prediction = [prediction]
        elif not isinstance(prediction, list):
            raise ValueError(f"Invalid prediction '{prediction}' for reference '{name}'.")
        # Read optional settings
        reduction = read_reduction(config.pop('reduction', 'mse'))
        weight = float(config.pop('weight', 1.0))
        bounds = config.pop('bounds', None)
        variance = bool(config.pop('variance', False))
        mean_dist = bool(config.pop('mean_dist', False))
        composite = bool(config.get('composite', False))
        noisy = bool(config.pop('noisy', None))
        # Read the reference and return the objective
        reference = Reference.read(name, config, settings.output_dir)
        return cls(
            name,
            reference,
            prediction,
            reduction,
            mean_dist=mean_dist,
            variance=variance,
            weight=weight,
            composite=composite,
            bounds=bounds,
            noisy=noisy,
        )


class ResponseFittingObjective(ResponseObjective):
    """Class for fitting of response-based objectives."""

    def prepare(self):
        """Prepare the objective for optimisation.

        For curve fitting, this involves preparing the reference data and updating both the
        flatten utility and the transformer.
        """
        super().prepare()
        objectives: list[FittingSingleObjective] = self.objectives
        for objective in objectives:
            objective.reference.prepare()
            # Update the latent transformer and the prediction transformer
            objective.latent_transformer = FixedTimeLatentTransformer(
                objective.reference.get_time()
            )
            objective.prediction_transform = PointwiseErrors(
                objective.reference.get_time(),
                objective.reference.get_data(),
            )

    @classmethod
    def read(
        cls: type[ObjectiveT],
        config: dict[str, Any],
        settings: Settings,
    ) -> ObjectiveT:
        """Read the objective from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Terms from the configuration dictionary.
        settings : Settings
            Global settings for the optimisation.

        Returns
        -------
        ObjectiveT
            Objective function to optimise.
        """
        # Read the solver
        if 'solver' not in config:
            raise ValueError("Missing solver for fitting objective.")
        solver = read_solver(config.pop('solver'), settings.parameters, settings.output_dir)
        # Read transformers
        transformers: dict[str, ResponseTransformer] = {}
        if 'transformers' in config:
            for name, transformer_config in config.pop('transformers').items():
                transformers[name] = read_response_transformer(transformer_config)
        # Pop the scalarisation flag
        scalarisation_conf = config.pop('scalarisation', None)
        # Pop the references
        if 'references' not in config:
            raise ValueError("Missing references for fitting objective.")
        references_config = config.pop('references')
        # Read the references and inject any additional configuration into the objectives
        objectives = [
            FittingSingleObjective.read(target_name, target_config | config, settings)
            for target_name, target_config in references_config.items()
        ]
        return cls(
            settings,
            solver,
            objectives,
            scalarisation=(
                read_scalarisation(scalarisation_conf, objectives)
                if scalarisation_conf is not None else None
            ),
            composite=bool(config.get('composite', False)),
            transformers=transformers,
        )
