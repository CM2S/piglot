"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from piglot.parameter import ParameterSet
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
from piglot.utils.composition.responses import FixedFlatteningUtility
from piglot.objectives.response_objective import ResponseSingleObjective, ResponseObjective


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
        return Reference(
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
        prediction: List[str],
        reduction: Reduction,
        weight: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            name,
            prediction,
            reduction,
            weight=weight,
            bounds=bounds,
            flatten_utility=FixedFlatteningUtility(reference.get_time()),
            prediction_transform=PointwiseErrors(reference.get_time(), reference.get_data()),
        )
        self.reference = reference

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
        lines: Dict[Line2D, str] = {}
        for prediction in self.prediction:
            response = raw_results[prediction]
            line, = axis.plot(response.get_time(), response.get_data(), label=prediction)
            lines[line] = prediction
        return lines

    @classmethod
    def read(cls, name: str, config: Dict[str, Any], output_dir: str) -> FittingSingleObjective:
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
        # Read the reference and return the objective
        reference = Reference.read(name, config, output_dir)
        return FittingSingleObjective(
            name,
            reference,
            prediction,
            reduction,
            weight=weight,
            bounds=bounds,
        )


class ResponseFittingObjective(ResponseObjective):
    """Class for fitting of response-based objectives."""

    def prepare(self):
        """Prepare the objective for optimisation.

        For curve fitting, this involves preparing the reference data and updating both the
        flatten utility and the transformer.
        """
        super().prepare()
        objectives: List[FittingSingleObjective] = self.objectives
        for objective in objectives:
            objective.reference.prepare()
            # Update the flattening utility and the prediction transformer
            objective.flatten_utility = FixedFlatteningUtility(objective.reference.get_time())
            objective.prediction_transform = PointwiseErrors(
                objective.reference.get_time(),
                objective.reference.get_data(),
            )

    @classmethod
    def read(
        cls,
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> ResponseFittingObjective:
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
        ResponseFittingObjective
            Objective function to optimise.
        """
        # Read the solver
        if 'solver' not in config:
            raise ValueError("Missing solver for fitting objective.")
        solver = read_solver(config['solver'], parameters, output_dir)
        # Read the references
        if 'references' not in config:
            raise ValueError("Missing references for fitting objective.")
        objectives = [
            FittingSingleObjective.read(target_name, target_config, output_dir)
            for target_name, target_config in config.pop('references').items()
        ]
        # Read transformers
        transformers: Dict[str, ResponseTransformer] = {}
        if 'transformers' in config:
            for name, transformer_config in config.pop('transformers').items():
                transformers[name] = read_response_transformer(transformer_config)
        return ResponseFittingObjective(
            parameters,
            solver,
            objectives,
            output_dir,
            scalarisation=(
                read_scalarisation(config['scalarisation'], objectives)
                if 'scalarisation' in config else None
            ),
            stochastic=bool(config.get('stochastic', False)),
            composite=bool(config.get('composite', False)),
            full_composite=bool(config.get('full_composite', True)),
            transformers=transformers,
        )
