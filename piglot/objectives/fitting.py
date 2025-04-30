"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import OutputResult
from piglot.utils.interpolators import Interpolator, read_interpolator
from piglot.utils.reductions import Reduction, read_reduction
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
        time_data: np.ndarray,
        data: np.ndarray,
    ) -> None:
        self.filename = filename
        self.x_data = time_data
        self.y_data = data

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
        x_col = int(config.pop('x_col', 0))
        y_col = int(config.pop('y_col', 1))
        transformer = config.pop('transformer', None)
        # Check if the processed data already exists
        output_filename = os.path.join(output_dir, 'references', filename)
        if os.path.exists(output_filename):
            data = np.genfromtxt(output_filename)
            return Reference(filename, data[:, 0], data[:, -1])
        # Read the data
        data = np.genfromtxt(filename, **config)
        points = data[:, x_col]
        values = data[:, y_col]
        # Transform the data if necessary
        if transformer is not None:
            points, values = read_response_transformer(transformer)(points, values)
        # Store the reference
        os.makedirs(os.path.join(output_dir, 'references'), exist_ok=True)
        np.savetxt(output_filename, np.stack((points, values), axis=1))
        return Reference(filename, points, values)


class FittingSingleObjective(ResponseSingleObjective):
    """Single objective for curve fitting optimisation objectives."""

    def __init__(
        self,
        name: str,
        reference: Reference,
        prediction: List[str],
        reduction: Reduction,
        interpolator: Interpolator,
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
            prediction_transform=PointwiseErrors(
                reference.get_time(),
                reference.get_data(),
                interpolator,
            ),
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
        interp_config = config.pop('interpolator', 'linear')
        # Read the reference and return the objective
        reference = Reference.read(name, config, output_dir)
        return FittingSingleObjective(
            name,
            reference,
            prediction,
            reduction,
            read_interpolator(interp_config),
            weight=weight,
            bounds=bounds,
        )


class ResponseFittingObjective(ResponseObjective):
    """Class for fitting of response-based objectives."""

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
