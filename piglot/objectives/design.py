"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import OutputResult
from piglot.utils.assorted import read_custom_module
from piglot.utils.reductions import read_reduction
from piglot.utils.scalarisations import read_scalarisation
from piglot.utils.composition.responses import EndpointFlattenUtility
from piglot.utils.response_transformer import ResponseTransformer, read_response_transformer
from piglot.objectives.response_objective import ResponseSingleObjective, ResponseObjective


class DesignSingleObjective(ResponseSingleObjective):
    """Single objective for design optimisation objectives."""

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
        # Plot the response
        lines: Dict[Line2D, str] = {}
        for prediction in self.prediction:
            response = raw_results[prediction]
            line, = axis.plot(response.get_time(), response.get_data(), label=prediction)
            lines[line] = prediction
        return lines

    @classmethod
    def read(cls, name: str, config: Dict[str, Any], output_dir: str) -> DesignSingleObjective:
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
            raise ValueError(f"Missing prediction for design target '{name}'.")
        # Sanitise prediction field
        prediction = config['prediction']
        if isinstance(prediction, str):
            prediction = [prediction]
        elif not isinstance(prediction, list):
            raise ValueError(f"Invalid prediction '{prediction}' for design target '{name}'.")
        # Read the quantity
        if 'quantity' not in config:
            raise ValueError(f"Missing quantity for design target '{name}'.")
        return DesignSingleObjective(
            name,
            prediction,
            read_reduction(config['quantity']),
            maximise=bool(config.get('maximise', False)),
            weight=float(config.get('weight', 1.0)),
            bounds=config.get('bounds', None),
            flatten_utility=(
                EndpointFlattenUtility(int(config['n_points'])) if 'n_points' in config else None
            ),
            prediction_transform=(
                read_response_transformer(config['transformers'])
                if 'transformers' in config else None
            ),
        )


class ResponseDesignObjective(ResponseObjective):
    """Class for design of response-based objectives."""

    @classmethod
    def read(
        cls,
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> ResponseDesignObjective:
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
        ResponseDesignObjective
            Objective function to optimise.
        """
        # Read the solver
        if 'solver' not in config:
            raise ValueError("Missing solver for design objective.")
        solver = read_solver(config['solver'], parameters, output_dir)
        # Read the targets
        if 'targets' not in config:
            raise ValueError("Missing targets for design objective.")
        objectives = [
            DesignSingleObjective.read(target_name, target_config, output_dir)
            for target_name, target_config in config.pop('targets').items()
        ]
        # Sanitise the objectives under composition
        composite = bool(config.get('composite', False))
        if composite:
            for objective in objectives:
                if objective.flatten_utility is None:
                    raise ValueError(
                        "All objectives must have a number of points specified for the composition."
                    )
        # Read transformers
        transformers: Dict[str, ResponseTransformer] = {}
        if 'transformers' in config:
            for name, transformer_config in config.pop('transformers').items():
                transformers[name] = read_response_transformer(transformer_config)
        # Read custom class (if any)
        target_class: type[ResponseDesignObjective] = ResponseDesignObjective
        if 'custom_class' in config:
            target_class = read_custom_module(config['custom_class'], ResponseDesignObjective)
        return target_class(
            parameters,
            solver,
            objectives,
            output_dir,
            scalarisation=(
                read_scalarisation(config['scalarisation'], objectives)
                if 'scalarisation' in config else None
            ),
            stochastic=bool(config.get('stochastic', False)),
            composite=composite,
            full_composite=bool(config.get('full_composite', True)),
            transformers=transformers,
        )
