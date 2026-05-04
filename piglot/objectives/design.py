"""Module for curve fitting objectives"""
from typing import Any, TypeVar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from piglot.settings import Settings
from piglot.solver import read_solver
from piglot.solver.solver import OutputResult
from piglot.utils.assorted import read_custom_module
from piglot.utils.reductions import read_reduction
from piglot.utils.scalarisations import read_scalarisation
from piglot.utils.composition import EndpointLatentTransformer
from piglot.utils.response_transformer import ResponseTransformer, read_response_transformer
from piglot.objectives.response_objective import ResponseSingleObjective, ResponseObjective


SingleObjT = TypeVar('SingleObjT', bound='DesignSingleObjective')
ObjectiveT = TypeVar('ObjectiveT', bound='ResponseDesignObjective')


class DesignSingleObjective(ResponseSingleObjective):
    """Single objective for design optimisation objectives."""

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
            Global settings for this problem.

        Returns
        -------
        SingleObjT
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
        # Read composition and set up latent transformer
        composite = bool(config.get('composite', False))
        latent_transformer = None
        if composite:
            if 'n_points' not in config:
                raise ValueError(
                    f"Missing number of points for design target '{name}' under composition."
                )
            latent_transformer = EndpointLatentTransformer(int(config['n_points']))
        return cls(
            name,
            prediction,
            read_reduction(config['quantity']),
            mean_dist=bool(config.get('mean_dist', False)),
            weight=float(config.get('weight', 1.0)),
            maximise=bool(config.get('maximise', False)),
            variance=bool(config.get('variance', False)),
            composite=bool(config.get('composite', False)),
            noisy=config.get('noisy', None),
            bounds=config.get('bounds', None),
            latent_transformer=latent_transformer,
            prediction_transform=(
                read_response_transformer(config['transformers'])
                if 'transformers' in config else None
            ),
        )


class ResponseDesignObjective(ResponseObjective):
    """Class for design of response-based objectives."""

    @classmethod
    def read(cls: type[ObjectiveT], config: dict[str, Any], settings: Settings) -> ObjectiveT:
        """Read the objective from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Terms from the configuration dictionary.
        settings : Settings
            Global settings for this problem.

        Returns
        -------
        ResponseDesignObjective
            Objective function to optimise.
        """
        # Read the solver
        if 'solver' not in config:
            raise ValueError("Missing solver for design objective.")
        solver = read_solver(config.pop('solver'), settings.parameters, settings.output_dir)
        # Read transformers
        transformers: dict[str, ResponseTransformer] = {}
        if 'transformers' in config:
            for name, transformer_config in config.pop('transformers').items():
                transformers[name] = read_response_transformer(transformer_config)
        # Read custom class (if any)
        target_class = cls
        if 'custom_class' in config:
            target_class = read_custom_module(config.pop('custom_class'), cls)
        # Read the targets
        if 'targets' not in config:
            raise ValueError("Missing targets for design objective.")
        objectives = [
            DesignSingleObjective.read(target_name, target_config | config, settings)
            for target_name, target_config in config.pop('targets').items()
        ]
        # Sanitise the objectives under composition
        composite = bool(config.pop('composite', False))
        if composite:
            for objective in objectives:
                if objective.latent_transformer is None:
                    raise ValueError(
                        "All objectives must have a number of points specified for the composition."
                    )
        return target_class(
            settings,
            solver,
            objectives,
            scalarisation=(
                read_scalarisation(config['scalarisation'], objectives)
                if 'scalarisation' in config else None
            ),
            composite=composite,
            transformers=transformers,
        )
