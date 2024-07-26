"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import Solver, OutputResult
from piglot.objective import Composition, DynamicPlotter, GenericObjective, ObjectiveResult
from piglot.utils.assorted import stats_interp_to_common_grid, read_custom_module
from piglot.utils.reductions import Reduction, NegateReduction, read_reduction
from piglot.utils.response_transformer import ResponseTransformer, read_response_transformer
from piglot.utils.composition.responses import ResponseComposition, EndpointFlattenUtility


class DesignTarget:
    """Container for a design objective target."""

    def __init__(
            self,
            name: str,
            prediction: Union[str, List[str]],
            quantity: Reduction,
            negate: bool = False,
            weight: float = 1.0,
            n_points: int = None,
            bounds: List[float] = None,
            ) -> None:
        # Sanitise prediction field
        if isinstance(prediction, str):
            prediction = [prediction]
        elif not isinstance(prediction, list):
            raise ValueError(f"Invalid prediction '{prediction}' for design target '{name}'.")
        self.name = name
        self.prediction = prediction
        self.quantity = NegateReduction(quantity) if negate else quantity
        self.weight = weight
        self.n_points = n_points
        self.flatten_utility = EndpointFlattenUtility(n_points) if n_points is not None else None
        self.bounds = bounds
        self.negate = negate

    @staticmethod
    def read(name: str, config: Dict[str, Any], output_dir: str) -> DesignTarget:
        """Read the design target from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the design target.
        config : Dict[str, Any]
            Configuration dictionary.
        output_dir: str
            Output directory.

        Returns
        -------
        DesignTarget
            Design target to use for this problem.
        """
        # Prediction parsing
        if 'prediction' not in config:
            raise ValueError(f"Missing prediction for design target '{name}'.")
        # Read the quantity
        if 'quantity' not in config:
            raise ValueError("Missing quantity for fitting objective.")
        return DesignTarget(
            name,
            config['prediction'],
            read_reduction(config['quantity']),
            negate=bool(config.get('negate', False)),
            weight=float(config.get('weight', 1.0)),
            n_points=int(config['n_points']) if 'n_points' in config else None,
            bounds=config.get('bounds', None),
        )


class CurrentPlot(DynamicPlotter):
    """Container for dynamically-updating plots."""

    def __init__(self, solver: Solver) -> None:
        """Constructor for dynamically-updating plots.

        Parameters
        ----------
        solver : Solver
            Solver to build the plot for.
        """
        self.solver = solver
        self.figs = {}
        self.preds = {}
        # Get current solver data
        result = solver.get_current_response()
        # Plot each response
        for name, response in result.items():
            fig, axis = plt.subplots()
            self.preds[name], = axis.plot(response.get_time(), response.get_data(), c='red')
            axis.set_title(name)
            axis.grid()
            self.figs[name] = (fig, axis)
        plt.show()
        for fig, _ in self.figs.values():
            fig.canvas.draw()
            fig.canvas.flush_events()

    def update(self) -> None:
        """Update the plot with the most recent data"""
        try:
            result = self.solver.get_current_response()
        except (FileNotFoundError, IndexError):
            pass
        for name, response in result.items():
            self.preds[name].set_xdata(response.get_time())
            self.preds[name].set_ydata(response.get_data())
            fig, axis = self.figs[name]
            axis.relim()
            axis.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()


class DesignObjective(GenericObjective):
    """Scalar design objective function."""

    def __init__(
        self,
        parameters: ParameterSet,
        solver: Solver,
        targets: List[DesignTarget],
        output_dir: str,
        stochastic: bool = False,
        composite: bool = False,
        multi_objective: bool = False,
        transformers: Dict[str, ResponseTransformer] = None,
        scalarisation: str = 'mean',
    ) -> None:
        super().__init__(
            parameters,
            stochastic,
            composition=(
                self.__composition(
                    not multi_objective,
                    stochastic,
                    targets,
                    scalarisation
                ) if composite else None
            ),
            num_objectives=len(targets) if multi_objective else 1,
            output_dir=output_dir,
        )
        self.solver = solver
        self.targets = targets
        self.transformers = transformers if transformers is not None else {}
        self.scalarisation = scalarisation

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    @staticmethod
    def __composition(
        scalarise: bool,
        stochastic: bool,
        targets: List[DesignTarget],
        scalarisation: str,
    ) -> Composition:
        """Build the composition utility for the design objective.

        Parameters
        ----------
        scalarise : bool
            Whether we should scalarise the objective.
        stochastic : bool
            Whether the objective is stochastic.
        targets : List[DesignTarget]
            List of design targets to consider.

        Returns
        -------
        Composition
            Composition utility for the design objective.
        """
        # Sanitise the number of points for each design target
        for target in targets:
            if target.n_points is None:
                raise ValueError(
                    "All targets must have a number of points specified for the composition."
                )
        # Sanitise scalarisation method
        if scalarisation not in ['mean', 'stch']:
            raise ValueError(f"Invalid scalarisation '{scalarisation}'. Use 'mean' or 'stch'.")

        return ResponseComposition(
            scalarise=scalarise,
            stochastic=stochastic,
            weights=[t.weight for t in targets],
            reductions=[t.quantity for t in targets],
            flatten_list=[t.flatten_utility for t in targets],
            scalarisation=scalarisation,
            bounds=[t.bounds for t in targets] if scalarisation == 'stch' else None,
            types=[t.negate for t in targets] if scalarisation == 'stch' else None,
        )

    @staticmethod
    def __interpolate_to_common_grid(
        response: OutputResult,
        n_points: int,
    ) -> OutputResult:
        """Interpolate the response to a common grid.

        Parameters
        ----------
        response : OutputResult
            Response to interpolate.
        n_points : int
            Number of points to interpolate to.

        Returns
        -------
        OutputResult
            Interpolated response.
        """
        # Interpolate to common grid
        time = np.linspace(np.min(response.get_time()), np.max(response.get_time()), n_points)
        data = np.interp(time, response.get_time(), response.get_data())
        return OutputResult(time, data)

    def _objective(self, values: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for design objectives.

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
        # Sanitise scalarisation method
        if self.scalarisation not in ['mean', 'stch']:
            raise ValueError(f"Invalid scalarisation '{self.scalarisation}'. Use 'mean' or 'stch'.")

        raw_responses = self.solver.solve(values, concurrent)
        # Transform responses
        for name, transformer in self.transformers.items():
            if name in raw_responses:
                raw_responses[name] = transformer.transform(raw_responses[name])
        # Interpolate responses to common grid and map to targets
        responses_interp = {
            target: [
                self.__interpolate_to_common_grid(raw_responses[pred], target.n_points)
                if target.n_points is not None else raw_responses[pred]
                for pred in target.prediction
            ]
            for target in self.targets
        }
        # Under composition, we delegate the computation to the composition utility
        if self.composition is not None:
            return self.composition.transform(values, list(responses_interp.values()))
        # Otherwise, compute the objective directly
        results = []
        variances = []
        for target, responses in responses_interp.items():
            # Evaluate target quantities for each response
            targets = [
                target.quantity.reduce(response.get_time(), response.get_data(), values)
                for response in responses
            ]
            # Build statistical model for the target
            results.append(target.weight * np.mean(targets))
            variances.append(target.weight * np.var(targets) / len(targets))

        return ObjectiveResult(
            values,
            results,
            self.scalarisation,
            variances if self.stochastic else None,
            [t.weight for t in self.targets] if self.scalarisation == 'stch' else None,
            [t.bounds for t in self.targets] if self.scalarisation == 'stch' else None,
            [t.negate for t in self.targets] if self.scalarisation == 'stch' else None,
        )

    def plot_case(self, case_hash: str, options: Dict[str, Any] = None) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        options : Dict[str, Any], optional
            Options for the plot, by default None

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        append_title = ''
        if options is not None and 'append_title' in options:
            append_title = f' ({options["append_title"]})'
        figures = []
        # Load all responses
        responses = self.solver.get_output_response(case_hash)
        # Transform responses if necessary
        for name, transformer in self.transformers.items():
            if name in responses:
                responses[name] = transformer.transform(responses[name])
        # Plot each target
        for target in self.targets:
            # Build figure with individual responses for this target
            fig, axis = plt.subplots()
            for pred in target.prediction:
                marker = 'o' if len(responses[pred].get_time()) < 2 else None
                axis.plot(
                    responses[pred].get_time(),
                    responses[pred].get_data(),
                    label=f'{pred}',
                    marker=marker,
                )
            # Under stochasticity, plot response stats and mean confidence interval
            if self.stochastic:
                stats = stats_interp_to_common_grid([
                    (responses[pred].get_time(), responses[pred].get_data())
                    for pred in target.prediction
                ])
                # Plot average and standard deviation
                axis.plot(stats['grid'], stats['average'], label='Average', c='black', ls='dashed')
                axis.fill_between(
                    stats['grid'],
                    stats['average'] - stats['std'],
                    stats['average'] + stats['std'],
                    label='Standard deviation',
                    color='black',
                    alpha=0.2,
                    ls='dashed',
                )
                axis.fill_between(
                    stats['grid'],
                    stats['average'] - stats['confidence'],
                    stats['average'] + stats['confidence'],
                    label='Confidence interval',
                    color='red',
                    alpha=0.2,
                )
            axis.set_title(target.name + append_title)
            axis.grid()
            axis.legend()
            figures.append(fig)
        return figures

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call.

        Returns
        -------
        List[DynamicPlotter]
            List of instances of updatable plots.
        """
        return [CurrentPlot(self.solver)]

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
            ) -> DesignObjective:
        """Read a design objective from a configuration dictionary.

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
        DesignObjective
            Objective function to optimise.
        """
        # Read the solver
        if 'solver' not in config:
            raise ValueError("Missing solver for fitting objective.")
        solver = read_solver(config['solver'], parameters, output_dir)
        # Read the targets
        if 'targets' not in config:
            raise ValueError("Missing targets for fitting objective.")
        targets: Dict[DesignTarget, List[str]] = {}
        for target_name, target_config in config.pop('targets').items():
            target = DesignTarget.read(target_name, target_config, output_dir)
            # Map the target to the solver cases
            targets[target] = []
            for field_name in solver.get_output_fields():
                if field_name in target.prediction:
                    targets[target].append(field_name)
            # Sanitise target: check if it is associated to at least one case
            if len(targets[target]) == 0:
                raise ValueError(f"Design target '{target_name}' is not associated to any case.")
        # Read transformers
        transformers: Dict[str, ResponseTransformer] = {}
        if 'transformers' in config:
            for name, transformer_config in config.pop('transformers').items():
                transformers[name] = read_response_transformer(transformer_config)
        # Read custom class (if any)
        target_class: type[DesignObjective] = DesignObjective
        if 'custom_class' in config:
            target_class = read_custom_module(config['custom_class'], DesignObjective)
        return target_class(
            parameters,
            solver,
            list(targets.keys()),
            output_dir,
            stochastic=bool(config.get('stochastic', False)),
            composite=bool(config.get('composite', False)),
            multi_objective=bool(config.get('multi_objective', False)),
            transformers=transformers,
            scalarisation=str(config.get('scalarisation', 'mean')),
        )
