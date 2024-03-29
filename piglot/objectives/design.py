"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Union, Type
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import Solver, OutputResult
from piglot.objective import Composition, DynamicPlotter, GenericObjective, ObjectiveResult
from piglot.objectives.compositions import UnflattenUtility, FixedLengthTransformer
from piglot.utils.assorted import stats_interp_to_common_grid
from piglot.utils.solver_utils import load_module_from_file


class Quantity(ABC):
    """Abstract class for quantities to be computed from a response."""

    def compute(self, time: np.ndarray, data: np.ndarray) -> float:
        """Compute this quantity for a given response.

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.

        Returns
        -------
        float
            Quantity value.
        """
        return self.compute_torch(torch.from_numpy(time), torch.from_numpy(data)).item()

    @abstractmethod
    def compute_torch(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Compute this quantity for a given response (with gradients).

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.

        Returns
        -------
        torch.Tensor
            Quantity value (with gradients).
        """


class MaxQuantity(Quantity):
    """Maximum value of a response."""

    def compute_torch(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Get the maximum of a given response (with gradients).

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.

        Returns
        -------
        torch.Tensor
            Quantity value (with gradients).
        """
        return torch.max(data, dim=-1)[0]


class MinQuantity(Quantity):
    """Minimum value of a response."""

    def compute_torch(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Get the minimum of a given response (with gradients).

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.

        Returns
        -------
        torch.Tensor
            Quantity value (with gradients).
        """
        return torch.min(data, dim=-1)[0]


class IntegralQuantity(Quantity):
    """Integral of a response."""

    def compute_torch(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Get the integral of a given response (with gradients).

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.

        Returns
        -------
        torch.Tensor
            Quantity value (with gradients).
        """
        return torch.trapz(data, time, dim=-1)


AVAILABLE_QUANTITIES: Dict[str, Type[Quantity]] = {
    'max': MaxQuantity,
    'min': MinQuantity,
    'integral': IntegralQuantity,
}


class DesignTarget:
    """Container for a design objective target."""

    def __init__(
            self,
            name: str,
            prediction: Union[str, List[str]],
            quantity: Quantity,
            negate: bool = False,
            weight: float = 1.0,
            n_points: int = None,
            ) -> None:
        # Sanitise prediction field
        if isinstance(prediction, str):
            prediction = [prediction]
        elif not isinstance(prediction, list):
            raise ValueError(f"Invalid prediction '{prediction}' for design target '{name}'.")
        self.name = name
        self.prediction = prediction
        self.quantity = quantity
        self.negate = negate
        self.weight = weight
        self.n_points = n_points
        self.transformer = FixedLengthTransformer(n_points)  # if n_points is not None else None

    def compute_quantity(self, time: np.ndarray, data: np.ndarray) -> float:
        """Compute the design quantity for the given results.

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.

        Returns
        -------
        float
            Design quantity for this response.
        """
        value = self.quantity.compute(time, data)
        return -value if self.negate else value

    def compute_quantity_torch(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Compute the design quantity for the given results.

        Parameters
        ----------
        time : torch.Tensor
            Time points of the response.
        data : torch.Tensor
            Data points of the response.

        Returns
        -------
        torch.Tensor
            Design quantity for this response.
        """
        value = self.quantity.compute_torch(time, data)
        return -value if self.negate else value

    def flatten(self, response: OutputResult) -> np.ndarray:
        """Flatten the response for this target.

        Parameters
        ----------
        response : OutputResult
            Response to flatten.

        Returns
        -------
        np.ndarray
            Flattened response.
        """
        return self.transformer.transform(response.get_time(), response.get_data())

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
        quantity = config['quantity']
        # Parse specification: simple or complete
        if isinstance(quantity, str):
            quantity = {'name': quantity}
        elif 'name' not in quantity:
            raise ValueError("Missing name in quantity specification.")
        # Parse script arguments, if passed
        if quantity['name'] == 'script':
            if 'script' not in quantity:
                raise ValueError("Missing script in quantity specification.")
            if 'class' not in quantity:
                raise ValueError("Missing class in quantity specification.")
            quantity_class: Type[Quantity] = load_module_from_file(quantity['script'],
                                                                   quantity['class'])
        else:
            quantity_class: Type[Quantity] = AVAILABLE_QUANTITIES[quantity['name']]
        return DesignTarget(
            name,
            config['prediction'],
            quantity_class(),
            negate=bool(config.get('negate', False)),
            weight=float(config.get('weight', 1.0)),
            n_points=int(config['n_points']) if 'n_points' in config else None,
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


class DesignComposition(Composition):
    """Optimisation composition function for design objectives."""

    def __init__(self, outmost: str, targets: List[DesignTarget]) -> None:
        # Sanitise the number of points for each design target
        for target in targets:
            if target.n_points is None:
                raise ValueError(
                    "All targets must have a number of points specified for the composition."
                )
        self.targets = targets
        self.unflatten_utility = UnflattenUtility([t.transformer.length() for t in self.targets])

    def composition(self, inner: np.ndarray) -> float:
        """Compute the outer function of the composition.

        Parameters
        ----------
        inner : np.ndarray
            Return value from the inner function.

        Returns
        -------
        float
            Scalar composition result.
        """
        unflatten = self.unflatten_utility.unflatten(inner)
        responses = [
            target.transformer.untransform(flattened)
            for target, flattened in zip(self.targets, unflatten)
        ]
        quantities = [
            target.compute_quantity(*response) * target.weight
            for response, target in zip(responses, self.targets)
        ]
        return np.mean(quantities, axis=0)

    def composition_torch(self, inner: torch.Tensor) -> torch.Tensor:
        """Compute the outer function of the composition with gradients.

        Parameters
        ----------
        inner : torch.Tensor
            Return value from the inner function.

        Returns
        -------
        torch.Tensor
            Scalar composition result.
        """
        unflatten = self.unflatten_utility.unflatten_torch(inner)
        responses = [
            target.transformer.untransform_torch(flattened)
            for target, flattened in zip(self.targets, unflatten)
        ]
        quantities = torch.stack([
            target.compute_quantity_torch(*response) * target.weight
            for response, target in zip(responses, self.targets)
        ], dim=-1)
        return torch.mean(quantities, dim=-1)


class DesignObjective(GenericObjective):
    """Scalar design objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: Solver,
            targets: List[DesignTarget],
            output_dir: str,
            stochastic: bool = False,
            composition: str = None,
            ) -> None:
        super().__init__(
            parameters,
            stochastic,
            composition=DesignComposition(composition, targets) if composition else None,
            output_dir=output_dir,
        )
        self.solver = solver
        self.targets = targets

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

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
        responses = self.solver.solve(values, concurrent)
        results = []
        variances = []
        if self.composition is None:
            for target in self.targets:
                targets = [
                    target.compute_quantity(responses[pred].get_time(), responses[pred].get_data())
                    for pred in target.prediction
                ]
                results.append(target.weight * np.mean(targets))
                variances.append(target.weight * np.var(targets) / len(targets))
        else:
            for target in self.targets:
                flat_responses = np.array([
                    target.flatten(responses[pred])
                    for pred in target.prediction
                ])
                results.append(np.mean(flat_responses, axis=0))
                variances.append(np.var(flat_responses, axis=0) / flat_responses.shape[0])
        return ObjectiveResult(results, variances if self.stochastic else None)

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
        # Plot each target
        for target in self.targets:
            # Build figure with individual responses for this target
            fig, axis = plt.subplots()
            for pred in target.prediction:
                axis.plot(responses[pred].get_time(), responses[pred].get_data(), label=f'{pred}')
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
        return DesignObjective(
            parameters,
            solver,
            list(targets.keys()),
            output_dir,
            stochastic=bool(config.get('stochastic', False)),
            composition=config.get('composite', None),
        )
