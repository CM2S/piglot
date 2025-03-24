"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from piglot.parameter import ParameterSet
from piglot.objective import DynamicPlotter
from piglot.objectives.fitting import FittingSingleObjective
from piglot.solver import read_solver
from piglot.solver.solver import OutputResult
from piglot.utils.assorted import trapezoidal_integration_weights
from piglot.utils.interpolators import Interpolator, read_interpolator
from piglot.utils.reductions import Reduction
from piglot.utils.response_transformer import FullFieldErrors
from piglot.utils.scalarisations import read_scalarisation
from piglot.utils.composition.responses import FixedFlatteningUtility
from piglot.objectives.response_objective import ResponseSingleObjective, ResponseObjective


class FullFieldReference:
    """Container for full-field reference solutions."""

    def __init__(
        self,
        name: str,
        coord_data: np.ndarray,
        field_data: np.ndarray,
    ) -> None:
        if len(coord_data.shape) != 2:
            raise ValueError("Coordinate data must be 2D.")
        if len(field_data.shape) != 2:
            raise ValueError("Field data must be 2D.")
        self.name = name
        self.coord_data = coord_data
        self.field_data = field_data

    def prepare(self) -> None:
        """Prepare the reference data."""

    def get_coords(self) -> np.ndarray:
        """Get the coordinate data (spatial + temporal) of the reference.

        Returns
        -------
        np.ndarray
            Coordinate data with shape: n_points x (n_dim + 1)
        """
        return self.coord_data

    def get_data(self) -> np.ndarray:
        """Get the full-field data of the reference.

        Returns
        -------
        np.ndarray
            Full-field data with shape: n_points x n_fields
        """
        return self.field_data

    def get_num_fields(self) -> int:
        """Get the number of fields in the reference.

        Returns
        -------
        int
            Number of fields.
        """
        return self.field_data.shape[-1]

    @staticmethod
    def read(filename: str, config: Dict[str, Any], output_dir: str) -> FullFieldReference:
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
        FullFieldReference
            Reference to use for this problem.
        """
        if 'coords' not in config:
            raise ValueError("Missing list of coordinate indices for reference.")
        if 'fields' not in config:
            raise ValueError("Missing list of field indices for reference.")
        data = np.genfromtxt(
             filename,
             skip_header=config.pop('skip_header', None),
             delimiter=config.pop('delimiter', None),
        )
        return FullFieldReference(
            filename,
            data[:, config['coords']],
            data[:, config['fields']],
        )


class FullFieldReduction(Reduction):
    """Reduction for full-field data."""

    def __init__(self, reference: FullFieldReference, reduction: str) -> None:
        super().__init__()
        if reduction not in ('square_integral', 'abs_integral', 'mse', 'mae'):
            raise ValueError(f"Invalid reduction '{reduction}' for full-field data.")
        # Assign modifier and compute reduction weights
        self.modifier = torch.square if reduction in ('square_integral', 'mse') else torch.abs
        if reduction.endswith('integral'):
            weights = torch.from_numpy(trapezoidal_integration_weights(reference.get_coords()))
        else:
            weights = torch.ones(reference.get_coords().shape[0])
        # Unit sum weights
        self.weights = (weights / torch.sum(weights)).unsqueeze(-1)

    def reduce_torch(
        self,
        time: torch.Tensor,
        data: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the input data to a single value (with gradients).

        Parameters
        ----------
        time : torch.Tensor
            Time points of the response.
        data : torch.Tensor
            Data points of the response.
        params : torch.Tensor
            Parameters for the given responses.

        Returns
        -------
        torch.Tensor
            Reduced value of the data.
        """
        return torch.sum(self.weights * self.modifier(data), dim=(-1, -2))


class FullFieldFittingSingleObjective(ResponseSingleObjective):
    """Single objective for curve fitting optimisation objectives."""

    def __init__(
        self,
        name: str,
        reference: FullFieldReference,
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
            flatten_utility=FixedFlatteningUtility(
                reference.get_coords(),
                (reference.get_num_fields(),),
            ),
            prediction_transform=FullFieldErrors(
                reference.get_coords(),
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
        raise NotImplementedError("Full-field fitting objectives do not support plotting.")

    @classmethod
    def read(
        cls,
        name: str,
        config: Dict[str, Any],
        output_dir: str,
    ) -> FullFieldFittingSingleObjective:
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
        weight = float(config.pop('weight', 1.0))
        bounds = config.pop('bounds', None)
        interp_config = config.pop('interpolator', 'unstructured_linear')
        # Read the reference and return the objective
        reference = FullFieldReference.read(name, config, output_dir)
        return FullFieldFittingSingleObjective(
            name,
            reference,
            prediction,
            FullFieldReduction(reference, config.pop('reduction', 'mse')),
            read_interpolator(interp_config),
            weight=weight,
            bounds=bounds,
        )


class FullFieldFittingObjective(ResponseObjective):
    """Class for fitting of response-based objectives."""

    @classmethod
    def read(
        cls,
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> FullFieldFittingObjective:
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
        FullFieldFittingObjective
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
            (
                FullFieldFittingSingleObjective.read(target_name, target_config, output_dir)
                if target_config.get('full_field', True)
                else FittingSingleObjective.read(target_name, target_config, output_dir)
            )
            for target_name, target_config in config.pop('references').items()
        ]
        return FullFieldFittingObjective(
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
        raise NotImplementedError("Full-field fitting objectives do not support case plotting.")

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of a updatable plots
        """
        raise NotImplementedError("Full-field fitting objectives do not support dynamic plotting.")
