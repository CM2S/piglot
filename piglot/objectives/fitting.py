"""Module for curve fitting objectives"""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from piglot.yaml_parser import parse_loss
from piglot.losses.loss import Loss
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import Solver, OutputResult, Case, OutputField
from piglot.utils.reduce_response import reduce_response
from piglot.objective import Objective, SingleObjective


class Reference:
    """Container for reference solutions."""

    def __init__(
            self,
            filename: str,
            prediction: str,
            x_col: int=1,
            y_col: int=2,
            skip_header: int=0,
            x_scale: float=1.0,
            y_scale: float=1.0,
            x_offset: float=0.0,
            y_offset: float=0.0,
            filter_tol: float=0.0,
            show: bool=False,
            loss: Loss=None,
            weight: float=1.0,
        ):
        self.filename = filename
        self.prediction = prediction
        self.data = np.genfromtxt(filename, skip_header=skip_header)[:,[x_col - 1, y_col - 1]]
        self.data[:,0] = x_offset + x_scale * self.data[:,0]
        self.data[:,1] = y_offset + y_scale * self.data[:,1]
        self.orig_data = np.copy(self.data)
        self.filter_tol = filter_tol
        self.show = show
        self.loss = loss
        self.weight = weight

    def prepare(self) -> None:
        """Prepare the reference data."""
        if self.has_filtering():
            print("Filtering reference ...", end='')
            num, error, (x, y) = reduce_response(self.data[:,0], self.data[:,1], self.filter_tol)
            self.data = np.array([x, y]).T
            print(f" done (from {self.orig_data.shape[0]} to {num} points, error = {error:.2e})")
            if self.show:
                _, ax = plt.subplots()
                ax.plot(self.orig_data[:,0], self.orig_data[:,1], label="Reference")
                ax.plot(self.data[:,0], self.data[:,1], c='r', ls='dashed')
                ax.scatter(self.data[:,0], self.data[:,1], c='r', label="Resampled")
                ax.legend()
                plt.show()

    def num_fields(self) -> int:
        """Get the number of reference fields.

        Returns
        -------
        int
            Number of reference fields.
        """
        return self.data.shape[1] - 1

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
        return self.data[:, 0]

    def get_data(self, field_idx: int=0) -> np.ndarray:
        """Get the data column of the reference.

        Parameters
        ----------
        field_idx : int
            Index of the field to output.

        Returns
        -------
        np.ndarray
            Data column.
        """
        return self.data[:, field_idx + 1]

    def get_orig_time(self) -> np.ndarray:
        """Get the original time column of the reference.

        Returns
        -------
        np.ndarray
            Original time column.
        """
        return self.orig_data[:, 0]

    def get_orig_data(self, field_idx: int=0) -> np.ndarray:
        """Get the original data column of the reference.

        Parameters
        ----------
        field_idx : int
            Index of the field to output.

        Returns
        -------
        np.ndarray
            Original data column.
        """
        return self.orig_data[:, field_idx + 1]

    def compute_loss(self, results: OutputResult) -> Any:
        """Compute the loss for the given results.

        Parameters
        ----------
        results : OutputResult
            Results to compute the loss for.

        Returns
        -------
        float
            Loss value.
        """
        return self.loss(
            self.get_time(),
            self.get_data(),
            results.get_time(),
            results.get_data(),
        )

    @staticmethod
    def read(filename: str, config: Dict[str, Any]) -> Reference:
        """Read the reference from the configuration dictionary.

        Parameters
        ----------
        filename : str
            Path to the reference file.
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Reference
            Reference to use for this problem.
        """
        if 'prediction' not in config:
            raise ValueError(f"Missing prediction for reference'{filename}'.")
        return Reference(
            filename,
            config['prediction'],
            x_col=int(config.get('x_col', 1)),
            y_col=int(config.get('y_col', 2)),
            skip_header=int(config.get('skip_header', 0)),
            x_scale=float(config.get('x_scale', 1.0)),
            y_scale=float(config.get('y_scale', 1.0)),
            x_offset=float(config.get('x_offset', 0.0)),
            y_offset=float(config.get('y_offset', 0.0)),
            filter_tol=float(config.get('filter_tol', 0.0)),
            show=bool(config.get('show', False)),
            loss=parse_loss(filename, config['loss']) if 'loss' in config else None,
            weight=float(config.get('weight', 1.0)),
        )


class FittingSolver:
    """Interface class between fitting objectives and solvers."""

    def __init__(
            self,
            solver: Solver,
            references: Dict[Reference, Dict[Case, OutputField]],
        ) -> None:
        self.solver = solver
        self.references = references

    def prepare(self) -> None:
        """Prepare the solver for optimisation."""
        for reference in self.references.keys():
            reference.prepare()
        self.solver.prepare()

    def solve(self, values: np.ndarray, concurrent: bool) -> Dict[Reference, List[OutputResult]]:
        """Evaluate the solver for the given set of parameter values and get the output results.

        Parameters
        ----------
        values : np.ndarray
            Parameter values to evaluate.
        concurrent : bool
            Whether this call may be concurrent to others.

        Returns
        -------
        Dict[Reference, List[OutputResult]]
            Output results.
        """
        result = self.solver.solve(values, concurrent)
        # Populate output results
        output = {reference: [] for reference in self.references.keys()}
        for reference, cases in self.references.items():
            for case, field in cases.items():
                output[reference].append(result[case][field])
        return output


class FittingSingleObjective(SingleObjective):
    """Scalar fitting objective function."""

    def __init__(
            self,
            parameters: ParameterSet,
            solver: FittingSolver,
            loss: Loss,
            output_dir: str,
        ) -> None:
        super().__init__(parameters, output_dir)
        self.solver = solver
        self.loss = loss

    def prepare(self) -> None:
        """Prepare the objective for optimisation."""
        super().prepare()
        self.solver.prepare()

    def _objective(self, values: np.ndarray, concurrent: bool=False) -> float:
        """Objective computation for analytical functions

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """
        # Run the solver
        output = self.solver.solve(values, concurrent)
        # Compute the loss
        loss = 0.0
        for reference, results in output.items():
            losses = [reference.compute_loss(result) for result in results]
            loss += reference.weight * np.mean(losses)
        return loss


def read_fitting_objective(
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> Objective:
    """Read a fitting objective from a configuration dictionary.

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
    Objective
        Objective function to optimise.
    """
    # Read the solver
    if not 'solver' in config:
        raise ValueError("Missing solver for fitting objective.")
    solver = read_solver(config['solver'], parameters, output_dir)
    # Read the references
    if not 'references' in config:
        raise ValueError("Missing references for fitting objective.")
    references: Dict[Reference, Dict[Case, OutputField]] = {}
    for reference_name, reference_config in config.pop('references').items():
        reference = Reference.read(reference_name, reference_config)
        # Map the reference to the solver cases
        references[reference] = {}
        for field_name, (case, field) in solver.get_output_fields().items():
            if field_name == reference.prediction:
                references[reference][case] = field
        # Sanitise reference: check if it is associated to at least one case
        if len(references[reference]) == 0:
            raise ValueError(f"Reference '{reference_name}' is not associated to any case.")
    # Read the (optional) loss
    loss = None
    if 'loss' in config:
        loss = parse_loss(output_dir, config['loss'])
        # Assign the default loss to references without a specific loss
        for reference in references:
            if reference.loss is None:
                reference.loss = loss
    # Ensure all references have a loss
    for reference in references:
        if reference.loss is None:
            raise ValueError(f"Missing loss for reference '{reference.filename}'")
    # Return the objective
    return FittingSingleObjective(parameters, FittingSolver(solver, references), loss, output_dir)
