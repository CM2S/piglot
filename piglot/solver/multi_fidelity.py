"""Module for multi-fidelity solvers."""
import os
from typing import Callable, Optional, Type, Any, TypeVar
from piglot.parameter import ParameterSet, ParameterValues
from piglot.solver import read_solver
from piglot.solver.solver import CaseResult, OutputResult, Solver


T = TypeVar("T", bound="MultiFidelitySolver")


class MultiFidelitySolver(Solver):
    """Wrapper for supporting multiple solvers with different levels of fidelity."""

    def __init__(
        self,
        parameters: ParameterSet,
        output_dir: str,
        tmp_dir: str,
        verbosity: Optional[str],
        solvers_config: dict[float, dict[str, Any]],
    ) -> None:
        # Sanitise fidelity parameter
        if not parameters.is_multi_fidelity():
            raise ValueError("No fidelity parameter found in parameter set.")
        if set(parameters.get_fidelities()) != set(solvers_config.keys()):
            raise ValueError("Fidelity parameter values do not match solver configurations.")
        self.fidelity_param = parameters.get_scalar_names()[parameters.get_scalar_fidelity_index()]

        # Inject verbosity into solvers
        if verbosity is not None:
            for config in solvers_config.values():
                if 'verbosity' not in config:
                    config['verbosity'] = verbosity

        # Build solvers from configuration
        self.solvers = {
            fidel: read_solver(config, parameters, os.path.join(output_dir, f"solver_{fidel}"))
            for fidel, config in solvers_config.items()
        }

        # Check if output fields between solvers are consistent
        self.output_fields: set[str] = None
        for solver in self.solvers.values():
            if self.output_fields is None:
                self.output_fields = set(solver.get_output_fields())
            elif self.output_fields != set(solver.get_output_fields()):
                raise ValueError("Inconsistent output fields between solvers.")

        super().__init__(parameters, output_dir, tmp_dir, verbosity)

    def prepare(self) -> None:
        """Prepare data for the optimisation."""
        for solver in self.solvers.values():
            os.makedirs(solver.output_dir, exist_ok=True)
            solver.prepare()

    def get_output_fields(self) -> list[str]:
        """Get all output fields.

        Returns
        -------
        list[str]
            Output fields.
        """
        return list(self.output_fields)

    def __try_solvers(self, func: Callable[[Solver], Any]) -> Any:
        exc = None
        for solver in self.solvers.values():
            try:
                return func(solver)
            except Exception as ex:
                exc = ex
                continue
        raise exc

    def get_case_result(self, param_hash: str) -> CaseResult:
        """Get the result for a given case.

        Parameters
        ----------
        param_hash : str
            Hash of the case to load.

        Returns
        -------
        CaseResult
            Result for this hash.
        """
        return self.__try_solvers(lambda solver: solver.get_case_result(param_hash))

    def get_case_params(self, param_hash: str) -> dict[str, float]:
        """Get the parameters for a given hash.

        Parameters
        ----------
        param_hash : str
            Hash of the case to load.

        Returns
        -------
        dict[str, float]
            Parameters for this hash.
        """
        return self.__try_solvers(lambda solver: solver.get_case_params(param_hash))

    def get_output_response(self, param_hash: str) -> dict[str, OutputResult]:
        """Get the responses from all output fields for a given case.

        Parameters
        ----------
        param_hash : str
            Hash of the case to load.

        Returns
        -------
        dict[str, OutputResult]
            Output responses.
        """
        return self.__try_solvers(lambda solver: solver.get_output_response(param_hash))

    def solve(self, values: ParameterValues, concurrent: bool) -> dict[str, OutputResult]:
        """Solve all cases for the given set of parameter values.

        Parameters
        ----------
        values : ParameterValues
            Named set of parameter values for this evaluation.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        dict[str, OutputResult]
            Evaluated results for each output field.
        """
        # Figure out which solver to use based on fidelity
        fidelity = values.scalar_values[self.fidelity_param]
        if fidelity not in self.solvers:
            raise ValueError(f"Unknown fidelity value {fidelity}.")
        return self.solvers[fidelity].solve(values, concurrent)

    @classmethod
    def read(
        cls: Type[T],
        config: dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> T:
        """Read the solver from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        Solver
            Solver to use for this problem.
        """
        if "fidelities" not in config:
            raise ValueError("Missing 'fidelities' keyword for multi-fidelity solver.")
        return cls(
            parameters,
            output_dir,
            os.path.join(output_dir, "tmp"),
            config.get("verbosity"),
            config["fidelities"]
        )
