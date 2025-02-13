"""Module with a sample multi-case solver interface."""
from __future__ import annotations
from typing import Dict, Any, List, Type
import time
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import CaseResult, OutputResult
from piglot.solver.multi_case_solver import MultiCaseSolver, Case


class SampleCase(Case):
    """Sample for a case used with multi-case solvers."""

    def __init__(
        self,
        name: str,
        output_name: str,
        multiplier: float,
    ) -> None:
        self.case_name = name
        self.output_name = output_name
        self.multiplier = multiplier

    def name(self) -> str:
        """Return the name of the case.

        Returns
        -------
        str
            Name of the case.
        """
        return self.case_name

    def get_fields(self) -> List[str]:
        """Get the fields to output for this case.

        Returns
        -------
        List[str]
            Fields to output for this case.
        """
        return [self.output_name]

    def run(
        self,
        parameters: ParameterSet,
        values: np.ndarray,
        tmp_dir: str,
    ) -> CaseResult:
        """Run the case for the given set of parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        values : np.ndarray
            Current parameters to evaluate.
        tmp_dir : str
            Temporary directory to run the problem.

        Returns
        -------
        CaseResult
            Result of the case.
        """
        begin_time = time.time()
        # Evaluate the response
        param_values = parameters.to_dict(values)
        grid = np.linspace(0, 1, len(param_values))
        curve = np.array(list(param_values.values())) * self.multiplier
        # Return the result
        run_time = time.time() - begin_time
        return CaseResult(
            begin_time=begin_time,
            run_time=run_time,
            values=values,
            success=True,
            param_hash=parameters.hash(values),
            responses={self.output_name: OutputResult(grid, curve)},
        )

    @classmethod
    def read(
        cls,
        name: str,
        config: Dict[str, Any],
    ) -> SampleCase:
        """Read the case from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the case.
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        SampleCase
            Case to use for this problem.
        """
        if 'multiplier' not in config:
            raise ValueError("Missing 'multiplier' in case configuration.")
        output_name = config.pop('output_name', name)
        return cls(name, output_name, config['multiplier'])


class SampleMultiCaseSolver(MultiCaseSolver):
    """Sample multi-case solver interface."""

    @classmethod
    def get_case_class(cls) -> Type[Case]:
        """Get the case class for this solver.

        Returns
        -------
        Type[Case]
            Case class for this solver.
        """
        return SampleCase
