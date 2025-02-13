"""Module for Curve solver."""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Type
import re
import time
import numpy as np
import sympy
from piglot.parameter import ParameterSet
from piglot.solver.solver import CaseResult, OutputResult
from piglot.solver.multi_case_solver import MultiCaseSolver, Case


class CurveCase(Case):
    """Case for the Curve solver."""

    def __init__(
        self,
        name: str,
        expression: str,
        parametric: str,
        bounds: Tuple[float, float],
        points: int,
    ) -> None:
        self.case_name = name
        self.expression = expression
        self.parametric = parametric
        self.bounds = bounds
        self.points = points

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
        return [self.case_name]

    def get_expression(self, parameters: ParameterSet, values: np.ndarray) -> str:
        """Get the expression for this case.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        values : np.ndarray
            Current parameters to evaluate.

        Returns
        -------
        str
            Expression for this case.
        """
        expression = self.expression
        param_value = parameters.to_dict(values)
        for parameter, value in param_value.items():
            expression = re.sub(r'\<' + parameter + r'\>', str(value), expression)
        return expression

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
        # Prepare symbols
        symbs = sympy.symbols([self.parametric] + [p.name for p in parameters])
        expression = sympy.lambdify(symbs, self.get_expression(parameters, values))
        param_values = parameters.to_dict(values)
        # Evaluate the expression on the grid
        grid = np.linspace(self.bounds[0], self.bounds[1], self.points)
        curve = np.array([expression(**param_values, **{self.parametric: x}) for x in grid])
        # Return the result
        run_time = time.time() - begin_time
        return CaseResult(
            begin_time,
            run_time,
            values,
            True,
            parameters.hash(values),
            {self.case_name: OutputResult(grid, curve)},
        )

    @classmethod
    def read(
        cls,
        name: str,
        config: Dict[str, Any],
    ) -> CurveCase:
        """Read the case from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the case.
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Case
            Case to use for this problem.
        """
        if 'expression' not in config:
            raise ValueError("Missing 'expression' in solver configuration.")
        if 'parametric' not in config:
            raise ValueError("Missing 'parametric' in solver configuration.")
        if 'bounds' not in config:
            raise ValueError("Missing 'bounds' in solver configuration.")
        points = int(config['points']) if 'points' in config else 100
        return cls(name, config['expression'], config['parametric'], config['bounds'], points)


class CurveSolver(MultiCaseSolver):
    """Curve solver."""

    def __init__(
        self,
        cases: List[Case],
        parameters: ParameterSet,
        output_dir: str,
        tmp_dir: str,
        verbosity: str,
        parallel: int = 1,
    ) -> None:
        """Constructor for the Curve solver class.

        Parameters
        ----------
        cases : List[Case]
            Cases to be run.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.
        verbosity: str
            Verbosity level for the solver.
        parallel : int
            Number of parallel processes to use.
        tmp_dir : str
            Path to the temporary directory.
        """
        super().__init__(cases, parameters, output_dir, tmp_dir, verbosity, parallel=parallel)

    @classmethod
    def get_case_class(cls) -> Type[Case]:
        """Get the case class for this solver.

        Returns
        -------
        Type[Case]
            Case class for this solver.
        """
        return CurveCase
