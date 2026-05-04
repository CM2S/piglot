"""Module for Curve solver."""
from typing import Dict, Any, List, Tuple, Type, Optional
import re
import time
import numpy as np
import sympy
from piglot.parameter import ParameterSet, ParameterValues
from piglot.solver.solver import CaseResult, OutputResult
from piglot.solver.multi_case_solver import MultiCaseSolver, Case
from piglot.utils.solver_utils import OutputStream


class CurveCase(Case):
    """Case for the Curve solver."""

    def __init__(
        self,
        name: str,
        expression: str,
        parametric: str,
        bounds: Tuple[float, float],
        points: int,
        variance: Optional[str] = None,
        lengthscale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.case_name = name
        self.expression = expression
        self.parametric = parametric
        self.bounds = bounds
        self.points = points
        self.variance = variance
        self.lengthscale = lengthscale
        self.seed = seed

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

    def get_expression(self, expression: str, values: ParameterValues) -> str:
        """Get the expression for this case.

        Parameters
        ----------
        expression : str
            Expression template for this case.
        values : np.ndarray
            Current parameters to evaluate.

        Returns
        -------
        str
            Expression for this case.
        """
        for parameter, value in values.scalar_values.items():
            expression = re.sub(r'\<' + parameter + r'\>', str(value), expression)
        return expression

    def run(self, values: ParameterValues, tmp_dir: str, stream: OutputStream) -> CaseResult:
        """Run the case for the given set of parameters.

        Parameters
        ----------
        values : ParameterValues
            Named set of parameter values for this evaluation.
        tmp_dir : str
            Temporary directory to run the problem.
        stream : OutputStream
            Output stream for this call.

        Returns
        -------
        CaseResult
            Result of the case.
        """
        begin_time = time.time()
        # Prepare symbols
        symbs = sympy.symbols(self.parametric)
        expression = sympy.lambdify(symbs, self.get_expression(self.expression, values))
        # Evaluate the expression on the grid
        grid = np.linspace(self.bounds[0], self.bounds[1], self.points)
        curve = np.array([expression(**{self.parametric: x}) for x in grid])
        # Check if this is a stochastic curve
        if self.variance is not None:
            var_expression = self.get_expression(self.variance, values)
            var_func = sympy.lambdify(symbs, var_expression)
            variances = np.array([var_func(**{self.parametric: x}) for x in grid])
            # Check if variances are valid
            if not np.all(variances > 0) or not np.all(np.isfinite(variances)):
                raise ValueError("Invalid variances computed.")
            # Generate noise
            rnd = np.random.RandomState(self.seed)  # pylint: disable=E1101
            if self.lengthscale is not None:
                # Use a multivariate normal with a squared exponential kernel
                kernel = np.exp(-np.square(np.subtract.outer(grid, grid) / self.lengthscale))
                covar = kernel * variances[:, None] * variances[None, :]
                noise = rnd.multivariate_normal(np.zeros(self.points), covar)
            else:
                # Gaussian i.i.d. noise
                noise = rnd.normal(0.0, np.sqrt(variances))
            curve += noise
        # Return the result
        run_time = time.time() - begin_time
        return CaseResult(
            begin_time,
            run_time,
            values.scalar_values,
            True,
            values.param_hash,
            {self.case_name: OutputResult(grid, curve)},
        )

    @classmethod
    def read(
        cls,
        name: str,
        config: Dict[str, Any],
    ) -> "CurveCase":
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
        return cls(
            name,
            config['expression'],
            config['parametric'],
            config['bounds'],
            points,
            config.get('variance', None),
            config.get('lengthscale', None),
            config.get('seed', None),
        )


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
