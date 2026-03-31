"""Provide synthetic test functions"""
from typing import Any, TypeVar
import os.path
import torch
import botorch.test_functions.synthetic
from botorch.test_functions.synthetic import SyntheticTestFunction
from piglot.parameter import ParameterValues
from piglot.settings import Settings
from piglot.objective import IndividualObjectiveResult
from piglot.objectives.simple_objective import SimpleObjective, SimpleIndividualObjective


IndividualT = TypeVar('IndividualT', bound='SyntheticIndividualObjective')


class SyntheticIndividualObjective(SimpleIndividualObjective):
    """Individual objective function derived from a synthetic test function."""

    def __init__(
        self,
        name: str,
        settings: Settings,
        function: str,
        weight: float = 1.0,
        maximise: bool = False,
        variance: bool = False,
        bounds: tuple[float, float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            weight=weight,
            maximise=maximise,
            variance=variance,
            composite=False,
            bounds=bounds,
        )
        test_functions = self.get_test_functions()
        if function not in test_functions:
            raise RuntimeError(
                f'Unknown function {function}. Must be one of {list(test_functions.keys())}'
            )
        self.func = test_functions[function](**kwargs)
        with open(os.path.join(settings.output_dir, 'optimum_value'), 'w', encoding='utf8') as file:
            file.write(f'{self.func.optimal_value}')

    @staticmethod
    def get_test_functions() -> dict[str, type[SyntheticTestFunction]]:
        """Return available test functions.

        Returns
        -------
        dict[str, type[SyntheticTestFunction]]
            Available test functions.
        """
        return {
            'ackley': botorch.test_functions.synthetic.Ackley,
            'beale': botorch.test_functions.synthetic.Beale,
            'branin': botorch.test_functions.synthetic.Branin,
            'bukin': botorch.test_functions.synthetic.Bukin,
            'cosine8': botorch.test_functions.synthetic.Cosine8,
            'drop_wave': botorch.test_functions.synthetic.DropWave,
            'dixon_price': botorch.test_functions.synthetic.DixonPrice,
            'egg_holder': botorch.test_functions.synthetic.EggHolder,
            'griewank': botorch.test_functions.synthetic.Griewank,
            'hartmann': botorch.test_functions.synthetic.Hartmann,
            'holder_table': botorch.test_functions.synthetic.HolderTable,
            'levy': botorch.test_functions.synthetic.Levy,
            'michalewicz': botorch.test_functions.synthetic.Michalewicz,
            'powell': botorch.test_functions.synthetic.Powell,
            'rastrigin': botorch.test_functions.synthetic.Rastrigin,
            'rosenbrock': botorch.test_functions.synthetic.Rosenbrock,
            'shekel': botorch.test_functions.synthetic.Shekel,
            'six_hump_camel': botorch.test_functions.synthetic.SixHumpCamel,
            'styblinski_tang': botorch.test_functions.synthetic.StyblinskiTang,
            'three_hump_camel': botorch.test_functions.synthetic.ThreeHumpCamel,
        }

    def evaluate(self, params: ParameterValues, concurrent: bool) -> IndividualObjectiveResult:
        """Evaluate objective value for the given results.

        Parameters
        ----------
        params : ParameterValues
            Named set of parameter values for this evaluation.
        concurrent : bool
            Whether this call may be concurrent to others.

        Returns
        -------
        IndividualObjectiveResult
            Objective value and variance for the given parameters.
        """
        params = torch.tensor(list(params.scalar_values.values()))
        value = self.func.evaluate_true(params).item()
        return IndividualObjectiveResult(value=value, variance=0 if not self.variance else None)

    @classmethod
    def read(
        cls: type[IndividualT], name: str, config: dict[str, Any], settings: Settings
    ) -> IndividualT:
        """Read the objective from a configuration dictionary.

        Parameters
        ----------
        name : str
            Name of this objective.
        config : dict[str, Any]
            Terms from the configuration dictionary.
        settings : Settings
            Settings for this problem.

        Returns
        -------
        IndividualT
            Objective function to optimise.
        """
        # Check for mandatory arguments
        if 'function' not in config:
            raise RuntimeError("Missing test function")
        function = config.pop('function')
        # Optional arguments
        weight = float(config.pop('weight', 1.0))
        maximise = bool(config.pop('maximise', False))
        variance = config.pop('variance', None)
        bounds = config.pop('bounds', None)
        return cls(
            name,
            settings,
            function,
            weight=weight,
            maximise=maximise,
            variance=variance,
            bounds=bounds,
            **config,
        )


class SyntheticObjective(SimpleObjective[SyntheticIndividualObjective]):
    """Objective function derived from a synthetic test function."""

    @classmethod
    def individual_objective_type(cls) -> type[SyntheticIndividualObjective]:
        """Return the type of the individual objective for this simple objective.

        Returns
        -------
        type[SyntheticIndividualObjective]
            Type of the individual objective for this simple objective.
        """
        return SyntheticIndividualObjective
