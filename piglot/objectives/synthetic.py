"""Provide synthetic test functions"""
from __future__ import annotations
from typing import Dict, Type, Any
import os.path
import numpy as np
import torch
import botorch.test_functions.synthetic
from botorch.test_functions.synthetic import SyntheticTestFunction
from piglot.parameter import ParameterSet
from piglot.objective import GenericObjective, ObjectiveResult
from piglot.utils.reductions import Reduction, read_reduction
from piglot.utils.composition.responses import ResponseComposition, FixedFlatteningUtility


class SyntheticObjective(GenericObjective):
    """Objective function derived from a synthetic test function."""

    def __init__(
            self,
            parameters: ParameterSet,
            name: str,
            output_dir: str,
            transform: str = None,
            composition: Reduction = None,
            **kwargs,
            ) -> None:
        super().__init__(
            parameters,
            stochastic=False,
            composition=self.__composition(composition) if composition is not None else None,
            output_dir=output_dir,
        )
        test_functions = self.get_test_functions()
        if name not in test_functions:
            raise RuntimeError(f'Unknown function {name}. Must be in {list(test_functions.keys())}')
        self.func = test_functions[name](**kwargs)
        self.transform = lambda x, y: x
        if transform == 'mse_composition':
            self.transform = lambda v, func: torch.square(torch.tensor([v - func.optimal_value]))
        with open(os.path.join(output_dir, 'optimum_value'), 'w', encoding='utf8') as file:
            file.write(f'{self.transform(self.func.optimal_value, self.func)}')

    @staticmethod
    def __composition(reduction: Reduction) -> ResponseComposition:
        """Create a response composition from a reduction.

        Parameters
        ----------
        reduction : Reduction
            Reduction to apply.

        Returns
        -------
        ResponseComposition
            Composition to apply.
        """
        return ResponseComposition(
            True,
            False,
            [1.0],
            [reduction],
            [FixedFlatteningUtility(np.array([0.0]))],
        )

    @staticmethod
    def get_test_functions() -> Dict[str, Type[SyntheticTestFunction]]:
        """Return available test functions.

        Returns
        -------
        Dict[str, Type[SyntheticTestFunction]]
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

    def _objective(self, values: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        """Objective computation for analytical functions.

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for.
        parallel : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        ObjectiveResult
            Objective value.
        """
        params = torch.from_numpy(values)
        value = self.func.evaluate_true(params)
        if self.composition is not None:
            value -= self.func.optimal_value
        elif self.transform is not None:
            value = self.transform(value, self.func)
        value = float(value.item())
        if self.composition is None:
            return ObjectiveResult(
                values,
                np.array([value]),
                np.array([value]),
                scalar_value=value,
            )
        final_value = self.composition.composition(np.array([value]), values)
        return ObjectiveResult(
            values,
            np.array([value]),
            np.array([final_value]),
            scalar_value=final_value.item(),
        )

    @staticmethod
    def read(
            config: Dict[str, Any],
            parameters: ParameterSet,
            output_dir: str,
            ) -> SyntheticObjective:
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
        SyntheticObjective
            Objective function to optimise.
        """
        # Check for mandatory arguments
        if 'function' not in config:
            raise RuntimeError("Missing test function")
        function = config.pop('function')
        # Optional arguments
        composition = None
        if 'composition' in config:
            composition = read_reduction(config.pop('composition'))
        return SyntheticObjective(
            parameters,
            function,
            output_dir,
            composition=composition,
            **config,
        )
