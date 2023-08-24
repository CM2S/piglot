"""Provide synthetic test functions"""
import os.path
from typing import Dict, Type
import numpy as np
import torch
import botorch.test_functions.synthetic
from botorch.test_functions.synthetic import SyntheticTestFunction
from piglot.parameter import ParameterSet
from piglot.objective import SingleObjective


class SyntheticObjective(SingleObjective):
    """Objective function derived from a synthetic test function"""

    def __init__(
            self,
            parameters: ParameterSet,
            name: str,
            output_dir: str,
            *args,
            **kwargs,
        ) -> None:
        super().__init__(parameters, output_dir)
        test_functions = self.get_test_functions()
        if name not in test_functions:
            raise RuntimeError(f'Unknown function {name}. Must be in {list(test_functions.keys())}')
        self.func = test_functions[name](*args, **kwargs)
        with open(os.path.join(output_dir, 'optimum_value'), 'w', encoding='utf8') as file:
            file.write(f'{self.func.optimal_value}')

    @staticmethod
    def get_test_functions() -> Dict[str, Type[SyntheticTestFunction]]:
        """Return available test functions

        Returns
        -------
        Dict[str, Type[SyntheticTestFunction]]
            Available test functions
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

    def _objective(self, values: np.ndarray, parallel: bool=False) -> float:
        """Objective computation for analytical functions

        Parameters
        ----------
        values : np.ndarray
            Set of parameters to evaluate the objective for
        parallel : bool, optional
            Whether this call may be concurrent to others, by default False

        Returns
        -------
        float
            Objective value
        """
        params = torch.tensor(self.parameters.denormalise(values))
        return self.func.evaluate_true(params).numpy()
