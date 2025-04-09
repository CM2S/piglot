from __future__ import annotations
from typing import Any, Callable, Optional, Tuple, Type, Dict
import unittest
import numpy as np
from piglot.parameter import ParameterSet
from piglot.optimiser import missing_method, ScalarOptimiser, InvalidOptimiserException
from piglot.objective import Objective, GenericObjective, ObjectiveResult


class DummyObjective(Objective):

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def prepare(self) -> None:
        pass

    @classmethod
    def read(
        cls: Type[DummyObjective],
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> DummyObjective:
        pass


class DummyGenericObjective(GenericObjective):

    def __init__(self, stochastic: bool):
        super().__init__(None, stochastic=stochastic)

    def _objective(self, values: np.ndarray, concurrent: bool = False) -> ObjectiveResult:
        pass

    @classmethod
    def read(
        cls: Type[DummyObjective],
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> DummyObjective:
        pass


class DummyScalarOptimiser(ScalarOptimiser):

    def __init__(self, objective: Objective):
        super().__init__('Dummy', objective)

    def _scalar_optimise(
        self,
        objective: Callable[[np.ndarray, Optional[bool]], float],
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        pass


class TestMissingMethod(unittest.TestCase):
    def setUp(self):
        self.optimiser = missing_method('optimiser', 'package')

    def test_missing_method(self):
        with self.assertRaises(ImportError) as ex:
            self.optimiser()
        self.assertEqual(ex.exception.args[0],
                         "optimiser is not available. You need to install package package!")


class TestScalarOptimiser(unittest.TestCase):

    def test_generic_objective(self):
        objective = DummyObjective()
        optimiser = DummyScalarOptimiser(objective)
        with self.assertRaises(InvalidOptimiserException) as ex:
            optimiser._validate_problem(objective)
        self.assertEqual(ex.exception.args[0],
                         'Generic objective required for this optimiser')

    def test_stochastic(self):
        objective = DummyGenericObjective(True)
        optimiser = DummyScalarOptimiser(objective)
        with self.assertRaises(InvalidOptimiserException) as ex:
            optimiser._validate_problem(objective)
        self.assertEqual(ex.exception.args[0],
                         'This optimiser does not support stochasticity')
