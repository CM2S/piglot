from __future__ import annotations
from typing import Any, Optional, Type
import os
import tempfile
import unittest
import numpy as np
from piglot.parameter import ParameterValues, read_parameters
from piglot.settings import Settings
from piglot.optimiser import OptimisationResult, OptimiserState, SimpleOptimiser
from piglot.objective import Objective, IndividualObjective, IndividualObjectiveResult


class DummyObjective(Objective):

    def __init__(self, settings: Settings, objectives: list[IndividualObjective], composite: bool = False):
        super().__init__(settings, objectives, scalarisation=None, composite=composite)

    def _objective(
        self, params: ParameterValues, concurrent: bool = False  # pylint: disable=unused-argument
    ) -> list[IndividualObjectiveResult]:
        x = params.scalar_values["x"]
        return [IndividualObjectiveResult(value=x * x) for _ in self.objectives]

    @classmethod
    def read(
        cls: Type[DummyObjective],
        config: dict[str, Any],
        settings: Settings,
    ) -> DummyObjective:
        return cls(settings, [IndividualObjective("obj")])


class DummySimpleOptimiser(SimpleOptimiser):

    def name(self) -> str:
        return "DummySimple"

    def _simple_optimise(
        self,
        num_iters: int,
        initial_guess: np.ndarray,
        bounds: list[tuple[float, float]],  # pylint: disable=unused-argument
        objective,
        callback,
    ) -> None:
        for i in range(num_iters):
            x = np.array([initial_guess[0] - 0.2 * i])
            objective(x)
            callback()


class TestOptimiserState(unittest.TestCase):

    def test_update_non_stochastic_only_on_improvement(self):
        state = OptimiserState()
        state.update(1, OptimisationResult(value=10.0, params=np.array([1.0])), stochastic=False)
        self.assertEqual(state.i_iter, 1)
        self.assertEqual(state.iters_without_improvement, 0)
        self.assertEqual(state.best_result.value, 10.0)

        state.update(2, OptimisationResult(value=11.0, params=np.array([1.1])), stochastic=False)
        self.assertEqual(state.i_iter, 2)
        self.assertEqual(state.iters_without_improvement, 1)
        self.assertEqual(state.best_result.value, 10.0)

        state.update(3, OptimisationResult(value=9.0, params=np.array([0.9])), stochastic=False)
        self.assertEqual(state.i_iter, 3)
        self.assertEqual(state.iters_without_improvement, 0)
        self.assertEqual(state.best_result.value, 9.0)

    def test_update_stochastic_always_refreshes_best(self):
        state = OptimiserState()
        state.update(1, OptimisationResult(value=5.0), stochastic=True)
        state.update(2, OptimisationResult(value=6.0), stochastic=True)
        self.assertEqual(state.best_result.value, 6.0)
        self.assertEqual(state.iters_without_improvement, 0)


class TestSimpleOptimiserValidation(unittest.TestCase):

    def make_settings(self, output_dir: str, iters: Optional[int] = 2) -> Settings:
        return Settings(
            output_dir=output_dir,
            parameters=read_parameters({"x": {"type": "real", "initial": 1.0, "lbound": 0.0, "ubound": 2.0}}),
            iters=iters,
            quiet=True,
        )

    def test_validate_problem_multi_objective(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self.make_settings(tmpdir)
            objective = DummyObjective(settings, [IndividualObjective("o1"), IndividualObjective("o2")])
            with self.assertRaises(ValueError) as ex:
                DummySimpleOptimiser(settings, objective)
            self.assertEqual(
                ex.exception.args[0],
                "This optimiser does not support multi-objective optimisation.",
            )

    def test_validate_problem_composite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self.make_settings(tmpdir)
            objective = DummyObjective(settings, [IndividualObjective("o1")], composite=True)
            with self.assertRaises(ValueError) as ex:
                DummySimpleOptimiser(settings, objective)
            self.assertEqual(
                ex.exception.args[0],
                "This optimiser does not support composite objectives.",
            )

    def test_validate_problem_variance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self.make_settings(tmpdir)
            objective = DummyObjective(settings, [IndividualObjective("o1", variance=True)])
            with self.assertRaises(ValueError) as ex:
                DummySimpleOptimiser(settings, objective)
            self.assertEqual(
                ex.exception.args[0],
                "This optimiser does not support objectives with variance.",
            )

    def test_requires_number_of_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self.make_settings(tmpdir, iters=None)
            objective = DummyObjective(settings, [IndividualObjective("o1")])
            with self.assertRaises(ValueError) as ex:
                DummySimpleOptimiser(settings, objective)
            self.assertEqual(ex.exception.args[0], "Number of iterations must be specified in the config file.")


class TestSimpleOptimiserRun(unittest.TestCase):

    def make_settings(self, output_dir: str) -> Settings:
        return Settings(
            output_dir=output_dir,
            parameters=read_parameters({"x": {"type": "real", "initial": 1.0, "lbound": 0.0, "ubound": 2.0}}),
            iters=3,
            quiet=True,
        )

    def test_optimise_returns_best_and_writes_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = self.make_settings(tmpdir)
            objective = DummyObjective(settings, [IndividualObjective("o1")])
            optimiser = DummySimpleOptimiser(settings, objective)

            result = optimiser.optimise()

            self.assertEqual(result.value, 0.36)
            np.testing.assert_array_almost_equal(result.params, np.array([0.6]))
            self.assertEqual(objective.num_calls, 3)
            self.assertIsNotNone(optimiser.state.best_result)
            self.assertEqual(optimiser.state.best_result.value, 0.36)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "history")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "progress")))
