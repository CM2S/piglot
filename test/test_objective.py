from typing import Any
import unittest
from piglot.objective import Objective


class DummyObjective(Objective):

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def prepare(self) -> None:
        pass


class TestDummyObjective(unittest.TestCase):
    def setUp(self):
        self.objective = DummyObjective()

    def test_missing_read(self):
        with self.assertRaises(NotImplementedError) as ex:
            self.objective.read(None, None, None)
        self.assertEqual(ex.exception.args[0],
                         "Reading not implemented for this objective")

    def test_missing_plot_case(self):
        with self.assertRaises(NotImplementedError) as ex:
            self.objective.plot_case(None)
        self.assertEqual(ex.exception.args[0],
                         "Single case plotting not implemented for this objective")

    def test_missing_plot_best(self):
        with self.assertRaises(NotImplementedError) as ex:
            self.objective.plot_best()
        self.assertEqual(ex.exception.args[0],
                         "Best case plotting not implemented for this objective")

    def test_missing_plot_current(self):
        with self.assertRaises(NotImplementedError) as ex:
            self.objective.plot_current()
        self.assertEqual(ex.exception.args[0],
                         "Current case plotting not implemented for this objective")

    def test_missing_get_history(self):
        with self.assertRaises(NotImplementedError) as ex:
            self.objective.get_history()
        self.assertEqual(ex.exception.args[0],
                         "Objective history not implemented for this objective")
