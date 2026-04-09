import unittest
import numpy as np
import torch

from piglot.parameter import (
    ComputedParameter,
    DiscreteParameter,
    Parameter,
    ParameterSet,
    RealParameter,
    legacy_converter,
    read_parameters,
)


class TestParameter(unittest.TestCase):
    def test_scalar_name_and_value(self):
        parameter = Parameter("test", optimisable=True, num_components=1)
        self.assertEqual(parameter.get_scalar_names(), ["test"])
        self.assertEqual(parameter.get_name_value_pair(np.array([1.5])), {"test": 1.5})
        self.assertEqual(parameter.get_value(np.array([2.5])), 2.5)

    def test_vector_name_and_value(self):
        parameter = Parameter("vec", optimisable=True, num_components=3)
        values = np.array([1.0, 2.0, 3.0])
        self.assertEqual(parameter.get_scalar_names(), ["vec_0", "vec_1", "vec_2"])
        self.assertEqual(
            parameter.get_name_value_pair(values),
            {"vec_0": 1.0, "vec_1": 2.0, "vec_2": 3.0},
        )
        np.testing.assert_array_equal(parameter.get_value(values), values)


class TestRealParameter(unittest.TestCase):
    def test_real_parameter_out_of_bounds_initial(self):
        with self.assertRaises(RuntimeError):
            RealParameter("x", initial=2.0, lbound=0.0, ubound=1.0)

    def test_real_parameter_bounds_initial_and_random(self):
        parameter = RealParameter("x", initial=0.5, lbound=0.0, ubound=1.0, num_components=2)
        np.testing.assert_array_equal(parameter.get_initial_vector(), np.array([0.5, 0.5]))
        np.testing.assert_array_equal(parameter.get_bounds(), np.array([[0.0, 1.0], [0.0, 1.0]]))
        random_values = parameter.get_random(np.random.default_rng(123))
        self.assertEqual(random_values.shape, (2,))
        self.assertTrue(np.all(random_values >= 0.0))
        self.assertTrue(np.all(random_values <= 1.0))

    def test_real_parameter_read(self):
        parameter = RealParameter.read(
            "x",
            {"type": "real", "initial": 0.5, "lbound": 0.0, "ubound": 1.0, "num_components": 2},
        )
        self.assertIsInstance(parameter, RealParameter)
        self.assertEqual(parameter.name, "x")
        self.assertEqual(parameter.num_components, 2)

    def test_real_parameter_read_missing_field(self):
        with self.assertRaises(RuntimeError):
            RealParameter.read("x", {"initial": 1.0, "lbound": 0.0})


class TestDiscreteParameter(unittest.TestCase):
    def test_discrete_parameter_invalid_initial(self):
        with self.assertRaises(RuntimeError):
            DiscreteParameter("k", initial=4.0, values=[1.0, 2.0, 3.0])

    def test_discrete_parameter_bounds_and_random(self):
        parameter = DiscreteParameter("k", initial=2.0, values=[1.0, 2.0, 3.0], num_components=2)
        np.testing.assert_array_equal(parameter.get_bounds(), np.array([[1.0, 3.0], [1.0, 3.0]]))
        random_values = parameter.get_random(np.random.default_rng(42))
        self.assertEqual(random_values.shape, (2,))
        self.assertTrue(set(random_values).issubset({1.0, 2.0, 3.0}))

    def test_discrete_parameter_read(self):
        parameter = DiscreteParameter.read(
            "k",
            {"type": "discrete", "initial": 2.0, "values": [1, 2, 3], "num_components": 3},
        )
        self.assertIsInstance(parameter, DiscreteParameter)
        self.assertEqual(parameter.num_components, 3)
        self.assertEqual(parameter.values, [1.0, 2.0, 3.0])


class TestComputedParameter(unittest.TestCase):
    def setUp(self):
        self.optim_params = [
            RealParameter("x", initial=1.0, lbound=0.0, ubound=2.0),
            RealParameter("y", initial=2.0, lbound=0.0, ubound=3.0),
        ]

    def test_scalar_expression(self):
        parameter = ComputedParameter("z", "x + y", self.optim_params)
        result = parameter.compute({"x": np.array([1.5]), "y": np.array([2.5])})
        np.testing.assert_array_equal(result, np.array([4.0]))
        self.assertEqual(parameter.num_components, 1)

    def test_vector_expression(self):
        optim_params = [RealParameter("v", initial=1.0, lbound=0.0, ubound=2.0, num_components=2)]
        parameter = ComputedParameter("w", "v * 2", optim_params)
        result = parameter.compute({"v": np.array([3.0, 4.0])})
        np.testing.assert_array_equal(result, np.array([6.0, 8.0]))
        self.assertEqual(parameter.num_components, 2)

    def test_invalid_compute_result_type(self):
        parameter = ComputedParameter("z", "x + y", self.optim_params)
        parameter.code = compile("'bad'", "<string>", "eval")
        with self.assertRaises(RuntimeError):
            parameter.compute({"x": np.array([1.0]), "y": np.array([2.0])})

    def test_read_missing_expression(self):
        with self.assertRaises(ValueError):
            ComputedParameter.read("z", {}, self.optim_params)


class TestParameterSet(unittest.TestCase):
    def setUp(self):
        self.optim_params = [
            RealParameter("x", initial=1.0, lbound=0.0, ubound=2.0),
            DiscreteParameter("k", initial=2.0, values=[1.0, 2.0, 3.0]),
            DiscreteParameter("d", initial=0.0, values=[0.0, 1.0], num_components=2),
        ]
        self.computed_params = [ComputedParameter("z", "x + k", self.optim_params)]
        self.parameter_set = ParameterSet(self.optim_params, self.computed_params)

    def test_iterator(self):
        target = self.optim_params
        for i, parameter in enumerate(self.parameter_set):
            self.assertEqual(parameter.name, target[i].name)
        self.assertEqual(len(self.parameter_set), len(target))
        self.assertEqual(self.parameter_set[0].name, "x")

    def test_parameter_counts_and_names(self):
        self.assertEqual(self.parameter_set.num_optim_parameters(), 4)
        self.assertEqual(self.parameter_set.num_discrete(), 3)
        self.assertEqual(self.parameter_set.get_names(), ["x", "k", "d", "z"])
        self.assertEqual(
            self.parameter_set.get_scalar_names(),
            ["x", "k", "d_0", "d_1", "z"],
        )
        self.assertEqual(
            self.parameter_set.get_scalar_names(include_computed=False),
            ["x", "k", "d_0", "d_1"],
        )

    def test_initial_random_and_bounds_vectors(self):
        np.testing.assert_array_equal(self.parameter_set.get_initial_vector(), np.array([1.0, 2.0, 0.0, 0.0]))
        random_vector = self.parameter_set.get_random_vector(np.random.default_rng(0))
        self.assertEqual(random_vector.shape, (4,))
        bounds = self.parameter_set.get_bounds()
        np.testing.assert_array_equal(
            bounds,
            np.array([[0.0, 2.0], [1.0, 3.0], [0.0, 1.0], [0.0, 1.0]]),
        )

    def test_discrete_combinations(self):
        combinations = self.parameter_set.get_discrete_combinations()
        self.assertEqual(len(combinations), 12)
        self.assertIn({1: 1.0, 2: 0.0, 3: 0.0}, combinations)
        self.assertIn({1: 3.0, 2: 1.0, 3: 1.0}, combinations)

    def test_to_torch_dict(self):
        values = torch.tensor([[1.0, 2.0, 0.0, 1.0], [0.5, 3.0, 1.0, 0.0]])
        mapped = self.parameter_set.to_torch_dict(values)
        self.assertEqual(set(mapped.keys()), {"x", "k", "d"})
        self.assertEqual(tuple(mapped["x"].shape), (2, 1))
        self.assertEqual(tuple(mapped["d"].shape), (2, 2))


class TestLegacyConverter(unittest.TestCase):
    def test_legacy_converter_list_spec(self):
        converted = legacy_converter("x", [1.0, 0.0, 2.0])
        self.assertEqual(
            converted,
            {"type": "real", "initial": 1.0, "lbound": 0.0, "ubound": 2.0},
        )

    def test_legacy_converter_invalid_non_dict(self):
        with self.assertRaises(TypeError):
            legacy_converter("x", 1)

    def test_legacy_converter_invalid_keys(self):
        with self.assertRaises(TypeError):
            legacy_converter("x", {1: "real"})

    def test_legacy_converter_missing_type(self):
        with self.assertRaises(ValueError):
            legacy_converter("x", {"initial": 1.0})


class TestReadParameters(unittest.TestCase):
    def test_read_parameters_with_real_and_computed(self):
        config = {
            "x": {"type": "real", "initial": 1.0, "lbound": 0.0, "ubound": 2.0},
            "z": {"type": "computed", "expression": "x * 2"},
        }
        parameters = read_parameters(config)
        self.assertIsInstance(parameters, ParameterSet)
        self.assertEqual(len(parameters.optim_parameters), 1)
        self.assertEqual(len(parameters.computed_parameters), 1)
        self.assertEqual(parameters.optim_parameters[0].name, "x")
        self.assertEqual(parameters.computed_parameters[0].name, "z")

    def test_read_parameters_with_legacy_spec(self):
        config = {"x": [1.0, 0.0, 2.0]}
        parameters = read_parameters(config)
        self.assertEqual(len(parameters.optim_parameters), 1)
        self.assertEqual(parameters.optim_parameters[0].name, "x")

    def test_read_parameters_unknown_type(self):
        with self.assertRaises(ValueError):
            read_parameters({"x": {"type": "unknown"}})


if __name__ == '__main__':
    unittest.main()
