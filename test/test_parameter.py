import unittest
import numpy as np
from piglot.parameter import Parameter, ParameterSet, DualParameterSet, read_parameters


class TestParameter(unittest.TestCase):
    def setUp(self):
        self.parameter = Parameter("test", 5, 0, 10)

    def test_bounds(self):
        with self.assertRaises(RuntimeError):
            Parameter("test", 5, 10, 0)

    def test_clip(self):
        self.assertEqual(self.parameter.clip(5), 5)
        self.assertEqual(self.parameter.clip(11), 10)
        self.assertEqual(self.parameter.clip(-1), 0)


class TestParameterSet(unittest.TestCase):
    def setUp(self):
        self.parameter_set = ParameterSet()
        self.parameter_set.add("test", 5, 0, 10)

    def test_add(self):
        with self.assertRaises(RuntimeError):
            self.parameter_set.add("test", 5, 0, 10)

    def test_hash(self):
        self.parameter_set.hash([5])

    def test_iterator(self):
        target = [Parameter("test", 5, 0, 10)]
        for i, parameter in enumerate(self.parameter_set):
            self.assertEqual(parameter.name, target[i].name)
            self.assertEqual(parameter.lbound, target[i].lbound)
            self.assertEqual(parameter.ubound, target[i].ubound)
            self.assertEqual(parameter.inital_value, target[i].inital_value)
        self.assertEqual(len(self.parameter_set), len(target))
        self.assertEqual(self.parameter_set[0].name, 'test')

    def test_clip(self):
        self.assertEqual(self.parameter_set.clip([5]), [5])
        self.assertEqual(self.parameter_set.clip([11]), [10])
        self.assertEqual(self.parameter_set.clip([-1]), [0])


class TestDualParameterSet(unittest.TestCase):
    def setUp(self):
        self.dual_parameter_set = DualParameterSet()
        self.dual_parameter_set.add("test", 5, 0, 10)
        self.dual_parameter_set.add_output("test_output", lambda test: test * 2)
        self.dual_parameter_set.clone_output('test')

    def test_add_output(self):
        with self.assertRaises(RuntimeError):
            self.dual_parameter_set.add_output("test_output", lambda test: test * 2)

    def test_to_output(self):
        self.assertTrue(np.all(self.dual_parameter_set.to_output([5]) == np.array([10, 5])))
        self.assertTrue(np.all(self.dual_parameter_set.to_output([10]) == np.array([20, 10])))
        self.assertTrue(np.all(self.dual_parameter_set.to_output([0]) == np.array([0, 0])))
        self.assertEqual(self.dual_parameter_set.to_dict([5]), {'test': 5, 'test_output': 10})


class TestReadParameters(unittest.TestCase):
    def test_read_parameters(self):
        with self.assertRaises(ValueError):
            parameters = read_parameters({})
        config = {
            'parameters': {
                'test_param': [5, 0, 10]
            },
            'output_parameters': {
                'test_output': 'test_param * 2'
            }
        }
        parameters = read_parameters(config)
        self.assertIsInstance(parameters, DualParameterSet)
        self.assertEqual(len(parameters.parameters), 1)
        self.assertEqual(len(parameters.output_parameters), 1)
        self.assertEqual(parameters.parameters[0].name, 'test_param')
        self.assertEqual(parameters.parameters[0].lbound, 0)
        self.assertEqual(parameters.parameters[0].ubound, 10)
        self.assertEqual(parameters.parameters[0].inital_value, 5)
        self.assertEqual(parameters.output_parameters[0].name, 'test_output')


if __name__ == '__main__':
    unittest.main()
