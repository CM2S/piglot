import unittest
import os
from piglot.utils.solver_utils import extract_parameters, has_keyword, has_parameter, \
    find_keyword, load_module_from_file
from piglot.parameter import ParameterSet

INPUT_FILE_1 = os.path.join("test", "solver_utils", "input_file_1.txt")
INPUT_FILE_2 = os.path.join("test", "solver_utils", "input_file_2.txt")
INPUT_FILE_3 = os.path.join("test", "solver_utils", "input_file_3.txt")
INPUT_FILE_4 = os.path.join("test", "solver_utils", "test_module.py")


class TestExtractParameters(unittest.TestCase):
    def test_extract_parameters(self):
        # Call the function with the test input file
        parameters = extract_parameters(INPUT_FILE_1)

        # Verify the returned ParameterSet
        self.assertIsInstance(parameters, ParameterSet)
        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0].inital_value, 1.0)
        self.assertEqual(parameters[0].lbound, 0.0)
        self.assertEqual(parameters[0].ubound, 2.0)
        self.assertEqual(parameters[1].inital_value, 2.0)
        self.assertEqual(parameters[1].lbound, 1.0)
        self.assertEqual(parameters[1].ubound, 3.0)

    def test_extract_parameters_fail_1(self):
        with self.assertRaises(RuntimeError) as ex:
            extract_parameters(INPUT_FILE_2)
        self.assertEqual(str(ex.exception), "Pattern parameter1 referenced but not defined!")

    def test_extract_parameters_fail_2(self):
        with self.assertRaises(RuntimeError) as ex:
            extract_parameters(INPUT_FILE_3)
        self.assertEqual(str(ex.exception), "Repeated pattern parameter1 in file!")

    def test_has_keyword(self):
        # Call the function with the test input file and a keyword that is in the file
        result = has_keyword(INPUT_FILE_1, '<parameter1')
        self.assertTrue(result)

        # Call the function with the test input file and a keyword that is not in the file
        result = has_keyword(INPUT_FILE_1, '<parameter3')
        self.assertFalse(result)

    def test_has_parameter(self):
        # Call the function with the test input file and a parameter that is in the file
        result = has_parameter(INPUT_FILE_1, '<parameter1')
        self.assertTrue(result)

        # Call the function with the test input file and a parameter that is not in the file
        result = has_parameter(INPUT_FILE_1, '<parameter3')
        self.assertFalse(result)

    def test_find_keyword(self):

        with open(INPUT_FILE_1, 'r', encoding='utf8') as file:
            # Call the function with the test input file and a keyword that is in the file
            result = find_keyword(file, '<parameter1')
            self.assertEqual(result.strip(), '<parameter1(1.0, 0.0, 2.0)>')

    def test_find_keyword_fail(self):
        with self.assertRaises(RuntimeError) as ex:
            find_keyword(INPUT_FILE_1, '<parameter3')
        self.assertEqual(str(ex.exception), "Keyword <parameter3 not found!")

    def test_load_module_from_file(self):
        # Define the attribute to load from the module
        attribute = 'test_attribute'

        # Call the function with the test Python file and the attribute
        result = load_module_from_file(INPUT_FILE_4, attribute)

        # Verify the returned object
        self.assertEqual(result, 'CM2S')


if __name__ == '__main__':
    unittest.main()
