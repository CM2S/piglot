import unittest
import os
import shutil
from piglot.bin.piglot import main as piglot_main
from piglot.bin.piglot_plot import main as piglot_plot_main


def get_first_hash(filename: str) -> str:
    """Extracts the first hash from a file.

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    str
        The first hash in the file.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        next(file)  # Skip the header line
        first_line = next(file)  # Get the second line
        first_hash = first_line.split()[-1]  # The hash is the last element on the line
    return first_hash


class TestExtractParameters(unittest.TestCase):
    def test_analytical_error(self):
        with self.assertRaises(RuntimeError) as ex:
            input_file = os.path.join("test", "examples_plots_assertions",
                                      "analytical_parameters_3.yaml")
            output_dir = os.path.join("test", "examples_plots_assertions",
                                      "analytical_parameters_3")
            piglot_main(input_file)
            filename = os.path.join(output_dir, 'func_calls')
            first_hash = get_first_hash(filename)
            piglot_plot_main([
                'case',
                input_file,
                first_hash,
                '--save_fig',
                os.path.join(output_dir, 'case_analytical.png'),
            ])
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        self.assertEqual(str(ex.exception), "Plotting only supported for one or two dimensions.")

    def test_gp_single_parameter(self):
        with self.assertRaises(ValueError) as ex:
            input_file = os.path.join("test", "examples_plots_assertions",
                                      "gp_error.yaml")
            output_dir = os.path.join("test", "examples_plots_assertions",
                                      "gp_error")
            piglot_main(input_file)
            piglot_plot_main([
                'gp',
                input_file,
                '--save_fig',
                os.path.join(output_dir, 'gp_error.png'),
            ])
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        self.assertEqual(str(ex.exception),
                         "Can only plot a Gaussian process regression for a single parameter.")

    def test_analytical_3d(self):
        with self.assertRaises(RuntimeError) as ex:
            input_file = os.path.join("test", "examples_plots_assertions",
                                      "test_analytical_3d.yaml")
            output_dir = os.path.join("test", "examples_plots_assertions",
                                      "test_analytical_3d")
            piglot_main(input_file)
            piglot_plot_main([
                'best',
                input_file,
                '--save_fig',
                os.path.join(output_dir, 'test_analytical_3d.png'),
            ])
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        self.assertEqual(str(ex.exception),
                         "Plotting only supported for one or two dimensions.")

    def test_analytical_3d_mo(self):
        with self.assertRaises(RuntimeError) as ex:
            input_file = os.path.join("test", "examples_plots_assertions",
                                      "test_analytical_3d_mo.yaml")
            output_dir = os.path.join("test", "examples_plots_assertions",
                                      "test_analytical_3d_mo")
            piglot_main(input_file)
            piglot_plot_main([
                'best',
                input_file,
                '--save_fig',
                os.path.join(output_dir, 'test_analytical_3d_mo.png'),
            ])
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        self.assertEqual(str(ex.exception),
                         "Plotting only supported for one or two dimensions.")

    def test_analytical_mo_3obj(self):
        with self.assertRaises(ValueError) as ex:
            input_file = os.path.join("test", "examples_plots_assertions",
                                      "test_analytical_mo_3obj.yaml")
            output_dir = os.path.join("test", "examples_plots_assertions",
                                      "test_analytical_mo_3obj")
            piglot_main(input_file)
            piglot_plot_main([
                'pareto',
                input_file,
                '--save_fig',
                os.path.join(output_dir, 'test_analytical_mo_3obj.png'),
            ])
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        self.assertEqual(str(ex.exception),
                         "Can only plot the Pareto front for a two-objective optimisation problem.")


if __name__ == '__main__':
    unittest.main()
