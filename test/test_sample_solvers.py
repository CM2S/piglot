from typing import List
import os
import sys
import shutil
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.bin.piglot_plot import main as piglot_plot_main
from piglot.utils.assorted import change_cwd
from piglot.solver import AVAILABLE_SOLVERS


def get_files(path: str) -> List[str]:
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.yaml')
    ]


@pytest.mark.parametrize('input_dir', get_files('test/examples_sample_solvers'))
def test_input_files(input_dir: str):
    # Load the example solver modules beforehand
    repo_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(repo_dir, 'examples', 'solver_example'))
    from single_solver import SampleSingleCaseSolver
    from multi_solver import SampleMultiCaseSolver
    from input_solver import SampleInputFileSolver
    # Inject the new solvers into piglot
    AVAILABLE_SOLVERS['sample_single_case'] = SampleSingleCaseSolver
    AVAILABLE_SOLVERS['sample_multi_case'] = SampleMultiCaseSolver
    AVAILABLE_SOLVERS['sample_input_file'] = SampleInputFileSolver
    # Run the cases
    with change_cwd(os.path.dirname(input_dir)):
        input_file = os.path.basename(input_dir)
        output_dir, _ = os.path.splitext(input_file)
        piglot_main(input_file)
        piglot_plot_main([
            'best',
            input_file,
            '--save_fig',
            os.path.join(output_dir, 'best.png'),
        ])
        output_dir, _ = os.path.splitext(input_file)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
