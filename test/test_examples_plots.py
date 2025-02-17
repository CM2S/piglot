from typing import List
import os
import shutil
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.bin.piglot_plot import main as piglot_plot_main
from piglot.utils.assorted import change_cwd


def get_files(path: str) -> List[str]:
    """Gets all the files in a directory.

    Parameters
    ----------
    path : str
        Path to the directory.

    Returns
    -------
    List[str]
        List of files in the directory.
    """
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.yaml')
    ]


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


@pytest.mark.parametrize('input_dir', get_files('test/examples_plots'))
def test_input_files(input_dir: str):
    with change_cwd(os.path.dirname(input_dir)):
        input_file = os.path.basename(input_dir)
        output_dir, _ = os.path.splitext(input_file)
        piglot_main(input_file)
        filename = os.path.join(output_dir, 'func_calls')
        first_hash = get_first_hash(filename)
        for kind in ('best', 'history', 'parameters', 'regret'):
            piglot_plot_main([
                kind,
                input_file,
                '--save_fig',
                os.path.join(output_dir, f'{kind}.png'),
            ])
        for kind in ('history', 'parameters'):
            piglot_plot_main([
                kind,
                input_file,
                '--save_fig',
                os.path.join(output_dir, f'{kind}.png'),
                '--log',
                '--best',
                '--time',
            ])
        piglot_plot_main([
            'regret',
            input_file,
            '--save_fig',
            os.path.join(output_dir, 'regret.png'),
            '--log',
            '--time',
        ])
        piglot_plot_main([
            'case',
            input_file,
            first_hash,
            '--save_fig',
            os.path.join(output_dir, 'case.png'),
        ])
        piglot_plot_main([
            'animation',
            input_file,
        ])
        if '_mo' in input_file:
            piglot_plot_main([
                'pareto',
                input_file,
                '--all',
                '--log',
                '--save_fig',
                os.path.join(output_dir, 'pareto.png'),
            ])
        if input_file.endswith('random.yaml'):
            piglot_plot_main([
                'surrogate',
                input_file,
            ])
        if 'test_analytical' not in input_file:
            piglot_plot_main([
                'gp',
                input_file,
                '--max_calls',
                '10',
                '--save_fig',
                os.path.join(output_dir, 'gp.png'),
            ])
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
