from typing import List
import os
import shutil
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.bin.piglot_plot import main as piglot_plot_main
from piglot.utils.assorted import change_cwd


def get_files(path: str) -> List[str]:
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.yaml')
    ]


@pytest.mark.parametrize('input_dir', get_files('test/examples_plots'))
def test_input_files(input_dir: str):
    with change_cwd(os.path.dirname(input_dir)):
        input_file = os.path.basename(input_dir)
        output_dir, _ = os.path.splitext(input_file)
        piglot_main(input_file)
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
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
