import os
import shutil
from tempfile import TemporaryDirectory
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.bin.piglot_plot import main as piglot_plot_main
from piglot.utils.assorted import change_cwd


def get_files(path: str) -> list[str]:
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.yaml')
    ]


@pytest.mark.parametrize('input_file', get_files('test/examples_plots'))
def test_examples_plots(input_file: str):
    with TemporaryDirectory() as tmpdir:
        shutil.copy(input_file, tmpdir)
        with change_cwd(tmpdir):
            basename = os.path.basename(input_file)
            piglot_main(basename)
            piglot_plot_main([
                'best',
                basename,
                '--save_fig',
                'best.png',
            ])
            if '1d' in input_file:
                piglot_plot_main([
                    'gp',
                    basename,
                    '--save_fig',
                    '1d.png',
                    '--num_func_samples',
                    '4'
                ])
