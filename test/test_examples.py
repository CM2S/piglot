import os
import shutil
from tempfile import TemporaryDirectory
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.utils.assorted import change_cwd


def get_files(path: str) -> list[str]:
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.yaml')
    ]


@pytest.mark.parametrize('input_file', get_files('test/examples'))
def test_examples(input_file: str):
    with TemporaryDirectory() as tmpdir:
        shutil.copy(input_file, tmpdir)
        with change_cwd(tmpdir):
            piglot_main(os.path.basename(input_file))
