from typing import List
import os
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.utils.assorted import change_cwd


def get_files(path: str) -> List[str]:
    return sorted([
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.yaml')
    ])


@pytest.mark.parametrize('input_dir', get_files('test/example_init_shot'))
def test_input_files(input_dir: str):
    with change_cwd(os.path.dirname(input_dir)):
        input_file = os.path.basename(input_dir)
        piglot_main(input_file)
