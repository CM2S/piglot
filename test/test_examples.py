from typing import List
import os
import shutil
import pytest
from piglot.bin.piglot import main as piglot_main


def get_files(path: str) -> List[str]:
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.yaml')
    ]


@pytest.mark.parametrize('input_file', get_files('test/examples'))
def test_input_files(input_file: str):
    piglot_main(input_file)
    output_dir, _ = os.path.splitext(input_file)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
