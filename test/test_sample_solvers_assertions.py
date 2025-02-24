from typing import List, Tuple, Dict
import os
import sys
import shutil
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.solver import AVAILABLE_SOLVERS
from piglot.utils.assorted import change_cwd


EXAMPLES_ASSERTIONS: Dict[str, Exception] = {
    'input_generator_deps.yaml': (
        ValueError,
        'Dependencies not supported with custom input data generators.',
    ),
    'input_generator_bad_tmp.yaml': (
        ValueError,
        'Input data temporary directory',
    ),
    'input_generator_bad_file.yaml': (
        RuntimeError,
        'does not exist in the temporary directory',
    ),
    'input_duplicate_fields.yaml': (
        ValueError,
        'Duplicate output field',
    ),
    'input_missing_file.yaml': (
        ValueError,
        'does not exist',
    ),
    'input_missing_field_name.yaml': (
        ValueError,
        'No name defined for field',
    ),
    'input_missing_fields.yaml': (
        ValueError,
        'No fields defined for case',
    ),
    'input_unsupported_field.yaml': (
        ValueError,
        'not supported for case',
    ),
    'multi_missing_multiplier.yaml': (
        ValueError,
        "Missing 'multiplier' in case configuration.",
    ),
    'multi_missing_cases.yaml': (
        ValueError,
        "Missing 'cases' in solver configuration.",
    ),
    'single_missing_name.yaml': (
        ValueError,
        'Missing output field name.',
    ),
    'single_bad_verbosity.yaml': (
        ValueError,
        'Invalid verbosity level:',
    ),
    'script_bad_output.yaml': (
        ValueError,
        'Unknown output field',
    ),
    'script_missing_output.yaml': (
        ValueError,
        'Missing output field',
    ),
    'script_empty_response.yaml': (
        ValueError,
        'All observed points are equal!',
    ),
}


def transform_files(path: str) -> List[Tuple[str, Exception]]:
    return [
        (os.path.join(path, file), exception, match)
        for file, (exception, match) in EXAMPLES_ASSERTIONS.items()
    ]


@pytest.mark.parametrize(
        'input_dir,expected,match',
        transform_files('test/examples_sample_solvers_assertions'),
)
def test_input_files(input_dir: str, expected: Exception, match: str):
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
    with change_cwd(os.path.dirname(input_dir)):
        input_file = os.path.basename(input_dir)
        with pytest.raises(expected, match=match):
            piglot_main(input_file)
        output_dir, _ = os.path.splitext(input_file)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
