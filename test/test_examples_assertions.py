from typing import List, Tuple, Dict
import os
import shutil
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.optimiser import InvalidOptimiserException


EXAMPLES_ASSERTIONS: Dict[str, Exception] = {
    'analytical_expression.yaml': (
        RuntimeError,
        'Missing analytical expression to minimise',
    ),
    'composite_bad_optimiser.yaml': (
        InvalidOptimiserException,
        'This optimiser does not support composition',
    ),
    'invalid_optimiser.yaml': (
        RuntimeError,
        "Unknown optimiser 'nelder-mead'.",
    ),
    'optimiser_missing_name.yaml': (
        RuntimeError,
        "Missing optimiser name.",
    ),
    'optimiser_random_invalid_kwargs.yaml': (
        TypeError,
        'dimension',
    ),
    'optimiser_random_sampling_typo.yaml': (
        ValueError,
        'Invalid sampling soboll!',
    ),
    'parameter_double_parameter.yaml': (
        ValueError,
        "Duplicate 'Young' key found in YAML.",
    ),
    'parameter_invalid_init_shot.yaml': (
        RuntimeError,
        'Initial shot outside of bounds',
    ),
    'objective_missing_name.yaml': (
        ValueError,
        "Missing name for objective.",
    ),
    'objective_unknown_name.yaml': (
        ValueError,
        "Unknown objective 'none'.",
    ),
    'synthetic_unknown_function.yaml': (
        RuntimeError,
        "Unknown function none.",
    ),
    'synthetic_missing_function.yaml': (
        RuntimeError,
        "Missing test function",
    ),
    'synthetic_unknown_composition.yaml': (
        RuntimeError,
        'Unknown composition none.',
    ),
    'missing_iters.yaml': (
        RuntimeError,
        "Missing number of iterations from the config file",
    ),
    'missing_objective.yaml': (
        RuntimeError,
        "Missing objective from the config file",
    ),
    'missing_optimiser.yaml': (
        RuntimeError,
        "Missing optimiser from the config file",
    ),
    'missing_parameters.yaml': (
        RuntimeError,
        "Missing parameters from the config file",
    ),
    'invalid_syntax.yaml': (
        RuntimeError,
        "Failed to parse the config file: YAML syntax seems invalid.",
    ),
    'dummy_invalid_parameters.yaml': (
        ValueError,
        "Invalid parameters: the parameters 'm' and 'c' are required.",
    ),
    'dummy_invalid_parameters2.yaml': (
        ValueError,
        "Invalid parameters: the parameters 'm' and 'c' are required.",
    ),
}


def transform_files(path: str) -> List[Tuple[str, Exception]]:
    return [
        (os.path.join(path, file), exception, match)
        for file, (exception, match) in EXAMPLES_ASSERTIONS.items()
    ]


@pytest.mark.parametrize('input_file,expected,match', transform_files('test/examples_assertions'))
def test_input_files(input_file: str, expected: Exception, match: str):
    with pytest.raises(expected, match=match):
        piglot_main(input_file)
    output_dir, _ = os.path.splitext(input_file)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
