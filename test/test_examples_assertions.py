from typing import List, Tuple, Dict
import os
import shutil
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.optimiser import InvalidOptimiserException
from piglot.utils.assorted import change_cwd


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
    'missing_solver_name.yaml': (
        ValueError,
        "Missing name for solver.",
    ),
    'wrong_solver_name.yaml': (
        ValueError,
        "Unknown solver 'curvefit'.",
    ),
    'fitting_without_solver.yaml': (
        ValueError,
        "Missing solver for fitting objective.",
    ),
    'invalid_pred.yaml': (
        ValueError,
        "Invalid prediction '2' for reference 'reference_curve.txt'.",
    ),
    'invalid_pred_design.yaml': (
        ValueError,
        "Invalid prediction '2' for design target 'maximum_force'.",
    ),
    'missing_references.yaml': (
        ValueError,
        "Missing references for fitting objective.",
    ),
    'missing_pred.yaml': (
        ValueError,
        "Missing prediction for reference 'reference_curve.txt'.",
    ),
    'unexistent_reference.yaml': (
        ValueError,
        "Reference 'reference_curve_2.txt' is not associated to any case.",
    ),
    'mae_reduction.yaml': (
        ValueError,
        "Invalid reduction 'mae' for fitting objective.",
    ),
    'design_missing_pred.yaml': (
        ValueError,
        "Missing prediction for design target 'maximum_force'.",
    ),
    'design_missing_quantity.yaml': (
        ValueError,
        "Missing quantity for fitting objective.",
    ),
    'design_missing_quantity_name.yaml': (
        ValueError,
        "Missing name in quantity specification.",
    ),
    'design_missing_quantity_script.yaml': (
        ValueError,
        "Missing script in quantity specification.",
    ),
    'design_missing_quantity_class.yaml': (
        ValueError,
        "Missing class in quantity specification.",
    ),
    'design_missing_solver.yaml': (
        ValueError,
        "Missing solver for fitting objective.",
    ),
    'design_missing_targets.yaml': (
        ValueError,
        "Missing targets for fitting objective.",
    ),
    'unexistent_targets.yaml': (
        ValueError,
        "Design target 'integral_quantity' is not associated to any case.",
    ),
    'bo_unkacq.yaml': (
        RuntimeError,
        "Unkown acquisition function ucbb",
    ),
    'bo_q.yaml': (
        RuntimeError,
        "Can only use q != 1 for quasi-Monte Carlo acquisitions",
    ),
    'duplicated_field.yaml': (
        ValueError,
        "Duplicate output field 'reaction_x'.",
    ),
    'bo_equalbounds.yaml': (
        RuntimeError,
        "All observed points are equal: add more initial samples",
    ),
}


def transform_files(path: str) -> List[Tuple[str, Exception]]:
    return [
        (os.path.join(path, file), exception, match)
        for file, (exception, match) in EXAMPLES_ASSERTIONS.items()
    ]


@pytest.mark.parametrize('input_dir,expected,match', transform_files('test/examples_assertions'))
def test_input_files(input_dir: str, expected: Exception, match: str):
    with change_cwd(os.path.dirname(input_dir)):
        input_file = os.path.basename(input_dir)
        with pytest.raises(expected, match=match):
            piglot_main(input_file)
        output_dir, _ = os.path.splitext(input_file)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
