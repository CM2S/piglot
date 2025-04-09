from typing import List, Tuple, Dict
import os
import shutil
import pytest
from piglot.bin.piglot import main as piglot_main
from piglot.optimiser import InvalidOptimiserException
from piglot.utils.assorted import change_cwd


EXAMPLES_ASSERTIONS: Dict[str, Exception] = {
    'test_analytical_acq_unknown.yaml': (
        RuntimeError,
        'Unkown acquisition function unknown',
    ),
    'analytical_expression.yaml': (
        RuntimeError,
        'Missing analytical expression to minimise',
    ),
    'analytical_mo_missing_expression.yaml': (
        RuntimeError,
        'Missing analytical expression to minimise',
    ),
    'analytical_random_missing_variance.yaml': (
        ValueError,
        'Random evaluations require variance',
    ),
    'analytical_negative_variance.yaml': (
        RuntimeError,
        'Negative variance not allowed',
    ),
    'analytical_composite_no_scalarisation.yaml': (
        ValueError,
        'Composite objectives require scalarisation',
    ),
    'analytical_missing_objectives.yaml': (
        RuntimeError,
        'Missing analytical objectives to optimise for',
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
        ValueError,
        'Reduction function "none" is not available.',
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
        "Missing prediction for fitting target 'reference_curve.txt'.",
    ),
    'unexistent_reference.yaml': (
        ValueError,
        "Undefined prediction case_2",
    ),
    'design_missing_pred.yaml': (
        ValueError,
        "Missing prediction for design target 'maximum_force'.",
    ),
    'design_missing_quantity.yaml': (
        ValueError,
        "Missing quantity for design target 'maximum_force'.",
    ),
    'design_composite_points.yaml': (
        ValueError,
        "All objectives must have a number of points specified for the composition.",
    ),
    'design_missing_quantity_name.yaml': (
        ValueError,
        "Need to pass the name of the reduction function.",
    ),
    'design_missing_quantity_script.yaml': (
        ValueError,
        "Missing 'script' field for reading the custom module script.",
    ),
    'design_missing_quantity_class.yaml': (
        ValueError,
        "Missing 'class' field for reading the custom module script.",
    ),
    'design_missing_solver.yaml': (
        ValueError,
        "Missing solver for design objective.",
    ),
    'design_missing_targets.yaml': (
        ValueError,
        "Missing targets for design objective.",
    ),
    'unexistent_targets.yaml': (
        ValueError,
        "Undefined prediction case_2",
    ),
    'bo_unkacq.yaml': (
        RuntimeError,
        "Unkown acquisition function ucbb",
    ),
    'duplicated_field.yaml': (
        ValueError,
        "Duplicate output field 'reaction_x'.",
    ),
    'bo_equalbounds.yaml': (
        ValueError,
        "All observed points are equal.",
    ),
    'query_analytical_composite.yaml': (
        InvalidOptimiserException,
        "This optimiser does not support composition",
    ),
    'query_analytical_stochastic.yaml': (
        InvalidOptimiserException,
        "This optimiser does not support stochasticity",
    ),
    'query_analytical_bad_num_params.yaml': (
        ValueError,
        "Number of parameters does not match the number of columns",
    ),
    'query_analytical_bad_num_iters.yaml': (
        ValueError,
        "Number of iterations does not match the number of rows",
    ),
    'query_analytical_bad_lbounds.yaml': (
        ValueError,
        "Parameter values outside lower bounds",
    ),
    'query_analytical_bad_ubounds.yaml': (
        ValueError,
        "Parameter values outside upper bounds",
    ),
    'curve_bad_composite_scalarisation.yaml': (
        ValueError,
        "Multi-objective composite problems require full composition",
    ),
    'curve_mo_bad_scalarisation.yaml': (
        ValueError,
        'Scalarisation function "missing" is not available.',
    ),
    'curve_mo_bad_scalarisation_2.yaml': (
        ValueError,
        'Scalarisation function "missing" is not available.',
    ),
    'curve_mo_bad_scalarisation_script.yaml': (
        ValueError,
        'Need to pass the file path for the "script" scalarisation.',
    ),
    'curve_mo_scalarisation_missing_name.yaml': (
        ValueError,
        'Need to pass the name of the scalarisation function.',
    ),
    'curve_bad_reduction.yaml': (
        ValueError,
        'Reduction function "missing" is not available.',
    ),
    'curve_bad_reduction_2.yaml': (
        ValueError,
        'Reduction function "missing" is not available.',
    ),
    'curve_bad_reduction_script.yaml': (
        ValueError,
        'Need to pass the file path for the "script" reduction.',
    ),
    'curve_reduction_missing_name.yaml': (
        ValueError,
        'Need to pass the name of the reduction function.',
    ),
    'curve_bad_reduction_grad.yaml': (
        ValueError,
        'Gradient check failed',
    ),
    'curve_bad_reduction_exception.yaml': (
        ValueError,
        'Test failed for reduction',
    ),
    'curve_bad_reduction_shape.yaml': (
        ValueError,
        'Bad shape after reduction for',
    ),
    'transformer_bad_name.yaml': (
        ValueError,
        'Response transformer "missing" is not available.',
    ),
    'transformer_bad_name_2.yaml': (
        ValueError,
        'Response transformer "missing" is not available.',
    ),
    'transformer_script.yaml': (
        ValueError,
        'Need to pass the file path for the "script" transformer.',
    ),
    'transformer_missing_name.yaml': (
        ValueError,
        'Need to pass the name of the response transformer.',
    ),
    'bad_solver_output_script.yaml': (
        RuntimeError,
        'Cannot read the configuration for a script-based output field.',
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
