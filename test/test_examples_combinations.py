import os
from typing import Any
from tempfile import TemporaryDirectory
import pytest
from piglot.bin.piglot import run_config
from piglot.bin.piglot_plot import main as piglot_plot_main
from piglot.utils.assorted import change_cwd
from piglot.utils.yaml_parser import read_yaml


def get_files(path: str) -> list[str]:
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.yaml')
    ]


def valid_problems(path: str) -> list[tuple[str, str]]:
    objectives = get_files(os.path.join(path, 'objectives'))
    scalar_optimisers = get_files(os.path.join(path, 'optimisers', 'scalar'))
    wrapper_optimisers = get_files(os.path.join(path, 'optimisers', 'wrappers'))
    generic_optimisers = get_files(os.path.join(path, 'optimisers', 'generic'))
    problems = []
    for objective in objectives:
        for wrapper in wrapper_optimisers:
            problems.append((objective, wrapper))
        for generic in generic_optimisers:
            if ('_mo' in objective) == ('_mo' in generic):
                problems.append((objective, generic))
        if not any(key in objective for key in ('composite', 'mo', 'stochastic')):
            for optimiser in scalar_optimisers:
                problems.append((objective, optimiser))
    return problems


def valid_plot_problems(path: str) -> list[tuple[str, str]]:
    return [
        (objective, optimiser)
        for objective, optimiser in valid_problems(path)
        if 'random' in optimiser and 'analytical' not in objective
    ]


def build_parameters(num_components: int) -> dict[str, Any]:
    return {
        'a': {
            'type': 'real',
            'initial': 0.0,
            'lbound': -2.0,
            'ubound': 2.0,
            'num_components': num_components,
        },
        'sum': {
            'type': 'computed',
            'expression': 'sum(abs(a))',
        }
    }


def build_problem(objective_path: str, optimiser_path: str) -> dict[str, Any]:
    objective = read_yaml(objective_path)
    optimiser = read_yaml(optimiser_path)
    base = {
        'parameters': build_parameters(4),
        'iters': 1 if 'botorch' in optimiser_path else 4,
    }
    problem = base | objective | optimiser
    return problem


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


def name(problem: tuple[str, str]) -> str:
    objective_path, optimiser_path = problem
    objective = os.path.basename(objective_path).replace('.yaml', '')
    optimiser = os.path.basename(optimiser_path).replace('.yaml', '')
    return f"{optimiser}_{objective}"


@pytest.mark.parametrize('problem', valid_problems('test/examples_combinations'), ids=name)
def test_combinations(problem: tuple[str, str]):
    objective_path, optimiser_path = problem
    config = build_problem(objective_path, optimiser_path)
    run_config(config)


@pytest.mark.parametrize('problem', valid_plot_problems('test/examples_combinations'), ids=name)
def test_combinations_plots(problem: tuple[str, str]):
    objective_path, optimiser_path = problem
    config = build_problem(objective_path, optimiser_path)
    with TemporaryDirectory() as tmpdir:
        with change_cwd(tmpdir):
            input_file = 'config.yaml'
            run_config(config, config_path=input_file)
            first_hash = get_first_hash(os.path.join('config', 'func_calls'))
            for kind in ('best', 'history', 'parameters'):
                piglot_plot_main([
                    kind,
                    input_file,
                    '--save_fig',
                    f'{kind}.png',
                ])
            for kind in ('history', 'parameters'):
                optional = ['--best'] if 'mo' not in objective_path else []
                piglot_plot_main([
                    kind,
                    input_file,
                    '--save_fig',
                    f'{kind}.png',
                    '--log',
                    '--time',
                ] + optional)
            piglot_plot_main([
                'case',
                input_file,
                first_hash,
                '--save_fig',
                'case.png',
            ])
            piglot_plot_main([
                'animation',
                input_file,
            ])
            if 'mo' in objective_path:
                piglot_plot_main([
                    'pareto',
                    input_file,
                    '--all',
                    '--log',
                    '--save_fig',
                    'pareto.png',
                ])
            else:
                piglot_plot_main([
                    'regret',
                    input_file,
                    '--save_fig',
                    'regret.png',
                    '--log',
                ])
                piglot_plot_main([
                    'regret',
                    input_file,
                    '--save_fig',
                    'regret.png',
                    '--log',
                    '--time',
                ])
