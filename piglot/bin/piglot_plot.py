"""Driving script for piglot's plotting utilities."""
from typing import List
import os
import time
import argparse
from tempfile import TemporaryDirectory
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from PIL import Image
import torch
from piglot.parameter import read_parameters
from piglot.objectives import read_objective
from piglot.utils.surrogate import get_model, optmise_posterior_mean
from piglot.utils.yaml_parser import parse_config_file


def cumulative_regret(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Compute the cumulative regret of a given function

    Parameters
    ----------
    values : np.ndarray
        Function values
    x_grid : np.ndarray
        Grid of x values

    Returns
    -------
    np.ndarray
        Cumulative regret
    """
    best = np.min(values)
    regret = [trapezoid(values[:i+1] - best, x_grid[:i+1]) for i in range(len(values))]
    return np.array(regret)


def pick_from_best(values: np.ndarray, variable: np.ndarray) -> np.ndarray:
    """Return the associated variable value associated with the current best objective

    Parameters
    ----------
    values : np.ndarray
        Objective values
    variable : np.ndarray
        Variable to output

    Returns
    -------
    np.ndarray
        Current best of the variable
    """
    return np.array([variable[np.argmin(values[:i+1])] for i in range(len(values))])


def plot_case(args):
    """Driver for single-case plotting.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    # Build piglot problem
    config = parse_config_file(args.config)
    objective = read_objective(config["objective"], read_parameters(config), config["output"])
    figs = objective.plot_case(args.case_hash)
    if args.save_fig:
        if len(figs) == 1:
            figs[0].savefig(args.save_fig)
        else:
            base, ext = os.path.splitext(args.save_fig)
            for i, fig in enumerate(figs):
                fig.savefig(f'{base}_{i}{ext}')
    else:
        plt.show()
    plt.close()


def plot_current(args):
    """Driver for plotting the currently running case.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    plt.ion()
    config = parse_config_file(args.config)
    objective = read_objective(config["objective"], read_parameters(config), config["output"])
    plots = objective.plot_current()
    while True:
        for plot in plots:
            plot.update()
        time.sleep(args.rate)


def plot_best(args):
    """Driver for plotting the best function call so far.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    # Build piglot problem
    config = parse_config_file(args.config)
    objective = read_objective(config["objective"], read_parameters(config), config["output"])
    figs = objective.plot_best()
    if args.save_fig:
        if len(figs) == 1:
            figs[0].savefig(args.save_fig)
        else:
            base, ext = os.path.splitext(args.save_fig)
            for i, fig in enumerate(figs):
                fig.savefig(f'{base}_{i}{ext}')
    else:
        plt.show()
    plt.close()


def plot_history(args):
    """Driver for plotting the history of a case.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    # Build piglot problem
    config = parse_config_file(args.config)
    objective = read_objective(config["objective"], read_parameters(config), config["output"])
    data = objective.get_history()
    fig, axis = plt.subplots()
    for name, data_dict in data.items():
        times = data_dict['time']
        values = data_dict['values']
        x_values = times if args.time else np.arange(len(times))
        y_values = pick_from_best(values, values) if args.best else values
        axis.plot(x_values, y_values, label=name)
    if args.log:
        axis.set_yscale('log')
    axis.set_xlabel("Elapsed time /s" if args.time else "Function calls")
    axis.set_ylabel("Objective")
    if len(data) > 1:
        axis.legend()
    axis.grid()
    fig.tight_layout()
    if args.save_fig:
        fig.savefig(args.save_fig)
    else:
        plt.show()
    plt.close()


def plot_parameters(args):
    """Driver for plotting the history of the parameters.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    # Build piglot problem
    config = parse_config_file(args.config)
    parameters = read_parameters(config)
    objective = read_objective(config["objective"], parameters, config["output"])
    data = objective.get_history()
    fig, axes = plt.subplots(nrows=len(parameters), sharex=True, squeeze=False)
    for idx, param in enumerate(parameters):
        axis = axes[idx, 0]
        for name, data_dict in data.items():
            times = data_dict['time']
            values = data_dict['values']
            param_values = data_dict['params']
            x_values = times if args.time else np.arange(len(times))
            params = param_values[:, idx]
            y_values = pick_from_best(values, params) if args.best else params
            axis.plot(x_values, y_values, label=f'{param.name}: {name}')
        if idx == len(parameters) - 1:
            axis.set_xlabel("Elapsed time /s" if args.time else "Function calls")
        axis.set_ylabel(param.name)
        axis.set_ylim(ymin=param.lbound, ymax=param.ubound)
        if len(data.items()) > 1:
            axis.legend()
        axis.grid()
    fig.tight_layout()
    if args.save_fig:
        fig.savefig(args.save_fig)
    else:
        plt.show()
    plt.close()


def plot_regret(args):
    """Driver for plotting the cumulative regret of a case.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    # Build piglot problem
    config = parse_config_file(args.config)
    objective = read_objective(config["objective"], read_parameters(config), config["output"])
    data = objective.get_history()
    fig, axis = plt.subplots()
    for name, data_dict in data.items():
        times = data_dict['time']
        values = data_dict['values']
        x_axis = times if args.time else np.arange(len(times))
        axis.plot(x_axis, cumulative_regret(values, x_axis), label=name)
    if args.log:
        axis.set_yscale('log')
    axis.set_xlabel("Elapsed time /s" if args.time else "Function calls")
    axis.set_ylabel("Cumulative regret")
    axis.legend()
    axis.grid()
    fig.tight_layout()
    if args.save_fig:
        fig.savefig(args.save_fig)
    else:
        plt.show()
    plt.close()


def plot_animation(args):
    """Driver for animating the response history.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    # Build piglot problem
    config = parse_config_file(args.config)
    objective = read_objective(config["objective"], read_parameters(config), config["output"])
    data = objective.get_history()
    # Hacky: we start by just plotting the first case to infer the number of plots per frame
    options = {'reference_limits': True}
    data_first = data[list(data.keys())[0]]
    first_figs = objective.plot_case(data_first['hashes'][0], {'reference_limits': True})
    num_plots = len(first_figs)
    for fig in first_figs:
        plt.close(fig)
    # We then plot and save each frame
    with TemporaryDirectory() as tmp_dir:
        # Export all frames to the temporary directory
        files = {}
        for obj_name, data_dict in data.items():
            param_hashes = data_dict['hashes']
            files[obj_name] = [[] for _ in range(num_plots)]
            for frame, param_hash in enumerate(param_hashes):
                options = {
                    'reference_limits': True,
                    'append_title': f'Iteration {frame}'
                }
                figs = objective.plot_case(param_hash, options)
                for idx, fig in enumerate(figs):
                    filename = os.path.join(tmp_dir, f'{obj_name}_{idx}-{frame}.png')
                    files[obj_name][idx].append(filename)
                    fig.savefig(filename)
                    plt.close(fig)
        # Build the final GIFs
        for obj_name, file_list in files.items():
            for idx in range(num_plots):
                images = [Image.open(filename) for filename in file_list[idx]]
                images[0].save(
                    os.path.join(config["output"], f'{obj_name}-{idx}.gif'),
                    format='GIF',
                    save_all=True,
                    append_images=images[1:],
                    duration=200,
                    loop=0,
                )


def plot_gp(args):
    """Driver for plotting a Gaussian process regression for the available data.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    # Build piglot problem
    config = parse_config_file(args.config)
    parameters = read_parameters(config)
    if len(parameters) != 1:
        raise ValueError("Can only plot a Gaussian process regression for a single parameter.")
    objective = read_objective(config["objective"], parameters, config["output"])
    data = objective.get_history()
    fig, axis = plt.subplots()
    x_min = min(par.lbound for par in parameters)
    x_max = max(par.ubound for par in parameters)
    x = torch.linspace(x_min, x_max, 1000)
    for name, data_dict in data.items():
        max_calls = len(data_dict['values'])
        if args.max_calls:
            max_calls = min(max_calls, args.max_calls)
        values = data_dict['values'][:max_calls]
        param_values = data_dict['params'][:max_calls]
        variances = data_dict['variances'][:max_calls] if 'variances' in data_dict else None
        model = get_model(param_values, values, variances)
        with torch.no_grad():
            posterior = model.posterior(x.unsqueeze(1))
            mean = posterior.mean.squeeze()
            variance = posterior.variance.squeeze()
        axis.plot(x, mean, label=f'{name}: Mean')
        axis.fill_between(x, mean - variance.sqrt(), mean + variance.sqrt(), alpha=0.5,
                          label=f'{name}: Std. dev.')
        if 'variances' in data_dict:
            axis.errorbar(param_values, values, yerr=np.sqrt(variances), color='black', fmt='o')
        else:
            axis.scatter(param_values, values, color='black')
    axis.set_xlim(x_min, x_max)
    axis.legend()
    axis.grid()
    axis.set_xlabel(parameters[0].name)
    axis.set_ylabel("Objective")
    fig.tight_layout()
    if args.save_fig:
        fig.savefig(args.save_fig)
    else:
        plt.show()
    plt.close()


def plot_pareto(args):
    """Driver for plotting the Pareto front for a given multi-objective optimisation problem.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    config = parse_config_file(args.config)
    objective = read_objective(config["objective"], read_parameters(config), config["output"])
    if objective.num_objectives != 2:
        raise ValueError("Can only plot the Pareto front for a two-objective optimisation problem.")
    data = objective.get_history()
    fig, ax = plt.subplots()
    # Read all the points and variances
    total_points = np.array([entry['values'] for entry in data.values()]).T
    variances = np.array([entry['variances'] for entry in data.values() if 'variances' in entry]).T
    has_variance = variances.size > 0
    # Separate the dominated points
    dominated = []
    nondominated = []
    pareto_data = pd.read_table(os.path.join(config["output"], 'pareto_front')).to_numpy()
    pareto = pareto_data[:, :objective.num_objectives]
    for i, point in enumerate(total_points):
        if np.isclose(point, pareto).all(axis=1).any():
            nondominated.append((point, variances[i, :] if has_variance else None))
        else:
            dominated.append((point, variances[i, :] if has_variance else None))
    # Sort the Pareto front by the first objective
    nondominated = sorted(nondominated, key=lambda x: x[0][0])
    # Plot the points
    if has_variance:
        ax.errorbar(
            [point[0][0] for point in nondominated],
            [point[0][1] for point in nondominated],
            xerr=np.sqrt([point[1][0] for point in nondominated]),
            yerr=np.sqrt([point[1][0] for point in nondominated]),
            c='r',
            fmt='-o',
            label='Pareto front',
        )
        if args.all:
            ax.errorbar(
                [point[0][0] for point in dominated],
                [point[0][1] for point in dominated],
                xerr=np.sqrt([point[1][0] for point in dominated]),
                yerr=np.sqrt([point[1][0] for point in dominated]),
                c='k',
                fmt='o',
                label='Dominated points',
            )
    else:
        ax.plot(
            [point[0][0] for point in nondominated],
            [point[0][1] for point in nondominated],
            c='r',
            ls='--',
            marker='o',
            label='Pareto front',
        )
        if args.all:
            ax.scatter(
                [point[0][0] for point in dominated],
                [point[0][1] for point in dominated],
                c='k',
                label='Dominated points',
            )
    if args.log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    if args.save_fig:
        fig.savefig(args.save_fig)
    else:
        plt.show()


def make_surrogate(args):
    """Driver for training a surrogate model for the available data.

    Parameters
    ----------
    args : dict
        Passed arguments.
    """
    # Build piglot problem
    config = parse_config_file(args.config)
    parameters = read_parameters(config)
    objective = read_objective(config["objective"], parameters, config["output"])
    data = objective.get_history()
    bounds = np.array([[par.lbound, par.ubound] for par in parameters]).T
    for name, data_dict in data.items():
        param_values = data_dict['params']
        values = data_dict['values']
        variances = data_dict['variances'] if 'variances' in data_dict else None
        if len(values) < 2:
            continue
        output_data = np.empty((len(values) - 2, bounds.shape[1]))
        for i in tqdm(range(2, len(values))):
            sliced_params = param_values[:i, :]
            sliced_values = values[:i]
            sliced_variances = variances[:i] if variances is not None else None
            model = get_model(sliced_params, sliced_values, sliced_variances)
            output_data[i - 2, :] = optmise_posterior_mean(model, bounds)
        np.savetxt(os.path.join(config['output'], f'{name}.csv'), output_data)


def main(passed_args: List[str] = None):
    """Entry point for this script."""
    # Global argument parser settings
    parser = argparse.ArgumentParser(
        prog='piglot-plot',
        description='Plotting utility for piglot',
    )
    subparsers = parser.add_subparsers(
        title='Available modes',
        description=("To get additional information for a given command and available "
                     "options, run piglot-plot command --help"),
    )

    # Standard plotting methods
    sp_case = subparsers.add_parser(
        'case',
        help='plot a single case file',
        description='Plot a single case file.',
    )
    sp_case.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_case.add_argument(
        'case_hash',
        type=str,
        help=("Hash of the case to plot."),
    )
    sp_case.add_argument(
        '--save_fig',
        default=None,
        type=str,
        help=("Path to save the generated figure. If used, graphical output is skipped."),
    )
    sp_case.set_defaults(func=plot_case)

    # Plotting current run
    sp_current = subparsers.add_parser(
        'current',
        help='plot the currently running function call',
        description=("Plot the currently running function call. This must be executed in "
                     "the same path as the running piglot instance."),
    )
    sp_current.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_current.add_argument(
        '--rate',
        default=1.0,
        type=float,
        help='Plot update rate, in seconds (defaults to 1.0).',
    )
    sp_current.set_defaults(func=plot_current)

    # Plotting best run so far
    sp_best = subparsers.add_parser(
        'best',
        help='plot the best response so far',
        description=("Plot the best response so far. This must be executed in "
                     "the same path as the running piglot instance."),
    )
    sp_best.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_best.add_argument(
        '--save_fig',
        default=None,
        type=str,
        help=("Path to save the generated figure. If used, graphical output is skipped."),
    )
    sp_best.set_defaults(func=plot_best)

    # Plotting the objective history
    sp_hist = subparsers.add_parser(
        'history',
        help='plot the objective history',
        description=("Plot the objective history. This must be executed in "
                     "the same path as the running piglot instance."),
    )
    sp_hist.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_hist.add_argument(
        '--save_fig',
        default=None,
        type=str,
        help=("Path to save the generated figure. If used, graphical output is skipped."),
    )
    sp_hist.add_argument(
        '--time',
        action='store_true',
        help=("Whether to plot the objective history w.r.t. the elapsed time."),
    )
    sp_hist.add_argument(
        '--log',
        action='store_true',
        help=("Plot with a log scale."),
    )
    sp_hist.add_argument(
        '--best',
        action='store_true',
        help=("Plot the best case so far."),
    )
    sp_hist.set_defaults(func=plot_history)

    # Plotting the parameter history
    sp_param = subparsers.add_parser(
        'parameters',
        help='plot the parameter history',
        description=("Plot the parameter history. This must be "
                     "executed in the same path as the running piglot instance."),
    )
    sp_param.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_param.add_argument(
        '--save_fig',
        default=None,
        type=str,
        help=("Path to save the generated figure. If used, graphical output is skipped."),
    )
    sp_param.add_argument(
        '--time',
        action='store_true',
        help="Plot w.r.t. the elapsed time"
    )
    sp_param.add_argument(
        '--log',
        action='store_true',
        help="Plot with a log scale"
    )
    sp_param.add_argument(
        '--best',
        action='store_true',
        help="Plot the best case so far.",
    )
    sp_param.set_defaults(func=plot_parameters)

    # Plotting cummulative regret measures
    sp_regret = subparsers.add_parser(
        'regret',
        help='plot the cummulative regret of a case',
        description=("Plot the cummulative regret of a case. This must be "
                     "executed in the same path as the running piglot instance."),
    )
    sp_regret.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_regret.add_argument(
        '--save_fig',
        default=None,
        type=str,
        help=("Path to save the generated figure. If used, graphical output is skipped."),
    )
    sp_regret.add_argument(
        '--time',
        action='store_true',
        help="Plot w.r.t. the elapsed time"
    )
    sp_regret.add_argument(
        '--log',
        action='store_true',
        help="Plot with a log scale"
    )
    sp_regret.set_defaults(func=plot_regret)

    # Animate the solution history
    sp_animation = subparsers.add_parser(
        'animation',
        help='animate the case histories',
        description=("Animate the case histories. This must be "
                     "executed in the same path as the running piglot instance."),
    )
    sp_animation.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_animation.set_defaults(func=plot_animation)

    # Gaussian process plotting
    sp_gp = subparsers.add_parser(
        'gp',
        help='plot a Gaussian process regression for the available data',
        description=("Plot a Gaussian process regression for the available data. This must be "
                     "executed in the same path as the running piglot instance."),
    )
    sp_gp.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_gp.add_argument(
        '--save_fig',
        default=None,
        type=str,
        help=("Path to save the generated figure. If used, graphical output is skipped."),
    )
    sp_gp.add_argument(
        '--max_calls',
        default=None,
        type=int,
        help=("Max number of calls to plot."),
    )
    sp_gp.set_defaults(func=plot_gp)

    # Pareto front plotting
    sp_pareto = subparsers.add_parser(
        'pareto',
        help='plot the Pareto front for a given multi-objective optimisation problem',
        description=("Plot the Pareto front for a given multi-objective optimisation problem. This "
                     "must be executed in the same path as the running piglot instance."),
    )
    sp_pareto.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_pareto.add_argument(
        '--save_fig',
        default=None,
        type=str,
        help=("Path to save the generated figure. If used, graphical output is skipped."),
    )
    sp_pareto.add_argument(
        '--log',
        action='store_true',
        help="Plot in a log scale."
    )
    sp_pareto.add_argument(
        '--all',
        action='store_true',
        help="Plot the both the Pareto front and the dominated points."
    )
    sp_pareto.set_defaults(func=plot_pareto)

    # Surrogate model training
    sp_surrogate = subparsers.add_parser(
        'surrogate',
        help='train a surrogate model for the available data',
        description=("Train a surrogate model for the available data. This must be "
                     "executed in the same path as the running piglot instance."),
    )
    sp_surrogate.add_argument(
        'config',
        type=str,
        help="Path for the used or generated configuration file.",
    )
    sp_surrogate.set_defaults(func=make_surrogate)

    args = parser.parse_args() if passed_args is None else parser.parse_args(passed_args)
    args.func(args)


if __name__ == '__main__':
    main()
