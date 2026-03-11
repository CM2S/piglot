"""Module for plotting optimisation histories."""
from argparse import Namespace, ArgumentParser
import os
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.integrate import trapezoid
from piglot.plots.module import PlottingModuleConfigFile
from piglot.utils.yaml_parser import ProblemConfig


class ObjectiveHistoryPlot(PlottingModuleConfigFile):
    """Plotting module for the objective function history."""

    def __init__(self) -> None:
        super().__init__(
            name="history",
            help_str="plot the objective function history",
            description="Plot the objective function history.",
            allow_save_fig=True,
        )

    def setup_parser(self, parser: ArgumentParser) -> ArgumentParser:
        """Set up the argument parser for this plotting module.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to set up.

        Returns
        -------
        ArgumentParser
            The argument parser with the added arguments.
        """
        parser = super().setup_parser(parser)
        parser.add_argument(
            "--time",
            action="store_true",
            help="Plot objective history against elapsed time",
        )
        parser.add_argument(
            "--log",
            action="store_true",
            help="Plot objective history on a logarithmic scale",
        )
        parser.add_argument(
            "--best",
            action="store_true",
            help="Plot the best objective value found so far at each iteration",
        )
        return parser

    def plot_run(self, problem: ProblemConfig, args: Namespace) -> list[Figure]:
        """Generate the plot based on the provided config file and arguments.

        Parameters
        ----------
        problem : ProblemConfig
            The optimisation problem built from the configuration file.
        args : Namespace
            The arguments parsed from the command line.

        Returns
        -------
        list[Figure]
            A list of generated figures.
        """
        # Load the history data from the problem's objective
        data = problem.objective.read_func_calls()

        # Set up the x axis
        if args.time:
            x = data.start_times + data.run_times
            x_label = "Elapsed Time (s)"
        else:
            x = np.arange(0, len(data.hashes))
            x_label = "Function evaluations"

        # Create the figure and axis
        nrows = problem.objective.num_objectives() if problem.objective.is_multi_objective() else 1
        fig, axes = plt.subplots(
            nrows=nrows,
            layout="constrained",
            sharex=True,
        )

        # Set up plotting function
        def uplot(ax: plt.Axes, y: np.ndarray, ylabel: str) -> None:
            if args.best:
                y = np.minimum.accumulate(y)
            ax.plot(x, y)
            ax.set_ylabel(ylabel)
            if args.log:
                ax.set_yscale("log")

        # Plot the objective history for each objective
        if problem.objective.is_multi_objective():
            for i in range(problem.objective.num_objectives()):
                uplot(axes[i], data.obj_values[:, i], f"Objective {i + 1}")
            axes[-1].set_xlabel(x_label)
        else:
            y = data.obj_values if problem.objective.scalarisation is None else data.scalar_values
            uplot(axes, y, "Objective Value")
            axes.set_xlabel(x_label)

        return [fig]


class ParameterHistoryPlot(PlottingModuleConfigFile):
    """Plotting module for the parameter history."""

    def __init__(self) -> None:
        super().__init__(
            name="parameters",
            help_str="plot the parameter history",
            description="Plot the parameter history.",
            allow_save_fig=True,
        )

    def setup_parser(self, parser: ArgumentParser) -> ArgumentParser:
        """Set up the argument parser for this plotting module.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to set up.

        Returns
        -------
        ArgumentParser
            The argument parser with the added arguments.
        """
        parser = super().setup_parser(parser)
        parser.add_argument(
            "--time",
            action="store_true",
            help="Plot parameter history against elapsed time",
        )
        parser.add_argument(
            "--log",
            action="store_true",
            help="Plot regret history on a logarithmic scale",
        )
        parser.add_argument(
            "--best",
            action="store_true",
            help="Plot the best parameter value found so far at each iteration",
        )
        return parser

    def plot_run(self, problem: ProblemConfig, args: Namespace) -> list[Figure]:
        """Generate the plot based on the provided config file and arguments.

        Parameters
        ----------
        problem : ProblemConfig
            The optimisation problem built from the configuration file.
        args : Namespace
            The arguments parsed from the command line.

        Returns
        -------
        list[Figure]
            A list of generated figures.
        """
        # Sanitise best: we cannot plot the best parameter values in multi-objective mode
        if args.best and problem.objective.is_multi_objective():
            raise ValueError("Cannot plot best parameter values for multi-objective problems.")

        # Load the history data from the problem's objective
        data = problem.objective.read_func_calls()

        # Set up the x axis
        if args.time:
            x = data.start_times + data.run_times
            x_label = "Elapsed Time (s)"
        else:
            x = np.arange(0, len(data.hashes))
            x_label = "Function evaluations"

        # Under best mode, find the indices to plot
        idx = np.arange(len(x))
        if args.best:
            obj = data.obj_values if problem.objective.scalarisation is None else data.scalar_values
            for i in range(len(obj)):
                idx[i] = np.argmin(obj[:i+1])

        # Create the figure and axis
        nrows = len(problem.settings.parameters)
        fig, axes = plt.subplots(
            nrows=nrows,
            layout="constrained",
            sharex=True,
            squeeze=False,
        )
        axes = axes.flatten()

        # Plot for each parameter
        for i, param in enumerate(problem.settings.parameters):
            y = data.params[idx, i]
            axes[i].plot(x, y)
            axes[i].set_ylabel(param.name)
            if args.log:
                axes[i].set_yscale("log")
        axes[-1].set_xlabel(x_label)

        return [fig]


class RegretHistoryPlot(PlottingModuleConfigFile):
    """Plotting module for the regret history."""

    def __init__(self) -> None:
        super().__init__(
            name="regret",
            help_str="plot the regret history",
            description="Plot the regret history.",
            allow_save_fig=True,
        )

    def setup_parser(self, parser: ArgumentParser) -> ArgumentParser:
        """Set up the argument parser for this plotting module.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to set up.

        Returns
        -------
        ArgumentParser
            The argument parser with the added arguments.
        """
        parser = super().setup_parser(parser)
        parser.add_argument(
            "--time",
            action="store_true",
            help="Plot regret history against elapsed time",
        )
        parser.add_argument(
            "--log",
            action="store_true",
            help="Plot regret history on a logarithmic scale",
        )
        return parser

    def plot_run(self, problem: ProblemConfig, args: Namespace) -> list[Figure]:
        """Generate the plot based on the provided config file and arguments.

        Parameters
        ----------
        problem : ProblemConfig
            The optimisation problem built from the configuration file.
        args : Namespace
            The arguments parsed from the command line.

        Returns
        -------
        list[Figure]
            A list of generated figures.
        """
        # Sanitise objective: we can only plot regret for single-objective problems
        if problem.objective.is_multi_objective():
            raise ValueError("Cannot plot regret history for multi-objective problems.")

        # Load the history data from the problem's objective
        data = problem.objective.read_func_calls()

        # Set up the x axis
        if args.time:
            x = data.start_times + data.run_times
            x_label = "Elapsed Time (s)"
        else:
            x = np.arange(0, len(data.hashes))
            x_label = "Function evaluations"

        # Compute the regret values
        obj = data.obj_values if problem.objective.scalarisation is None else data.scalar_values
        best = np.min(obj)
        regret = np.array([trapezoid(obj[:i+1] - best, x[:i+1]) for i in range(len(obj))])

        # Create the figure and axis
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(x, regret)
        ax.set_ylabel("Cumulative regret")
        if args.log:
            ax.set_yscale("log")
        ax.set_xlabel(x_label)

        return [fig]


class AnimationPlot(PlottingModuleConfigFile):
    """Plotting module for the animation of the optimisation history."""

    def __init__(self) -> None:
        super().__init__(
            name="animation",
            help_str="animate the optimisation history",
            description="Animate the optimisation history.",
            allow_save_fig=True,
            show_fig=False,
        )

    def setup_parser(self, parser: ArgumentParser) -> ArgumentParser:
        """Set up the argument parser for this plotting module.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to set up.

        Returns
        -------
        ArgumentParser
            The argument parser with the added arguments.
        """
        parser = super().setup_parser(parser)
        parser.add_argument(
            "--duration",
            type=int,
            default=200,
            help="Duration of each frame in milliseconds.",
        )
        return parser

    def plot_run(self, problem: ProblemConfig, args: Namespace) -> list[Figure]:
        """Generate the plot based on the provided config file and arguments.

        Parameters
        ----------
        problem : ProblemConfig
            The optimisation problem built from the configuration file.
        args : Namespace
            The arguments parsed from the command line.

        Returns
        -------
        list[Figure]
            A list of generated figures.
        """
        # Hacky: we start by plotting the best case to figure out the number of figures per plot
        first_figs = problem.objective.plot_best()
        num_plots = len(first_figs)
        for fig in first_figs:
            plt.close(fig)

        # Load the history data from the problem's objective
        data = problem.objective.read_func_calls()

        with TemporaryDirectory() as tmp_dir:
            # Export all frames to the temporary directory
            files: list[list[str]] = [[] for _ in range(num_plots)]
            for i, case_hash in enumerate(data.hashes):
                figs = problem.objective.plot_case(
                    case_hash, animation=True, append_title=f'Iteration {i}'
                )
                for idx, fig in enumerate(figs):
                    filename = os.path.join(tmp_dir, f'{idx}-{i}.png')
                    files[idx].append(filename)
                    fig.savefig(filename)
                    plt.close(fig)

            # Build the final GIFs
            output_dir = os.path.join(problem.settings.output_dir)
            for idx in range(num_plots):
                images = [Image.open(filename) for filename in files[idx]]
                images[0].save(
                    os.path.join(output_dir, f'Fig{idx}.gif'),
                    format='GIF',
                    save_all=True,
                    append_images=images[1:],
                    duration=args.duration,
                    loop=0,
                )

        return []
