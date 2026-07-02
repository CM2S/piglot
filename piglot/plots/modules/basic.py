"""Basic plotting modules for piglot."""
from argparse import Namespace, ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.plots.module import PlottingModuleConfigFile
from piglot.utils.yaml_parser import ProblemConfig


class CasePlot(PlottingModuleConfigFile):
    """Plotting module for a single function call case."""

    def __init__(self) -> None:
        super().__init__(
            name="case",
            help_str="plot a given function call given the parameter hash",
            description="Plot a given function call given the parameter hash.",
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
            "case_hash",
            type=str,
            help="Hash of the parameter set to plot.",
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
        return problem.objective.plot_case(args.case_hash)


class BestCasePlot(PlottingModuleConfigFile):
    """Plotting module for the best function call case."""

    def __init__(self) -> None:
        super().__init__(
            name="best",
            help_str="plot the best function call case",
            description="Plot the best function call case.",
            allow_save_fig=True,
        )

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
        return problem.objective.plot_best()


class ParetoPlot(PlottingModuleConfigFile):
    """Plotting module for Pareto front."""

    def __init__(self) -> None:
        super().__init__(
            name="pareto",
            help_str="plot the Pareto front",
            description="Plot the Pareto front.",
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
            '--log',
            action='store_true',
            help="Plot in a log scale."
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help="Plot the both the Pareto front and the dominated points."
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
        # Initial sanity check
        if not problem.objective.is_multi_objective():
            raise ValueError("Pareto plotting can only be used for multi-objective problems.")
        if problem.objective.num_objectives() != 2:
            raise ValueError("Pareto plotting can only be used for 2-objective problems.")
        if not isinstance(problem.optimiser, GenericOptimiser):
            raise ValueError("Pareto plotting can only be used with data-driven optimisers.")

        # Load data from the previous run and fetch MO state
        problem.optimiser.campaign.load()
        mo_state = problem.optimiser.campaign.state.mo_state
        evaluations = np.array(
            [obs.result.obj_values.tolist() for obs in problem.optimiser.campaign.dataset.data]
        )
        if mo_state is None:
            raise ValueError("No multi-objective state data available for Pareto plotting.")
    
        fig, ax = plt.subplots(layout='constrained')
        ax.plot(
            mo_state.pareto_y.cpu()[:, 0],
            mo_state.pareto_y.cpu()[:, 1],
            label='Pareto front',
            ls='--',
        )
        ax.scatter(
            mo_state.ref_point.cpu()[0],
            mo_state.ref_point.cpu()[1],
            label='Reference point',
            marker='x',
        )
        if args.all:
            ax.scatter(evaluations[:, 0], evaluations[:, 1], label='Dominated points')
        if args.log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.legend()
        return [fig]
