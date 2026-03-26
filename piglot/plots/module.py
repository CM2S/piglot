"""Definition of plotting modules for piglot objectives."""
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from piglot.utils.yaml_parser import ProblemConfig, build_problem


class PlottingModule(ABC):
    """Abstract base class for plotting modules in piglot objectives."""

    def __init__(self, name: str, help_str: str, description: str) -> None:
        self.name = name
        self.help = help_str
        self.description = description

    @abstractmethod
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

    @abstractmethod
    def plot(self, args: Namespace) -> None:
        """Generate the plot based on the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments parsed from the command line.
        """


class PlottingModuleConfigFile(PlottingModule):
    """Abstract base class for plotting modules that require a configuration file."""

    def __init__(
        self,
        name: str,
        help_str: str,
        description: str,
        allow_save_fig: bool = False,
        show_fig: bool = True,
    ) -> None:
        super().__init__(name, help_str, description)
        self.allow_save_fig = allow_save_fig
        self.show_fig = show_fig

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
        parser.add_argument(
            'config',
            type=str,
            help="Path for the configuration file.",
        )
        if self.allow_save_fig:
            parser.add_argument(
                '--save_fig',
                type=str,
                default=None,
                help="Path to save the generated figure. If used, graphical output is skipped.",
            )
        return parser

    def plot(self, args: Namespace) -> None:
        """Generate the plot based on the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments parsed from the command line.
        """
        # Validate the configuration file path
        config = args.config
        if not os.path.isfile(config):
            raise FileNotFoundError(f"Configuration file {config} not found.")

        # Read the problem configuration from the file and generate the figure(s)
        problem, _ = build_problem(config)
        figures = self.plot_run(problem, args)

        # Save the figure(s) if required
        if self.allow_save_fig and args.save_fig is not None:
            # If multiple figures are generated, save them with an index suffix
            base, ext = os.path.splitext(args.save_fig)
            for i, fig in enumerate(figures):
                fig_path = args.save_fig if len(figures) == 1 else f"{base}_{i}{ext}"
                fig.savefig(fig_path)

        # Show the figure(s) if required
        elif self.show_fig:
            plt.show()

    @abstractmethod
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
