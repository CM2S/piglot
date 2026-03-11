"""Module for Gaussian process regression plots."""
from argparse import Namespace, ArgumentParser
from matplotlib.figure import Figure
from piglot.plots.module import PlottingModuleConfigFile
from piglot.utils.yaml_parser import ProblemConfig


class GPPlot(PlottingModuleConfigFile):
    """Plotting module for Gaussian process regression."""

    def __init__(self) -> None:
        super().__init__(
            name="gp",
            help_str="plot Gaussian process regression results",
            description="Plot Gaussian process regression results.",
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
            "--max_calls",
            type=int,
            default=None,
            help="Maximum number of function calls to plot.",
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
        raise NotImplementedError("GPPlot is not implemented yet.")


class MCGPPlot(PlottingModuleConfigFile):
    """Plotting module for Gaussian process regression (with Monte Carlo sampling)."""

    def __init__(self) -> None:
        super().__init__(
            name="mcgp",
            help_str="plot Gaussian process regression results (with Monte Carlo sampling)",
            description="Plot Gaussian process regression results (with Monte Carlo sampling).",
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
            "--max_calls",
            type=int,
            default=None,
            help="Maximum number of function calls to plot.",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=512,
            help="Number of Monte Carlo samples to use.",
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
        raise NotImplementedError("MCGPPlot is not implemented yet.")
