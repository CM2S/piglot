"""Driving script for piglot's plotting utilities."""
import argparse
from piglot.plots.module import PlottingModuleConfigFile
from piglot.plots.modules.basic import CasePlot, BestCasePlot, ParetoPlot
from piglot.plots.modules.gp import GPPlot, CompositeGPPlot
from piglot.plots.modules.inference import InferencePlot, CornerPlot
from piglot.plots.modules.history import (
    ObjectiveHistoryPlot,
    ParameterHistoryPlot,
    RegretHistoryPlot,
    AnimationPlot,
)
from piglot.plots.modules.response_samples import ResponseSamplePlot


AVAILABLE_PLOTTING_MODULES: list[type[PlottingModuleConfigFile]] = [
    CasePlot,
    BestCasePlot,
    ParetoPlot,
    GPPlot,
    CompositeGPPlot,
    ObjectiveHistoryPlot,
    ParameterHistoryPlot,
    RegretHistoryPlot,
    AnimationPlot,
    InferencePlot,
    CornerPlot,
    ResponseSamplePlot,
]


def main(passed_args: list[str] = None):
    """Entry point for the plotting utility.

    Parameters
    ----------
    passed_args : list[str], optional
        List of arguments to parse. If None, the arguments will be parsed from the command line.
    """

    # Global argument parser settings
    parser = argparse.ArgumentParser(
        prog='piglot-plot',
        description='Plotting utility for piglot',
    )
    subparsers = parser.add_subparsers(
        title='Available modes',
        description=(
            "To get additional information for a given command and available options, run "
            "piglot-plot command --help"
        ),
    )

    # Inject the available plotting modules into the argument parser
    for module_cls in AVAILABLE_PLOTTING_MODULES:
        module = module_cls()

        # Set up the subparser for this module
        module_parser = subparsers.add_parser(
            module.name,
            help=module.help,
            description=module.description,
        )

        # Set up the module-specific arguments and function to call
        module_parser = module.setup_parser(module_parser)
        module_parser.set_defaults(func=module.plot)

    # Parse the arguments and call the appropriate plotting function
    args = parser.parse_args() if passed_args is None else parser.parse_args(passed_args)
    args.func(args)


if __name__ == '__main__':
    main()
