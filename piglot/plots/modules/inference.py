"""Module for Bayesian inference plots."""
from argparse import Namespace, ArgumentParser
from typing import Optional
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from piglot.data.dataset import RawDataset
from piglot.data.sampling import draw_function_samples, PathwiseSamplingModel, find_pathwise_optima
from piglot.data.surrogate import GPModel, SurrogateSettings
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.plots.module import PlottingModuleConfigFile
from piglot.utils.yaml_parser import ProblemConfig


class InferencePlot(PlottingModuleConfigFile):
    """Plotting module for Bayesian inference."""

    def __init__(self) -> None:
        super().__init__(
            name="inference",
            help_str="plot Bayesian inference results",
            description="Plot Bayesian inference results.",
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
            "--sampler",
            choices=["naive", "pathwise"],
            default="naive",
            help="Specify the sampling method to use.",
        )
        parser.add_argument(
            "--num_rounds",
            type=int,
            default=128,
            help="Number of sampling rounds",
        )
        parser.add_argument(
            "--num_samples_per_round",
            type=int,
            default=64,
            help="Number of function samples per round to sample.",
        )
        parser.add_argument(
            "--num_grid_points",
            type=int,
            default=1024,
            help="Number of grid points to use for sampling.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility.",
        )
        parser.add_argument(
            "--sample_from",
            choices=["mean", "objective"],
            default="mean",
            help="Specify whether to sample from the mean estimator or the objective.",
        )
        parser.add_argument(
            "--bins",
            type=int,
            default=20,
            help="Number of bins to use for the histograms.",
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
        if not isinstance(problem.optimiser, GenericOptimiser):
            raise ValueError("Can only run Bayesian inference with generic optimisers.")
        if problem.objective.is_multi_objective():
            raise ValueError("Can only run Bayesian inference with single-objective problems.")
        campaign = problem.optimiser.campaign

        # Load optimiser data
        campaign.load()
        model = campaign.get_model()

        # Build function samples for the model
        grids = []
        samples = []
        for i in tqdm(range(args.num_rounds)):
            if args.sampler == "naive":
                grid, sample = draw_function_samples(
                    model,
                    args.num_grid_points,
                    args.num_samples_per_round,
                    seed=args.seed + i if args.seed is not None else None,
                    observation_noise=(args.sample_from == "objective"),
                )
                # Find minimum for each sample
                indices = torch.argmin(sample, dim=1)
                for sample_num, idx in enumerate(indices):
                    grids.append(grid[idx, :])
                    samples.append(sample[sample_num, idx].item())
            else:
                path_model = PathwiseSamplingModel(
                    model,
                    args.num_samples_per_round,
                    observation_noise=(args.sample_from == "objective"),
                )
                bounds = torch.from_numpy(problem.settings.parameters.get_bounds().transpose())
                points, values = find_pathwise_optima(path_model, bounds)
                for i in range(points.shape[0]):
                    grids.append(points[i, :])
                    samples.append(values[i].item())
        grids = torch.stack(grids, dim=0)
        samples = torch.tensor(samples)

        # Build the histograms: values
        figures = []
        fig, ax = plt.subplots(layout='constrained')
        ax.hist(samples, bins=args.bins, density=True)
        ax.set_xlabel("Objective values")
        ax.set_ylabel("Probability")
        figures.append(fig)

        # Build the histograms: parameters
        for i in range(grids.shape[-1]):
            bounds = problem.settings.parameters[i].get_bounds()
            lbound, ubound = bounds[0, 0], bounds[0, 1]
            fig, ax = plt.subplots(layout='constrained')
            ax.hist(grids[..., i], bins=args.bins, density=True, range=(lbound, ubound))
            ax.set_xlabel(problem.settings.parameters[i].name)
            ax.set_ylabel("Probability")
            ax.set_xlim(lbound, ubound)
            figures.append(fig)

        return figures
