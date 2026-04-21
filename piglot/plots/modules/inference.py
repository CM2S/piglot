"""Module for Bayesian inference plots."""
from argparse import Namespace, ArgumentParser
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from piglot.data.sampling import draw_function_samples, PathwiseSamplingModel, find_pathwise_optima
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.plots.module import PlottingModuleConfigFile
from piglot.plots.modules.gp import GPPlot, CompositeGPPlot
from piglot.utils.tabular import TabularFile, TabularFloatColumn
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
            default=None,
            help=(
                "Number of bins to use for the histograms. "
                "If None, the number of bins will be determined automatically.",
            ),
        )
        parser.add_argument(
            "--save_samples",
            type=str,
            default=None,
            help="Path to save the sampled data.",
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

        # Save the sampled data if a path is provided
        if args.save_samples is not None:
            names = problem.settings.parameters.get_scalar_names()
            columns = [
                TabularFloatColumn("Objective", 16),
                *[TabularFloatColumn(name, 16) for name in names],
            ]
            tabular_file = TabularFile(args.save_samples, columns)
            tabular_file.prepare()
            for i in range(grids.shape[0]):
                param_values = grids[i, :].tolist()
                values = problem.settings.parameters.to_values(np.array(param_values))
                tabular_file.write_row([samples[i].item()] + list(values.scalar_values.values()))

        # Set up number of bins
        def num_bins(data: torch.Tensor, range: tuple[float, float] = None) -> int:
            if args.bins is not None:
                return args.bins
            if range is None:
                range = (torch.min(data).item(), torch.max(data).item())
            # Freedman–Diaconis rule
            h = 2 * (torch.quantile(data, 0.75) - torch.quantile(data, 0.25)) * len(data) ** (-1/3)
            return min(128, int((range[1] - range[0]) / h))

        # Build the histograms: values
        figures = []
        fig, ax = plt.subplots(layout='constrained')
        ax.hist(samples, bins=num_bins(samples), density=True)
        ax.set_xlabel("Objective values")
        ax.set_ylabel("Probability")
        figures.append(fig)

        # Build the histograms: parameters
        param_names = problem.settings.parameters.get_scalar_names()
        bounds = problem.settings.parameters.get_bounds()
        for i in range(grids.shape[-1]):
            lbound, ubound = bounds[i, 0], bounds[i, 1]
            fig, ax = plt.subplots(layout='constrained')
            ax.hist(
                grids[..., i],
                bins=num_bins(grids[..., i], range=(lbound, ubound)),
                density=True,
                range=(lbound, ubound),
            )
            ax.set_xlabel(param_names[i])
            ax.set_ylabel("Probability")
            ax.set_xlim(lbound, ubound)
            figures.append(fig)

        # For 1D problems, make the joint plot with the GP and the histograms
        if grids.shape[-1] == 1:
            fig, axes = plt.subplots(
                nrows=2, ncols=2, sharex='col', sharey='row', layout='constrained'
            )

            # Plot the GP
            gp_plot = CompositeGPPlot() if problem.objective.is_composite() else GPPlot()
            parser = ArgumentParser()
            new_args = gp_plot.setup_parser(parser).parse_args([args.config])
            gp_plot.single_plot(axes[0, :1], problem, new_args)
            axes[0, 0].set_xlabel(None)
            axes[0, 0].set_title(None)
            axes[0, 0].set_ylabel("Objective value")

            # Find the bounds for each axis based on the GP plot
            param_lbound, param_ubound = axes[0, 0].get_xlim()
            obj_lbound, obj_ubound = axes[0, 0].get_ylim()

            # Plot parameter distribution
            bounds = problem.settings.parameters[0].get_bounds()
            lbound, ubound = bounds[0, 0], bounds[0, 1]
            axes[1, 0].hist(
                grids[..., 0],
                bins=num_bins(grids[..., 0], range=(param_lbound, param_ubound)),
                density=True,
                range=(param_lbound, param_ubound),
            )
            axes[1, 0].set_xlabel(problem.settings.parameters[0].name)
            axes[1, 0].set_ylabel("Probability")

            # Plot objective distribution (rotated)
            axes[0, 1].hist(
                samples,
                bins=num_bins(samples, range=(obj_lbound, obj_ubound)),
                density=True,
                orientation='horizontal',
                range=(obj_lbound, obj_ubound),
            )
            axes[0, 1].set_xlabel("Probability")

            # Clean up the unused axis and the temporary GP figure
            axes[1, 1].axis('off')
            figures.append(fig)

        return figures
