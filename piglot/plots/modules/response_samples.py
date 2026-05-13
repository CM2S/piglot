"""Module for plotting response samples from compositional dataset."""
from argparse import Namespace, ArgumentParser
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from piglot.data.sampling import draw_function_samples, PathwiseSamplingModel, find_pathwise_optima
from piglot.objectives.response_objective import ResponseObjective, ResponseSingleObjective
from piglot.optimisers.generic.optimiser import GenericOptimiser
from piglot.plots.module import PlottingModuleConfigFile
from piglot.plots.modules.gp import GPPlot, CompositeGPPlot
from piglot.solver.solver import OutputResult
from piglot.utils.tabular import TabularFile, TabularFloatColumn
from piglot.utils.yaml_parser import ProblemConfig


class ResponseSamplePlot(PlottingModuleConfigFile):
    """Plotting module for response samples."""

    def __init__(self) -> None:
        super().__init__(
            name="responses",
            help_str="plot response samples from composite models",
            description="Plot response samples from composite models.",
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
            "--num_samples",
            type=int,
            default=8,
            help="Number of samples to draw.",
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
            "--point",
            type=float,
            nargs="+",
            default=None,
            help="Sample at this point's coordinates.",
        )
        parser.add_argument(
            "--shade",
            action="store_true",
            help="Whether to shade the area under the response curves.",
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
        if not problem.objective.is_composite():
            raise ValueError("Can only plot response samples for composite objectives.")
        if not isinstance(problem.optimiser, GenericOptimiser):
            raise ValueError("Can only plot response samples for generic optimisers.")
        if not isinstance(problem.objective, ResponseObjective):
            raise ValueError("Can only plot response samples for response-based objectives.")
        campaign = problem.optimiser.campaign
        objective = problem.objective

        # Load optimiser data
        campaign.load()
        model = campaign.get_model()

        # Figure out the point at which to sample
        if args.point is not None:
            num_params = problem.settings.parameters.num_optim_parameters()
            if len(args.point) != num_params:
                raise ValueError(
                    f"The number of coordinates provided ({len(args.point)}) does not match the "
                    f"number of optimisation parameters ({num_params})."
                )
            point = torch.tensor(args.point, dtype=torch.float64)
        else:
            # Find the best point according to the optimiser
            point = torch.from_numpy(campaign.state.best_params).to(dtype=torch.float64)

        # Draw latent response samples from the model and unpack them
        sample_shape = torch.Size([args.num_samples])
        with torch.no_grad():
            samples = model.latent_samples(
                point.view(1, -1),
                sample_shape=sample_shape,
                seed=args.seed,
                observation_noise=args.sample_from == "objective",
            )
        latent_responses = objective.concat_utility.split(samples)

        # Build the output for each objective
        figures = []
        for responses, objective in zip(latent_responses, objective.objectives):
            fig, ax = plt.subplots(layout='constrained')

            # Reconstruct the responses from the latent samples
            results = objective.responses_from_latent(responses)
            if objective.prediction_transform is not None:
                results = [objective.prediction_transform.untransform(r) for r in results]

            # Plot the reconstructed responses
            if args.shade:
                time = np.mean([r.get_time() for r in results], axis=0)
                mean = np.mean([r.get_data() for r in results], axis=0)
                lb = np.quantile([r.get_data() for r in results], 0.025, axis=0)
                ub = np.quantile([r.get_data() for r in results], 0.975, axis=0)
                objective.plot_raw_responses(ax, [OutputResult(time, mean)])
                ax.fill_between(time, lb, ub, alpha=0.3)
            else:
                objective.plot_raw_responses(ax, results)

            # Axes setup
            ax.set_xlim(
                np.min([r.get_time() for r in results]), np.max([r.get_time() for r in results])
            )
            ax.grid()
            figures.append(fig)

        return figures
