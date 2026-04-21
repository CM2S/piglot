"""Module for Gaussian process regression plots."""
from argparse import Namespace, ArgumentParser
from typing import Optional
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import torch
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from piglot.data.dataset import RawDataset
from piglot.data.surrogate import GPModel, SurrogateSettings
from piglot.optimisers.generic.optimiser import GenericOptimiser
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
        parser.add_argument(
            "--num_points",
            type=int,
            default=1024,
            help="Number of points to use for plotting.",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=512,
            help="Number of Monte Carlo samples to use for plotting.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility.",
        )
        parser.add_argument(
            "--num_func_samples",
            type=int,
            default=0,
            help="Number of function samples to use for plotting.",
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
        num_objectives = (
            problem.objective.num_problem.objectives()
            if problem.objective.is_multi_objective() else 1
        )
        figures = []
        axes = []
        for _ in range(num_objectives):
            fig, ax = plt.subplots(layout="constrained")
            figures.append(fig)
            axes.append(ax)
        self.single_plot(axes, problem, args)
        return figures

    @classmethod
    def single_plot(cls, axes: list[plt.Axes], problem: ProblemConfig, args: Namespace) -> None:
        """Generate a single plot for the given GP model.

        Parameters
        ----------
        axes : list[plt.Axes]
            The axes to plot on.
        problem : ProblemConfig
            The optimisation problem built from the configuration file.
        args : Namespace
            The arguments parsed from the command line.
        """

        # Build x-grid
        parameters = problem.settings.parameters
        x_min, x_max = parameters.get_bounds()[0]
        x = torch.linspace(x_min, x_max, args.num_points).view(-1, 1, 1)

        # Read function calls
        func_calls = problem.objective.read_func_calls()
        num_objectives = 1
        if func_calls.obj_values.ndim > 1:
            num_objectives = func_calls.obj_values.shape[1]

        # Build the data to plot
        data: dict[str, tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        if num_objectives > 1:
            # Multiple objectives
            for i in range(num_objectives):
                data[f'Objective_{i + 1}'] = (
                    func_calls.params,
                    func_calls.obj_values[:, i],
                    None if func_calls.obj_variances is None else func_calls.obj_variances[:, i],
                )
            # Scalarised value (if available)
            if func_calls.scalar_values is not None:
                data['Scalarised Objective'] = (
                    func_calls.params,
                    func_calls.scalar_values,
                    None if func_calls.scalar_variances is None else func_calls.scalar_variances,
                )
        else:
            # Single objective
            data['Objective'] = (
                func_calls.params,
                func_calls.obj_values,
                None if func_calls.obj_variances is None else func_calls.obj_variances,
            )

        # Sanitise number of axes
        if len(axes) != len(data):
            raise ValueError("Number of axes does not match number of objectives.")

        # Plot each case
        for idx, (name, (x_vals, y_vals, y_vars)) in enumerate(data.items()):
            # Clamp to max_calls and build dataset
            max_calls = len(x_vals) if args.max_calls is None else min(len(x_vals), args.max_calls)
            x_vals = x_vals[:max_calls]
            y_vals = y_vals[:max_calls]
            y_vars = None if y_vars is None else y_vars[:max_calls]
            dataset = RawDataset(
                x_vals.reshape(-1, 1),
                y_vals.reshape(-1, 1),
                None if y_vars is None else y_vars.reshape(-1, 1, 1),
            )

            # Fit the GP to the data
            settings = SurrogateSettings(noise='none' if y_vars is None else 'fixed')
            model = GPModel(dataset, settings)

            # Sample from the model
            sample_shape = torch.Size([args.num_samples])
            with torch.no_grad():
                samples = model.sample(x, sample_shape=sample_shape, seed=args.seed)
                noisy_samples = model.sample(
                    x, sample_shape=sample_shape, seed=args.seed, observation_noise=True
                )

            # Derive mean and confidence intervals
            # For Gaussian posteriors, the f and y means should be identical (plot for completeness)
            f_mean = torch.mean(samples, dim=0).squeeze()
            y_mean = torch.mean(noisy_samples, dim=0).squeeze()
            f_lb = torch.quantile(samples, 0.025, dim=0).squeeze()
            f_ub = torch.quantile(samples, 0.975, dim=0).squeeze()
            y_lb = torch.quantile(noisy_samples, 0.025, dim=0).squeeze()
            y_ub = torch.quantile(noisy_samples, 0.975, dim=0).squeeze()

            # Plot the GP
            ax = axes[idx]
            ax.plot(x.squeeze(), f_mean, label='f mean')
            ax.fill_between(x.squeeze(), f_lb, f_ub, alpha=0.2, label='f 95% CI')

            # Plot the observations
            if y_vars is None:
                ax.scatter(x_vals, y_vals, color='k', label='Observations')
            else:
                y_std = np.sqrt(y_vars)
                ax.plot(x.squeeze(), y_mean, label='y mean')
                ax.fill_between(x.squeeze(), y_lb, y_ub, alpha=0.2, label='y 95% CI')
                ax.errorbar(
                    x_vals, y_vals, yerr=2 * y_std, fmt='o', color='k', label='Observations'
                )

            # If function samples are requested, plot them
            if args.num_func_samples > 0:
                func_sample_shape = torch.Size([args.num_func_samples])
                with torch.no_grad():
                    func_samples = model.sample(
                        x.view(-1, 1), sample_shape=func_sample_shape, seed=args.seed
                    )
                for i in range(args.num_func_samples):
                    ax.plot(x.squeeze(), func_samples[i].squeeze(), color='k', alpha=0.2)

            ax.set_xlabel(problem.settings.parameters[0].name)
            ax.set_xlim(x_min, x_max)
            ax.set_title(name)
            ax.legend()


class CompositeGPPlot(PlottingModuleConfigFile):
    """Plotting module for Gaussian process regression (with composite objectives)."""

    def __init__(self) -> None:
        super().__init__(
            name="composite_gp",
            help_str="plot Gaussian process regression results (with composite objectives)",
            description="Plot Gaussian process regression results (with composite objectives).",
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
            "--num_points",
            type=int,
            default=1024,
            help="Number of points to use for plotting.",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=512,
            help="Number of Monte Carlo samples to use for plotting.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility.",
        )
        parser.add_argument(
            "--num_func_samples",
            type=int,
            default=0,
            help="Number of function samples to use for plotting.",
        )
        parser.add_argument(
            "--sample_from",
            choices=["mean", "objective"],
            default="mean",
            help="Specify whether to sample from the mean estimator or the objective.",
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
        num_objectives = (
            problem.objective.num_problem.objectives()
            if problem.objective.is_multi_objective() else 1
        )
        figures = []
        axes = []
        for _ in range(num_objectives):
            fig, ax = plt.subplots(layout="constrained")
            figures.append(fig)
            axes.append(ax)
        self.single_plot(axes, problem, args)
        return figures

    @classmethod
    def single_plot(cls, axes: list[plt.Axes], problem: ProblemConfig, args: Namespace) -> None:
        """Generate a single plot for the given GP model.

        Parameters
        ----------
        axes : list[plt.Axes]
            The axes to plot on.
        problem : ProblemConfig
            The optimisation problem built from the configuration file.
        args : Namespace
            The arguments parsed from the command line.
        """
        if problem.settings.parameters.num_optim_parameters() != 1:
            raise ValueError("Can only plot a Gaussian process regression for a single parameter.")
        if not problem.objective.is_composite():
            raise ValueError("Can only plot this GP for composite objectives.")
        if not isinstance(problem.optimiser, GenericOptimiser):
            raise ValueError("Can only plot a Gaussian process regression for generic optimisers.")
        campaign = problem.optimiser.campaign
        objective = problem.objective

        # Build x-grid
        parameters = problem.settings.parameters
        x_min, x_max = parameters.get_bounds()[0]
        x = torch.linspace(x_min, x_max, args.num_points).view(-1, 1, 1)

        # Load optimiser data
        campaign.load()
        model = campaign.get_model()

        # Sample from the model
        sample_shape = torch.Size([args.num_samples])
        with torch.no_grad():
            samples = model.objective_samples(x, sample_shape=sample_shape, seed=args.seed)
            noisy_samples = model.objective_samples(
                x, sample_shape=sample_shape, seed=args.seed, observation_noise=True
            )
            objective_of_mean = objective.composition(
                model.latent_samples(x, sample_shape=sample_shape, seed=args.seed).mean(dim=0), x
            )

        # Ensure sample shapes are consistent with the problem
        num_objectives = objective.num_objectives() if objective.is_multi_objective() else 1
        samples = samples.reshape(args.num_samples, args.num_points, 1, num_objectives)
        noisy_samples = noisy_samples.reshape(args.num_samples, args.num_points, 1, num_objectives)
        objective_of_mean = objective_of_mean.reshape(args.num_points, 1, num_objectives)

        # Derive mean and confidence intervals
        f_mean = torch.mean(samples, dim=0)
        y_mean = torch.mean(noisy_samples, dim=0)
        f_lb = torch.quantile(samples, 0.025, dim=0)
        f_ub = torch.quantile(samples, 0.975, dim=0)
        y_lb = torch.quantile(noisy_samples, 0.025, dim=0)
        y_ub = torch.quantile(noisy_samples, 0.975, dim=0)

        # If function samples are requested, evaluate them
        if args.num_func_samples > 0:
            func_sample_shape = torch.Size([args.num_func_samples])
            with torch.no_grad():
                func_samples = model.objective_samples(
                    x.view(-1, 1),
                    sample_shape=func_sample_shape,
                    seed=args.seed,
                    observation_noise=(args.sample_from == "objective"),
                )
            func_samples = func_samples.reshape(
                args.num_func_samples, args.num_points, 1, num_objectives
            )

        # Build list of observations
        dataset = campaign.dataset.export_raw()
        obs_x_vals = dataset.inputs
        if dataset.covariances is None:
            obs_y_vals = objective.composition(dataset.outputs, obs_x_vals).reshape(
                -1, num_objectives
            )
            obs_y_lb = None
            obs_y_ub = None
            obs_y_mean_vals = None
        else:
            obs_latent_samples = torch.stack(
                [
                    MultivariateNormalQMCEngine(
                        dataset.outputs[i, ...], dataset.covariances[i, ...], seed=0
                    ).draw(args.num_samples)
                    for i in range(dataset.outputs.shape[0])
                ],
                dim=1,
            )
            obj_samples = objective.composition(obs_latent_samples, obs_x_vals).reshape(
                -1, dataset.outputs.shape[0], num_objectives
            )
            obs_y_vals = torch.mean(obj_samples, dim=0)
            obs_y_lb = torch.quantile(obj_samples, 0.025, dim=0)
            obs_y_ub = torch.quantile(obj_samples, 0.975, dim=0)
            obs_y_mean_vals = objective.composition(dataset.outputs, obs_x_vals).reshape(
                -1, dataset.outputs.shape[0], num_objectives
            )

        # Sanitise number of axes
        if len(axes) != num_objectives:
            raise ValueError("Number of axes does not match number of objectives")

        # Plot for each objective
        for i in range(num_objectives):
            ax = axes[i]

            # Set up the plot depending on whether this is stochastic or not
            if obs_y_lb is None:
                ax.plot(x.squeeze(), f_mean[..., i].squeeze(), label='f mean')
                ax.fill_between(
                    x.squeeze(),
                    f_lb[..., i].squeeze(),
                    f_ub[..., i].squeeze(),
                    alpha=0.2,
                    label='f 95% CI',
                )
                ax.scatter(
                    obs_x_vals.squeeze(),
                    obs_y_vals[..., i].squeeze(),
                    color='k',
                    label='Observations',
                )
            else:
                # Distribution from the mean estimator
                ax.plot(
                    x.squeeze(),
                    f_mean[..., i].squeeze(),
                    label=r'$\mathbb{E}[f(\theta, \mu_p(\theta))]$',
                )
                ax.fill_between(
                    x.squeeze(),
                    f_lb[..., i].squeeze(),
                    f_ub[..., i].squeeze(),
                    alpha=0.2,
                )
                # Distribution from the latent posterior (with noise)
                p_y, = ax.plot(
                    x.squeeze(),
                    y_mean[..., i].squeeze(),
                    label=r'$\mathbb{E}[f(\theta, p(\theta))]$',
                )
                ax.fill_between(
                    x.squeeze(),
                    y_lb[..., i].squeeze(),
                    y_ub[..., i].squeeze(),
                    alpha=0.2,
                )
                # Objective of the mean
                p_m, = ax.plot(
                    x.squeeze(),
                    objective_of_mean[..., i].squeeze(),
                    label=r'$f(\theta, \mathbb{E}[p(\theta)])$',
                )

                # Observations
                yerr = torch.stack(
                    [
                        obs_y_vals[..., i].squeeze() - obs_y_lb[..., i].squeeze(),
                        obs_y_ub[..., i].squeeze() - obs_y_vals[..., i].squeeze()
                    ],
                    dim=0,
                )
                ax.errorbar(
                    obs_x_vals.squeeze(),
                    obs_y_vals[..., i].squeeze(),
                    yerr=yerr,
                    fmt='o',
                    color=p_y.get_color(),
                    label=r'Observations - $p(f(\theta, p(\theta)))$',
                )
                ax.scatter(
                    obs_x_vals.squeeze(),
                    obs_y_mean_vals[..., i].squeeze(),
                    color=p_m.get_color(),
                    label=r'Observations - $p(f(\theta, \mu_p(\theta)))$',
                )

            # If function samples are requested, plot them
            if args.num_func_samples > 0:
                label = (
                    r"$p(f(\theta, p(\theta)))$"
                    if args.sample_from == "objective"
                    else r"$p(f(\theta, \mu_p(\theta)))$"
                )
                for j in range(args.num_func_samples):
                    ax.plot(
                        x.squeeze(),
                        func_samples[j, ..., i].squeeze(),
                        color='k',
                        alpha=0.2,
                        label=f'Function Samples - {label}' if j == 0 else None,
                    )

            ax.set_xlabel(problem.settings.parameters[0].name)
            ax.set_xlim(x_min, x_max)
            ax.set_title(f"Objective {i+1}" if num_objectives > 1 else "Objective")
            ax.legend()
