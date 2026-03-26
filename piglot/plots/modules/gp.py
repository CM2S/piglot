"""Module for Gaussian process regression plots."""
from argparse import Namespace, ArgumentParser
from typing import Optional
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import torch
from piglot.data.dataset import RawDataset
from piglot.data.surrogate import GPModel, SurrogateSettings
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
        if len(problem.settings.parameters) != 1:
            raise ValueError("Can only plot a Gaussian process regression for a single parameter.")

        # Build x-grid
        parameters = problem.settings.parameters
        x_min = min(par.lbound for par in parameters)
        x_max = max(par.ubound for par in parameters)
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

        # Plot each case
        figures: list[Figure] = []
        for name, (x_vals, y_vals, y_vars) in data.items():
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
            fig, ax = plt.subplots(layout='constrained')
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
            figures.append(fig)

        return figures


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
