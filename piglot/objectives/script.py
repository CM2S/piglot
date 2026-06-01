"""Module for script-based objectives."""
from abc import abstractmethod
from typing import Any, Optional, TypeVar
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
from piglot.objective import IndividualObjectiveResult
from piglot.parameter import ParameterValues
from piglot.settings import Settings
from piglot.objectives.simple_objective import SimpleObjective, SimpleIndividualObjective
from piglot.utils.assorted import read_custom_module


IndividualT = TypeVar('IndividualT', bound='ScriptIndividualObjective')


class ScriptIndividualObjective(SimpleIndividualObjective):
    """Script-based individual objective."""

    _composite: bool = False

    def __init__(
        self,
        name: str,
        settings: Settings,
        weight: float = 1.0,
        maximise: bool = False,
        variance: bool = False,
        composite: bool = False,
        noisy: bool = False,
        bounds: tuple[float, float] = None,
    ) -> None:
        super().__init__(name, weight, maximise, variance, composite, noisy, bounds)
        self.settings = settings

    @abstractmethod
    def evaluate(self, params: ParameterValues, concurrent: bool) -> IndividualObjectiveResult:
        """Evaluate the objective for a set of parameters.

        Parameters
        ----------
        params : ParameterValues
            Named set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others.

        Returns
        -------
        IndividualObjectiveResult
            Objective value.
        """

    def plot(self, values: np.ndarray, **kwargs) -> Figure:
        """Plot the objective.

        Parameters
        ----------
        values : np.ndarray
            Parameter values to plot for.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        Figure
            Figure with the plot.
        """
        raise NotImplementedError("Plotting not implemented for this script-based objective.")

    @classmethod
    def read(
        cls: type[IndividualT], name: str, config: dict[str, Any], settings: Settings
    ) -> IndividualT:
        """Read the objective from a configuration dictionary.

        Parameters
        ----------
        name : str
            Name of this objective.
        config : dict[str, Any]
            Terms from the configuration dictionary.
        settings : Settings
            Settings for this problem.

        Returns
        -------
        IndividualT
            Objective function to optimise.
        """
        cls = read_custom_module(config, cls)
        return cls(
            name,
            settings,
            weight=float(config.get('weight', 1.0)),
            maximise=bool(config.get('maximise', False)),
            variance=config.get('variance', None),
            bounds=config.get('bounds', None),
            noisy=bool(config.get('noisy', False)),
            composite=cls._composite,
        )


class ScriptIndividualCompositeObjective(ScriptIndividualObjective):
    """Script-based composite individual objective."""

    _composite: bool = True

    @abstractmethod
    def composition(self, latent: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Composition function for this objective, if supported.

        Parameters
        ----------
        latent : torch.Tensor
            Latent space values from the inner function of shape `(batch_shape) x n_latent`.
        params : dict[str, torch.Tensor]
            Named parameters for the given result. Each tensor has shape
            `(batch_shape) x n_components`.

        Returns
        -------
        torch.Tensor
            Composition result of shape `(batch_shape)`.
        """

    @abstractmethod
    def latent_size(self) -> int:
        """Return the size of the latent space for this objective.

        Returns
        -------
        int
            Size of the latent space.
        """


class ScriptObjective(SimpleObjective[ScriptIndividualObjective]):
    """Objective function derived from a script."""

    def plot_case(self, case_hash: str, **kwargs) -> list[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting function

        Returns
        -------
        list[Figure]
            List of figures with the plot
        """
        # Read function calls file and find parameters associated with the hash
        data = self.read_func_calls()
        idx = data.hashes.index(case_hash)
        params = data.params[idx, :]
        return [obj.plot(params, **kwargs) for obj in self.objectives]

    @classmethod
    def individual_objective_type(cls) -> type[ScriptIndividualObjective]:
        """Get the type of the individual objective for this simple objective.

        Returns
        -------
        type[ScriptIndividualObjective]
            Individual objective type for this simple objective.
        """
        return ScriptIndividualObjective
