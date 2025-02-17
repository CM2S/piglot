"""Module for scalarisation of objectives."""
from typing import Optional, Tuple, Dict, Union, Any, Type, List
from abc import ABC, abstractmethod
import torch
from botorch.utils.sampling import draw_sobol_normal_samples
from piglot.utils.assorted import read_custom_module
from piglot.objective import Scalarisation, IndividualObjective


class MonteCarloScalarisation(Scalarisation, ABC):
    """Base class for non-linear scalarisations requiring Monte Carlo variance estimations."""

    def __init__(
        self,
        objectives: List[IndividualObjective],
        num_samples: int = 512,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(objectives)
        self.num_samples = num_samples
        self.seed = seed

    @abstractmethod
    def _scalarise_sample(self, values: torch.Tensor) -> torch.Tensor:
        """Scalarise a batch of objectives samples for Monte Carlo estimation of variance.

        Parameters
        ----------
        values : torch.Tensor
            A "num_samples x (batch_shape) x num_objectives" tensor with the objective values.

        Returns
        -------
        torch.Tensor
            A "num_samples x (batch_shape)" tensor with the scalarised objective values.
        """

    def scalarise_torch(
        self,
        values: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scalarise a set of objectives with gradients.

        Parameters
        ----------
        values : torch.Tensor
            Mean objective values.
        variances : Optional[torch.Tensor]
            Optional variances of the objectives.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Mean and variance of the scalarised objective.
        """
        if variances is None:
            return self._scalarise_sample(values), None
        # Sample from the given normal distribution
        z_samples = draw_sobol_normal_samples(1, self.num_samples, seed=self.seed).to(values.device)
        samples = values + z_samples.expand(-1, *[1 for _ in values.shape]) * variances.sqrt()
        # Return mean and variance of the scalarised samples
        scalarised_samples = self._scalarise_sample(samples)
        return torch.mean(scalarised_samples, dim=0), torch.var(scalarised_samples, dim=0)


class MeanScalarisation(Scalarisation):
    """Scalarise using the mean of the objectives."""

    def scalarise_torch(
        self,
        values: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scalarise a set of objectives with gradients.

        Parameters
        ----------
        values : torch.Tensor
            Mean objective values.
        variances : Optional[torch.Tensor]
            Optional variances of the objectives.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Mean and variance of the scalarised objective.
        """
        if variances is None:
            return torch.mean(values * self.weights, dim=-1), None
        return (
            torch.mean(values * self.weights, dim=-1),
            torch.sum(variances * self.weights.square(), dim=-1) / (values.shape[-1] ** 2),
        )


class SumScalarisation(Scalarisation):
    """Scalarise using the sum of the objectives."""

    def scalarise_torch(
        self,
        values: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scalarise a set of objectives with gradients.

        Parameters
        ----------
        values : torch.Tensor
            Mean objective values.
        variances : Optional[torch.Tensor]
            Optional variances of the objectives.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Mean and variance of the scalarised objective.
        """
        if variances is None:
            return torch.sum(values * self.weights, dim=-1), None
        return (
            torch.sum(values * self.weights, dim=-1),
            torch.sum(variances * self.weights.square(), dim=-1),
        )


class MaxScalarisation(MonteCarloScalarisation):
    """Scalarise using the maximum of the objectives."""

    def _scalarise_sample(self, values: torch.Tensor) -> torch.Tensor:
        """Scalarise a batch of objectives samples for Monte Carlo estimation of variance.

        Parameters
        ----------
        values : torch.Tensor
            A "num_samples x (batch_shape) x num_objectives" tensor with the objective values.

        Returns
        -------
        torch.Tensor
            A "num_samples x (batch_shape)" tensor with the scalarised objective values.
        """
        return torch.amax(values * self.weights, dim=-1)


class MinScalarisation(MonteCarloScalarisation):
    """Scalarise using the minimum of the objectives."""

    def _scalarise_sample(self, values: torch.Tensor) -> torch.Tensor:
        """Scalarise a batch of objectives samples for Monte Carlo estimation of variance.

        Parameters
        ----------
        values : torch.Tensor
            A "num_samples x (batch_shape) x num_objectives" tensor with the objective values.

        Returns
        -------
        torch.Tensor
            A "num_samples x (batch_shape)" tensor with the scalarised objective values.
        """
        return torch.amin(values * self.weights, dim=-1)


AVALIABLE_SCALARISATIONS: Dict[str, Type[Scalarisation]] = {
    'mean': MeanScalarisation,
    'sum': SumScalarisation,
    'max': MaxScalarisation,
    'min': MinScalarisation,
}


def read_scalarisation(
    config: Union[str, Dict[str, Any]],
    objectives: List[IndividualObjective],
) -> Scalarisation:
    """Read a scalarisation function from a configuration.

    Parameters
    ----------
    config : Union[str, Dict[str, Any]]
        Configuration of the scalarisation function.
    objectives : List[IndividualObjective]
        List of objectives to scalarise.

    Returns
    -------
    Scalarisation
        Scalarisation function.
    """
    # Parse the scalarisation in the simple format
    if isinstance(config, str):
        name = config
        if name == 'script':
            raise ValueError('Need to pass the file path for the "script" scalarisation.')
        if name not in AVALIABLE_SCALARISATIONS:
            raise ValueError(f'Scalarisation function "{name}" is not available.')
        return AVALIABLE_SCALARISATIONS[name](objectives)
    # Detailed format
    if 'name' not in config:
        raise ValueError('Need to pass the name of the scalarisation function.')
    name = config.pop('name')
    # Read script scalarisation
    if name == 'script':
        return read_custom_module(config, Scalarisation)(objectives)
    if name not in AVALIABLE_SCALARISATIONS:
        raise ValueError(f'Scalarisation function "{name}" is not available.')
    return AVALIABLE_SCALARISATIONS[name](objectives, **config)
