"""Module for risk-aware acquisition functions in BoTorch."""
from __future__ import annotations
from typing import Optional, Tuple, Type, TypeVar, Dict
from abc import ABC, abstractmethod
from math import ceil
import torch
from torch import Tensor
from botorch.acquisition import (
    MCAcquisitionFunction,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from piglot.optimisers.botorch.fantasy_acquisitions import qFantasyAcqusition


T = TypeVar("T", bound="qFantasyAcqusition")


class RiskMeasure(ABC):
    """Base class for risk measures."""

    def tuple_reduce(self, samples: Tensor, dim: Tuple[int]) -> Tensor:
        """Reduce samples over multiple dimensions.

        Parameters
        ----------
        samples : Tensor
            Samples to reduce.
        dim : Tuple[int]
            Dimensions to reduce over.

        Returns
        -------
        Tensor
            Reduced samples.
        """
        for d in dim:
            samples = self(samples, dim=d, keepdim=True)
        return samples.squeeze(*dim)

    @abstractmethod
    def __call__(self, samples: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        """Compute the risk metric by reducing over a single dimension.

        Parameters
        ----------
        samples : Tensor
            Samples to reduce.
        dim : int
            Dimension to reduce over.
        keepdim : bool, optional
            Whether to keep the reduced dimension, by default False.

        Returns
        -------
        Tensor
            Risk measure.
        """


class VaR(RiskMeasure):
    """Value-at-risk risk measure."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, samples: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        """Compute the VaR risk metric by reducing over a single dimension.

        Parameters
        ----------
        samples : Tensor
            Samples to reduce.
        dim : int
            Dimension to reduce over.
        keepdim : bool, optional
            Whether to keep the reduced dimension, by default False.

        Returns
        -------
        Tensor
            Risk measure.
        """
        return samples.quantile(self.alpha, dim=dim, keepdim=keepdim)


class CVaR(RiskMeasure):
    """Conditional Value-at-risk risk measure."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, samples: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        """Compute the CVaR risk metric by reducing over a single dimension.

        Parameters
        ----------
        samples : Tensor
            Samples to reduce.
        dim : int
            Dimension to reduce over.
        keepdim : bool, optional
            Whether to keep the reduced dimension, by default False.

        Returns
        -------
        Tensor
            Risk measure.
        """
        return torch.topk(
            samples,
            k=ceil(samples.shape[dim] * self.alpha),
            largest=False,
            dim=dim,
        ).values.mean(dim=dim, keepdim=keepdim)


class WorstCase(RiskMeasure):
    """Worst-case risk measure."""

    def __call__(self, samples: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        """Compute the worst-case risk metric by reducing over a single dimension.

        Parameters
        ----------
        samples : Tensor
            Samples to reduce.
        dim : int
            Dimension to reduce over.
        keepdim : bool, optional
            Whether to keep the reduced dimension, by default False.

        Returns
        -------
        Tensor
            Risk measure.
        """
        return samples.amin(dim=dim, keepdim=keepdim)


AVALIALBE_RISK_MEASURES: Dict[str, Type[RiskMeasure]] = {
    "var": VaR,
    "cvar": CVaR,
    "worst": WorstCase,
}


def get_risk_measure(name: str, **options) -> RiskMeasure:
    """Get a risk measure by name.

    Parameters
    ----------
    name : str
        Name of the risk measure.
    options : Dict[str, Any]
        Options for the risk measure.

    Returns
    -------
    RiskMeasure
        Risk measure.
    """
    if name not in AVALIALBE_RISK_MEASURES:
        raise UnsupportedError(f"Risk measure {name} is not supported.")
    cls = AVALIALBE_RISK_MEASURES[name]

    # Sanitise the options for the VaR and CVaR risk measures
    if name in ('var', 'cvar'):
        if 'alpha' not in options:
            raise ValueError(f"Risk measure {name} requires an alpha value.")
        return cls(alpha=options['alpha'])
    return cls()


def make_risk_acquisition(cls: Type[T], risk_measure: RiskMeasure, *args, **kwargs) -> T:
    """Create a fantasy-based risk acquisition function from a base class and a risk measure.

    Parameters
    ----------
    cls : Type[T]
        Base class for the acquisition function.
    risk_measure : RiskMeasure
        Risk measure to use.

    Returns
    -------
    T
        Risk acquisition function.
    """
    return cls(*args, reduction=risk_measure, observation_noise=True, **kwargs)


class qSimpleRegretRisk(MCAcquisitionFunction):
    """Simple regret for a risk measure."""

    def __init__(
        self,
        model: Model,
        risk_measure: RiskMeasure,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.risk_measure = risk_measure

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(
            X=X,
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples=samples, X=X)
        return self.risk_measure.tuple_reduce(
            torch.amax(obj, dim=-1),
            dim=tuple(range(len(self.sample_shape))),
        )


class qKnowledgeGradientRisk(qMultiFidelityKnowledgeGradient):
    """Knowledge gradient for risk-aware optimisation."""

    def __init__(self, *args, risk_measure: RiskMeasure, **kwargs) -> None:
        super().__init__(
            *args,
            valfunc_cls=qSimpleRegretRisk,
            valfunc_argfac=lambda model: {"risk_measure": risk_measure},
            **kwargs,
        )
