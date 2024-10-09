"""Module with fantasy-based acquisition functions for BoTorch models."""
from __future__ import annotations
from typing import Optional, Callable
from functools import partial
import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform, match_batch_shape
from botorch.utils.safe_math import log_fatplus, logmeanexp, fatmax
from torch import Tensor


class qFantasyAcqusition(MCAcquisitionFunction):
    """Base class for fantasy-based acquisition functions.

    This family of acquisitions uses fantasies to approximate the posterior after observing the
    objective at X. We then compute the MC estimate of the objective with the fantasised posterior
    samples and reduce them to a scalar value.
    """

    def __init__(
        self,
        model: Model,
        sample_forward: Callable[[Tensor], Tensor],
        num_fantasies: Optional[int] = 64,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        observation_noise: bool = False,
        noiseless_fantasies: bool = False,
        final_reduction: Optional[Callable[[Tensor, int], Tensor]] = None,
        q_reduction: Optional[Callable[[Tensor, int], Tensor]] = None,
        input_transform: Optional[Callable[[Tensor], Tensor]] = None,
        reduction: Optional[Callable[[Tensor, int], Tensor]] = None,
    ) -> None:
        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super(MCAcquisitionFunction, self).__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        elif objective is not None and not isinstance(
            objective, MCAcquisitionObjective
        ):
            raise UnsupportedError(
                "Objectives that are not an `MCAcquisitionObjective` are not supported."
            )

        if objective is None and model.num_outputs != 1:
            if posterior_transform is None:
                raise UnsupportedError(
                    "Must specify an objective or a posterior transform when using "
                    "a multi-output model."
                )
            elif not posterior_transform.scalarize:
                raise UnsupportedError(
                    "If using a multi-output model without an objective, "
                    "posterior_transform must scalarize the output."
                )
        if final_reduction is None:
            final_reduction = torch.mean
        if q_reduction is None:
            q_reduction = torch.amax
        if reduction is None:
            reduction = torch.mean
        self.sample_forward = sample_forward
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.set_X_pending(X_pending)
        self.X_pending: Tensor = self.X_pending
        self.inner_sampler = inner_sampler
        self.num_fantasies: int = num_fantasies
        self.observation_noise = observation_noise
        self.noiseless_fantasies = noiseless_fantasies
        self.final_reduction = final_reduction
        self.q_reduction = q_reduction
        self.input_transform = input_transform
        self.reduction = reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition function on the given input.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape `b x q x d` or `q x d`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `b` or `1`."""
        if self.input_transform is not None:
            X = self.input_transform(X)

        # Construct the fantasy model of shape `num_fantasies x b` and get samples
        zero_noise = torch.zeros(X.shape[:-1] + torch.Size([self.model.num_outputs])).to(X)
        fantasy_model = self.model.fantasize(
            X=X,
            sampler=self.sampler,
            observation_noise=zero_noise if self.noiseless_fantasies else None,
        )
        fantasy_posterior = fantasy_model.posterior(
            X,
            observation_noise=self.observation_noise,
            posterior_transform=self.posterior_transform,
        )
        fantasy_samples = self.inner_sampler(fantasy_posterior)
        fantasy_obj = self.objective(samples=fantasy_samples, X=X)

        # Compute reduction in objective for each fantasy model
        reduced_obj = self.reduction(fantasy_obj, dim=0)
        quantity = self.sample_forward(reduced_obj)

        # Maximum over the q-batch and reduce over the fantasy models
        q_reduced = self.q_reduction(quantity, dim=-1)
        return self.final_reduction(q_reduced, dim=0)


class qFantasySimpleRegret(qFantasyAcqusition):
    """Simple regret (using fantasies)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            sample_forward=lambda samples: samples,
            **kwargs,
        )


class qFantasyExpectedImprovement(qFantasyAcqusition):
    """Expected improvement acquisition (using fantasies)."""

    def __init__(self, *args, best_f, **kwargs) -> None:
        super().__init__(
            *args,
            sample_forward=lambda samples: (samples - best_f).clamp_min(0),
            **kwargs,
        )


class qFantasyLogExpectedImprovement(qFantasyAcqusition):
    """Log expected improvement acquisition (using fantasies)."""

    def __init__(
        self,
        *args,
        best_f: float,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            sample_forward=lambda samples: log_fatplus(samples - best_f, tau=1e-6),
            final_reduction=logmeanexp,
            q_reduction=partial(fatmax, tau=1e-2),
            **kwargs,
        )


class qFantasyNoisyExpectedImprovement(qFantasyAcqusition):
    """Noisy expected improvement acquisition (using fantasies)."""

    @staticmethod
    def _input_transform(X: Tensor, X_baseline: Tensor) -> Tensor:
        return torch.cat([X, match_batch_shape(X_baseline, X)], dim=-2)

    @staticmethod
    def _q_reduction(samples: Tensor, q_baseline: int, dim: int) -> Tensor:
        q = samples.shape[-1] - q_baseline
        obj_new = samples[..., :q].amax(dim=dim)
        obj_old = samples[..., q:].amax(dim=dim)
        return (obj_new - obj_old).clamp_min(0)

    def __init__(
        self,
        *args,
        X_baseline: Tensor,
        max_frac: float = 0.5,
        prune_baseline: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            sample_forward=lambda samples: samples,
            **kwargs,
        )
        if prune_baseline:
            X_baseline = prune_inferior_points(
                self.model,
                X_baseline,
                objective=self.objective,
                posterior_transform=self.posterior_transform,
                sampler=self.sampler,
                max_frac=max_frac,
            )
        self.input_transform = partial(self._input_transform, X_baseline=X_baseline)
        self.q_reduction = partial(self._q_reduction, q_baseline=X_baseline.shape[-2])


class qFantasyLogNoisyExpectedImprovement(qFantasyAcqusition):
    """Log noisy expected improvement acquisition (using fantasies)."""

    @staticmethod
    def _input_transform(X: Tensor, X_baseline: Tensor) -> Tensor:
        return torch.cat([X, match_batch_shape(X_baseline, X)], dim=-2)

    @staticmethod
    def _q_reduction(samples: Tensor, q_baseline: int, dim: int) -> Tensor:
        q = samples.shape[-1] - q_baseline
        obj_new = fatmax(samples[..., :q], dim=dim, tau=1e-2)
        obj_old = fatmax(samples[..., q:], dim=dim, tau=1e-2)
        return log_fatplus(obj_new - obj_old, tau=1e-6)

    def __init__(
        self,
        *args,
        X_baseline: Tensor,
        max_frac: float = 0.5,
        prune_baseline: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            sample_forward=lambda samples: samples,
            final_reduction=logmeanexp,
            **kwargs,
        )
        if prune_baseline:
            X_baseline = prune_inferior_points(
                self.model,
                X_baseline,
                objective=self.objective,
                posterior_transform=self.posterior_transform,
                sampler=self.sampler,
                max_frac=max_frac,
            )
        self.input_transform = partial(self._input_transform, X_baseline=X_baseline)
        self.q_reduction = partial(self._q_reduction, q_baseline=X_baseline.shape[-2])
