"""Module for multi-fidelity optimisation utilities."""
from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor
from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import t_batch_mode_transform, concatenate_pending_points
from botorch.utils.safe_math import log_fatplus, logmeanexp, fatmax
from piglot.parameter import ParameterSet
from piglot.data.surrogate import fit_single_task_gp
from piglot.data.dataset import ObjectiveDataset


class CostModel(ABC):

    def __init__(self, parameters: ParameterSet, dataset: ObjectiveDataset) -> None:
        self.parameters = parameters
        self.dataset = dataset

    @abstractmethod
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the cost for the given input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape `(batch_shape) x q x d`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_shape) x q` representing the cost for each input.
        """


class FixedCostModel(CostModel):
    """Fixed cost model."""

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the cost for the given input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape `(batch_shape) x q x d`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_shape) x q` representing the cost for each input.
        """
        fidelities = X[..., self.parameters.get_scalar_fidelity_index()]
        fixed_costs = self.parameters.get_fidelity_parameter().cost
        return torch.tensor(
            [fixed_costs[f.item()] for f in fidelities.flatten()]
        ).to(X).reshape_as(fidelities)


class InferredCostModel(CostModel):
    """Inferred cost model."""

    def __init__(self, parameters: ParameterSet, dataset: ObjectiveDataset) -> None:
        super().__init__(parameters, dataset)

        # Extract observations of each fidelity
        idx = self.parameters.get_scalar_fidelity_index()
        observations = {
            fidelity: dataset.subset(lambda obs: obs.params[idx] == fidelity).data
            for fidelity in parameters.get_fidelities()
        }

        # Compute the average cost for each fidelity
        self.costs = {
            fid: sum(obs.elapsed_time for obs in data) / len(data) if len(data) > 0 else 1e-6
            for fid, data in observations.items()
        }

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the cost for the given input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape `(batch_shape) x q x d`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_shape) x q` representing the cost for each input.
        """
        fidelities = X[..., self.parameters.get_scalar_fidelity_index()]
        return torch.tensor(
            [self.costs[f.item()] for f in fidelities.flatten()]
        ).to(X).reshape_as(fidelities)


class FullyInferredCostModel(CostModel):
    """Fully inferred cost model."""

    def __init__(self, parameters: ParameterSet, dataset: ObjectiveDataset) -> None:
        super().__init__(parameters, dataset)

        # Build a parameter-cost dataset and fit a GP
        inputs = torch.tensor([obs.params.tolist() for obs in dataset.data], dtype=torch.float64)
        costs = torch.tensor(
            [obs.elapsed_time for obs in dataset.data], dtype=torch.float64
        ).unsqueeze(-1)
        self.model = fit_single_task_gp(inputs, costs)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the cost for the given input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape `(batch_shape) x q x d`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_shape) x q` representing the cost for each input.
        """
        return self.model.posterior(X).mean.squeeze(-1).to(X)


def build_cost_model(parameters: ParameterSet, dataset: ObjectiveDataset) -> CostModel:
    """Build a cost model for multi-fidelity optimisation.

    Parameters
    ----------
    parameters : ParameterSet
        Parameters to optimise.
    dataset : ObjectiveDataset
        Dataset containing the objective values.

    Returns
    -------
    CostModel
        Multi-fidelity cost model.
    """
    models: dict[str, type[CostModel]] = {
        "fixed": FixedCostModel,
        "infer": InferredCostModel,
        "infer_full": FullyInferredCostModel,
    }
    cls = models[parameters.get_fidelity_parameter().cost_model]
    return cls(parameters, dataset)


def project_to_target_fidelity(X: torch.Tensor, fidelity_index: int, target: float) -> torch.Tensor:
    """Project the input tensor to the target fidelity dimension.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape `(batch_shape) x d`.
    fidelity_index : int
        Index of the fidelity dimension.
    target : float
        Target fidelity value.

    Returns
    -------
    torch.Tensor
        Tensor with the target fidelity value set at the specified index.
    """
    new_fidelity = torch.full_like(X[..., fidelity_index], target).unsqueeze(-1)
    return torch.cat([X[..., :fidelity_index], new_fidelity, X[..., fidelity_index + 1:]], dim=-1)


class MultiFidelityAcquisition(MCAcquisitionFunction):
    """Base class for multi-fidelity acquisition functions."""

    def __init__(
        self,
        model: Model,
        sampler: MCSampler,
        fidelity_dim: int,
        cost_model: CostModel,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super(MCAcquisitionFunction, self).__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        if objective is not None and not isinstance(objective, MCAcquisitionObjective):
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
        self.objective = objective
        self.fidelity_dim = fidelity_dim
        self.cost_model = cost_model
        self.posterior_transform = posterior_transform
        self.set_X_pending(X_pending)
        self.X_pending: Tensor


class qMultiFidelityExpectedImprovement(MultiFidelityAcquisition):
    """Multi-fidelity expected improvement acquisition."""

    def __init__(
        self,
        model: Model,
        best_f: dict[float, float],
        sampler: MCSampler,
        fidelity_dim: int,
        cost_model: CostModel,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            fidelity_dim=fidelity_dim,
            cost_model=cost_model,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.best_f = best_f

    def get_best_f(self, X: Tensor) -> Tensor:
        """Compute the best_f tensor based on the fidelity values in X.
        
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape `(batch_shape) x q x d` or `q x d`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_shape) x q` with the best_f values corresponding to the
            fidelity values in X.
        """
        fidelities = X[..., self.fidelity_dim]
        best_f = torch.tensor([self.best_f[f.item()] for f in fidelities.flatten()])
        return best_f.to(X).reshape_as(fidelities)

    @concatenate_pending_points
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
            Output tensor of shape `b` or `1`.
        """
        # Build the `(batch_shape) x q` best_f tensor based on fidelity values
        best_f = self.get_best_f(X)

        # Evaluate model posterior and draw samples
        posterior = self.model.posterior(X, posterior_transform=self.posterior_transform)
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)

        # EI sample forward
        cost = self.cost_model(X)
        improvement = torch.clamp_min(obj - best_f, 0) / cost
        q_reduced = torch.amax(improvement, dim=-1)
        return torch.mean(q_reduced, dim=0)


class qMultiFidelityLogExpectedImprovement(qMultiFidelityExpectedImprovement):
    """Multi-fidelity log expected improvement acquisition."""

    _log = True

    @concatenate_pending_points
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
            Output tensor of shape `b` or `1`.
        """
        # Build the `(batch_shape) x q` best_f tensor based on fidelity values
        best_f = self.get_best_f(X)

        # Evaluate model posterior and draw samples
        posterior = self.model.posterior(X, posterior_transform=self.posterior_transform)
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)

        # LogEI sample forward
        cost = self.cost_model(X)
        improvement = log_fatplus(obj - best_f, tau=1e-6) - torch.log(cost)
        q_reduced = fatmax(improvement, tau=1e-2, dim=-1)
        return logmeanexp(q_reduced, dim=0)
