"""Bayesian optimiser module (using BoTorch)."""
import warnings
import math
import time
from typing import Optional, Union, Any, Callable
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from scipy.interpolate import interp1d
try:
    from scipy.stats import qmc
except ImportError:
    qmc = None
try:
    import torch
    from torch import Tensor
    import botorch
    from botorch.models.gp_regression_fidelity import FixedNoiseMultiFidelityGP, SingleTaskMultiFidelityGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import UpperConfidenceBound, qUpperConfidenceBound, PosteriorMean
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qSimpleRegret
    from botorch.acquisition.monte_carlo import MCAcquisitionFunction
    from botorch.acquisition import ProbabilityOfImprovement, qProbabilityOfImprovement
    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
    from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
    from botorch.acquisition.cost_aware import GenericCostAwareUtility
    from botorch.acquisition.utils import project_to_target_fidelity
    from botorch.optim import optimize_acqf, optimize_acqf_mixed
    from botorch.sampling import SobolQMCNormalSampler
    from piglot.objective import MultiFidelitySingleObjective
    from piglot.optimisers.optimiser import ScalarMultiFidelityOptimiser
    from botorch.models.model import Model
    from botorch.sampling.base import MCSampler
    from botorch.utils.transforms import (
        concatenate_pending_points,
        t_batch_mode_transform,
    )
    from botorch.acquisition.objective import (
        MCAcquisitionObjective,
        PosteriorTransform,
    )
    from botorch.acquisition.monte_carlo import MCAcquisitionFunction
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimisers.optimiser import missing_method
    ScalarOptimiser = missing_method("Bayesian optimisation (BoTorch)", "botorch")



SUPPORTED_ACQUISITIONS = (
    'qmfkg',
    'qmfkg_adaptive_ucb',
    'qmfkg_adaptive_ei',
    'ucb',
    'ei',
    'pi',
    'qkg',
    'la_mf_ei',
    'la_mf_ucb',
    'la_mf_mean',
    'la_asd',
    'mf_ei',
    'mf_ucb',
    'foumani',
)



def fit_mll_pytorch_loop(mll: ExactMarginalLogLikelihood, n_iters=100):
    mll.train()
    mll.model.likelihood.train()
    optimizer = torch.optim.Adam(mll.model.parameters(), lr=0.1)
    for _ in range(n_iters):
        optimizer.zero_grad()
        output = mll.model(mll.model.train_inputs[0])
        loss = -torch.mean(mll(output, mll.model.train_targets))
        loss.backward()
        optimizer.step()
    mll.model.eval()
    mll.model.likelihood.eval()



class qMaxFidelityExpectedImprovement(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X[:,:,-1] = 1
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei


class qVarianceExpectedImprovement(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        # stdev = math.sqrt(math.pi / 2) * (obj - mean).abs().mean(dim=0)
        improving = torch.where(obj > self.best_f.unsqueeze(-1).to(obj), 1, 0)
        obj = improving * (obj - mean)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei




class qMultiFidelityAcquisition(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        acquisition: MCAcquisitionFunction,
        fidelities: Tensor,
        cost_model: Callable[[Tensor], Tensor],
        fidelity_dim: int,
        acquisition_kwargs=None,
        improvement_func: Callable[[Tensor], Tensor]=lambda value: value,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.acquisition = acquisition
        self.cost_model = cost_model
        self.fidelities = fidelities
        self.fidelity_dim = fidelity_dim
        self.acquisition_kwargs = acquisition_kwargs
        self.improvement_func = improvement_func

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # Evaluate acquisition for this fidelity
        acq_func = self.acquisition(self.model, **self.acquisition_kwargs)
        imp = self.improvement_func(acq_func(X))
        # Evaluate expected improvement of the acquisition at the target fidelity
        X_hf = torch.clone(X)
        X_hf[:,:,self.fidelity_dim] = 1
        mean_acq = PosteriorMean(self.model)
        acq_hf = mean_acq(X_hf)
        fidelities = X[:,:,self.fidelity_dim].squeeze()
        fantasy_model = self.model.fantasize(X=X, sampler=self.sampler)
        # fantasy_acq_func = self.acquisition(fantasy_model, **self.acquisition_kwargs)
        # fantasy_acq = fantasy_acq_func(X_hf)
        fantasy_acq_func = PosteriorMean(fantasy_model)
        fantasy_acq = fantasy_acq_func(X_hf)
        fantasy_imp = fantasy_acq.mean(0) - acq_hf
        costs = torch.tensor(self.cost_model(fidelities.detach().numpy()))
        return (fidelities * imp + (1 - fidelities) * fantasy_imp) / costs


class qMixedMultiFidelityAcquisition(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        lf_acquisition: MCAcquisitionFunction,
        hf_acquisition: MCAcquisitionFunction,
        fidelities: Tensor,
        cost_model,
        fidelity_dim: int,
        lf_acquisition_kwargs=None,
        hf_acquisition_kwargs=None,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.lf_acquisition = lf_acquisition
        self.hf_acquisition = hf_acquisition
        self.lf_acquisition_kwargs = lf_acquisition_kwargs
        self.hf_acquisition_kwargs = hf_acquisition_kwargs
        self.cost_model = cost_model
        self.fidelities = fidelities
        self.fidelity_dim = fidelity_dim

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # Build low and high-fidelity acquisitions
        hf_acq_func = self.hf_acquisition(self.model, **self.hf_acquisition_kwargs)
        lf_acq_func = self.lf_acquisition(self.model, **self.lf_acquisition_kwargs)
        # Evaluate expected improvement of the acquisition at the target fidelity
        X_hf = torch.clone(X)
        X_hf[:,:,self.fidelity_dim] = 1
        acq_hf = lf_acq_func(X_hf)
        fidelities = X[:,:,self.fidelity_dim].squeeze()
        fantasy_model = self.model.fantasize(X=X, sampler=self.sampler)
        fantasy_acq_func = self.lf_acquisition(fantasy_model, **self.lf_acquisition_kwargs)
        fantasy_acq = fantasy_acq_func(X_hf)
        fantasy_imp = fantasy_acq.mean(0) - acq_hf
        costs = torch.tensor(self.cost_model(fidelities.detach().numpy()))
        # return fantasy_imp / costs
        return (fidelities * hf_acq_func(X) + (1 - fidelities) * fantasy_imp) / costs



class qPosteriorVariance(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        negate: bool = False,
        scale: float = 1.0,
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
        self.mult = -scale if negate else scale

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        ucb_samples = math.sqrt(math.pi / 2) * (obj - mean).abs()
        return self.mult * ucb_samples.max(dim=-1)[0].mean(dim=0)




class qMultiFidelityExpectedImprovement(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        fidelities: Tensor,
        cost_model,
        fidelity_dim: int,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
        self.cost_model = cost_model
        self.fidelities = fidelities
        self.fidelity_dim = fidelity_dim


    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # Evaluate qEI for this fidelity
        posterior = self.model.posterior(X=X, posterior_transform=self.posterior_transform)
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        # Evaluate expected improvement of qEI at the target fidelity
        X_hf = torch.clone(X)
        X_hf[:,:,self.fidelity_dim] = 1
        fidelities = X[:,:,self.fidelity_dim].squeeze()
        fantasy_model = self.model.fantasize(X=X, sampler=self.sampler)
        fantasy_posterior = fantasy_model.posterior(X=X_hf, posterior_transform=self.posterior_transform)
        hf_samples = self.get_posterior_samples(fantasy_posterior)
        hf_obj = self.objective(hf_samples, X=X_hf)
        # improvement = hf_obj - self.best_f.to(hf_obj)
        # expected = improvement.mean(dim=0).clamp_min(0)
        # expected = improvement.clamp_min(0).mean(dim=0)
        expected = (hf_obj.mean(dim=0, keepdim=True) - hf_obj).abs().mean(dim=0)
        q_mfei = expected.max(dim=-1)[0].mean(dim=0)
        costs = torch.tensor(self.cost_model(fidelities.detach().numpy()))
        return q_mfei / costs
        # return (q_ei * fidelities + q_mfei * (1 - fidelities)) / costs

    # @concatenate_pending_points
    # @t_batch_mode_transform()
    # def forward(self, X: Tensor) -> Tensor:
    #     # Evaluate qEI for this fidelity
    #     posterior = self.model.posterior(X=X, posterior_transform=self.posterior_transform)
    #     samples = self.get_posterior_samples(posterior)
    #     obj = self.objective(samples, X=X)
    #     obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
    #     q_ei = obj.max(dim=-1)[0].mean(dim=0)
    #     # Evaluate expected improvement of qEI at the target fidelity
    #     X_hf = torch.clone(X)
    #     fidelities = X[:,:,self.fidelity_dim].squeeze()
    #     X_hf[:,:,self.fidelity_dim] = 1
    #     fantasy_model = self.model.fantasize(X=X, sampler=self.sampler)
    #     fantasy_posterior = fantasy_model.posterior(X=X_hf, posterior_transform=self.posterior_transform)
    #     hf_samples = self.get_posterior_samples(fantasy_posterior)
    #     hf_obj = self.objective(hf_samples, X=X_hf)
    #     hf_obj = (hf_obj - self.best_f.unsqueeze(-1).to(hf_obj)).mean(dim=0)
    #     q_ei_as = hf_obj.max(dim=-1)[0].mean(dim=0)
    #     costs = torch.tensor(self.cost_model(fidelities.detach().numpy()))
    #     return q_ei * fidelities + (q_ei_as - q_ei) * (1 - fidelities) / (costs + 4)
    #     # return q_ei * fidelities + q_ei * (q_ei_as - q_ei) * (1 - fidelities) / costs



class BayesDataset:

    def __init__(self, n_dim, n_outputs, bounds, export=None, dtype=torch.float64):
        self.dtype = dtype
        self.n_points = 0
        self.n_dim = n_dim
        self.n_outputs = n_outputs
        self.params = torch.empty((0, n_dim), dtype=dtype)
        self.values = torch.empty((0, n_outputs), dtype=dtype)
        self.variances = torch.empty((0, n_outputs), dtype=dtype)
        self.fidelities = torch.empty((0, 1), dtype=dtype)
        self.lbounds = torch.tensor(bounds[:, 0], dtype=dtype)
        self.ubounds = torch.tensor(bounds[:, 1], dtype=dtype)
        self.export = export

    def save(self, output):
        # Build a joint tensor with all data for the highest fidelity
        mask = self.high_fidelity_mask()
        joint = torch.cat([self.params[mask,:], self.values[mask,:], self.variances[mask,:]], dim=1)
        torch.save(joint, output)

    def fidelity_mask(self, fidelity):
        return torch.isclose(self.fidelities, fidelity * torch.ones(1, dtype=self.dtype))[:,0]

    def high_fidelity_mask(self):
        return torch.isclose(self.fidelities, torch.ones(1, dtype=self.dtype))[:,0]

    def push(self, params, values, variances, fidelity):
        torch_params = torch.tensor(params, dtype=self.dtype).unsqueeze(0)
        torch_value = torch.tensor([values], dtype=self.dtype).unsqueeze(0)
        torch_variance = torch.tensor([variances], dtype=self.dtype).unsqueeze(0)
        torch_fidelity = torch.tensor([fidelity], dtype=self.dtype).unsqueeze(0)
        self.params = torch.cat([self.params, torch_params], dim=0)
        self.values = torch.cat([self.values, torch_value], dim=0)
        self.variances = torch.cat([self.variances, torch_variance], dim=0)
        self.fidelities = torch.cat([self.fidelities, torch_fidelity], dim=0)
        self.n_points += 1
        # Update the dataset file after every push
        if self.export:
            self.save(self.export)

    def get_params_value_pairs(self, fidelity=None):
        mask = self.high_fidelity_mask() if fidelity is None else \
               torch.isclose(self.fidelities, fidelity * torch.ones(1, dtype=self.dtype))[:,0]
        return self.params[mask].cpu().numpy(), self.values[mask].cpu().numpy()



class BayesianBoTorchMF(ScalarMultiFidelityOptimiser):

    def __init__(self, n_initial=5, acquisition='ucb', log_space=False, def_variance=0,
                 beta=0.5, beta_final=None, noisy=False, q=1, seed=42,
                 export=None, n_test=0):
        super().__init__('BoTorch')
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.log_space = log_space
        self.def_variance = def_variance
        self.beta = beta
        self.beta_final = beta if beta_final is None else beta_final
        self.noisy = bool(noisy)
        self.q = q
        self.seed = seed
        self.export = export
        self.n_test = n_test
        if self.acquisition not in SUPPORTED_ACQUISITIONS:
            raise RuntimeError(f"Unkown acquisition function {self.acquisition}")
        if not self.acquisition.startswith('q') and self.q != 1:
            raise RuntimeError("Can only use q != 1 for quasi-Monte Carlo acquisitions")
        torch.set_num_threads(1)
        self.timings = []

    def get_candidates(self, n_dim, dataset: BayesDataset, beta, test_dataset: BayesDataset, objective: MultiFidelitySingleObjective):
        # Get data needed for unit-cube space mapping and standardisation
        X_delta = (dataset.ubounds - dataset.lbounds)
        y_avg = torch.mean(dataset.values, dim=-2)
        y_std = torch.std(dataset.values, dim=-2)

        # Take particular care if we only have one point to avoid divisions by zero
        if dataset.n_points == 1:
            y_std = 1

        # Build unit cube space and standardised values
        X_cube = (dataset.params - dataset.lbounds) / X_delta
        y_standard = (dataset.values - y_avg) / y_std
        var_standard = dataset.variances / y_std

        # Clamp variances to prevent warnings from GPyTorch
        var_standard = torch.clamp_min(var_standard, 1e-6)

        # Build the model
        fidelity_dim = n_dim
        X_cube_mf = torch.cat([X_cube, dataset.fidelities], dim=1)
        if self.noisy:
            model = SingleTaskMultiFidelityGP(
                X_cube_mf,
                y_standard,
                # data_fidelity=fidelity_dim,
                iteration_fidelity=fidelity_dim,
            )
        else:
            model = FixedNoiseMultiFidelityGP(
                X_cube_mf,
                y_standard,
                var_standard,
                # data_fidelity=fidelity_dim,
                iteration_fidelity=fidelity_dim,
            )

        # Fit the GP (in case of trouble, we fall back to an Adam-based optimiser)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except botorch.exceptions.ModelFittingError:
            warnings.warn('Optimisation of the MLL failed, falling back to PyTorch optimiser')
            fit_mll_pytorch_loop(mll)

        # Evaluate GP performance with the test dataset
        cv_error = None
        if self.n_test > 0:
            X_test_cube = (test_dataset.params - dataset.lbounds) / X_delta
            y_test_standard = (test_dataset.values - y_avg) / y_std
            with torch.no_grad():
                posterior = model.posterior(X_test_cube)
                cv_error = ((posterior.mean - y_test_standard) ** 2).mean()
        
        # Find best point in the unit-cube and standardised dataset (at highest fidelity)
        y_best = {fid: max(y_standard[dataset.fidelity_mask(fid),:]) for fid in objective.fidelities}
        idx_best = {fid: np.argmax(y_standard[dataset.fidelity_mask(fid),:]) for fid in objective.fidelities}
        arg_best = {fid: X_cube_mf[dataset.fidelity_mask(fid),:][idx_best[fid]].unsqueeze(0) for fid in objective.fidelities}
        y_best_hf = y_best[max(objective.fidelities)].item()

        # Build cost model for the fidelities
        costs = [objective.cost(fid) + 1.0 * np.mean(self.timings) for fid in objective.fidelities]
        cost_model = interp1d(objective.fidelities, costs)

        # Common data for optimisers
        num_restarts = 12
        raw_samples = max(256, 16 * n_dim * n_dim)
        sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)
        bounds = torch.stack((torch.zeros(n_dim + 1, dtype=dataset.dtype),
                              torch.ones(n_dim + 1, dtype=dataset.dtype)))

        # Obtain next point based on the multi-fidelity strategy
        if self.acquisition in ('qmfkg', 'qmfkg_adaptive_ucb', 'qmfkg_adaptive_ei'):
            # Multi-fidelity knowledge gradient (with or without the adaptive strategy)
            num_fantasies = 64
            sampler = SobolQMCNormalSampler(torch.Size([num_fantasies]), seed=self.seed)
            # Build cost model
            def cost_ut(X, deltas):
                fidelities = X[:,:,fidelity_dim].detach().numpy()
                costs = torch.tensor(cost_model(fidelities)).sum(dim=-1)
                return deltas / costs
            cost_utility = GenericCostAwareUtility(cost_ut)
            # Find current best posterior mean (at maximum fidelity)
            _, current_value = optimize_acqf_mixed(
                PosteriorMean(model),
                bounds=bounds,
                q=1,
                num_restarts=2 * num_restarts,
                raw_samples=2 * raw_samples,
                fixed_features_list=[{fidelity_dim: max(objective.fidelities)}],
                options={"sample_around_best": True},
            )
            # Build and optimise the MFKG acquisition (for discrete fidelities)
            acq = qMultiFidelityKnowledgeGradient(
                model,
                num_fantasies=num_fantasies,
                sampler=sampler,
                current_value=current_value,
                cost_aware_utility=cost_utility,
                project=lambda X: project_to_target_fidelity(
                    X=X,
                    target_fidelities={fidelity_dim: max(objective.fidelities)},
                ),
            )
            candidates, value = optimize_acqf_mixed(
                acq,
                bounds=bounds,
                q=self.q,
                fixed_features_list=[{fidelity_dim: fid} for fid in objective.fidelities],
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={
                    "batch_limit": 12,
                    "maxiter": 200,
                },
            )
            # Evaluate the adaptive strategy, if requested
            if self.acquisition in ('qmfkg_adaptive_ucb', 'qmfkg_adaptive_ei'):
                # Build the apropriate adaptive acquisition and improvement function
                if self.acquisition == 'qmfkg_adaptive_ei':
                    adapt_acq = ExpectedImprovement(model, y_best_hf)
                    adapt_imp = lambda val: val
                elif self.acquisition == 'qmfkg_adaptive_ucb':
                    adapt_acq = UpperConfidenceBound(model, self.beta)
                    adapt_imp = lambda val: val - y_best_hf
                # Optimise the adaptive acquisition function (at target fidelity)
                adapt_candidates, adapt_value = optimize_acqf_mixed(
                    adapt_acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    fixed_features_list=[{fidelity_dim: max(objective.fidelities)}],
                    options={"sample_around_best": True},
                )
                adapt_improvement = adapt_imp(adapt_value) / cost_model(max(objective.fidelities))
                # Switch to this candidate when it outperforms the MFKG one
                if adapt_improvement.item() > value.item():
                    candidates = adapt_candidates
                    value = adapt_improvement

        # Look-ahead multi-fidelity strategy
        elif self.acquisition in ('la_mf_ei', 'la_mf_ucb', 'la_mf_mean'):
            # Build list of supported options
            acq_type = {
                'la_mf_ei': ExpectedImprovement,
                'la_mf_ucb': UpperConfidenceBound,
                'la_mf_mean': PosteriorMean,
            }
            acq_kwargs = {
                'la_mf_ei': {'best_f': y_best_hf},
                'la_mf_ucb': {'beta': self.beta},
                'la_mf_mean': {},
            }
            acq_improvement_func = {
                'la_mf_ei': lambda x: x,
                'la_mf_ucb': lambda x: torch.clamp_min(x - y_best_hf, 0),
                'la_mf_mean': lambda x: torch.clamp_min(x - y_best_hf, 0),
            }
            # Build and optimise the acquisition function
            sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)
            acq = qMultiFidelityAcquisition(
                model,
                acq_type[self.acquisition],
                objective.fidelities,
                cost_model,
                fidelity_dim,
                sampler=sampler,
                acquisition_kwargs=acq_kwargs[self.acquisition],
                improvement_func=acq_improvement_func[self.acquisition],
            )
            points = {}
            improvements = {}
            for fidelity in objective.fidelities:
                points[fidelity], improvements[fidelity] = optimize_acqf_mixed(
                    acq,
                    bounds=bounds,
                    q=self.q,
                    fixed_features_list=[{fidelity_dim: fidelity}],
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"sample_around_best": True},
                )
            print(improvements)
            candidates = points[max(improvements, key=improvements.get)]
            # candidates, value = optimize_acqf_mixed(
            #     acq,
            #     bounds=bounds,
            #     q=self.q,
            #     fixed_features_list=[{fidelity_dim: fid} for fid in objective.fidelities],
            #     num_restarts=num_restarts,
            #     raw_samples=raw_samples,
            #     options={"sample_around_best": True},
            # )

        # Cost-weighted improvement-based multi-fidelity strategy
        elif self.acquisition in ('mf_ei', 'mf_ucb'):
            # Generate acquistion builders and improvement function
            acq_builder = {
                'mf_ei': lambda model, fidelity: ExpectedImprovement(model, y_best[fidelity]),
                'mf_ucb': lambda model, fidelity: UpperConfidenceBound(model, self.beta),
            }[self.acquisition]
            improvement_func = {
                'mf_ei': lambda value, fidelity: value,
                'mf_ucb': lambda value, fidelity: value - y_best[fidelity],
            }[self.acquisition]
            # Individually optimise each fidelity with a different acquisition
            points = {}
            improvements = {}
            for fidelity in objective.fidelities:
                acq = acq_builder(model, fidelity)
                acq_best = acq(arg_best[fidelity].unsqueeze(0)).detach()
                points[fidelity], value = optimize_acqf_mixed(
                    acq,
                    bounds=bounds,
                    q=self.q,
                    fixed_features_list=[{fidelity_dim: fidelity}],
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"sample_around_best": True},
                )
                # improvements[fidelity] = (value - acq_best) / cost_model(fidelity)
                improvements[fidelity] = improvement_func(value, fidelity) / cost_model(fidelity)
            # Select best cost-weighted improvement
            print(improvements)
            candidates = points[max(improvements, key=improvements.get)]

        elif self.acquisition == 'foumani':
            points = {}
            improvements = {}
            for fidelity in objective.fidelities:
                if fidelity < max(objective.fidelities):
                    # acq = qVarianceExpectedImprovement(model, y_best[fidelity], sampler=sampler)
                    acq = ExpectedImprovement(model, y_best[fidelity])
                else:
                    acq = ProbabilityOfImprovement(model, y_best_hf)
                points[fidelity], value = optimize_acqf_mixed(
                    acq,
                    bounds=bounds,
                    q=self.q,
                    fixed_features_list=[{fidelity_dim: fidelity}],
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"sample_around_best": True},
                )
                improvements[fidelity] = value / cost_model(fidelity)
            print(improvements)
            candidates = points[max(improvements, key=improvements.get)]


        elif self.acquisition == 'la_asd':
            acq = qMultiStepLookahead(
                model,
                batch_sizes=[1],
                num_fantasies=[32],
                valfunc_cls=[None, FixedFeatureAcquisitionFunction],
                valfunc_argfacs=[
                    None,
                    lambda model, X: {
                        'acq_function': qExpectedImprovement(model, y_best_hf),
                        'd': n_dim + 1,
                        'columns': [fidelity_dim],
                        'values': [1],
                    }
                ],
                inner_mc_samples=[None, 64],
            )
            q_prime = acq.get_augmented_q_batch_size(self.q)
            points = {}
            improvements = {}
            for fidelity in objective.fidelities:
                points[fidelity], value = optimize_acqf(
                    acq,
                    bounds=bounds,
                    q=q_prime,
                    # fixed_features_list=[{fidelity_dim: fidelity}],
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"sample_around_best": True},
                )
                print(improvements)
                improvements[fidelity] = value / cost_model(fidelity)
            candidates = points[max(improvements, key=improvements.get)]
            


        # Use standard acquisition functions at the highest fidelity
        else:
            # Build the acquisition function
            q_prime = 1
            if self.acquisition == 'ucb':
                acq = UpperConfidenceBound(model, beta)
            elif self.acquisition == 'ei':
                acq = ExpectedImprovement(model, y_best_hf)
            elif self.acquisition == 'pi':
                acq = ProbabilityOfImprovement(model, y_best_hf)
            elif self.acquisition == 'qkg':
                num_restarts = 6
                raw_samples = 128
                sampler = SobolQMCNormalSampler(torch.Size([64]), seed=self.seed)
                acq = qKnowledgeGradient(model, sampler=sampler)
            elif self.acquisition == 'la_ei':
                acq = qMultiStepLookahead(
                    model,
                    batch_sizes=[1],
                    num_fantasies=[32],
                    valfunc_cls=[None, qExpectedImprovement],
                    valfunc_argfacs=[None, lambda model, X: {'best_f': y_best_hf}],
                    objective=objective,
                    inner_mc_samples=[None, 64],
                )
                q_prime = acq.get_augmented_q_batch_size(self.q)
            # Optimise the acquisition function at highest fidelity
            candidates, value = optimize_acqf_mixed(
                acq,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                fixed_features_list=[{fidelity_dim: max(objective.fidelities)}],
                options={"sample_around_best": True},
            )











#             # current_value = y_best[max(objective.fidelities)]
#             candidate_ei, current_value = optimize_acqf_mixed(
#                 PosteriorMean(model),
#                 bounds=bounds,
#                 q=1,
#                 num_restarts=num_restarts,
#                 raw_samples=raw_samples,
#                 fixed_features_list=[{n_dim: max(objective.fidelities)}],
#                 options={"sample_around_best": True},
#             )
#             expected_improvement = current_value - y_best[max(objective.fidelities)]
#             candidate_ei, expected_improvement = optimize_acqf_mixed(
#                 ExpectedImprovement(model, y_best[max(objective.fidelities)].item()),
#                 bounds=bounds,
#                 q=1,
#                 num_restarts=num_restarts,
#                 raw_samples=raw_samples,
#                 fixed_features_list=[{n_dim: max(objective.fidelities)}],
#                 options={"sample_around_best": True},
#             )
#             cw_expected_improvement = expected_improvement / cost_model(max(objective.fidelities))
#             print(f'Current value: {current_value.item()} Best: {y_best[max(objective.fidelities)].item()} EI: {expected_improvement.item()}')
#             print(f'Cost-weighted gains: qKG: {value.item()} EI: {cw_expected_improvement.item()}')

#             sel_acq = 'qKG'
#             if cw_expected_improvement.item() >= value.item():
#                 candidates = candidate_ei
#                 value = cw_expected_improvement
#                 sel_acq = 'EI'
#             target_fidelity = candidates[0, fidelity_dim].item()
#             print(f'Selecting fidelity {target_fidelity} ({sel_acq}: {value.item()})')

#         # Build the acquisition function

#         # Find next candidate(s)
#         norm_improvement = {}
#         candidates_fidelities = {}



#         # hf_candidates, _ = optimize_acqf_mixed(
#         #         acq,
#         #         bounds=bounds,
#         #         q=self.q,
#         #         fixed_features_list=[{n_dim: 1.0}],
#         #         num_restarts=num_restarts,
#         #         raw_samples=raw_samples,
#         #         options={"sample_around_best": True},
#         #     )
#         # candidates = hf_candidates
#         # for fidelity in objective.fidelities:
#         #     # Manual fantasisation
#         #     candidates[:,-1] = fidelity
#         #     with warnings.catch_warnings():
#         #         warnings.simplefilter("ignore")
#         #         fantasy = model.fantasize(candidates, sampler)
#         #     improvement = (model.posterior(hf_candidates).variance - fantasy.posterior(hf_candidates).variance.mean(dim=0)).detach().item()
#         #     norm_improvement[fidelity] = improvement / objective.cost(fidelity)
#         #     candidates_fidelities[fidelity] = candidates
#         #     print(f'Fidelity {fidelity:<3.1f}: Cost: {objective.cost(fidelity):<5.3f} Candidate: {candidates[:,:-1].squeeze()} Improvement: {improvement:<6.4e} (norm. {norm_improvement[fidelity]:<6.4e})')


            
                    

#         # for fidelity in objective.fidelities:
#         #     improv_function = UpperConfidenceBound(model, beta)
#         #     # improv_function = ExpectedImprovement(model, y_best[fidelity])
#         #     candidates, _ = optimize_acqf_mixed(
#         #         improv_function,
#         #         bounds=bounds,
#         #         q=self.q,
#         #         fixed_features_list=[{n_dim: fidelity}],
#         #         num_restarts=num_restarts,
#         #         raw_samples=raw_samples,
#         #         options={"sample_around_best": True},
#         #     )
#         #     improvement = (improv_function(candidates) - y_best[fidelity]).detach().item()
#         #     # improvement = (improv_function(candidates) - improv_function(arg_best[fidelity])).detach().item()
#         #     # improvement = improv_function(candidates).detach().item()

#         #     # # Manual fantasisation
#         #     # with warnings.catch_warnings():
#         #     #     warnings.simplefilter("ignore")
#         #     #     fantasy = model.fantasize(candidates, sampler)
#         #     # high_fidelity_candidate = candidates
#         #     # high_fidelity_candidate[:,-1] = 1.0
#         #     # improvement = (fantasy.posterior(high_fidelity_candidate).mean.mean() - model.posterior(high_fidelity_candidate).mean).detach().item()
#         #     # # improvement = (fantasy.posterior(high_fidelity_candidate).mean.mean() - y_best[fidelity]).detach().item()

#         #     norm_improvement[fidelity] = improvement / (objective.cost(fidelity) + 1.0 * np.mean(self.timings))
#         #     candidates_fidelities[fidelity] = candidates
#         #     print(f'Fidelity {fidelity:<3.1f}: Cost: {objective.cost(fidelity):<5.3e} Improvement: {improvement:<6.4e} (norm. {norm_improvement[fidelity]:<6.4e})')
#         #     # print(f'Fidelity {fidelity:<3.1f}: Cost: {objective.cost(fidelity):<5.3f} Candidate: {candidates[:,:-1].squeeze()} Improvement: {improvement:<6.4e} (norm. {norm_improvement[fidelity]:<6.4e})')

#         # target_fidelity = max(norm_improvement, key=norm_improvement.get)
#         # # target_fidelity = min(objective.fidelities)
#         # candidates = candidates_fidelities[target_fidelity]
#         # print(f'Selecting fidelity {target_fidelity} ({max(norm_improvement.values())})')

#         # acq = qMultiFidelityExpectedImprovement(model, y_best[max(objective.fidelities)], objective.fidelities, cost_model, n_dim, sampler=sampler)
#         # acq = qMultiFidelityAcquisition(
#         #     model,
#         #     qExpectedImprovement,
#         #     objective.fidelities,
#         #     cost_model,
#         #     n_dim,
#         #     acquisition_kwargs={
#         #         'best_f': y_best[max(objective.fidelities)],
#         #         'sampler': sampler,
#         #     },
#         #     sampler=sampler,
#         # )
#         # acq = qMultiFidelityAcquisition(
#         #     model,
#         #     qSimpleRegret,
#         #     objective.fidelities,
#         #     cost_model,
#         #     n_dim,
#         #     acquisition_kwargs={
#         #         'sampler': sampler,
#         #     },
#         #     sampler=sampler,
#         # )
#         # acq = qMultiFidelityAcquisition(
#         #     model,
#         #     qPosteriorVariance,
#         #     objective.fidelities,
#         #     cost_model,
#         #     n_dim,
#         #     acquisition_kwargs={
#         #         'negate': True,
#         #         'sampler': sampler,
#         #     },
#         #     sampler=sampler,
#         # )
#         # acq = qMultiFidelityAcquisition(
#         #     model,
#         #     qUpperConfidenceBound,
#         #     objective.fidelities,
#         #     cost_model,
#         #     n_dim,
#         #     acquisition_kwargs={
#         #         'beta': 1.0,
#         #         'sampler': sampler,
#         #     },
#         #     sampler=sampler,
#         # )

#         # acq = qMixedMultiFidelityAcquisition(
#         #     model,
#         #     qPosteriorVariance,
#         #     qExpectedImprovement,
#         #     objective.fidelities,
#         #     cost_model,
#         #     n_dim,
#         #     lf_acquisition_kwargs={
#         #         'negate': True,
#         #         'scale': 0.05,
#         #         'sampler': sampler,
#         #     },
#         #     hf_acquisition_kwargs={
#         #         'beta': self.beta,
#         #         'best_f': y_best[max(objective.fidelities)],
#         #         'sampler': sampler,
#         #     },
#         #     sampler=sampler,
#         # )
#         # for fidelity in objective.fidelities:
#         #     # acq = qMultiFidelityAcquisition(
#         #     #     model,
#         #     #     qExpectedImprovement,
#         #     #     objective.fidelities,
#         #     #     cost_model,
#         #     #     n_dim,
#         #     #     acquisition_kwargs={
#         #     #         'best_f': y_best[fidelity],
#         #     #         'sampler': sampler,
#         #     #     },
#         #     #     sampler=sampler,
#         #     # )
#         #     cost = cost_model(fidelity) / cost_model(max(objective.fidelities))
#         #     acq = qUpperConfidenceBound(model, self.beta / cost, sampler=sampler)
#         #     candidates, value = optimize_acqf_mixed(
#         #         acq,
#         #         bounds=bounds,
#         #         q=self.q,
#         #         fixed_features_list=[{n_dim: fidelity}],
#         #         num_restarts=num_restarts,
#         #         raw_samples=raw_samples,
#         #         options={"sample_around_best": True},
#         #     )
#         #     norm_improvement[fidelity] = value - y_best[fidelity]
#         #     candidates_fidelities[fidelity] = candidates
#         #     print(f'Fidelity {fidelity:<3.1f}: Cost: {cost_model(fidelity):<5.3f} Candidate: {candidates[:,:-1].squeeze()} Value: {value:<6.4f} ({norm_improvement[fidelity]})')

#         # print(y_standard[dataset.high_fidelity_mask(),:])
#         # print(y_standard[dataset.high_fidelity_mask(),:] * y_std + y_avg)

#         # target_fidelity = max(norm_improvement, key=norm_improvement.get)
#         # candidates = candidates_fidelities[target_fidelity]
#         # print(f'Selecting fidelity {target_fidelity} ({norm_improvement[target_fidelity].item()})')
#         # # input()



# ################################################################################################################################
# ########       MFKG
# ################################################################################################################################
#         num_fantasies = 64

#         # current_value = y_best[max(objective.fidelities)]
#         candidate_ei, current_value = optimize_acqf_mixed(
#             PosteriorMean(model),
#             bounds=bounds,
#             q=1,
#             num_restarts=num_restarts,
#             raw_samples=raw_samples,
#             fixed_features_list=[{n_dim: max(objective.fidelities)}],
#             options={"sample_around_best": True},
#         )
#         expected_improvement = current_value - y_best[max(objective.fidelities)]
#         fidelity_dim = n_dim
#         def cost_ut(X, deltas):
#             fidelities = X[:,:,fidelity_dim].detach().numpy()
#             costs = torch.tensor(cost_model(fidelities)).sum(dim=-1)
#             return deltas / costs
#         cost_utility = GenericCostAwareUtility(cost_ut)
#         sampler = SobolQMCNormalSampler(torch.Size([num_fantasies]), seed=self.seed)
#         acq = qMultiFidelityKnowledgeGradient(
#             model,
#             num_fantasies=num_fantasies,
#             sampler=sampler,
#             current_value=current_value,
#             cost_aware_utility=cost_utility,
#             project=lambda X: project_to_target_fidelity(X=X, target_fidelities={fidelity_dim: max(objective.fidelities)}),
#         )
#         candidates, value = optimize_acqf_mixed(
#             acq,
#             bounds=bounds,
#             q=self.q,
#             fixed_features_list=[{n_dim: fid} for fid in objective.fidelities],
#             num_restarts=12,
#             raw_samples=256,
#             options={
#                 "batch_limit": 12,
#                 "maxiter": 200,
#             },
#         )
#         candidate_ei, expected_improvement = optimize_acqf_mixed(
#             ExpectedImprovement(model, y_best[max(objective.fidelities)].item()),
#             bounds=bounds,
#             q=1,
#             num_restarts=num_restarts,
#             raw_samples=raw_samples,
#             fixed_features_list=[{n_dim: max(objective.fidelities)}],
#             options={"sample_around_best": True},
#         )
#         cw_expected_improvement = expected_improvement / cost_model(max(objective.fidelities))
#         print(f'Current value: {current_value.item()} Best: {y_best[max(objective.fidelities)].item()} EI: {expected_improvement.item()}')
#         print(f'Cost-weighted gains: qKG: {value.item()} EI: {cw_expected_improvement.item()}')

#         sel_acq = 'qKG'
#         if cw_expected_improvement.item() >= value.item():
#             candidates = candidate_ei
#             value = cw_expected_improvement
#             sel_acq = 'EI'
#         target_fidelity = candidates[0, fidelity_dim].item()
#         print(f'Selecting fidelity {target_fidelity} ({sel_acq}: {value.item()})')

        
        # Re-map to original space and remove fidelity from the candidate
        target_fidelity = candidates[:, fidelity_dim].item()

        # self.__plot(model, objective.fidelities, dataset, y_avg, y_std)
        candidates_map = torch.empty((self.q, n_dim))
        for i in range(self.q):
            candidates_map[i, :n_dim] = dataset.lbounds + candidates[i, :n_dim] * X_delta
        return candidates_map.cpu().numpy(), target_fidelity, cv_error


    def __plot(self, model, fidelities, dataset: BayesDataset, y_avg, y_std):
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x_vector = torch.linspace(0, 1, 1000)
        x_transf = 50 + 450 * x_vector
        for i, fidelity in enumerate(fidelities):
            x_complete = torch.stack((x_vector, fidelity * torch.ones_like(x_vector)), dim=1)
            posterior = model.posterior(x_complete)
            mean = posterior.mean.detach().squeeze() * y_std.squeeze() + y_avg.squeeze()
            vars = posterior.variance.detach().squeeze() * y_std.squeeze() * y_std.squeeze()
            ubound = mean + 2 * torch.sqrt(vars)
            lbound = mean - 2 * torch.sqrt(vars)
            fid_mask = dataset.fidelity_mask(fidelity)
            x_fid = 50 + 450 * (dataset.params[fid_mask,0] + 1) / 2
            y_fid = dataset.values[fid_mask,0]
            ax.plot(x_transf, mean, label=f'Mean (fidelity={fidelity})', color=cm.Set1(i))
            ax.fill_between(x_transf, lbound, ubound, fc=cm.Set1(i), alpha=0.2)
            ax.scatter(x_fid, y_fid, label=f'Observations (fidelity={fidelity})', color=cm.Set1(i))
        ax.set_xlabel('Yield stress')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()

    def _eval_candidates(self, func, candidates, fidelity):
        # Single candidate case
        if self.q == 1:
            return [func(candidate, fidelity) for candidate in candidates]

        # Multi-candidate: run cases in parallel
        pool = Pool(self.q)
        return pool.map(lambda x: func(x, fidelity, unique=True), candidates)
    
    def _get_best_point(self, dataset: BayesDataset):
        params, losses = dataset.get_params_value_pairs()
        idx = np.argmin(losses)
        return params[idx, :], losses[idx, :].squeeze()

    def _get_random_points(self, n_points, n_dim, seed, bound):
        if qmc is None:
            points = np.random.default_rng(seed=seed).random([n_points, n_dim])
        else:
            points = qmc.Sobol(n_dim, seed=seed).random(n_points)
        return [point * (bound[:, 1] - bound[:, 0]) + bound[:, 0] for point in points]

    def _optimise(
        self,
        objective: MultiFidelitySingleObjective,
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ):
        """
        Parameters
        ----------
        func : callable
            function to optimize
        n_dim : integer
            dimension, i.e., number of parameters to optimize
        n_iter : integer
            maximum number of iterations
        bound : array
            first column corresponding to the lower bound, and second column to the
            upper bound
        init_shot : list
            initial shot for the optimization problem

        Returns
        -------
        best_value : float
            best loss function value
        best_solution : list
            best parameter solution
        """

        # Negate the loss function to convert problem to a maximisation
        loss_transformer = lambda x: -np.log(x) if self.log_space else -x
        inv_loss_transformer = lambda x: np.exp(-x) if self.log_space else -x

        fidelities = objective.fidelities

        # Build initial dataset with the initial shot
        dataset = BayesDataset(n_dim, 1, bound, export=self.export)
        for fidelity in fidelities:
            dataset.push(init_shot, loss_transformer(objective(init_shot, fidelity)), self.def_variance, fidelity)

        # If requested, sample some random points before starting (in parallel if possible)
        random_points = self._get_random_points(self.n_initial, n_dim, self.seed, bound)
        init_losses = self._eval_candidates(objective, random_points, min(fidelities))
        for i, loss in enumerate(init_losses):
            dataset.push(random_points[i], loss_transformer(loss), self.def_variance, min(fidelities))
        # init_losses = self._eval_candidates(objective, random_points, max(fidelities))
        # for i, loss in enumerate(init_losses):
        #     dataset.push(random_points[i], loss_transformer(loss), self.def_variance, max(fidelities))

        # Build test dataset (in parallel if possible)
        test_dataset = BayesDataset(n_dim, 1, bound)
        if self.n_test > 0:
            test_points = self._get_random_points(self.n_test, n_dim, self.seed + 1, bound)
            test_losses = self._eval_candidates(objective, test_points, min(fidelities))
            for i, loss in enumerate(test_losses):
                test_dataset.push(test_points[i], loss_transformer(loss), self.def_variance, min(fidelities))

        # Find current best point to return to the driver
        best_params, best_loss = self._get_best_point(dataset)
        best_loss = inv_loss_transformer(best_loss)
        self._progress_check(0, best_loss, best_params)
        self.timings = [0]

        # Optimisation loop
        for i_iter in range(n_iter):
            beta = (self.beta * (n_iter - i_iter - 1) + self.beta_final * i_iter) / n_iter

            # Generate and evaluate candidates (in parallel if possible)
            begin = time.perf_counter()
            candidates, fidelity, cv_error = self.get_candidates(n_dim, dataset, beta, test_dataset, objective)
            self.timings.append(time.perf_counter() - begin)
            losses = self._eval_candidates(objective, candidates, fidelity)

            # Find best value for this batch and update dataset
            if np.isclose(fidelity, max(objective.fidelities)):
                best_idx = np.argmin(losses)
                best_params = candidates[best_idx, :]
                best_loss = losses[best_idx]
            for i, loss in enumerate(losses):
                dataset.push(candidates[i, :], loss_transformer(loss), self.def_variance, fidelity)

            # Update progress
            extra = f'Val. {cv_error:6.4}' if cv_error else None
            if self._progress_check(i_iter + 1, best_loss, best_params, extra_info=extra):
                break

        # Return optimisation result
        best_params, best_loss = self._get_best_point(dataset)
        return best_params, inv_loss_transformer(best_loss)
