"""Bayesian optimiser module under composite optimisation (using BoTorch)."""
import time
from typing import Optional, Union, Any, Callable
import warnings
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from scipy.interpolate import interp1d
from piglot.objective import Composition, MultiFidelityCompositeObjective
try:
    from scipy.stats import qmc
except ImportError:
    qmc = None
try:
    import torch
    from torch import Tensor
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import botorch
    from botorch.models import FixedNoiseGP, SingleTaskGP
    from botorch.models.gp_regression_fidelity import FixedNoiseMultiFidelityGP, SingleTaskMultiFidelityGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qUpperConfidenceBound, qSimpleRegret
    from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement
    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
    from botorch.acquisition.monte_carlo import MCAcquisitionFunction
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.optim import optimize_acqf, optimize_acqf_mixed
    from botorch.sampling import SobolQMCNormalSampler
    from piglot.optimisers.optimiser import CompositeMultiFidelityOptimiser
    from botorch.acquisition.cost_aware import GenericCostAwareUtility
    from botorch.acquisition.utils import project_to_target_fidelity
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
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimisers.optimiser import missing_method
    CompositeOptimiser = missing_method("Bayesian optimisation (BoTorch)", "botorch")




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
    'foumani',
    'mf_ucb',
    'mf_ei',
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
        mean_acq = qSimpleRegret(self.model, sampler=self.sampler)
        acq_hf = mean_acq(X_hf)
        fidelities = X[:,:,self.fidelity_dim].squeeze()
        fantasy_model = self.model.fantasize(X=X, sampler=self.sampler)
        # fantasy_acq_func = self.acquisition(fantasy_model, **self.acquisition_kwargs)
        # fantasy_acq = fantasy_acq_func(X_hf)
        fantasy_acq_func = qSimpleRegret(fantasy_model, sampler=self.sampler)
        fantasy_acq = fantasy_acq_func(X_hf)
        fantasy_imp = fantasy_acq.mean(0) - acq_hf
        costs = torch.tensor(self.cost_model(fidelities.detach().numpy()))
        return (fidelities * imp + (1 - fidelities) * fantasy_imp) / costs



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

    def load(self, filename, fidelity):
        joint = torch.load(filename)
        idx1 = self.n_dim
        idx2 = self.n_dim + self.n_outputs
        for point in joint:
            point_np = point.numpy()
            self.push(point_np[:idx1], point_np[idx1:idx2], point_np[idx2:], fidelity=fidelity)

    def save(self, output):
        # Build a joint tensor with all data for the highest fidelity
        mask = self.high_fidelity_mask()
        joint = torch.cat([self.params[mask,:], self.values[mask,:], self.variances[mask,:]], dim=1)
        torch.save(joint, output)

    def fidelity_mask(self, fidelity):
        return torch.isclose(self.fidelities, fidelity * torch.ones(1, dtype=self.dtype))[:,0]

    def high_fidelity_mask(self):
        return torch.isclose(self.fidelities, torch.ones(1, dtype=self.dtype))[:,0]

    def push(self, params, values, variances, fidelity=1.0):
        torch_params = torch.tensor(params, dtype=self.dtype).unsqueeze(0)
        torch_value = torch.tensor(values, dtype=self.dtype).unsqueeze(0)
        torch_variance = torch.tensor(variances, dtype=self.dtype).unsqueeze(0)
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



class BayesianBoTorchMultiFidelityComposite(CompositeMultiFidelityOptimiser):

    def __init__(self, n_initial=5, acquisition='ucb', log_space=False, def_variance=0,
                 beta=0.5, beta_final=None, noisy=False, q=1, seed=42, load_file=None,
                 export=None, fidelities=None, n_test=0):
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
        self.load_file = load_file
        self.export = export
        self.fidelities = fidelities
        self.n_test = n_test
        self.multi_fidelity_run = fidelities is not None
        if self.acquisition not in SUPPORTED_ACQUISITIONS:
            raise RuntimeError(f"Unkown acquisition function {self.acquisition}")
        torch.set_num_threads(1)
        self.timings = []

    def get_candidates(self, n_dim, dataset: BayesDataset, beta, test_dataset: BayesDataset, objective: MultiFidelityCompositeObjective):
        # Get data needed for unit-cube space mapping and standardisation
        X_delta = (dataset.ubounds - dataset.lbounds)
        y_avg = torch.mean(dataset.values, dim=-2)
        y_std = torch.std(dataset.values, dim=-2)
        y_abs_avg = torch.mean(torch.abs(dataset.values), dim=-2)

        # Take particular care if we only have one point to avoid divisions by zero
        if dataset.n_points == 1:
            y_std = 1

        # Remove points that have near-null variance: not relevant to the model
        mask = torch.abs(y_std / y_abs_avg) > 1e-6
        y_avg = y_avg[mask]
        y_std = y_std[mask]

        # Ensure we have at least one point after the previous step
        if not torch.any(mask):
            raise RuntimeError("All observed points are equal: add more initial samples")

        # Build unit cube space and standardised values
        X_cube = (dataset.params - dataset.lbounds) / X_delta
        y_standard = (dataset.values[:,mask] - y_avg) / y_std
        var_standard = dataset.variances[:,mask] / y_std

        # Clamp variances to prevent warnings from GPyTorch
        var_standard = torch.clamp_min(var_standard, 1e-6)

        # Handy loss function using the standardised dataset
        def loss_func(value):
            return -objective.composition.composition_torch(value * y_std + y_avg)

        # Build the GP: append the fidelity to the dataset in multi-fidelity runs
        X_cube_mf = torch.cat([X_cube, dataset.fidelities], dim=1)
        if self.noisy:
            model = SingleTaskMultiFidelityGP(
                X_cube_mf,
                y_standard,
                data_fidelity=n_dim
            )
        else:
            model = FixedNoiseMultiFidelityGP(
                X_cube_mf,
                y_standard,
                var_standard,
                data_fidelity=n_dim
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
            y_test_standard = (test_dataset.values[:,mask] - y_avg) / y_std
            if self.multi_fidelity_run:
                X_test_cube = torch.cat([X_test_cube, test_dataset.fidelities], dim=1)
            with torch.no_grad():
                posterior = model.posterior(X_test_cube)
                cv_error = ((posterior.mean - y_test_standard) ** 2).mean()

        # Find best point in the unit-cube and standardised dataset (at highest fidelity)
        losses_fid = {fid: loss_func(y_standard[dataset.fidelity_mask(fid),:]) for fid in objective.fidelities}
        y_best = {fid: max(losses_fid[fid]) for fid in objective.fidelities}
        idx_best = {fid: np.argmax(losses_fid[fid]) for fid in objective.fidelities}
        arg_best = {fid: X_cube_mf[dataset.fidelity_mask(fid),:][idx_best[fid]].unsqueeze(0) for fid in objective.fidelities}
        y_best_hf = y_best[max(objective.fidelities)].item()
    
        # # Find best point in the unit-cube and standardised dataset (at highest fidelity)
        # losses = [loss_func(y) for y in y_standard[dataset.high_fidelity_mask(),:]]
        # y_best = max(losses)

        # Build cost model for the fidelities
        costs = [objective.cost(fid) + 1.0 * np.mean(self.timings) for fid in objective.fidelities]
        cost_model = interp1d(objective.fidelities, costs)

        # Common data for optimisers
        num_restarts = 12
        fidelity_dim = n_dim
        raw_samples = max(256, 16 * n_dim * n_dim)
        sampler = SobolQMCNormalSampler(torch.Size([128]), seed=self.seed)
        bounds = torch.stack((torch.zeros(n_dim + 1, dtype=dataset.dtype),
                              torch.ones(n_dim + 1, dtype=dataset.dtype)))

        # Obtain next point based on the multi-fidelity strategy
        mc_obj = GenericMCObjective(loss_func)
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
                qSimpleRegret(model, objective=mc_obj),
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
                objective=mc_obj,
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
                    adapt_acq = qExpectedImprovement(model, y_best_hf, sampler=sampler, objective=mc_obj)
                    adapt_imp = lambda val: val
                elif self.acquisition == 'qmfkg_adaptive_ucb':
                    adapt_acq = qUpperConfidenceBound(model, self.beta, sampler=sampler, objective=mc_obj)
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
                'la_mf_ei': qExpectedImprovement,
                'la_mf_ucb': qUpperConfidenceBound,
                'la_mf_mean': qSimpleRegret,
            }
            acq_kwargs = {
                'la_mf_ei': {
                    'best_f': y_best_hf,
                    'sampler': sampler,
                },
                'la_mf_ucb': {
                    'beta': self.beta,
                    'sampler': sampler,
                },
                'la_mf_mean': {
                    'sampler': sampler,
                },
            }
            acq_improvement_func = {
                'la_mf_ei': lambda x: x,
                'la_mf_ucb': lambda x: torch.clamp_min(x - y_best_hf, 0),
                'la_mf_mean': lambda x: torch.clamp_min(x - y_best_hf, 0),
            }
            # Build and optimise the acquisition function
            acq = qMultiFidelityAcquisition(
                model,
                acq_type[self.acquisition],
                objective.fidelities,
                cost_model,
                fidelity_dim,
                sampler=sampler,
                objective=mc_obj,
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
                'mf_ei': lambda model, fidelity: qExpectedImprovement(model, y_best[fidelity], sampler=sampler, objective=mc_obj),
                'mf_ucb': lambda model, fidelity: qUpperConfidenceBound(model, self.beta, sampler=sampler, objective=mc_obj),
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
                    acq = qVarianceExpectedImprovement(model, y_best[fidelity], sampler=sampler, objective=mc_obj)
                    # acq = ExpectedImprovement(model, y_best[fidelity])
                else:
                    acq = qProbabilityOfImprovement(model, y_best_hf, sampler=sampler, objective=mc_obj)
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
            


        # Use standard acquisition functions at the highest fidelity
        else:
            # Build the acquisition function
            if self.acquisition == 'ucb':
                acq = qUpperConfidenceBound(model, beta, sampler=sampler, objective=mc_obj)
            elif self.acquisition == 'ei':
                acq = qExpectedImprovement(model, y_best_hf, sampler=sampler, objective=mc_obj)
            elif self.acquisition == 'pi':
                acq = qProbabilityOfImprovement(model, y_best_hf, sampler=sampler, objective=mc_obj)
            elif self.acquisition == 'qkg':
                num_restarts = 6
                raw_samples = 128
                sampler = SobolQMCNormalSampler(torch.Size([64]), seed=self.seed)
                acq = qKnowledgeGradient(model, sampler=sampler, objective=mc_obj)
            # Optimise the acquisition function at highest fidelity
            candidates, value = optimize_acqf_mixed(
                acq,
                bounds=bounds,
                q=self.q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                fixed_features_list=[{fidelity_dim: max(objective.fidelities)}],
                options={"sample_around_best": True},
            )
        
        # Re-map to original space and remove fidelity from the candidate
        target_fidelity = candidates[:, fidelity_dim].item()

        # Re-map to original space (and remove fidelity if needed)
        candidates_map = torch.empty((self.q, n_dim))
        for i in range(self.q):
            candidates_map[i, :] = dataset.lbounds + candidates[i, :n_dim] * X_delta
        return candidates_map.cpu().numpy(), target_fidelity, cv_error


    def _eval_candidates(self, func, candidates, fidelity):
        # Single candidate case
        if self.q == 1:
            return [func(candidate, fidelity=fidelity) for candidate in candidates]

        # Multi-candidate: run cases in parallel
        pool = Pool(self.q)
        return pool.map(lambda x: func(x, fidelity=fidelity, unique=True), candidates)
    
    def _get_best_point(self, dataset: BayesDataset, composition: Composition):
        params, values = dataset.get_params_value_pairs()
        losses = [composition(value) for value in values]
        idx = np.argmax(losses)
        return params[idx, :], losses[idx]

    def _get_random_points(self, n_points, n_dim, seed, bound):
        if qmc is None:
            points = np.random.default_rng(seed=seed).random([n_points, n_dim])
        else:
            points = qmc.Sobol(n_dim, seed=seed).random(n_points)
        return [point * (bound[:, 1] - bound[:, 0]) + bound[:, 0] for point in points]

    def _optimise(
        self,
        objective: MultiFidelityCompositeObjective,
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

        fidelities = objective.fidelities

        # Evaluate initial shot: for single-fidelity acquisitions, only do it at max fidelity
        if self.acquisition in ('ucb', 'ei', 'pi', 'qkg'):
            init_responses = {max(fidelities): objective(init_shot, max(fidelities))}
        else:
            init_responses = {fid: objective(init_shot, fid) for fid in fidelities}

        # Use initial shot to infer number of dimensions
        n_outputs = max([len(response) for response in init_responses.values()])
        def_variance = np.ones(n_outputs) * self.def_variance

        # Build initial dataset with the initial shot
        dataset = BayesDataset(n_dim, n_outputs, bound, self.export)
        for fid, response in init_responses.items():
            dataset.push(init_shot, response, def_variance, fid)

        # If requested, sample some random points before starting (in parallel if possible)
        random_points = self._get_random_points(self.n_initial, n_dim, self.seed, bound)
        init_responses = self._eval_candidates(objective, random_points, min(fidelities))
        for i, response in enumerate(init_responses):
            dataset.push(random_points[i], response, def_variance, min(fidelities))

        # Build test dataset
        test_dataset = BayesDataset(n_dim, n_outputs, bound)
        test_points = self._get_random_points(self.n_test, n_dim, self.seed + 1, bound)
        test_responses = self._eval_candidates(objective, test_points, min(fidelities))
        for i, response in enumerate(test_responses):
            test_dataset.push(test_points[i], response, def_variance, min(fidelities))

        # Find current best point to return to the driver
        best_params, best_loss = self._get_best_point(dataset, objective.composition)
        self._progress_check(0, best_loss, best_params)
        self.timings = [0]

        # Optimisation loop
        for i_iter in range(n_iter):
            beta = (self.beta * (n_iter - i_iter - 1) + self.beta_final * i_iter) / n_iter

            # Generate and evaluate candidates (in parallel if possible)
            begin = time.perf_counter()
            candidates, fidelity, cv_error = self.get_candidates(n_dim, dataset, beta, test_dataset,
                                                                 objective)
            self.timings.append(time.perf_counter() - begin)
            responses = self._eval_candidates(objective, candidates, fidelity)
            losses = [objective.composition(response) for response in responses]

            # Find best value for this batch and update dataset
            best_idx = np.argmin(losses)
            best_loss, best_params = losses[best_idx], candidates[best_idx, :]
            for i, response in enumerate(responses):
                dataset.push(candidates[i, :], response, def_variance, fidelity)

            # Update progress
            extra = f'Val. {cv_error:6.4}' if cv_error else None
            if self._progress_check(i_iter + 1, best_loss, best_params, extra_info=extra):
                break

        # Return optimisation result
        best_params, best_loss = self._get_best_point(dataset, objective.composition)
        return best_params, best_loss
