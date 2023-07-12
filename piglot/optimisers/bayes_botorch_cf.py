"""Bayesian optimiser module under composite optimisation (using BoTorch)."""
import warnings
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
try:
    from scipy.stats import qmc
except ImportError:
    qmc = None
try:
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import botorch
    from botorch.models import FixedNoiseGP, SingleTaskGP
    from botorch.models.gp_regression_fidelity import FixedNoiseMultiFidelityGP, SingleTaskMultiFidelityGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qUpperConfidenceBound
    from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement
    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.optim import optimize_acqf, optimize_acqf_mixed
    from botorch.sampling import SobolQMCNormalSampler
    from piglot.optimisers.optimiser import Optimiser
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimisers.optimiser import missing_method
    Optimiser = missing_method("Bayesian optimisation (BoTorch)", "botorch")



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



class BayesianBoTorchComposite(Optimiser):

    def __init__(self, n_initial=5, acquisition='ucb', log_space=False, def_variance=0,
                 beta=0.5, beta_final=None, noisy=False, q=1, seed=42, load_file=None,
                 export=None, fidelities=None, n_test=0):
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
        self.name = 'BoTorch'
        if self.acquisition not in ('ucb', 'ei', 'pi', 'kg', 'qucb', 'qei', 'qpi', 'qkg'):
            raise RuntimeError(f"Unkown acquisition function {self.acquisition}")
        torch.set_num_threads(1)

    @staticmethod
    def loss_func_torch(samples):
        return -samples.pow(2).mean(dim=-1)

    @staticmethod
    def loss_func_numpy(samples):
        return np.mean(np.square(samples))

    def get_candidates(self, n_dim, dataset: BayesDataset, beta, test_dataset: BayesDataset):
        # Get data needed for unit-cube space mapping and standardisation
        X_delta = (dataset.ubounds - dataset.lbounds)
        y_avg = torch.mean(dataset.values, dim=-2)
        y_std = torch.std(dataset.values, dim=-2)

        # Take particular care if we only have one point to avoid divisions by zero
        if dataset.n_points == 1:
            y_std = 1

        # Remove points that have near-null variance: not relevant to the model
        mask = torch.abs(y_std * y_avg) > 1e-6
        y_avg = y_avg[mask]
        y_std = y_std[mask]

        # Build unit cube space and standardised values
        X_cube = (dataset.params - dataset.lbounds) / X_delta
        y_standard = (dataset.values[:,mask] - y_avg) / y_std
        var_standard = dataset.variances[:,mask] / y_std

        # Clamp variances to prevent warnings from GPyTorch
        var_standard = torch.clamp_min(var_standard, 1e-6)

        # Handy loss function using the standardised dataset
        def loss_func(value):
            return self.loss_func_torch(value * y_std + y_avg)

        # Build the GP: append the fidelity to the dataset in multi-fidelity runs
        if self.multi_fidelity_run:
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
        else:
            if self.noisy:
                model = SingleTaskGP(X_cube, y_standard)
            else:
                model = FixedNoiseGP(X_cube, y_standard, var_standard)

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
        losses = [loss_func(y) for y in y_standard[dataset.high_fidelity_mask(),:]]
        y_best = max(losses)

        # Build the acquisition function with the composite objective
        num_restarts = 12
        raw_samples = max(256, 16 * n_dim * n_dim)
        objective = GenericMCObjective(loss_func)
        sampler = SobolQMCNormalSampler(torch.Size([512]), seed=self.seed)
        if self.acquisition in ('ucb', 'qucb'):
            acq = qUpperConfidenceBound(model, beta, sampler=sampler, objective=objective)
        elif self.acquisition in ('ei', 'qei'):
            acq = qExpectedImprovement(model, y_best, sampler=sampler, objective=objective)
        elif self.acquisition in ('pi', 'qpi'):
            acq = qProbabilityOfImprovement(model, y_best, sampler=sampler, objective=objective)
        elif self.acquisition in ('kg', 'qkg'):
            num_restarts = 6
            raw_samples = 64
            sampler = SobolQMCNormalSampler(torch.Size([64]), seed=self.seed)
            acq = qKnowledgeGradient(model, sampler=sampler, objective=objective)

        # Find next candidate(s)
        if self.multi_fidelity_run:
            bounds = torch.stack((torch.zeros(n_dim + 1, dtype=dataset.dtype),
                                  torch.ones(n_dim + 1, dtype=dataset.dtype)))
            candidates, _ = optimize_acqf_mixed(
                acq,
                bounds=bounds,
                q=self.q,
                fixed_features_list=[{n_dim: 1.0}],
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"sample_around_best": True},
            )
        else:
            bounds = torch.stack((torch.zeros(n_dim, dtype=dataset.dtype),
                                  torch.ones(n_dim, dtype=dataset.dtype)))
            candidates, _ = optimize_acqf(
                acq,
                bounds=bounds,
                q=self.q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"sample_around_best": True},
            )

        # Re-map to original space (and remove fidelity if needed)
        candidates_map = torch.empty((self.q, n_dim))
        for i in range(self.q):
            candidates_map[i, :] = dataset.lbounds + candidates[i, :n_dim] * X_delta
        return candidates_map.cpu().numpy(), cv_error


    def _eval_candidates(self, func, candidates):
        # Single candidate case
        if self.q == 1:
            return [func(candidate) for candidate in candidates]

        # Multi-candidate: run cases in parallel
        pool = Pool(self.q)
        return pool.map(lambda x: func(x, unique=True), candidates)
    
    def _get_best_point(self, dataset: BayesDataset):
        params, values = dataset.get_params_value_pairs()
        losses = [self.loss_func_numpy(value) for value in values]
        idx = np.argmax(losses)
        return params[idx, :], losses[idx]

    def _get_random_points(self, n_points, n_dim, seed, bound):
        if qmc is None:
            points = np.random.default_rng(seed=seed).random([n_points, n_dim])
        else:
            points = qmc.Sobol(n_dim, seed=seed).random(n_points)
        return [point * (bound[:, 1] - bound[:, 0]) + bound[:, 0] for point in points]


    def _optimise(self, func, n_dim, n_iter, bound, init_shot):
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

        # Evaluate initial shot and use it to infer number of dimensions
        init_response = func(init_shot)
        n_outputs = len(init_response)
        def_variance = np.ones(n_outputs) * self.def_variance

        # Build initial dataset with the initial shot
        dataset = BayesDataset(n_dim, n_outputs, bound, self.export)
        dataset.push(init_shot, init_response, def_variance)

        # If requested, sample some random points before starting (in parallel if possible)
        random_points = self._get_random_points(self.n_initial, n_dim, self.seed, bound)
        init_responses = self._eval_candidates(func, random_points)
        for i, response in enumerate(init_responses):
            dataset.push(random_points[i], response, def_variance)

        # If specified, load data from the input file (at final fidelity)
        if self.load_file:
            dataset.load(self.load_file, 1.0)

        # Load any multi-fidelity data
        if self.fidelities:
            for filename, fidelity in self.fidelities.items():
                dataset.load(filename, fidelity)

        # Build test dataset
        test_dataset = BayesDataset(n_dim, n_outputs, bound)
        test_points = self._get_random_points(self.n_test, n_dim, self.seed + 1, bound)
        test_responses = self._eval_candidates(func, test_points)
        for i, response in enumerate(test_responses):
            test_dataset.push(test_points[i], response, def_variance)

        # Find current best point to return to the driver
        best_params, best_loss = self._get_best_point(dataset)
        self._progress_check(0, best_loss, best_params)

        # Optimisation loop
        for i_iter in range(n_iter):
            beta = (self.beta * (n_iter - i_iter - 1) + self.beta_final * i_iter) / n_iter

            # Generate and evaluate candidates (in parallel if possible)
            candidates, cv_error = self.get_candidates(n_dim, dataset, beta, test_dataset)
            responses = self._eval_candidates(func, candidates)
            losses = [self.loss_func_numpy(response) for response in responses]

            # Find best value for this batch and update dataset
            best_idx = np.argmin(losses)
            best_loss, best_params = losses[best_idx], candidates[best_idx, :]
            for i, response in enumerate(responses):
                dataset.push(candidates[i, :], response, def_variance)

            # Update progress
            extra = f'Val. {cv_error:6.4}' if cv_error else None
            if self._progress_check(i_iter + 1, best_loss, best_params, extra_info=extra):
                break

        # Return optimisation result
        best_params, best_loss = self._get_best_point(dataset)
        return best_params, best_loss
