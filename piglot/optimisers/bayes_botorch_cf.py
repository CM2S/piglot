"""Bayesian optimiser module under composite optimisation (using BoTorch)."""
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
try:
    from scipy.stats import qmc
except ImportError:
    qmc = None
try:
    import torch
    from botorch.models import FixedNoiseGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import qUpperConfidenceBound
    from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement
    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.optim import optimize_acqf
    from piglot.optimisers.optimiser import Optimiser
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimisers.optimiser import missing_method
    Optimiser = missing_method("Bayesian optimisation (BoTorch)", "botorch")


class BayesDataset:

    def __init__(self, n_dim, n_outputs, bounds, dtype=torch.float64):
        self.dtype = dtype
        self.n_points = 0
        self.n_outputs = n_outputs
        self.params = torch.empty((0, n_dim), dtype=dtype)
        self.values = torch.empty((0, n_outputs), dtype=dtype)
        self.variances = torch.empty((0, n_outputs), dtype=dtype)
        self.lbounds = torch.tensor(bounds[:, 0], dtype=dtype)
        self.ubounds = torch.tensor(bounds[:, 1], dtype=dtype)

    def push(self, params, values, variances):
        torch_params = torch.tensor(params, dtype=self.dtype).unsqueeze(0)
        torch_value = torch.tensor(values, dtype=self.dtype).unsqueeze(0)
        torch_variance = torch.tensor(variances, dtype=self.dtype).unsqueeze(0)
        self.params = torch.cat([self.params, torch_params], dim=0)
        self.values = torch.cat([self.values, torch_value], dim=0)
        self.variances = torch.cat([self.variances, torch_variance], dim=0)
        self.n_points += 1



class BayesianBoTorchComposite(Optimiser):

    def __init__(self, n_initial=5, acquisition='ucb', log_space=False, def_variance=0,
                 beta=0.5, q=1, effort=1.0, beta_final=None, load_file=None):
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.log_space = log_space
        self.def_variance = def_variance
        self.beta = beta
        self.q = q
        self.effort = effort
        self.beta_final = beta if beta_final is None else beta_final
        self.load_file = load_file
        self.name = 'BoTorch'
        if self.acquisition not in ('ucb', 'ei', 'pi', 'kg'):
            raise Exception(f"Unkown acquisition function {self.acquisition}")
        torch.set_num_threads(1)

    @staticmethod
    def loss_func_torch(samples):
        return -samples.pow(2).mean(dim=-1)

    @staticmethod
    def loss_func_numpy(samples):
        return np.mean(np.square(samples))

    def get_candidate(self, n_dim, dataset, beta):
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

        # Find best point in the unit-cube and standardised dataset
        loss_func = lambda x: self.loss_func_torch(x * y_std + y_avg)
        losses = [loss_func(y) for y in y_std]
        y_best = np.max(losses)

        # Build and fit the GP
        gp = FixedNoiseGP(X_cube, y_standard, var_standard)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Build the acquisition function with the composite objective
        effort = self.effort / self.q
        raw_samples = int(128 * effort)
        num_restarts = int(12 * effort)
        objective = GenericMCObjective(loss_func)
        if self.acquisition == 'ucb':
            acq = qUpperConfidenceBound(gp, beta, objective=objective)
        elif self.acquisition == 'ei':
            acq = qExpectedImprovement(gp, y_best, objective=objective)
        elif self.acquisition == 'pi':
            acq = qProbabilityOfImprovement(gp, y_best, objective=objective)
        elif self.acquisition == 'kg':
            raw_samples = int(64 * effort)
            num_restarts = int(6 * effort)
            num_fantasies = int(32 * effort)
            acq = qKnowledgeGradient(gp, num_fantasies=num_fantasies, objective=objective)

        # Find next candidate
        bounds = torch.stack((torch.zeros(n_dim, dtype=dataset.dtype),
                              torch.ones(n_dim, dtype=dataset.dtype)))
        candidates, acq_value = optimize_acqf(acq,
                                              bounds=bounds,
                                              q=self.q,
                                              num_restarts=num_restarts,
                                              raw_samples=raw_samples)

        # Re-map to original space
        for i in range(self.q):
            candidates[i, :] = dataset.lbounds + candidates[i, :] * X_delta
        acq_value = y_avg + acq_value * y_std
        return [candidates[i, :].cpu().numpy() for i in range(self.q)], \
               [acq_value[i].cpu().numpy() for i in range(self.q)]
    

    def _eval_candidates(self, func, candidates):
        # Single candidate case
        if self.q == 1:
            return [func(candidates[0])]
        
        # Multi-candidate: run cases in parallel
        pool = Pool(self.q)
        parallel_func = lambda x: func(x, unique=True)
        return pool.map(parallel_func, [candidates[i] for i in range(self.q)])
        

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

        # Negate the loss function to convert problem to a maximisation
        loss_transformer = lambda x: -np.log(x) if self.log_space else -x
        inv_loss_transformer = lambda x: np.exp(-x) if self.log_space else -x

        # Evaluate initial shot and use it to infer number of dimensions
        init_value = loss_transformer(func(init_shot))
        n_outputs = len(init_value)
        def_variance = np.ones(n_outputs) * self.def_variance

        # Build initial dataset with the initial shot
        dataset = BayesDataset(n_dim, n_outputs, bound)
        dataset.push(init_shot, init_value, def_variance)

        # If requested, sample some random points before starting
        rng = np.random.default_rng(seed=42) if qmc is None else qmc.Sobol(n_dim, seed=42)
        points = []
        for _ in range(self.n_initial):
            random = rng.random([n_dim]) if qmc is None else rng.random().squeeze()
            points.append(random * (bound[:, 1] - bound[:, 0]) + bound[:, 0])

        # Run the initial set of points (in parallel if possible)
        values = self._eval_candidates(func, points)
        for i, value in enumerate(values):
            dataset.push(points[i], loss_transformer(value), def_variance)

        # If specified, load data from the input file
        if self.load_file:
            input_data = np.genfromtxt(self.load_file)
            for row in input_data:
                params, loss = row[:-n_outputs], row[-n_outputs]
                dataset.push(params, loss_transformer(loss), def_variance)

        # Optimisation loop
        for i in range(n_iter):
            beta = (self.beta * (n_iter - i - 1) + self.beta_final * i) / n_iter
            candidates, _ = self.get_candidate(n_dim, dataset, beta)
            values = self._eval_candidates(func, candidates)
            # Add to dataset and find best point of this batch
            batch_best, batch_best_candidate = np.inf, None
            for j in range(self.q):
                value = values[j]
                dataset.push(candidates[j], loss_transformer(value), def_variance)
                if self.loss_func_numpy(value) < batch_best:
                    batch_best = self.loss_func_numpy(value)
                    batch_best_candidate = candidates[j]
            if self._progress_check(i + 1, batch_best, batch_best_candidate):
                break

        # Return optimisation result
        losses = self.loss_func_torch(dataset.values)
        best_params = dataset.params[torch.argmax(losses),:].cpu().numpy()
        best_loss = inv_loss_transformer(torch.max(losses)).cpu().numpy()
        return best_params, self.loss_func_numpy(best_loss)
