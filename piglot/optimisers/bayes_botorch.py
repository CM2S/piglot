"""Bayesian optimiser module (using BoTorch)."""
import numpy as np
try:
    from scipy.stats import qmc
except ImportError:
    qmc = None
try:
    import torch
    from botorch.models import FixedNoiseGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import UpperConfidenceBound
    from botorch.optim import optimize_acqf
    from piglot.optimisers.optimiser import Optimiser
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimisers.optimiser import missing_method
    Optimiser = missing_method("Bayesian optimisation (BoTorch)", "botorch")


class BayesDataset:

    def __init__(self, n_dim, bounds, dtype=torch.float64):
        self.dtype = dtype
        self.n_points = 0
        self.params = torch.empty((0, n_dim), dtype=dtype)
        self.values = torch.empty((0, 1), dtype=dtype)
        self.variances = torch.empty((0, 1), dtype=dtype)
        self.lbounds = torch.tensor(bounds[:, 0], dtype=dtype)
        self.ubounds = torch.tensor(bounds[:, 1], dtype=dtype)

    def push(self, params, value, variance):
        torch_params = torch.tensor(params, dtype=self.dtype).unsqueeze(0)
        torch_value = torch.tensor([value], dtype=self.dtype).unsqueeze(0)
        torch_variance = torch.tensor([variance], dtype=self.dtype).unsqueeze(0)
        self.params = torch.cat([self.params, torch_params], dim=0)
        self.values = torch.cat([self.values, torch_value], dim=0)
        self.variances = torch.cat([self.variances, torch_variance], dim=0)
        self.n_points += 1



class BayesianBoTorch(Optimiser):

    def __init__(self, n_initial=5, log_space=False, def_variance=0, beta=0.5,
                 beta_final=None, load_file=None):
        self.n_initial = n_initial
        self.log_space = log_space
        self.def_variance = def_variance
        self.beta = beta
        self.beta_final = beta if beta_final is None else beta_final
        self.load_file = load_file
        self.name = 'BoTorch'
        torch.set_num_threads(1)

    def get_candidate(self, n_dim, dataset, beta):
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

        # Build and fit the GP
        gp = FixedNoiseGP(X_cube, y_standard, var_standard)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Find next candidate
        UCB = UpperConfidenceBound(gp, beta=beta)
        bounds = torch.stack((torch.zeros(n_dim), torch.ones(n_dim)))
        candidate, acq_value = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=50, raw_samples=200)

        # Re-map to original space
        candidate = dataset.lbounds + candidate * X_delta
        acq_value = y_avg + acq_value * y_std
        return candidate.cpu().numpy().squeeze(), acq_value.cpu().numpy().squeeze()

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

        # Build initial dataset with the initial shot
        dataset = BayesDataset(n_dim, bound)
        dataset.push(init_shot, loss_transformer(func(init_shot)), self.def_variance)

        # If requested, sample some random points before starting
        rng = np.random.default_rng(seed=42) if qmc is None else qmc.Sobol(n_dim, seed=42)
        for _ in range(self.n_initial):
            random = rng.random([n_dim]) if qmc is None else rng.random().squeeze()
            point = random * (bound[:, 1] - bound[:, 0]) + bound[:, 0]
            dataset.push(point, loss_transformer(func(point)), self.def_variance)

        # If specified, load data from the input file
        if self.load_file:
            input_data = np.genfromtxt(self.load_file)
            for row in input_data:
                params, loss = row[:-1], row[-1]
                dataset.push(params, loss_transformer(loss), self.def_variance)

        # Optimisation loop
        for i in range(n_iter):
            beta = (self.beta * (n_iter - i - 1) + self.beta_final * i) / n_iter
            candidate, _ = self.get_candidate(n_dim, dataset, beta)
            value = func(candidate)
            dataset.push(candidate, loss_transformer(value), self.def_variance)
            if self._progress_check(i + 1, value, candidate):
                break

        # Return optimisation result
        best_params = dataset.params[torch.argmax(dataset.values),:].cpu().numpy()
        best_loss = inv_loss_transformer(torch.max(dataset.values))
        return best_params, best_loss
