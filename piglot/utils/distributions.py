"""Wrapper module for PyTorch distributions."""
from typing import Any
import torch
from torch.distributions import (
    Distribution,
    Categorical,
    Beta,
    Cauchy,
    Chi2,
    Exponential,
    FisherSnedecor,
    Gamma,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Kumaraswamy,
    Laplace,
    LogNormal,
    Normal,
    Pareto,
    Poisson,
    StudentT,
    Uniform,
    VonMises,
    Weibull,
)


class ClosedUniform(Uniform):
    """Wrapper to force a closed interval for Uniform distribution."""

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        lb = self.low.le(value).type_as(self.low)
        ub = self.high.ge(value).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)


REAL_DISTRIBUTIONS: dict[str, type[Distribution]] = {
    'beta': Beta,
    'cauchy': Cauchy,
    'chi2': Chi2,
    'exponential': Exponential,
    'fisher_snedecor': FisherSnedecor,
    'gamma': Gamma,
    'gumbel': Gumbel,
    'half_cauchy': HalfCauchy,
    'half_normal': HalfNormal,
    'inverse_gamma': InverseGamma,
    'kumaraswamy': Kumaraswamy,
    'laplace': Laplace,
    'log_normal': LogNormal,
    'normal': Normal,
    'pareto': Pareto,
    'poisson': Poisson,
    'student_t': StudentT,
    'uniform': ClosedUniform,
    'von_mises': VonMises,
    'weibull': Weibull,
}


def read_real_distribution(config: dict[str, Any]) -> Distribution:
    """Read a real-valued distribution from a configuration dictionary.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Distribution
        PyTorch distribution instance.
    """
    if 'name' not in config:
        raise ValueError("Missing name for distribution.")
    name = config.pop('name')
    if name not in REAL_DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution '{name}'.")
    return REAL_DISTRIBUTIONS[name](**config, validate_args=True)


def get_discrete_distribution(probs: list[float]) -> Distribution:
    """Get a discrete distribution from a list of probabilities.

    Parameters
    ----------
    probs : list[float]
        List of probabilities.

    Returns
    -------
    Distribution
        PyTorch Categorical distribution instance.
    """
    return Categorical(probs=torch.tensor(probs), validate_args=True)
