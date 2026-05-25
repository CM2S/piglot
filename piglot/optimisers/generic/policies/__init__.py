"""Module for generic optimisation policies in piglot."""
from piglot.optimisers.generic.campaign import CandidatePolicy
from piglot.optimisers.generic.policies.acquisition import AcquisitionCandidatePolicy
from piglot.optimisers.generic.policies.initial import InitialCandidatePolicy
from piglot.optimisers.generic.policies.optima import OptimaCandidatePolicy
from piglot.optimisers.generic.policies.query import QueryCandidatePolicy
from piglot.optimisers.generic.policies.random import RandomCandidatePolicy
from piglot.optimisers.generic.policies.thompson import ThompsonSamplingCandidatePolicy


AVAILABLE_POLICIES: dict[str, type[CandidatePolicy]] = {
    'acquisition': AcquisitionCandidatePolicy,
    'initial': InitialCandidatePolicy,
    'optima': OptimaCandidatePolicy,
    'query': QueryCandidatePolicy,
    'random': RandomCandidatePolicy,
    'thompson': ThompsonSamplingCandidatePolicy,
}


def read_policy(config: dict[str, any]) -> CandidatePolicy:
    """Read a candidate policy from the given configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary for the candidate policy.

    Returns
    -------
    CandidatePolicy
        The candidate policy created from the configuration.
    """
    if 'name' not in config:
        raise ValueError('Missing name for candidate policy in configuration.')
    name = config.pop('name')
    if name not in AVAILABLE_POLICIES:
        raise ValueError(f'Unknown candidate policy: {name}')
    return AVAILABLE_POLICIES[name].read(config)
