"""Main piglot module."""

import piglot.optimisers
from piglot.optimisers.optimiser import Optimiser

def optimiser(name, *args, **kwargs) -> Optimiser:
    """Returns one of the optimisers in piglot.

    Parameters
    ----------
    name : name
        Name of the method to use.

    Returns
    -------
    Optimiser
        Optimiser instance.
    """
    if name not in piglot.optimisers.AVAILABLE_OPTIMISERS:
        raise NameError(f"Method {name} unknown! Check available methods.")
    return piglot.optimisers.AVAILABLE_OPTIMISERS[name](*args, **kwargs)
