from __future__ import annotations
import numpy as np
from piglot.solver.solver import OutputResult
from piglot.objectives.design import Quantity

class MinQuantity(Quantity):
    """Minimum value of a response."""

    def compute(self, result: OutputResult) -> float:
        """Get the minimum of a given response.

        Parameters
        ----------
        result : OutputResult
            Output result to compute the quantity for.

        Returns
        -------
        float
            Quantity value.
        """
        return np.min(result.get_data())
