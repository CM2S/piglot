from typing import Optional, Tuple
import torch
from piglot.objective import Scalarisation


class SampleScalarisation(Scalarisation):

    def scalarise_torch(
        self,
        values: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scalarise a set of objectives with gradients.

        Parameters
        ----------
        values : torch.Tensor
            Mean objective values.
        variances : Optional[torch.Tensor]
            Optional variances of the objectives.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Mean and variance of the scalarised objective.
        """
        if variances is None:
            return torch.mean(values * self.weights, dim=-1), None
        return (
            torch.mean(values * self.weights, dim=-1),
            torch.sum(variances * self.weights.square(), dim=-1) / (values.shape[-1] ** 2),
        )
