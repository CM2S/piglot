import torch
from piglot.utils.reductions import Reduction


class BadReductionException(Reduction):

    def reduce_torch(
        self,
        time: torch.Tensor,
        data: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the input data to a single value (with gradients).

        Parameters
        ----------
        time : torch.Tensor
            Time points of the response.
        data : torch.Tensor
            Data points of the response.
        params : torch.Tensor
            Parameters for the given responses.

        Returns
        -------
        torch.Tensor
            Reduced value of the data.
        """
        raise ValueError("This is a bad reduction function.")


class BadReductionShape(Reduction):

    def reduce_torch(
        self,
        time: torch.Tensor,
        data: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the input data to a single value (with gradients).

        Parameters
        ----------
        time : torch.Tensor
            Time points of the response.
        data : torch.Tensor
            Data points of the response.
        params : torch.Tensor
            Parameters for the given responses.

        Returns
        -------
        torch.Tensor
            Reduced value of the data.
        """
        return torch.mean(data, dim=0)


class BadReductionGrad(Reduction):

    def reduce_torch(
        self,
        time: torch.Tensor,
        data: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the input data to a single value (with gradients).

        Parameters
        ----------
        time : torch.Tensor
            Time points of the response.
        data : torch.Tensor
            Data points of the response.
        params : torch.Tensor
            Parameters for the given responses.

        Returns
        -------
        torch.Tensor
            Reduced value of the data.
        """
        return torch.mean(torch.from_numpy(data.detach().numpy()), dim=-1)
