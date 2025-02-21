"""Module for defining reduction functions for responses."""
from typing import Callable, Dict, Any, Union
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.autograd.gradcheck import gradcheck, GradcheckError
from piglot.utils.assorted import read_custom_module


class Reduction(ABC):
    """Abstract class for defining reduction functions."""

    @abstractmethod
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

    def reduce(self, time: np.ndarray, data: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Reduce the input data to a single value.

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.
        params : np.ndarray
            Parameters for the given responses.

        Returns
        -------
        np.ndarray
            Reduced value of the data.
        """
        return self.reduce_torch(
            torch.from_numpy(time),
            torch.from_numpy(data),
            torch.from_numpy(params),
        ).numpy(force=True)

    def test_reduction(self) -> None:
        """Test the reduction function to check batch processing."""
        # Sanitise the shape after applying the reduction
        test_params = [2, 6]
        test_shapes = [(4,), (2, 4), (3, 2, 4), (6, 3, 2, 4)]
        for num_params in test_params:
            for shape in test_shapes:
                time = torch.arange(shape[-1]).repeat(shape[:-1] + (1,))
                data = torch.randn(*shape)
                params = torch.randn(num_params).repeat(shape[:-1] + (1,))
                try:
                    reduced = self.reduce_torch(time, data, params)
                except Exception as exc:
                    raise ValueError(f"Test failed for reduction {type(self)}.") from exc
                if reduced.shape != shape[:-1]:
                    raise ValueError(
                        f"Bad shape after reduction for {type(self)}. "
                        f"While reducing a tensor of shape {shape}, "
                        f"got {reduced.shape} instead of {shape[:-1]}."
                    )
        # Check if the gradient is computed
        time = torch.tensor([[0, 1], [1, 2]], requires_grad=True, dtype=torch.float64)
        data = torch.tensor([[2, 3], [4, 3]], requires_grad=True, dtype=torch.float64)
        params = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float64)
        try:
            if not gradcheck(self.reduce_torch, (time, data, params)):
                raise ValueError(f"Gradient check failed for {type(self)}.")
        except GradcheckError as exc:
            raise ValueError(f"Gradient check failed for {type(self)}.") from exc


class NegateReduction(Reduction):
    """Negate the result of another reduction function."""

    def __init__(self, reduction: Reduction) -> None:
        self.reduction = reduction

    def reduce_torch(
        self,
        time: torch.Tensor,
        data: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the input data to a single value.

        Parameters
        ----------
        time : np.ndarray
            Time points of the response.
        data : np.ndarray
            Data points of the response.
        params : np.ndarray
            Parameters for the given responses.

        Returns
        -------
        np.ndarray
            Reduced value of the data.
        """
        return -self.reduction.reduce_torch(time, data, params)


class SimpleReduction(Reduction):
    """Reduction function defined from a lambda function (without using the parameters)."""

    def __init__(self, reduction: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        self.reduction = reduction

    def reduce_torch(
        self,
        time: torch.Tensor,
        data: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the input data to a single value.

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
        return self.reduction(time, data)


AVAILABLE_REDUCTIONS: Dict[str, Reduction] = {
    'mean': SimpleReduction(lambda time, data: torch.mean(data, dim=-1)),
    'max': SimpleReduction(lambda time, data: torch.amax(data, dim=-1)),
    'min': SimpleReduction(lambda time, data: torch.amin(data, dim=-1)),
    'sum': SimpleReduction(lambda time, data: torch.sum(data, dim=-1)),
    'std': SimpleReduction(lambda time, data: torch.std(data, dim=-1)),
    'var': SimpleReduction(lambda time, data: torch.var(data, dim=-1)),
    'mse': SimpleReduction(lambda time, data: torch.mean(torch.square(data), dim=-1)),
    'mae': SimpleReduction(lambda time, data: torch.mean(torch.abs(data), dim=-1)),
    'last': SimpleReduction(lambda time, data: data[..., -1]),
    'first': SimpleReduction(lambda time, data: data[..., 0]),
    'max_abs': SimpleReduction(lambda time, data: torch.amax(torch.abs(data), dim=-1)),
    'min_abs': SimpleReduction(lambda time, data: torch.amin(torch.abs(data), dim=-1)),
    'integral': SimpleReduction(lambda time, data: torch.trapz(data, time, dim=-1)),
    'square_integral': SimpleReduction(
        lambda time, data: torch.trapz(torch.square(data), time, dim=-1),
    ),
    'abs_integral': SimpleReduction(
        lambda time, data: torch.trapz(torch.abs(data), time, dim=-1),
    ),
}
# TODO: Add test for non-existing 'script' reduction


def read_reduction(config: Union[str, Dict[str, Any]]) -> Reduction:
    """Read a reduction function from a configuration.

    Parameters
    ----------
    config : Union[str, Dict[str, Any]]
        Configuration of the reduction function.

    Returns
    -------
    Reduction
        Reduction function.
    """
    # Parse the reduction in the simple format
    if isinstance(config, str):
        name = config
        if name == 'script':
            raise ValueError('Need to pass the file path for the "script" reduction.')
        if name not in AVAILABLE_REDUCTIONS:
            raise ValueError(f'Reduction function "{name}" is not available.')
        return AVAILABLE_REDUCTIONS[name]
    # Detailed format
    if 'name' not in config:
        raise ValueError('Need to pass the name of the reduction function.')
    name = config['name']
    # Read script reduction
    if name == 'script':
        instance = read_custom_module(config, Reduction)()
        # Sanitise external reductions
        if not bool(config.get('skip_test', False)):
            instance.test_reduction()
        return instance
    if name not in AVAILABLE_REDUCTIONS:
        raise ValueError(f'Reduction function "{name}" is not available.')
    return AVAILABLE_REDUCTIONS[name]
