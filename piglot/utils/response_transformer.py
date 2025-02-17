"""Module for defining transformations for responses."""
from typing import Union, Dict, Any, Type, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from piglot.utils.assorted import read_custom_module
from piglot.solver.solver import OutputResult
from piglot.utils.responses import interpolate_response


class ResponseTransformer(ABC):
    """Abstract class for defining transformation functions."""

    @abstractmethod
    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        OutputResult
            Transformed time and data points of the response.
        """

    def __call__(self, x_old: np.ndarray, y_old: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a response function.

        Parameters
        ----------
        x_old : np.ndarray
            Original time grid.
        y_old : np.ndarray
            Original values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed time grid and values.
        """
        response = OutputResult(x_old, y_old)
        response = self.transform(response)
        return response.time, response.data


class ChainResponse(ResponseTransformer):
    """Chain of response transformers."""

    def __init__(self, transformers: List[Any]):
        self.transformers = [read_response_transformer(t) for t in transformers]

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        Output
            Transformed time and data points of the response.
        """
        for transformer in self.transformers:
            response = transformer.transform(response)
        return response


class MinimumResponse(ResponseTransformer):
    """Minimum of a response transformer."""

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        OutputResult
            Transformed time and data points of the response.
        """
        return OutputResult(np.array([0.0]), np.array([np.min(response.data)]))


class MaximumResponse(ResponseTransformer):
    """Maximum of a response transformer."""

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        OutputResult
            Transformed time and data points of the response.
        """
        return OutputResult(np.array([0.0]), np.array([np.max(response.data)]))


class NegateResponse(ResponseTransformer):
    """Negate a response transformer."""

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        OutputResult
            Transformed time and data points of the response.
        """
        return OutputResult(response.time, -response.data)


class SquareResponse(ResponseTransformer):
    """Square a response transformer."""

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        Output
            Transformed time and data points of the response.
        """
        return OutputResult(response.time, np.square(response.data))


class AffineTransformResponse(ResponseTransformer):
    """Affine transformation of a response transformer."""

    def __init__(
        self,
        x_scale: float = 1.0,
        x_offset: float = 0.0,
        y_scale: float = 1.0,
        y_offset: float = 0.0,
    ):
        self.x_scale = x_scale
        self.x_offset = x_offset
        self.y_scale = y_scale
        self.y_offset = y_offset

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        OutputResult
            Transformed time and data points of the response.
        """
        return OutputResult(
            self.x_scale * response.time + self.x_offset,
            self.y_scale * response.data + self.y_offset,
        )


class ClipResponse(ResponseTransformer):
    """Clip x and y values of the response to given bounds."""

    def __init__(
        self,
        x_min: float = -np.inf,
        x_max: float = np.inf,
        y_min: float = -np.inf,
        y_max: float = np.inf,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        OutputResult
            Transformed time and data points of the response.
        """
        return OutputResult(
            np.clip(response.time, self.x_min, self.x_max),
            np.clip(response.data, self.y_min, self.y_max),
        )


class PointwiseErrors(ResponseTransformer):
    """Compute the pointwise errors between the response and a reference."""

    def __init__(self, reference_time: np.ndarray, reference_data: np.ndarray) -> None:
        self.reference_time = reference_time
        self.reference_data = reference_data

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        OutputResult
            Transformed time and data points of the response.
        """
        # Interpolate response to the reference grid
        resp_interp = interpolate_response(
            response.get_time(),
            response.get_data(),
            self.reference_time,
        )
        # Compute normalised error
        factor = np.mean(np.abs(self.reference_data))
        return OutputResult(self.reference_time, (resp_interp - self.reference_data) / factor)


AVAILABLE_RESPONSE_TRANSFORMERS: Dict[str, Type[ResponseTransformer]] = {
    'min': MinimumResponse,
    'max': MaximumResponse,
    'negate': NegateResponse,
    'square': SquareResponse,
    'chain': ChainResponse,
    'clip': ClipResponse,
    'affine': AffineTransformResponse,
}


def read_response_transformer(config: Union[str, Dict[str, Any]]) -> ResponseTransformer:
    """Read a response transformer from a configuration.

    Parameters
    ----------
    config : Union[str, Dict[str, Any]]
        Configuration of the response transformer.

    Returns
    -------
    ResponseTransformer
        Response transformer.
    """
    # Parse the transformer in the simple format
    if isinstance(config, str):
        name = config
        if name == 'script':
            raise ValueError('Need to pass the file path for the "script" transformer.')
        if name not in AVAILABLE_RESPONSE_TRANSFORMERS:
            raise ValueError(f'Response transformer "{name}" is not available.')
        return AVAILABLE_RESPONSE_TRANSFORMERS[name]()
    # Detailed format
    if 'name' not in config:
        raise ValueError('Need to pass the name of the response transformer.')
    name = config.pop('name')
    # Read script transformer
    if name == 'script':
        return read_custom_module(config, ResponseTransformer)()
    if name not in AVAILABLE_RESPONSE_TRANSFORMERS:
        raise ValueError(f'Response transformer "{name}" is not available.')
    return AVAILABLE_RESPONSE_TRANSFORMERS[name](**config)
