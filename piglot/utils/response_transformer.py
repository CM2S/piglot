"""Module for defining transformations for responses."""
from typing import Union, Dict, Any, Type, List
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
        scale_x: float = 1.0,
        offset_x: float = 0.0,
        scale_y: float = 1.0,
        offset_y: float = 0.0,
    ):
        self.scale_x = scale_x
        self.offset_x = offset_x
        self.scale_y = scale_y
        self.offset_y = offset_y

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
            self.scale_x * response.time + self.offset_x,
            self.scale_y * response.data + self.offset_y,
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
