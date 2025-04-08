"""Module for defining transformations for responses."""
from typing import Union, Dict, Any, Type, List, Tuple
from abc import ABC, abstractmethod
from functools import partial
import sys
import time
import numpy as np
from piglot.utils.assorted import read_custom_module, trapezoidal_integration_weights
from piglot.solver.solver import OutputResult, FullFieldOutputResult
from piglot.utils.interpolators import Interpolator, OneDimensionalInterpolator, read_interpolator
from piglot.utils.responses import reduce_points, reduce_points_clusters, get_error


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


class ReduceResponse(ResponseTransformer):
    """Reduce the number of points in the response."""

    def __init__(
        self,
        filter_tol: float,
        method: str = 'clusters',
        interpolator: Union[str, Dict[str, Any]] = 'unstructured_linear',
        integral_error: bool = False,
    ) -> None:
        if method not in ['clusters', 'global']:
            raise ValueError('Unknown method for reducing the response.')
        self.filter_tol = float(filter_tol)
        self.method = method
        self.interpolator = read_interpolator(interpolator)
        self.integral_error = integral_error

    def __progress_callback(self, num_points: int, error: float, orig_points: int) -> bool:
        """Progress callback for the reduction process."""
        print(
            f"\rReducing from {orig_points} to {num_points} points (rel error: {error:6.4e}) ...",
            end='',
        )
        sys.stdout.flush()
        return False

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
        # Ensure the coordinates are two-dimensional
        points = response.get_time()
        if points.ndim == 1:
            points = points[:, np.newaxis]
        # And likewise for the values
        values = response.get_data()
        if values.ndim == 1:
            values = values[:, np.newaxis]
        # Check if we need to compute the weights for the integration
        weights = 1 / np.sum(np.square(values), axis=0)
        if self.integral_error:
            weights = weights * trapezoidal_integration_weights(points).reshape(-1, 1)
        # Reduce the points
        self.__progress_callback(points.shape[0], 0.0, points.shape[0])
        reduce_func = reduce_points_clusters if self.method == 'clusters' else reduce_points
        elapsed = time.perf_counter()
        reduced_points, reduced_values = reduce_func(
            points,
            values,
            points,
            values,
            self.filter_tol,
            self.interpolator,
            weights=weights,
            progress_callback=partial(self.__progress_callback, orig_points=points.shape[0]),
        )
        elapsed = time.perf_counter() - elapsed
        # Get the reduction error and output for the user
        error = get_error(
            reduced_points, reduced_values, points, values, self.interpolator, weights=weights,
        )
        print(
            f'\rReduced from {points.shape[0]} to {reduced_points.shape[0]} points in '
            f'{elapsed:.2f}s (relative error: {error:6.4e})'
        )
        # Return with appropriate dimensions
        return OutputResult(
            reduced_points[:, 0] if response.get_time().ndim == 1 else reduced_points,
            reduced_values[:, 0] if response.get_data().ndim == 1 else reduced_values,
        )


class PointwiseErrors(ResponseTransformer):
    """Compute the pointwise errors between the response and a reference."""

    def __init__(
        self,
        reference_time: np.ndarray,
        reference_data: np.ndarray,
        interpolator: Interpolator,
    ) -> None:
        if not isinstance(interpolator, OneDimensionalInterpolator):
            raise ValueError('Pointwise errors require a one-dimensional interpolator.')
        self.reference_time = reference_time
        self.reference_data = reference_data
        self.interpolator = interpolator

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
        resp_interp = self.interpolator(
            self.reference_time,
            response.get_time(),
            response.get_data(),
        )
        # Compute normalised error
        factor = np.mean(np.abs(self.reference_data))
        return OutputResult(self.reference_time, (resp_interp - self.reference_data) / factor)


class FullFieldErrors(ResponseTransformer):
    """Compute the pointwise full-field errors between the data and a reference."""

    def __init__(
        self,
        reference_coords: np.ndarray,
        reference_data: np.ndarray,
        interpolator: Interpolator,
    ) -> None:
        if isinstance(interpolator, OneDimensionalInterpolator):
            raise ValueError('Full-field errors require a spatial interpolator.')
        self.reference_coords = reference_coords
        self.reference_data = reference_data
        self.interpolator = interpolator
        self.norm_factor = np.mean(np.abs(reference_data), axis=0)

    def transform(self, response: OutputResult) -> FullFieldOutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        FullFieldOutputResult
            Transformed time and data points of the response.
        """
        if not isinstance(response, FullFieldOutputResult):
            raise ValueError('Full-field problems require full-field outputs from the solver.')
        # Compute normalised error of the spatially and temporally interpolated data
        interpolated_data = self.interpolator(
            self.reference_coords,
            response.get_coords(),
            response.get_data(),
        )
        return FullFieldOutputResult(
            self.reference_coords,
            (interpolated_data - self.reference_data) / self.norm_factor,
        )


AVAILABLE_RESPONSE_TRANSFORMERS: Dict[str, Type[ResponseTransformer]] = {
    'min': MinimumResponse,
    'max': MaximumResponse,
    'negate': NegateResponse,
    'square': SquareResponse,
    'chain': ChainResponse,
    'clip': ClipResponse,
    'affine': AffineTransformResponse,
    'reduction': ReduceResponse,
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
