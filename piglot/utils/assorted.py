"""Assorted utilities."""
from typing import List, Dict, Tuple, Type, TypeVar, Any
import os
import math
import warnings
import contextlib
import importlib
import importlib.util
import numpy as np
from scipy.stats import t
from scipy.spatial import Delaunay  # pylint: disable=E0611


def pretty_time(elapsed_sec: float) -> str:
    """Return a human-readable representation of a given elapsed time

    Parameters
    ----------
    elapsed_sec : float
        Elapsed time, in seconds

    Returns
    -------
    str
        Pretty elapsed time string
    """
    mults = {
        'y': 60*60*24*365,
        'd': 60*60*24,
        'h': 60*60,
        'm': 60,
        's': 1,
    }
    time_str = ''
    for suffix, factor in mults.items():
        count = elapsed_sec // factor
        if count > 0:
            time_str += str(int(elapsed_sec / factor)) + suffix
        elapsed_sec %= factor
    if time_str == '':
        time_str = f'{elapsed_sec:.5f}s'
    return time_str


def reverse_pretty_time(time_str: str) -> float:
    """Return an elapsed time from its human-readable representation

    Parameters
    ----------
    time_str : str
        Pretty elapsed time string

    Returns
    -------
    str
        Elapsed time, in seconds
    """
    mults = {
        'y': 60*60*24*365,
        'd': 60*60*24,
        'h': 60*60,
        'm': 60,
        's': 1,
    }
    value = 0.0
    for suffix, factor in mults.items():
        if suffix in time_str:
            left, time_str = time_str.split(suffix)
            value += float(left) * factor
    return value


def filter_close_points(data: np.ndarray, tol: float) -> np.ndarray:
    """Remove points from an array that are too close to each other.

    Parameters
    ----------
    data : np.ndarray
        Data array.
    tol : float
        Tolerance to use during removal.

    Returns
    -------
    np.ndarray
        Reduced array.
    """
    delta = np.diff(data)
    return data[np.insert(delta > tol, 0, True)]


def stats_interp_to_common_grid(
        responses: List[Tuple[np.ndarray, np.ndarray]],
        ) -> Dict[str, np.ndarray]:
    """Interpolate a set of response to a common grid and compute several statistics.

    Parameters
    ----------
    responses : List[Tuple[np.ndarray, np.ndarray]]
        List of responses to interpolate.

    Returns
    -------
    Dict[str, np.ndarray]
        Results for the interpolated responses.
    """
    # Get the common grid: join all points and filter out close ones
    grid_total = np.concatenate([response[0] for response in responses])
    grid_range = np.max(grid_total) - np.min(grid_total)
    grid = filter_close_points(np.sort(grid_total), grid_range * 1e-6)
    # Interpolate all responses to the common grid
    interp_responses = np.array([
        np.interp(grid, response[0], response[1], left=np.nan, right=np.nan)
        for response in responses
    ])
    # Return all relevant quantities
    num_points = np.count_nonzero(~np.isnan(interp_responses), axis=0)
    conf = t.interval(0.95, num_points - 1)[1]
    return {
        'grid': grid,
        'num_points': num_points,
        'responses': interp_responses,
        'average': np.nanmean(interp_responses, axis=0),
        'variance': np.nanvar(interp_responses, axis=0),
        'std': np.nanstd(interp_responses, axis=0),
        'min': np.nanmin(interp_responses, axis=0),
        'max': np.nanmax(interp_responses, axis=0),
        'median': np.nanmedian(interp_responses, axis=0),
        'confidence': conf * np.nanstd(interp_responses, axis=0) / np.sqrt(num_points),
    }


@contextlib.contextmanager
def change_cwd(path: str):
    """Context manager to temporarily change the current working directory.

    Adapted from https://stackoverflow.com/a/75049063

    Parameters
    ----------
    path : str
        New working directory.
    """
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


T = TypeVar('T')


def read_custom_module(config: Dict[str, Any], cls: Type[T]) -> Type[T]:
    """Read a custom module from a configuration spec.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration of the custom module.
    cls : Type
        Base class of the module to load.

    Returns
    -------
    Type
        Custom module type read from the script.
    """
    # Sanitise the configuration
    if 'script' not in config:
        raise ValueError("Missing 'script' field for reading the custom module script.")
    if 'class' not in config:
        raise ValueError("Missing 'class' field for reading the custom module script.")
    # Load the module
    module_name = f'piglot_{os.path.basename(config["script"]).replace(".", "_")}'
    spec = importlib.util.spec_from_file_location(module_name, config['script'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_class = getattr(module, config['class'])
    # Sanitise the class
    if not issubclass(module_class, cls):
        raise ValueError(f"Custom class '{module_class}' is not a subclass of '{cls}'.")
    return module_class


def simplex_volume(vertices: np.ndarray) -> float:
    """Compute the volume of a simplex from its vertices.

    Parameters
    ----------
    vertices : np.ndarray
        Vertices of the simplex (n_points x n_dim) = (n_dim + 1 x n_dim).

    Returns
    -------
    float
        Volume of the simplex.
    """
    vertices_matrix = vertices[1:, :] - vertices[0, :]
    return np.abs(np.linalg.det(vertices_matrix.T)) / math.factorial(vertices.shape[1])


def project_to_full_rank_space(coords: np.ndarray, tol: float = 1e-12) -> Tuple[int, np.ndarray]:
    """Project a list of coordinates to its full-rank space.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the points to project (n_points x n_dim).
    tol : float, optional
        Tolerance for singular values, by default 1e-12.

    Returns
    -------
    Tuple[int, np.ndarray]
        Rank and full-rank coordinates (n_points x n_dim*).
    """
    difference = coords[1:, :] - coords[0, :]
    _, R = np.linalg.qr(difference)  # pylint: disable=C0103
    independent_dims = np.abs(np.diag(R)) > tol
    return np.sum(independent_dims), coords[:, independent_dims]


def trapezoidal_integration_weights(coords: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute the weights for a multi-dimensional trapezoidal integration.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the points to integrate (n_points x n_dim).
    tol : float, optional
        Tolerance for rank reduction, by default 1e-12.

    Returns
    -------
    np.ndarray
        Weights for trapezoidal integration (n_points).
    """
    # If required, project the points to the full-rank space
    new_coords = coords
    rank, tmp = project_to_full_rank_space(coords, tol=tol)
    if rank < coords.shape[1]:
        warnings.warn(
            "Rank-deficient coordinates for triangulation, projected and reduced from "
            f"{coords.shape[1]} to {rank} dimensions. Integration may be inaccurate."
        )
        new_coords = tmp
    # Check if we have sufficient points to integrate
    # If not, simply return a vector to average all points
    n_points = new_coords.shape[0]
    n_dim = new_coords.shape[1]
    if n_points <= n_dim + 1:
        return np.ones(n_points) / n_points
    # If we fall back to a single dimension, return the trapezoidal weights
    if n_dim == 1:
        dx = np.abs(np.diff(new_coords, axis=0).flatten())
        return np.concatenate([[dx[0] / 2], (dx[:-1] + dx[1:]) / 2, [dx[-1] / 2]])
    # Compute the Delaunay triangulation and get the weights based on the simplex volume
    delaunay = Delaunay(new_coords, qhull_options='QJ')  # QJ required to include all points
    weights = np.zeros(n_points)
    for simplex in delaunay.simplices:
        weights[simplex] += simplex_volume(delaunay.points[simplex, :])
    return weights
