"""Module for reducing the number of points in a reference response."""
from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
import scipy.optimize
from scipy.spatial import ConvexHull  # pylint: disable=E0611
from sklearn.cluster import KMeans
from piglot.utils.interpolators import Interpolator


def get_error(
    new_points: np.ndarray,
    new_values: np.ndarray,
    orig_points: np.ndarray,
    orig_values: np.ndarray,
    interpolator: Interpolator,
    weights: np.ndarray = None,
    fill_value: float = 0.0,
) -> float:
    """Compute the interpolation error of a set of points.

    Parameters
    ----------
    new_points : np.ndarray
        New coordinates of the points (n_points x n_dim).
    new_values : np.ndarray
        New values at the points (n_points x n_values).
    orig_points : np.ndarray
        Original coordinates of the points (n_points x n_dim).
    orig_values : np.ndarray
        Original values at the points (n_points x n_values).
    interpolator : Interpolator
        Interpolator to use for the error computation.
    weights : np.ndarray, optional
        Weights for the interpolation error, by default None
    fill_value : float, optional
        Value to fill NaNs in the errors, by default None

    Returns
    -------
    float
        Interpolation error with the point removed.
    """
    interp_values = interpolator(orig_points, new_points, new_values)
    errors = np.square(orig_values - interp_values)
    if np.any(np.isnan(errors)):
        if fill_value is None:
            return np.inf
        errors[np.isnan(errors)] = fill_value
    if weights is not None:
        errors *= weights
    return np.sum(errors).item()


def split_convex_hull(
    points: np.ndarray,
    values: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Split a set of points into the convex hull and the rest.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of the points (n_points x n_dim).
    values : np.ndarray
        Values at the points (n_points x n_values).

    Returns
    -------
    Tuple[int, np.ndarray, np.ndarray]
        Number of points in the convex hull and arranged coordinates and values.
    """
    # Check if we have enough points to form a convex hull
    n_points, n_dim = points.shape
    if n_points <= n_dim + 1:
        return n_points, points, values

    # Find the convex hull of the points to determine the boundary
    if points.shape[-1] == 1:
        # For 1D, the convex hull is the min and max points
        hull_idx = [np.argmin(points), np.argmax(points)]
    else:
        try:
            hull_idx = list(ConvexHull(points).vertices)
        except scipy.spatial.QhullError:  # pylint: disable=E1101
            # No way to continue if the convex hull fails
            return n_points, points, values

    # Arrange the points so that the convex hull is the first set of points in the list
    sorted_idx = hull_idx + [i for i in range(n_points) if i not in hull_idx]
    return len(hull_idx), np.copy(points[sorted_idx, :]), np.copy(values[sorted_idx, :])


def find_best_point(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    interpolator: Interpolator,
    weights: np.ndarray = None,
    start_idx: int = 0,
    baseline_error: float = 0.0,
) -> Tuple[int, float]:
    """Find the point to remove that minimises the interpolation error.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of the points to reduce (n_points x n_dim).
    values : np.ndarray
        Values at the points to reduce (n_points x n_values).
    ref_points : np.ndarray
        Reference coordinates of the points (n_points x n_dim).
    ref_values : np.ndarray
        Reference values at the points (n_points x n_values).
    interpolator : Interpolator
        Interpolator to use for the error computation.
    weights : np.ndarray, optional
        Weights for the interpolation error, by default None.
    start_idx : int, optional
        Index to start the search from, by default 0.
    baseline_error : float, optional
        Error of the original dataset, by default 0.0.

    Returns
    -------
    Tuple[int, float]
        Index of the best point to remove and the associated error.
    """
    errors = [
        get_error(
            np.delete(points, i, axis=0),
            np.delete(values, i, axis=0),
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
        )
        for i in range(start_idx, points.shape[0])
    ]
    if len(errors) == 0:
        return -1, baseline_error
    idx = np.argmin(errors)
    return idx + start_idx, errors[idx] + baseline_error


def reduce_points(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    progress_callback: Callable[[int, float], bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce the number of points in an unstructured dataset.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of the points to reduce (n_points x n_dim).
    values : np.ndarray
        Values at the points to reduce (n_points x n_values).
    ref_points : np.ndarray
        Reference coordinates of the points (n_points x n_dim).
    ref_values : np.ndarray
        Reference values at the points (n_points x n_values).
    tol : float
        Stop reducing when the error is above this tolerance.
    interpolator : Interpolator
        Interpolator to use for the error computation.
    baseline_error : float, optional
        Error of the original dataset, by default 0.0.
    weights : np.ndarray, optional
        Weights for the interpolation error, by default None.
    progress_callback : Callable[[int, float], bool], optional
        Callback function to report progress and check if the computation should stop.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    n_points, n_dim = points.shape
    num_hull, new_points, new_values = split_convex_hull(points, values)
    if num_hull == n_points:
        return points, values

    # At each iteration, remove the point with the smallest error
    n_interior = n_points - num_hull
    for _ in range(n_interior):
        idx, global_error = find_best_point(
            new_points,
            new_values,
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
            start_idx=num_hull,
            baseline_error=baseline_error,
        )
        # Check if removing this point increases the error above the tolerance
        if global_error > tol:
            break
        # Remove the point with the smallest error
        new_points = np.delete(new_points, idx, axis=0)
        new_values = np.delete(new_values, idx, axis=0)
        if progress_callback is not None:
            if progress_callback(new_points.shape[0], global_error):
                break

    # Re-sort for 1D interpolation before returning
    if n_dim == 1:
        idx = np.argsort(new_points[:, 0])
        new_points = new_points[idx, :]
        new_values = new_values[idx, :]
    return new_points, new_values


def reduce_points_clusters(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    n_clusters: int = None,
    progress_callback: Callable[[int, float], bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce the number of points in an unstructured dataset.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of the points to reduce (n_points x n_dim).
    values : np.ndarray
        Values at the points to reduce (n_points x n_values).
    ref_points : np.ndarray
        Reference coordinates of the points (n_points x n_dim).
    ref_values : np.ndarray
        Reference values at the points (n_points x n_values).
    tol : float
        Stop reducing when the error is above this tolerance.
    interpolator : Interpolator
        Interpolator to use for the error computation.
    baseline_error : float, optional
        Error of the original dataset, by default 0.0.
    weights : np.ndarray, optional
        Weights for the interpolation error, by default None.
    n_clusters : int, optional
        Number of clusters to use, by default None
    progress_callback : Callable[[int, float], bool], optional
        Callback function to report progress and check if the computation should stop.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Default value for the number of clusters
    n_points, n_dim = points.shape
    if n_clusters is None:
        n_clusters = n_points // (4 * (n_dim + 1))

    # Check if we have enough points to reduce
    if n_clusters < 2:
        return reduce_points(
            points,
            values,
            ref_points,
            ref_values,
            tol,
            interpolator,
            baseline_error=baseline_error,
            weights=weights,
            progress_callback=progress_callback,
        )

    # Cluster the points and values
    clustering = KMeans(n_clusters=n_clusters, random_state=0)
    mean = np.min(points, axis=0)
    std = np.max(points, axis=0) - mean
    points_std = (points - mean) / std
    ref_points_std = (ref_points - mean) / std
    clustering.fit(points_std)
    labels = clustering.predict(points_std)
    ref_labels = clustering.predict(ref_points_std)

    # Select each cluster
    num_hull_clusters = [0] * n_clusters
    num_interior_clusters = [0] * n_clusters
    point_clusters = [points[labels == i, :] for i in range(n_clusters)]
    value_clusters = [values[labels == i, :] for i in range(n_clusters)]
    ref_point_clusters = [ref_points[ref_labels == i, :] for i in range(n_clusters)]
    ref_value_clusters = [ref_values[ref_labels == i, :] for i in range(n_clusters)]
    weights_clusters = [weights] * n_clusters
    if weights is not None and weights.shape[0] == n_points and weights.shape[1] == n_dim:
        weights_clusters = [weights[labels == i] for i in range(n_clusters)]

    # Split the convex hull within each cluster
    for i in range(n_clusters):
        num_hull_clusters[i], point_clusters[i], value_clusters[i] = split_convex_hull(
            point_clusters[i],
            value_clusters[i],
        )
        num_interior_clusters[i] = point_clusters[i].shape[0] - num_hull_clusters[i]

    # As long as we have still points to process, keep going
    while all(num_interior > 0 for num_interior in num_interior_clusters):
        # Find best point of each cluster
        best_points = [
            find_best_point(
                point_clusters[i],
                value_clusters[i],
                ref_point_clusters[i],
                ref_value_clusters[i],
                interpolator,
                weights=weights_clusters[i],
                start_idx=num_hull_clusters[i],
            )
            for i in range(n_clusters)
        ]
        # Compute joint error and check if we can remove the best points
        global_error = baseline_error + sum(error for _, error in best_points)
        if global_error > tol / n_clusters:
            break
        # Remove the best points
        for i, best_point in enumerate(best_points):
            idx = best_point[0]
            if idx < 0:
                continue
            num_interior_clusters[i] -= 1
            point_clusters[i] = np.delete(point_clusters[i], idx, axis=0)
            value_clusters[i] = np.delete(value_clusters[i], idx, axis=0)
        if progress_callback is not None:
            if progress_callback(sum(p.shape[0] for p in point_clusters), global_error):
                return (
                    np.concatenate(point_clusters, axis=0),
                    np.concatenate(value_clusters, axis=0),
                )

    # Join the reduced data and cluster again
    new_points = np.concatenate(point_clusters, axis=0)
    new_values = np.concatenate(value_clusters, axis=0)
    return reduce_points_clusters(
        new_points,
        new_values,
        ref_points,
        ref_values,
        tol,
        interpolator,
        baseline_error=baseline_error,
        weights=weights,
        n_clusters=n_clusters - 1 if n_clusters < 16 else n_clusters // 2,
        progress_callback=progress_callback,
    )
