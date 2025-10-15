"""Module for reducing the number of points in a reference response."""
from __future__ import annotations
from typing import List, Set, Tuple, Callable, Optional
from itertools import chain, combinations
import math
import numpy as np
import scipy.optimize
from scipy.spatial import ConvexHull, Delaunay  # pylint: disable=E0611
from sklearn.cluster import KMeans
from piglot.utils.interpolators import Interpolator
from piglot.utils.triangulation import Triangulation
from piglot.utils.root_finding import PiecewiseDistribution, probabilistic_minimum_search


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
    # return np.max(errors).item() * orig_points.shape[0]
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
    hull_idx = convex_hull_indices(points)
    if hull_idx is None:
        return n_points, points, values

    # Arrange the points so that the convex hull is the first set of points in the list
    sorted_idx = hull_idx.tolist() + [i for i in range(n_points) if i not in hull_idx]
    return len(hull_idx), np.copy(points[sorted_idx, :]), np.copy(values[sorted_idx, :])


def convex_hull_indices(
    points: np.ndarray,
) -> np.ndarray:
    """List the indices of the points in the convex hull.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of the points (n_points x n_dim).

    Returns
    -------
    np.ndarray
        Indices of the points in the convex hull.
    """
    # Check if we have enough points to form a convex hull
    n_points, n_dim = points.shape
    if n_points <= n_dim + 1:
        return np.arange(n_points)

    # For 1D, the convex hull is the min and max points
    if points.shape[-1] == 1:
        return np.array([np.argmin(points), np.argmax(points)])

    # Find the convex hull of the points to determine the boundary
    try:
        return ConvexHull(points).vertices
    except scipy.spatial.QhullError:  # pylint: disable=E1101
        return None


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
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
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
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
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
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
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
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
        Callback function to report progress and check if the computation should stop.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Default value for the number of clusters
    n_points, n_dim = points.shape
    if n_clusters is None:
        n_clusters = n_points // 10

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

    # Standardise the points and reference points into a common space
    # We aim to get points that have more or less the same spacings in each dimension
    train_data = points
    ref_train_data = ref_points
    mean = np.mean(train_data, axis=0)
    avg_dist = np.array([
        np.mean(np.diff(np.unique(np.sort(train_data[:, i]))))
        for i in range(train_data.shape[1])
    ])
    train_data = (train_data - mean) / avg_dist
    ref_train_data = (ref_train_data - mean) / avg_dist

    # Cluster the points and values
    clustering = KMeans(n_clusters=n_clusters, random_state=0)
    clustering.fit(train_data)
    labels = clustering.predict(train_data)
    ref_labels = clustering.predict(ref_train_data)

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
    while np.sum(num_interior > 0 for num_interior in num_interior_clusters) > 0.5 * n_clusters:
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


def simplex_gradient(
    points: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Compute the gradient of the linear interpolator for a given simplex.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of the simplex points (n_dim + 1 x n_dim).
    values : np.ndarray
        Values at the points (n_dim + 1 x n_values).

    Returns
    -------
    np.ndarray
        Gradient of the interpolator at each point (n_values x n_dim).
    """
    edge_vectors = (points[1:, :] - points[0, :]).T
    return np.linalg.solve(edge_vectors.T, values[1:, :] - values[0, :]).T


def simplex_removal_error(
    triangulation: Triangulation,
    simplex_idx: int,
    points: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Compute the maximum expected error when removing a simplex.

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation object containing the simplices and neighbors.
    simplex_idx : int
        Index of the simplex to remove.
    points : np.ndarray
        Coordinates of the points (n_points x n_dim).
    values : np.ndarray
        Values at the points (n_points x n_values).

    Returns
    -------
    np.ndarray
        Maximum expected error when removing the simplex (n_values).
    """
    simplex_points = triangulation.simplices[simplex_idx]
    other_idx = list(set(chain.from_iterable(
        triangulation.point_neighbours(p) for p in simplex_points
    )))
    extrapolated_values = triangulation.interpolate(
        points[simplex_points, :],
        values[simplex_points, :],
        points[other_idx, :]
    )
    true_values = values[other_idx, :]
    return np.max(np.abs(extrapolated_values - true_values), axis=0)


def reduce_points_simplices_mc(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
    num_iters: int = 100,
    seed: int = 0,
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
        Target tolerance for the error.
    interpolator : Interpolator
        Interpolator to use for the error computation.
    baseline_error : float, optional
        Error of the original dataset, by default 0.0.
    weights : np.ndarray, optional
        Weights for the interpolation error, by default None.
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
        Callback function to report progress and check if the computation should stop.
    num_iters : int, optional
        Number of iterations for the sampling stage, by default 100.
    seed : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Compute the convex hull of the points
    hull_idx = convex_hull_indices(points)
    if hull_idx is None:
        return points, values

    # Standardise the values
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std_values = (values - mean) / np.where(std == 0, 1.0, std)

    # Triangulation of the points
    if progress_callback:
        progress_callback(points.shape[0], baseline_error, 'triangulating')
    tri = Triangulation(points)
    interior_point_idx = np.setdiff1d(np.arange(points.shape[0]), hull_idx)

    errors = np.zeros(tri.simplices.shape[0])
    boundary_simplices = []
    for i in range(tri.simplices.shape[0]):
        # Check if the simplex is an interior simplex
        if not np.all(np.isin(tri.simplices[i], interior_point_idx)):
            boundary_simplices.append(i)
            continue
        errors[i] = np.max(
            simplex_removal_error(
                tri,
                i,
                tri.transformed_points,
                std_values,
            ),
        )
        if progress_callback:
            progress_callback(
                points.shape[0],
                baseline_error,
                f'scoring simplices {i + 1}/{tri.simplices.shape[0]}',
            )

    # Estimate pontual errors from the simplex errors
    pontual_errors = np.zeros(points.shape[0])
    for i in interior_point_idx:
        for p in tri.vertex_to_simplices(i):
            pontual_errors[i] = max(pontual_errors[i], errors[p])

    # Monte Carlo removal of points based on the pontual errors
    rng = np.random.default_rng(seed)
    probs = pontual_errors[interior_point_idx]
    probs += 1e-8 * np.min(probs[probs > 0])  # Avoid zero probabilities
    probs /= np.sum(probs)

    # Store progress and best solution history (using mutable types)
    iter_count = [0]
    best_solution: List[Tuple[float, np.ndarray]] = []

    def closure(point_ratio: float) -> float:
        num_points = int(len(interior_point_idx) * point_ratio)
        sampled_points = rng.choice(
            interior_point_idx,
            size=num_points,
            replace=False,
            p=probs,
        ).tolist() + hull_idx.tolist()
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[sampled_points] = True
        error = get_error(
            points[mask],
            values[mask],
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
        )

        # Fetch best solution so far
        if len(best_solution) > 0:
            best_error, best_mask = best_solution[-1]
        else:
            best_error = 0.0
            best_mask = np.ones(points.shape[0], dtype=bool)

        # Update best solution if the new solution is valid and either:
        # i) Has fewer points (irrespective of the error)
        # ii) Has a smaller error (with the same number of points)
        if baseline_error + error <= tol:
            point_diff = np.sum(best_mask) - np.sum(mask)
            if point_diff > 0 or (point_diff == 0 and baseline_error + error < best_error):
                best_error = baseline_error + error
                best_mask = mask
                best_solution.append((best_error, best_mask))

        # Progress update
        iter_count[0] += 1
        if progress_callback:
            progress_callback(
                np.sum(best_mask),
                best_error,
                f'sampling {iter_count[0]}/{num_iters}',
            )
        return baseline_error + error - tol

    # # Estimate the prior distribution of the point ratio
    # error_priors = closure(0.0) * np.cumsum(probs)[::-1]
    # prior = PiecewiseDistribution(
    #     points=np.linspace(0, 1, len(interior_point_idx) + 1),
    #     values=np.exp(-np.square(error_priors - tol)),
    # )

    # Sample the point ratio using a probabilistic bisection method
    probabilistic_minimum_search(closure, max_iter=num_iters)

    # Find the best solution
    if len(best_solution) > 0:
        _, best_mask = best_solution[-1]
    else:
        best_mask = np.ones(points.shape[0], dtype=bool)
    return points[best_mask], values[best_mask]


def reduce_points_simplices_neighbours(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
    depth: int = 1,
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
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
        Callback function to report progress and check if the computation should stop.
    depth : int, optional
        Current search depth, by default 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Compute the convex hull of the points
    hull_idx = convex_hull_indices(points)
    if hull_idx is None:
        return points, values

    # Standardise the values
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std_values = (values - mean) / np.where(std == 0, 1.0, std)

    # Triangulation of the points
    if progress_callback:
        progress_callback(points.shape[0], baseline_error, f'depth {depth}: triangulating')
    tri = Triangulation(points)
    interior_point_idx = np.setdiff1d(np.arange(points.shape[0]), hull_idx)

    errors = {}
    for i in range(tri.simplices.shape[0]):
        # Check if the simplex is an interior simplex
        if not np.all(np.isin(tri.simplices[i], interior_point_idx)):
            continue
        errors[i] = np.max(
            simplex_removal_error(
                tri,
                i,
                tri.transformed_points,
                std_values,
            ),
        )
        if progress_callback:
            progress_callback(
                points.shape[0],
                baseline_error,
                f'depth {depth}: scoring simplices {i + 1}/{tri.simplices.shape[0]}',
            )

    # Find a subset of non-neighboring simplices with the lowest errors
    idx = 0
    accepted_simplices = []
    removed_simplices = [False for _ in range(tri.simplices.shape[0])]
    sorted_simplices = [
        a for a, b in sorted(errors.items(), key=lambda x: x[1])
    ][:len(errors) // 2 + 1]
    for idx, curr_simplex in enumerate(sorted_simplices):
        if progress_callback:
            progress_callback(
                points.shape[0],
                baseline_error,
                f'depth {depth}: removing simplex {idx + 1}/{len(sorted_simplices)}',
            )
        if removed_simplices[curr_simplex]:
            continue
        accepted_simplices.append(curr_simplex)
        for point in tri.simplices[curr_simplex, :]:
            for neighbour in tri.point_neighbours(point, depth=1):
                for simplex in tri.vertex_to_simplices(neighbour):
                    if simplex not in accepted_simplices:
                        removed_simplices[simplex] = True
    sorted_simplices = accepted_simplices

    # Pseudo bissection method to find the maximum number of points to remove
    mask = np.ones(points.shape[0], dtype=bool)
    num_steps = int(np.ceil(np.log2(len(sorted_simplices) + 1)) + 1)
    valid_current = 0
    best_error = 0.0
    current = len(sorted_simplices)
    if progress_callback:
        progress_callback(
            np.sum(mask),
            baseline_error,
            f'depth {depth}: removing points using bissection - step 0/{num_steps}',
        )
    for i in range(num_steps):
        mask[:] = True
        curr_simplices = sorted_simplices[:current]
        mask[np.unique(tri.simplices[curr_simplices, :].flatten())] = False
        error = get_error(
            points[mask],
            values[mask],
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
        )
        # Choose update direction based on the error
        if baseline_error + error > tol:
            current //= 2
        else:
            valid_current = current
            best_error = error
            break
        # Update the progress callback
        if progress_callback:
            progress_callback(
                np.sum(mask),
                baseline_error + best_error,
                f'depth {depth}: removing points using bissection - step {i + 1}/{num_steps}',
            )

    # Return the best solution
    mask[:] = True
    curr_simplices = sorted_simplices[:valid_current]
    mask[np.unique(tri.simplices[curr_simplices, :].flatten())] = False
    if np.all(mask):
        return points, values
    return reduce_points_simplices_neighbours(
        points[mask],
        values[mask],
        ref_points,
        ref_values,
        tol,
        interpolator,
        baseline_error,
        weights,
        progress_callback,
        depth=depth + 1,
    )


def reduce_points_simplices(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
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
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
        Callback function to report progress and check if the computation should stop.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Compute the convex hull of the points
    hull_idx = convex_hull_indices(points)
    if hull_idx is None:
        return points, values

    # Standardise the values
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std_values = (values - mean) / np.where(std == 0, 1.0, std)

    # Triangulation of the points
    if progress_callback:
        progress_callback(points.shape[0], baseline_error, 'triangulating')
    tri = Triangulation(points)
    interior_point_idx = np.setdiff1d(np.arange(points.shape[0]), hull_idx)

    errors = {}
    for i in range(tri.simplices.shape[0]):
        # Check if the simplex is an interior simplex
        if not np.all(np.isin(tri.simplices[i], interior_point_idx)):
            continue
        errors[i] = np.max(
            simplex_removal_error(
                tri,
                i,
                tri.transformed_points,
                std_values,
            ),
        )
        if progress_callback:
            progress_callback(
                points.shape[0],
                baseline_error,
                f'scoring simplices {i + 1}/{tri.simplices.shape[0]}',
            )

    # Pseudo bissection method to find the maximum number of points to remove
    sorted_simplices = [a for a, b in sorted(errors.items(), key=lambda x: x[1])]

    mask = np.ones(points.shape[0], dtype=bool)
    num_steps = int(np.ceil(np.log2(len(sorted_simplices))) + 1)
    valid_current = 0
    best_error = 0.0
    current = len(sorted_simplices) // 2
    step = len(sorted_simplices) // 4
    if progress_callback:
        progress_callback(
            np.sum(mask),
            baseline_error,
            f'removing points using bissection - step 0/{num_steps}',
        )
    for i in range(num_steps):
        mask[:] = True
        curr_simplices = sorted_simplices[:current]
        mask[np.unique(tri.simplices[curr_simplices, :].flatten())] = False
        error = get_error(
            points[mask],
            values[mask],
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
        )
        # Choose update direction based on the error
        if baseline_error + error < tol:
            if current > valid_current:
                valid_current = current
                best_error = error
            current += step
        else:
            current -= step
        step //= 2
        # Update the progress callback
        if progress_callback:
            progress_callback(
                np.sum(mask),
                baseline_error + best_error,
                f'removing points using bissection - step {i + 1}/{num_steps}',
            )

    # Return the best solution
    mask[:] = True
    curr_simplices = sorted_simplices[:valid_current]
    mask[np.unique(tri.simplices[curr_simplices, :].flatten())] = False
    return points[mask], values[mask]


def reduce_points_clusters_recursive(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    n_clusters: int = None,
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
    seed: int = 0,
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
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
        Callback function to report progress and check if the computation should stop.
    seed : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Default value for the number of clusters
    n_points, n_dim = points.shape
    if n_clusters is None:
        n_clusters = n_points // 500

    while n_clusters > n_points:
        n_clusters //= 2

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

    # Standardise the points and reference points into a common space
    # We aim to get points that have more or less the same spacings in each dimension
    train_data = points
    ref_train_data = ref_points
    mean = np.mean(train_data, axis=0)
    avg_dist = np.array([
        np.mean(np.diff(np.unique(np.sort(train_data[:, i]))))
        for i in range(train_data.shape[1])
    ])
    train_data = (train_data - mean) / avg_dist
    ref_train_data = (ref_train_data - mean) / avg_dist

    # Cluster the points and values
    clustering = KMeans(n_clusters=n_clusters, random_state=seed)
    clustering.fit(train_data)
    labels = clustering.predict(train_data)
    ref_labels = clustering.predict(ref_train_data)

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

    # As long as we have still points to process, reduce each cluster recursively
    if all(num_interior > 0 for num_interior in num_interior_clusters):
        for i in range(n_clusters):
            point_clusters[i], value_clusters[i] = reduce_points_clusters_recursive(
                point_clusters[i],
                value_clusters[i],
                ref_point_clusters[i],
                ref_value_clusters[i],
                tol,
                interpolator,
                baseline_error=baseline_error,
                weights=weights_clusters[i],
                n_clusters=n_clusters // 2 if n_clusters > 16 else n_clusters,
                # progress_callback=progress_callback,
            )

        # Compute the global error after reducing each cluster
        global_error = baseline_error + sum(
            get_error(
                point_clusters[i],
                value_clusters[i],
                ref_point_clusters[i],
                ref_value_clusters[i],
                interpolator,
                weights=weights_clusters[i],
            )
            for i in range(n_clusters)
        )
        # Check if we can stop the reduction
        if progress_callback is not None:
            if progress_callback(sum(p.shape[0] for p in point_clusters), global_error):
                return (
                    np.concatenate(point_clusters, axis=0),
                    np.concatenate(value_clusters, axis=0),
                )

    # Join the reduced data and cluster again
    new_points = np.concatenate(point_clusters, axis=0)
    new_values = np.concatenate(value_clusters, axis=0)
    return reduce_points_clusters_recursive(
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
        seed=seed,
    )


def reduce_points_bruteforce(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
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
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
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

    def compute_error(indices: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
        return indices, get_error(
            np.concatenate([new_points[:num_hull, :], new_points[indices, :]]),
            np.concatenate([new_values[:num_hull, :], new_values[indices, :]]),
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
        )

    n_interior = n_points - num_hull
    candidates = list(range(num_hull, n_points))
    best_points = np.copy(new_points)
    best_values = np.copy(new_values)
    for i in range(n_interior):
        best_error = np.inf
        best_candidates = None
        total = math.comb(n_interior, i + 1)
        for count, combination in enumerate(combinations(candidates, i + 1)):
            indices, error = compute_error(combination)
            if error < best_error:
                best_error = error
                best_candidates = indices
                # if best_error <= tol:
                #     break
            if progress_callback:
                progress_callback(
                    n_points,
                    baseline_error,
                    f'Bruteforce ({i + num_hull + 1}): {count}/{total} best: {best_error}',
                )
        best_points = np.concatenate([new_points[:num_hull, :], new_points[best_candidates, :]])
        best_values = np.concatenate([new_values[:num_hull, :], new_values[best_candidates, :]])
        if best_error <= tol:
            break
    if n_dim == 1:
        idx = np.argsort(best_points[:, 0])
        best_points = best_points[idx, :]
        best_values = best_values[idx, :]
    return best_points, best_values



def reduce_points_simplices_mc_simplices(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
    num_iters: int = 100,
    seed: int = 0,
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
        Target tolerance for the error.
    interpolator : Interpolator
        Interpolator to use for the error computation.
    baseline_error : float, optional
        Error of the original dataset, by default 0.0.
    weights : np.ndarray, optional
        Weights for the interpolation error, by default None.
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
        Callback function to report progress and check if the computation should stop.
    num_iters : int, optional
        Number of iterations for the sampling stage, by default 100.
    seed : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Compute the convex hull of the points
    hull_idx = convex_hull_indices(points)
    if hull_idx is None:
        return points, values

    # Standardise the values
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std_values = (values - mean) / np.where(std == 0, 1.0, std)

    # Triangulation of the points
    if progress_callback:
        progress_callback(points.shape[0], baseline_error, 'triangulating')
    tri = Triangulation(points)
    interior_point_idx = np.setdiff1d(np.arange(points.shape[0]), hull_idx)

    errors = np.zeros(tri.simplices.shape[0])
    boundary_simplices = []
    interior_simplex_idx = []
    for i in range(tri.simplices.shape[0]):
        # Check if the simplex is an interior simplex
        if not np.all(np.isin(tri.simplices[i], interior_point_idx)):
            boundary_simplices.append(i)
            continue
        interior_simplex_idx.append(i)
        errors[i] = np.max(
            simplex_removal_error(
                tri,
                i,
                tri.transformed_points,
                std_values,
            ),
        )
        if progress_callback:
            progress_callback(
                points.shape[0],
                baseline_error,
                f'scoring simplices {i + 1}/{tri.simplices.shape[0]}',
            )

    # Monte Carlo removal of simplices based on the errors
    rng = np.random.default_rng(seed)
    probs = errors[interior_simplex_idx]
    probs += 1e-8 * np.min(probs[probs > 0])  # Avoid zero probabilities
    probs /= np.sum(probs)

    # Store progress and best solution history (using mutable types)
    iter_count = [0]
    best_solution: List[Tuple[float, np.ndarray]] = []

    def closure(point_ratio: float) -> float:
        num_simplices = int(len(interior_simplex_idx) * point_ratio)
        sampled_simplices = rng.choice(
            interior_simplex_idx,
            size=num_simplices,
            replace=False,
            p=probs,
        ).tolist() + boundary_simplices
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[np.unique(tri.simplices[sampled_simplices, :].flatten())] = True
        error = get_error(
            points[mask],
            values[mask],
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
        )

        # Fetch best solution so far
        if len(best_solution) > 0:
            best_error, best_mask = best_solution[-1]
        else:
            best_error = 0.0
            best_mask = np.ones(points.shape[0], dtype=bool)

        # Update best solution if the new solution is valid and either:
        # i) Has fewer points (irrespective of the error)
        # ii) Has a smaller error (with the same number of points)
        if baseline_error + error <= tol:
            point_diff = np.sum(best_mask) - np.sum(mask)
            if point_diff > 0 or (point_diff == 0 and baseline_error + error < best_error):
                best_error = baseline_error + error
                best_mask = mask
                best_solution.append((best_error, best_mask))

        # Progress update
        iter_count[0] += 1
        if progress_callback:
            progress_callback(
                np.sum(best_mask),
                best_error,
                f'sampling {iter_count[0]}/{num_iters}',
            )
        return baseline_error + error - tol

    # # Estimate the prior distribution of the point ratio
    # error_priors = closure(0.0) * np.cumsum(probs)[::-1]
    # prior = PiecewiseDistribution(
    #     points=np.linspace(0, 1, len(interior_simplex_idx) + 1),
    #     values=np.exp(-np.square(error_priors - tol)),
    # )

    # Sample the point ratio using a probabilistic bisection method
    probabilistic_minimum_search(closure, max_iter=num_iters)

    # Find the best solution
    if len(best_solution) > 0:
        _, best_mask = best_solution[-1]
    else:
        best_mask = np.ones(points.shape[0], dtype=bool)
    return points[best_mask], values[best_mask]


def reduce_points_simplices_neighbours_points(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
    depth: int = 1,
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
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
        Callback function to report progress and check if the computation should stop.
    depth : int, optional
        Current search depth, by default 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Compute the convex hull of the points
    hull_idx = convex_hull_indices(points)
    if hull_idx is None:
        return points, values

    # Standardise the values
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std_values = (values - mean) / np.where(std == 0, 1.0, std)

    # Triangulation of the points
    if progress_callback:
        progress_callback(points.shape[0], baseline_error, f'depth {depth}: triangulating')
    tri = Triangulation(points)
    interior_point_idx = np.setdiff1d(np.arange(points.shape[0]), hull_idx)

    errors = {}
    for i in range(tri.simplices.shape[0]):
        # Check if the simplex is an interior simplex
        if not np.all(np.isin(tri.simplices[i], interior_point_idx)):
            continue
        errors[i] = np.max(
            simplex_removal_error(
                tri,
                i,
                tri.transformed_points,
                std_values,
            ),
        )
        if progress_callback:
            progress_callback(
                points.shape[0],
                baseline_error,
                f'depth {depth}: scoring simplices {i + 1}/{tri.simplices.shape[0]}',
            )

    # Find a subset of non-neighboring simplices with the lowest errors
    idx = 0
    accepted_simplices = []
    removed_simplices = [False for _ in range(tri.simplices.shape[0])]
    sorted_simplices = [
        a for a, b in sorted(errors.items(), key=lambda x: x[1])
    ][:len(errors) // 2 + 1]
    for idx, curr_simplex in enumerate(sorted_simplices):
        if progress_callback:
            progress_callback(
                points.shape[0],
                baseline_error,
                f'depth {depth}: removing simplex {idx + 1}/{len(sorted_simplices)}',
            )
        if removed_simplices[curr_simplex]:
            continue
        accepted_simplices.append(curr_simplex)
        for point in tri.simplices[curr_simplex, :]:
            for neighbour in tri.point_neighbours(point, depth=1):
                for simplex in tri.vertex_to_simplices(neighbour):
                    if simplex not in accepted_simplices:
                        removed_simplices[simplex] = True
    sorted_simplices = accepted_simplices

    # Estimate pontual errors from the simplex errors
    pontual_errors = np.zeros(points.shape[0])
    for i in interior_point_idx:
        for p in tri.vertex_to_simplices(i):
            pontual_errors[i] = max(pontual_errors[i], errors[p])

    # Pseudo bissection method to find the maximum number of points to remove
    mask = np.ones(points.shape[0], dtype=bool)
    num_steps = int(np.ceil(np.log2(len(sorted_simplices) + 1)) + 1)
    valid_current = 0
    best_error = 0.0
    current = len(sorted_simplices)
    if progress_callback:
        progress_callback(
            np.sum(mask),
            baseline_error,
            f'depth {depth}: removing points using bissection - step 0/{num_steps}',
        )
    for i in range(num_steps):
        mask[:] = True
        curr_simplices = sorted_simplices[:current]
        mask[np.unique(tri.simplices[curr_simplices, :].flatten())] = False
        error = get_error(
            points[mask],
            values[mask],
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
        )
        # Choose update direction based on the error
        if baseline_error + error > tol:
            current //= 2
        else:
            valid_current = current
            best_error = error
            break
        # Update the progress callback
        if progress_callback:
            progress_callback(
                np.sum(mask),
                baseline_error + best_error,
                f'depth {depth}: removing points using bissection - step {i + 1}/{num_steps}',
            )

    # Return the best solution
    mask[:] = True
    curr_simplices = sorted_simplices[:valid_current]
    mask[np.unique(tri.simplices[curr_simplices, :].flatten())] = False
    if np.all(mask):
        return points, values
    return reduce_points_simplices_neighbours(
        points[mask],
        values[mask],
        ref_points,
        ref_values,
        tol,
        interpolator,
        baseline_error,
        weights,
        progress_callback,
        depth=depth + 1,
    )


def reduce_points_simplices_points(
    points: np.ndarray,
    values: np.ndarray,
    ref_points: np.ndarray,
    ref_values: np.ndarray,
    tol: float,
    interpolator: Interpolator,
    baseline_error: float = 0.0,
    weights: np.ndarray = None,
    progress_callback: Callable[[int, float, Optional[str]], bool] = None,
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
    progress_callback : Callable[[int, float, Optional[str]], bool], optional
        Callback function to report progress and check if the computation should stop.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced coordinates and values (n_points x n_dim and n_points x n_values).
    """
    # Compute the convex hull of the points
    hull_idx = convex_hull_indices(points)
    if hull_idx is None:
        return points, values

    # Standardise the values
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std_values = (values - mean) / np.where(std == 0, 1.0, std)

    # Triangulation of the points
    if progress_callback:
        progress_callback(points.shape[0], baseline_error, 'triangulating')
    tri = Triangulation(points)
    interior_point_idx = np.setdiff1d(np.arange(points.shape[0]), hull_idx)

    errors = {}
    for i in range(tri.simplices.shape[0]):
        # Check if the simplex is an interior simplex
        if not np.all(np.isin(tri.simplices[i], interior_point_idx)):
            continue
        errors[i] = np.max(
            simplex_removal_error(
                tri,
                i,
                tri.transformed_points,
                std_values,
            ),
        )
        if progress_callback:
            progress_callback(
                points.shape[0],
                baseline_error,
                f'scoring simplices {i + 1}/{tri.simplices.shape[0]}',
            )

    # Estimate pontual errors from the simplex errors
    pontual_errors = {}
    for i in interior_point_idx:
        pontual_errors[i] = 0.0
        for p in tri.vertex_to_simplices(i):
            pontual_errors[i] = max(pontual_errors[i], errors[p])

    # Pseudo bissection method to find the maximum number of points to remove
    sorted_points = [a for a, b in sorted(pontual_errors.items(), key=lambda x: x[1])]

    mask = np.ones(points.shape[0], dtype=bool)
    num_steps = int(np.ceil(np.log2(len(sorted_points))) + 1)
    valid_current = 0
    best_error = 0.0
    current = len(sorted_points) // 2
    step = len(sorted_points) // 4
    if progress_callback:
        progress_callback(
            np.sum(mask),
            baseline_error,
            f'removing points using bissection - step 0/{num_steps}',
        )
    for i in range(num_steps):
        mask[:] = True
        mask[sorted_points[:current]] = False
        error = get_error(
            points[mask],
            values[mask],
            ref_points,
            ref_values,
            interpolator,
            weights=weights,
        )
        # Choose update direction based on the error
        if baseline_error + error < tol:
            if current > valid_current:
                valid_current = current
                best_error = error
            current += step
        else:
            current -= step
        step //= 2
        # Update the progress callback
        if progress_callback:
            progress_callback(
                np.sum(mask),
                baseline_error + best_error,
                f'removing points using bissection - step {i + 1}/{num_steps}',
            )

    # Return the best solution
    mask[:] = True
    mask[sorted_points[:valid_current]] = False
    return points[mask], values[mask]
