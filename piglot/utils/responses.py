"""Module for reducing the number of points in a response function"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import scipy.optimize


class ResamplingLoss:
    """Loss for resampling a response function"""

    def __init__(self, ref_x: np.ndarray, ref_y: np.ndarray, n_points: int) -> None:
        self.ref_x = ref_x
        self.ref_y = ref_y
        self.n_points = n_points
        self.min_x = np.min(ref_x)
        self.max_x = np.max(ref_x)

    def __call__(self, new_x: np.ndarray):
        # Using the new points, build the new grid
        new_x = np.concatenate([np.array([self.min_x]), np.sort(new_x), np.array([self.max_x])])
        new_y = np.interp(new_x, self.ref_x, self.ref_y)
        # Interpolate the new grid to the reference grid
        new_y_ref = np.interp(self.ref_x, new_x, new_y, left=new_y[0], right=new_y[-1])
        # Compute the integrated loss
        errors = np.square(new_y_ref - self.ref_y)
        loss = np.trapz(errors, self.ref_x) / np.trapz(np.square(self.ref_y), self.ref_x)
        return loss


def errors_interps(
        x_new: np.ndarray,
        y_new: np.ndarray,
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        ) -> np.ndarray:
    """Compute the error associated with removing each point from the grid

    Parameters
    ----------
    x_new : np.ndarray
        New time grid
    y_new : np.ndarray
        Values on the new grid
    x_ref : np.ndarray
        Old time grid
    y_ref : np.ndarray
        Values on the old grid

    Returns
    -------
    np.ndarray
        Error associated with removing each point
    """
    errors = []
    for i in range(len(x_new) - 2):
        x_deleted = np.delete(x_new, i + 1)
        y_deleted = np.delete(y_new, i + 1)
        y_ref_interp = np.interp(x_ref, x_deleted, y_deleted)
        errors.append(np.trapz(np.square(y_ref - y_ref_interp), x_ref))
    return errors


def reduce_response(
        x_old: np.ndarray,
        y_old: np.ndarray,
        tol: float,
        ) -> Tuple[int, float, Tuple[np.ndarray, np.ndarray]]:
    """Reduce the number of points in a response function

    Parameters
    ----------
    x_old : np.ndarray
        Original time grid
    y_old : np.ndarray
        Values in the original grid
    tol : float
        Maximum acceptable error

    Returns
    -------
    Tuple[int, float, Tuple[np.ndarray, np.ndarray]]
        Number of points, error, and new grid
    """

    # Ensure that the grid is sorted (for np.interp to work)
    idx = np.argsort(x_old)
    x_old = x_old[idx]
    y_old = y_old[idx]

    # Shortcut if we have way too many points
    x_new = np.linspace(np.min(x_old), np.max(x_old), 1000) if len(x_old) > 1000 else np.copy(x_old)
    y_new = np.interp(x_new, x_old, y_old)
    x_min, x_max = np.min(x_old), np.max(x_old)

    # Remove points until the error is below the tolerance or we run out of points
    while len(x_new) > 3:
        # Compute the error associated with removing each point
        error = errors_interps(x_new, y_new, x_old, y_old)
        idx = np.argmin(error)
        # Remove the point with the smallest error
        x_bk = np.copy(x_new)
        x_new = np.delete(x_new, idx + 1)
        # Compute the error after removing this point
        y_new = np.interp(x_new, x_old, y_old)
        y_interp = np.interp(x_old, x_new, y_new)
        y_error = np.trapz(np.square(y_old - y_interp), x_old) / np.trapz(np.square(y_old), x_old)
        # Check if we have reached the tolerance
        if y_error >= tol:
            x_new = x_bk
            break

    # Refine the solution: move the interior points to minimise the error
    x_init = x_new[1:-1]
    n_points = len(x_new)
    bounds = [(x_min, x_max)] * (n_points - 2)
    loss_func = ResamplingLoss(x_old, y_old, n_points)
    result = scipy.optimize.minimize(loss_func, x_init, bounds=bounds)
    x_new = np.concatenate([np.array([x_min]), np.sort(result.x), np.array([x_max])])
    y_new = np.interp(x_new, x_old, y_old)
    y_interp = np.interp(x_old, x_new, y_new)
    y_error = np.trapz(np.square(y_old - y_interp), x_old) / np.trapz(np.square(y_old), x_old)

    return n_points, y_error, (x_new, y_new)


def interpolate_response(
        x_resp: np.ndarray,
        y_resp: np.ndarray,
        x_grid: np.ndarray,
        ) -> np.ndarray:
    """Interpolate a response function.

    Parameters
    ----------
    x_resp : np.ndarray
        Original time grid.
    y_resp : np.ndarray
        Values in the original grid.
    x_grid : np.ndarray
        New time grid.

    Returns
    -------
    np.ndarray
        Values on the new grid.
    """
    # Do we have sufficient points to interpolate?
    if len(x_resp) < 2:
        return np.ones_like(x_grid) * y_resp.item()
    # Filter out points with the same x coordinate (to prevent issues during interpolation)
    mask = np.append(np.abs(np.diff(x_resp)) > 1e-16, np.array([True]), axis=0)
    x_resp = x_resp[mask]
    y_resp = y_resp[mask]
    # Re-check the number of points
    if len(x_resp) < 2:
        return np.ones_like(x_grid) * y_resp.item()
    # Ensure the grid is sorted
    idx = np.argsort(x_resp)
    x_resp = x_resp[idx]
    y_resp = y_resp[idx]
    # Interpolate
    return np.interp(
        x_grid,
        x_resp,
        y_resp,
    )
