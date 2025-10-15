"""Module for root-finding algorithms."""
from typing import Callable
import numpy as np


class PiecewiseDistribution:
    """Class representing a piecewise distribution for 1D stochastic root-finding."""

    def __init__(self, points: np.ndarray = None, values: np.ndarray = None) -> None:
        if (points is None) != (values is None):
            raise ValueError("Both points and values must be provided together.")
        self.points = np.array([0., 1.]) if points is None else points
        self.values = np.array([1.]) if values is None else values
        if self.points.ndim != 1:
            raise ValueError("Points must be a 1D array.")
        if self.values.ndim != 1:
            raise ValueError("Values must be a 1D array.")
        if self.points.shape[0] != self.values.shape[0] + 1:
            raise ValueError(
                "Invalid number of points and values. Must have len(points) = len(values) + 1."
            )
        self.values /= np.sum(np.diff(self.points) * self.values)

    def pdf(self, x: float) -> float:
        """Evaluate the probability density function at a given point."""
        if x < self.points[0] or x > self.points[-1]:
            return 0.0
        # Find the interval containing x
        idx = np.searchsorted(self.points, x) - 1
        return self.values[idx] if 0 <= idx < len(self.values) else 0.0

    def cdf(self, x: float) -> float:
        """Evaluate the cumulative distribution function at a given point."""
        if x <= self.points[0]:
            return 0.0
        if x >= self.points[-1]:
            return 1.0
        # Compute the area of each segment
        areas = np.cumsum(np.diff(self.points) * self.values)
        areas = np.insert(areas, 0, 0)  # Insert 0 at the beginning
        # Find the interval containing x and linearly interpolate the position
        idx = np.searchsorted(areas, x) - 1
        return np.interp(x, areas[idx:idx + 2], self.points[idx:idx + 2])

    def split_and_scale(self, x: float, left: float, right: float) -> None:
        """Split the distribution at a given point and scale the left and right segments.

        Parameters
        ----------
        x : float
            Point to split the distribution at.
        left : float
            Scale factor for the left segment.
        right : float
            Scale factor for the right segment.
        """
        if x <= self.points[0] or x >= self.points[-1]:
            raise ValueError("Split point must be within the range of points.")

        # Find the interval containing x and create a new point
        idx = np.searchsorted(self.points, x) - 1
        self.points = np.insert(self.points, idx + 1, x)
        self.values = np.insert(self.values, idx, self.values[idx])

        # Scale the left and right segments
        self.values[:idx + 1] *= left
        self.values[idx + 1:] *= right

    def plot(self) -> None:
        """Plot the piecewise distribution."""
        import matplotlib.pyplot as plt

        plt.step(self.points, np.append(self.values, self.values[-1]), where='post')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.title('Piecewise Distribution')
        plt.grid()
        plt.show()


def probabilistic_bisection(
    func: Callable[[float], float],
    max_iter: int = 100,
    prob_c: float = 0.6,
    prior: PiecewiseDistribution = None,
    quantile: float = 0.5,
) -> float:
    """Perform probabilistic bisection to find a root of a function.

    Parameters
    ----------
    func : Callable[[float], float]
        Function for which to find the root. Domain should be the same as the prior, and should
        return positive values on the left-hand side of the root and negative values on the
        right-hand side.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    prob_c : float, optional
        Probability of observing correct sign (must be > 0.5 and <= 1), by default 0.6.
    prior : PiecewiseDistribution, optional
        Prior distribution representing the initial belief about the root's location.
        If None, a uniform distribution over [0, 1] is used.
    quantile : float, optional
        Quantile to evaluate at each call, by default 0.5.

    Returns
    -------
    float
        Estimated location of the root.
    """
    if prob_c <= 0.5 or prob_c > 1.0:
        raise ValueError("prob_c must be in the interval ]0.5, 1].")
    prob_q = 1 - prob_c
    distribution = PiecewiseDistribution() if prior is None else prior
    for _ in range(max_iter):
        # Compute the median of the current distribution
        median = distribution.cdf(quantile)
        f_median = func(median)

        # Update the distribution based on the sign of the function at the median
        if f_median > 0:
            distribution.split_and_scale(median, left=2 * prob_c, right=2 * prob_q)
        else:
            distribution.split_and_scale(median, left=2 * prob_q, right=2 * prob_c)

    # Return the median as the best estimate after max_iter
    return distribution.cdf(quantile)


def probabilistic_minimum_search(
    func: Callable[[float], float],
    max_iter: int = 100,
    prob_c: float = 0.55,
    prior: PiecewiseDistribution = None,
    quantile: float = 0.5,
) -> float:
    """Perform probabilistic search to find a minimum of a function.

    Parameters
    ----------
    func : Callable[[float], float]
        Function for which to find the root. Domain should be the same as the prior, and should
        return positive values on the left-hand side of the root and negative values on the
        right-hand side.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    prob_c : float, optional
        Probability of observing correct sign (must be > 0.5 and <= 1), by default 0.6.
    prior : PiecewiseDistribution, optional
        Prior distribution representing the initial belief about the root's location.
        If None, a uniform distribution over [0, 1] is used.
    quantile : float, optional
        Quantile to evaluate at each call, by default 0.5.

    Returns
    -------
    float
        Estimated location of the root.
    """
    if prob_c <= 0.5 or prob_c > 1.0:
        raise ValueError("prob_c must be in the interval ]0.5, 1].")
    distribution = PiecewiseDistribution() if prior is None else prior
    for _ in range(max_iter):
        # Compute the median of the current distribution
        median = distribution.cdf(quantile)
        f_median = func(median)

        # Update the distribution based on the sign of the function at the median
        if f_median > 0:
            distribution.split_and_scale(median, left=2 * (1 - prob_c), right=2 * prob_c)
        else:
            distribution.split_and_scale(median, left=2.0, right=0.0)

    # Return the median as the best estimate after max_iter
    return distribution.cdf(quantile)
