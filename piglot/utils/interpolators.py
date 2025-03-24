"""Module for response interpolators."""
from typing import Any, Dict, Union, Type, TypeVar
from abc import ABC, abstractmethod
import numpy as np
import scipy.spatial
from scipy.spatial import KDTree
from scipy.interpolate import (
    LinearNDInterpolator,
    make_interp_spline,
    NearestNDInterpolator,
    RBFInterpolator as ScipyRBFInterpolator,
)


T = TypeVar('T')


class Interpolator(ABC):
    """Base class for response interpolators."""

    @abstractmethod
    def __call__(self, x_new: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Interpolate the response at the given points.

        Parameters
        ----------
        x_new : np.ndarray
            Points to interpolate at.
        x_ref : np.ndarray
            Reference points.
        y_ref : np.ndarray
            Reference values.

        Returns
        -------
        np.ndarray
            Interpolated values.
        """

    @classmethod
    @abstractmethod
    def read(cls: Type[T], config: Dict[str, Any]) -> T:
        """Read the interpolator from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.

        Returns
        -------
        Interpolator
            Interpolator instance.
        """


class OneDimensionalInterpolator(Interpolator):
    """Base class for one-dimensional interpolators."""


class SplineInterpolator(OneDimensionalInterpolator):
    """B-spline-based one-dimensional interpolator.

    The values outside the domain are controlled via the `left` and `right` parameters and the
    `extrapolate` flag. When extrapolate is set, the sides whose value is not None will be
    extrapolated. When it is not set, `left` is set to the value at the leftmost point and `right`
    is set to the value at the rightmost point.
    """

    def __init__(
        self,
        k: int = 3,
        left: float = None,
        right: float = None,
        extrapolate: bool = True,
    ) -> None:
        self.k = k
        self.left = left
        self.right = right
        self.extrapolate = extrapolate

    def __call__(self, x_new: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Interpolate the response at the given points.

        Parameters
        ----------
        x_new : np.ndarray
            Points to interpolate at: (n_points x 1) or (n_points).
        x_ref : np.ndarray
            Reference points: (n_ref x 1) or (n_ref).
        y_ref : np.ndarray
            Reference values: (n_ref x n_values) or (n_ref).

        Returns
        -------
        np.ndarray
            Interpolated values (n_points x n_values) or (n_points).
        """
        # Sanity checks on the input data
        if any(len(a.shape) > 2 for a in (x_new, x_ref, y_ref)):
            raise ValueError('Invalid data shape for interpolation.')
        if x_ref.shape[0] != y_ref.shape[0]:
            raise ValueError('Incompatible dimensions for interpolation.')

        # If needed, convert 2D arrays to 1D
        if len(x_new.shape) == 2:
            if x_new.shape[1] != 1:
                raise ValueError('Invalid data shape for 1D interpolation.')
            x_new = x_new[:, 0]
        if len(x_ref.shape) == 2:
            if x_ref.shape[1] != 1:
                raise ValueError('Invalid data shape for 1D interpolation.')
            x_ref = x_ref[:, 0]
        # ... and the inverse for y_ref
        vector_y = len(y_ref.shape) == 1
        if vector_y:
            y_ref = y_ref[:, None]

        # Do we have points?
        if len(x_ref) == 0:
            return np.zeros((x_new.shape[0], y_ref.shape[1]))

        # Sort grid and remove duplicates
        idx = np.argsort(x_ref)
        x_ref = x_ref[idx]
        y_ref = y_ref[idx, :]
        mask = np.insert(np.diff(x_ref) > 0, 0, True)
        x_ref = x_ref[mask]
        y_ref = y_ref[mask, :]

        # Do we still have enough points?
        if len(x_ref) == 0:
            return np.zeros((x_new.shape[0], y_ref.shape[1]))

        # Tricky case of a single point: fill the left and right sides
        if len(x_ref) == 1:
            left = y_ref[0, :] if self.left is None else self.left
            right = y_ref[0, :] if self.right is None else self.right
            return np.where(x_new[:, None] < x_ref[0], left, right)

        # 1-dimensional interpolation
        interpolator = make_interp_spline(x_ref, y_ref, k=1)
        values = interpolator(x_new)

        # Handle points outside the reference range
        left_outside = x_new < x_ref[0]
        right_outside = x_new > x_ref[-1]
        if self.extrapolate:
            if self.left is not None:
                values[left_outside, :] = self.left
            if self.right is not None:
                values[right_outside, :] = self.right
        else:
            values[left_outside, :] = y_ref[0, :] if self.left is None else self.left
            values[right_outside, :] = y_ref[-1, :] if self.right is None else self.right

        # And squeeze the result back if needed
        return values[:, 0] if vector_y else values

    @classmethod
    def read(cls: Type[T], config: Dict[str, Any]) -> T:
        """Read the interpolator from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.

        Returns
        -------
        Interpolator
            Interpolator instance.
        """
        return cls(
            int(config.get('k', 3)),
            config.get('left', None),
            config.get('right', None),
            bool(config.get('extrapolate', True)),
        )


class LinearInterpolator(SplineInterpolator):
    """Linear one-dimensional interpolator.

    This simply wraps the spline interpolator with `k=1`.
    The values outside the domain are controlled via the `left` and `right` parameters and the
    `extrapolate` flag. When extrapolate is set, the sides whose value is not None will be
    extrapolated. When it is not set, `left` is set to the value at the leftmost point and `right`
    is set to the value at the rightmost point.
    """

    def __init__(
        self,
        left: float = None,
        right: float = None,
        extrapolate: bool = True,
    ) -> None:
        super().__init__(k=1, left=left, right=right, extrapolate=extrapolate)
        self.left = left
        self.right = right
        self.extrapolate = extrapolate

    @classmethod
    def read(cls: Type[T], config: Dict[str, Any]) -> T:
        """Read the interpolator from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.

        Returns
        -------
        Interpolator
            Interpolator instance.
        """
        return cls(
            config.get('left', None),
            config.get('right', None),
            bool(config.get('extrapolate', True)),
        )


class UnstructuredLinearInterpolator(Interpolator):
    """Linear interpolator for n-dimensional unstructured data.

    This class does not perform extrapolation. Points outside the convex hull of the reference
    points will be assigned NaN values.
    """

    def __call__(self, x_new: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Interpolate the response at the given points.

        Parameters
        ----------
        x_new : np.ndarray
            Points to interpolate at (n_points x n_dim).
        x_ref : np.ndarray
            Reference points (n_ref x n_dim).
        y_ref : np.ndarray
            Reference values (n_ref x n_values).

        Returns
        -------
        np.ndarray
            Interpolated values (n_points x n_values).
        """
        # Sanity checks on the input data
        if any(len(a.shape) != 2 for a in (x_new, x_ref, y_ref)):
            raise ValueError('Invalid data shape for interpolation. All arrays must be 2D.')
        if x_new.shape[-1] != x_ref.shape[-1] or x_ref.shape[0] != y_ref.shape[0]:
            raise ValueError('Incompatible dimensions for interpolation.')
        _, n_dim = x_new.shape

        # 1-dimensional interpolation is a special case
        if n_dim == 1:
            interpolator = LinearInterpolator(left=np.nan, right=np.nan, extrapolate=False)
            return interpolator(x_new, x_ref, y_ref)

        # n-dimensional interpolation: construct the triangulation using scipy
        try:
            interpolator = LinearNDInterpolator(x_ref, y_ref)
        except scipy.spatial.QhullError as ex:  # pylint: disable=E1101
            raise ValueError(
                'Triangulation failed for n-dimensional interpolation. This usually happens '
                'when the points are collinear or too close to each other.'
            ) from ex
        return interpolator(x_new)

    @classmethod
    def read(cls: Type[T], config: Dict[str, Any]) -> T:
        """Read the interpolator from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.

        Returns
        -------
        Interpolator
            Interpolator instance.
        """
        return cls()


class FastUnstructuredLinearInterpolator(Interpolator):
    """Linear interpolator for n-dimensional unstructured data using nearest neighbours.

    We filter the domain to a neighbourhood of the point to interpolate, and then perform a linear
    interpolation using the nearest neighbours. This should perform better when the number of
    reference points is large.

    This class does not perform extrapolation. Points outside the convex hull of the reference
    points will be assigned NaN values.
    """

    def __init__(self, n_simplices: int) -> None:
        self.n_simplices = n_simplices
        self.base_interpolator = UnstructuredLinearInterpolator()

    @staticmethod
    def _find_neighbourhood(
        x: np.ndarray,
        kdtree: KDTree,
        x_ref: np.ndarray,
        num_neighbours: int,
    ) -> np.ndarray:
        """Find the neighbourhood of a point in the reference grid.

        Parameters
        ----------
        x : np.ndarray
            Point to interpolate at (n_dim).
        kdtree : KDTree
            k-d tree for the reference points.
        x_ref : np.ndarray
            Reference points (n_ref x n_dim).
        num_neighbours : int
            Number of neighbours to consider.

        Returns
        -------
        np.ndarray
            Indices of the neighbours.
        """
        # Iteratively search for a valid neighbourhood
        _, n_dim = x_ref.shape
        while num_neighbours <= len(x_ref):
            _, idx = kdtree.query(x, k=num_neighbours)
            # Count the number of points below and above the point for each dimension
            x_below = np.sum(x_ref[idx, :] <= x, axis=0)
            x_above = np.sum(x_ref[idx, :] >= x, axis=0)
            # As an heuristic, ensure we have at least n_dim points below and above
            if np.all(x_below >= n_dim) and np.all(x_above >= n_dim):
                return idx
            num_neighbours *= 2
        raise ValueError('Could not find a valid neighbourhood for interpolation')

    def __call__(self, x_new: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Interpolate the response at the given points.

        Parameters
        ----------
        x_new : np.ndarray
            Points to interpolate at (n_points x n_dim).
        x_ref : np.ndarray
            Reference points (n_ref x n_dim).
        y_ref : np.ndarray
            Reference values (n_ref x n_values).

        Returns
        -------
        np.ndarray
            Interpolated values (n_points x n_values).
        """
        # Sanity checks on the input data
        if any(len(a.shape) != 2 for a in (x_new, x_ref, y_ref)):
            raise ValueError('Invalid data shape for interpolation. All arrays must be 2D.')
        if x_new.shape[-1] != x_ref.shape[-1] or x_ref.shape[0] != y_ref.shape[0]:
            raise ValueError('Incompatible dimensions for interpolation.')
        n_new_points, n_dim = x_new.shape
        n_ref_points = x_ref.shape[0]

        # Try to find a decent scaling factor to make the average distance between points more or
        # less constant among dimensions
        factor = np.mean(np.diff(np.sort(x_ref, axis=0), axis=0), axis=0)

        # Build the k-d tree for the reference points
        kdtree = KDTree(x_ref / factor)

        # Interpolate each new point
        values = np.empty((n_new_points, y_ref.shape[1]))
        for i, x in enumerate(x_new):
            # Iteratively search for a valid neighbourhood
            num_neighbours = min(self.n_simplices * (n_dim + 1), n_ref_points)
            while num_neighbours <= n_ref_points:
                idx = self._find_neighbourhood(x / factor, kdtree, x_ref / factor, num_neighbours)
                try:
                    value = self.base_interpolator(x[None, :], x_ref[idx, :], y_ref[idx, :])
                    if np.all(np.isfinite(value)):
                        break
                except ValueError:
                    pass
                num_neighbours *= 2
            else:
                raise ValueError('Could not find a valid neighbourhood for interpolation.')
            values[i, :] = value[0, :]
        return values

    @classmethod
    def read(cls: Type[T], config: Dict[str, Any]) -> T:
        """Read the interpolator from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.

        Returns
        -------
        Interpolator
            Interpolator instance.
        """
        return cls(int(config.pop('n_simplices', 32)))


class UnstructuredNearestInterpolator(Interpolator):
    """Nearest neighbour interpolator for n-dimensional unstructured data."""

    def __call__(self, x_new: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Interpolate the response at the given points.

        Parameters
        ----------
        x_new : np.ndarray
            Points to interpolate at (n_points x n_dim).
        x_ref : np.ndarray
            Reference points (n_ref x n_dim).
        y_ref : np.ndarray
            Reference values (n_ref x n_values).

        Returns
        -------
        np.ndarray
            Interpolated values (n_points x n_values).
        """
        # Sanity checks on the input data
        if any(len(a.shape) != 2 for a in (x_new, x_ref, y_ref)):
            raise ValueError('Invalid data shape for interpolation. All arrays must be 2D.')
        if x_new.shape[-1] != x_ref.shape[-1] or x_ref.shape[0] != y_ref.shape[0]:
            raise ValueError('Incompatible dimensions for interpolation.')

        # Straightforward nearest neighbour search
        return NearestNDInterpolator(x_ref, y_ref)(x_new)

    @classmethod
    def read(cls: Type[T], config: Dict[str, Any]) -> T:
        """Read the interpolator from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.

        Returns
        -------
        Interpolator
            Interpolator instance.
        """
        return cls()


class RBFInterpolator(Interpolator):
    """Radial basis function interpolator for n-dimensional unstructured data."""

    def __init__(self, neighbours: int = None, **kwargs) -> None:
        self.neighbours = neighbours
        self.options = kwargs

    def __call__(self, x_new: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Interpolate the response at the given points.

        Parameters
        ----------
        x_new : np.ndarray
            Points to interpolate at (n_points x n_dim).
        x_ref : np.ndarray
            Reference points (n_ref x n_dim).
        y_ref : np.ndarray
            Reference values (n_ref x n_values).

        Returns
        -------
        np.ndarray
            Interpolated values (n_points x n_values).
        """
        # Sanity checks on the input data
        if any(len(a.shape) != 2 for a in (x_new, x_ref, y_ref)):
            raise ValueError('Invalid data shape for interpolation. All arrays must be 2D.')
        if x_new.shape[-1] != x_ref.shape[-1] or x_ref.shape[0] != y_ref.shape[0]:
            raise ValueError('Incompatible dimensions for interpolation.')
        n_points, n_dim = x_ref.shape

        # Heuristic for the number of neighbours
        if self.neighbours is None:
            neighbours = min(64 * (n_dim + 1), n_points)
        elif self.neighbours <= 0:
            neighbours = n_points
        else:
            neighbours = min(self.neighbours, n_points)

        # Try to find a decent scaling factor to make the average distance between points more or
        # less constant among dimensions
        factor = np.mean(np.diff(np.sort(x_ref, axis=0), axis=0), axis=0)

        # Straightforward RBF interpolation
        interpolator = ScipyRBFInterpolator(
            x_ref / factor,
            y_ref,
            neighbors=neighbours,
            **self.options,
        )
        return interpolator(x_new / factor)

    @classmethod
    def read(cls: Type[T], config: Dict[str, Any]) -> T:
        """Read the interpolator from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Terms from the configuration dictionary.

        Returns
        -------
        Interpolator
            Interpolator instance.
        """
        neighbours = config.pop('neighbours', None)
        return cls(neighbours, **config)


AVAILABLE_INTERPOLATORS: Dict[str, Interpolator] = {
    'spline': SplineInterpolator,
    'linear': LinearInterpolator,
    'unstructured_linear': UnstructuredLinearInterpolator,
    'fast_unstructured_linear': FastUnstructuredLinearInterpolator,
    'unstructured_nearest': UnstructuredNearestInterpolator,
    'rbf': RBFInterpolator,
}


def read_interpolator(config: Union[str, Dict[str, Any]]) -> Interpolator:
    """Read an interpolator from a configuration dictionary.

    Parameters
    ----------
    config : Union[str, Dict[str, Any]]
        Terms from the configuration dictionary.

    Returns
    -------
    Interpolator
        Interpolator instance.
    """
    # Simple specification
    if isinstance(config, str):
        name = config
        if name not in AVAILABLE_INTERPOLATORS:
            raise ValueError(f'Interpolator "{name}" is not available.')
        return AVAILABLE_INTERPOLATORS[name].read({})
    # Detailed specification
    if 'name' not in config:
        raise ValueError('Need to pass the name of the interpolator.')
    name = config.pop('name')
    if name not in AVAILABLE_INTERPOLATORS:
        raise ValueError(f'Interpolator "{name}" is not available.')
    return AVAILABLE_INTERPOLATORS[name].read(config)
