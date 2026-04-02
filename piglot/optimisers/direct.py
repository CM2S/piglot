"""DIRECT optimiser module."""
from typing import Callable, Any
import copy
import numpy as np
from piglot.objective import Objective
from piglot.optimiser import SimpleOptimiser
from piglot.settings import Settings


class Rectangle:
    """Rectangle class for using with DIRECT.

    Methods
    -------
    diagonal(self):
        Returns the distance between the center and the furthest vertex.
    """
    def __init__(self, size: np.ndarray, center: np.ndarray, func_val: float) -> None:
        """Constructor for the Rectangle class.

        Parameters
        ----------
        size : np.ndarray
            Dimensions of the rectangle.
        center : np.ndarray
            Coordinates of the centre of the rectable.
        func_val : float
            Function value at the centre of the rectangle.
        """
        self.size = size
        self.center = center
        self.func_val = func_val

    def diagonal(self) -> float:
        """Returns the distance between the center and the furthest vertex.

        Returns
        -------
        float
            Distance between the center and the furthest vertex
        """
        return np.linalg.norm(self.size / 2)


class DIRECT(SimpleOptimiser):
    """DIRECT method for optimisation.

    Reference:
    https://doi.org/10.1007/BF00941892
    """

    def __init__(self, settings: Settings, objective: Objective, epsilon: float = 0.0) -> None:
        super().__init__(settings, objective, normalise_params=True)
        self.epsilon = epsilon
        self.K = 0

    def name(self) -> str:
        """Name of the optimiser.

        Returns
        -------
        str
            Name of the optimiser.
        """
        return "DIRECT"

    def __divide_rectangle(
        self, n_dim: int, rectangles: list[Rectangle], j: int, func: Callable[[np.ndarray], float]
    ) -> tuple[np.ndarray, float]:
        """Method for rectangle division.

        Parameters
        ----------
        n_dim : int
            Number of dimensions of the hyperrectangle.
        rectangles : list[Rectangle]
            Array of current rectangles. This will be modified.
        j : int
            Index of the rectangle to subdivide
        func : Callable[[np.ndarray], float]
            Function to call on new rectangle centres.

        Returns
        -------
        best_point : np.ndarray
            From all evaluated points in this call, returns the best solution
        best_value : float
            From all evaluated points in this call, returns the best loss
        """
        # Find dimensions with largest size
        max_size = np.max(rectangles[j].size)
        max_dims = np.flatnonzero(rectangles[j].size == max_size)
        # Build list of points to sample
        delta = max_size / 3
        new_points = []
        new_samples = []
        w_vec = []
        dir_vector = np.zeros(n_dim)

        # Slope function
        def slope(d1: tuple[np.ndarray, float], d2: tuple[np.ndarray, float]) -> float:
            return np.abs(d1[1] - d2[1]) / np.linalg.norm(d1[0] - d2[0])
        for i in max_dims:
            delta_vec = copy.deepcopy(dir_vector)
            delta_vec[i] = delta
            p1 = rectangles[j].center + delta_vec
            p2 = rectangles[j].center - delta_vec
            fp1 = func(p1)
            fp2 = func(p2)
            new_points.append((p1, p2))
            new_samples.append((fp1, fp2))
            w_vec.append(min(fp1, fp2))
            # update slope
            self.K = max(self.K, max(slope((p1, fp1), (r.center, r.func_val)) for r in rectangles))
            self.K = max(self.K, max(slope((p2, fp2), (r.center, r.func_val)) for r in rectangles))
        # Sort dimensions to subdivide
        sorted_dims = np.argsort(w_vec)
        # Subdivide each dimension
        for i in sorted_dims:
            # Size for new rectangles
            new_size = copy.deepcopy(rectangles[j].size)
            new_size[max_dims[i]] /= 3
            # Create new rectangles
            rectangles.append(Rectangle(new_size, new_points[i][0], new_samples[i][0]))
            rectangles.append(Rectangle(new_size, new_points[i][1], new_samples[i][1]))
            # Shrink existing rectangle
            rectangles[j].size = new_size
        # Return new best function call
        i_best = np.argmin(w_vec)
        return new_points[i_best][np.argmin(w_vec[i_best])], w_vec[i_best]

    def __potential_optimisers(self, rectangles: list[Rectangle], best_value: float) -> list[int]:
        """Builds the set of potential optimisers.

        Parameters
        ----------
        rectangles : list[Rectangle]
            Array of rectangles.
        best_value : float
            Current best value.

        Returns
        -------
        list[int]
            Set of potential optimiser rectangles.
        """
        # Sort rectangles firstly by size then by function value
        # (this works because Python sorting is stable)
        rectangles.sort(key=lambda x: x.func_val)
        rectangles.sort(key=lambda x: x.diagonal())

        # First pass: select only the best candidates for each distance
        candidates = []
        last_dist = None
        for i, rectangle in enumerate(rectangles):
            if rectangle.diagonal() != last_dist:
                last_dist = rectangle.diagonal()
                candidates.append(i)

        # Second pass: filter the candidates by the slope condition
        # (we add the best rectangle by default)
        candidates.sort(key=lambda x: rectangles[x].func_val)
        last_rect = candidates[0]
        potential = [last_rect]
        for j in candidates[1:]:
            # Slope condition: for increasing distances, the slope must always increase
            # between two points
            if rectangles[j].diagonal() > rectangles[last_rect].diagonal():
                # Add current point
                last_rect = j
                potential.append(j)

        # Third pass: filter points after a slope decrease
        def slope_between(x: int, y: int) -> float:
            return ((rectangles[y].func_val - rectangles[x].func_val) /
                    (rectangles[y].diagonal() - rectangles[x].diagonal()))
        slopes_bad = True
        while slopes_bad:
            slopes_bad = False
            for i, j in enumerate(potential[1:-1]):
                slope_l = slope_between(potential[i+1], potential[i])
                slope_r = slope_between(potential[i+2], potential[i+1])
                if slope_r < slope_l:
                    del potential[i+1]
                    slopes_bad = True
                    break
        # Early return if we get only 2 points: a pair is always convex
        if len(potential) < 3:
            return potential

        # Fourth pass: after convex hull is found, filter on the second condition
        final_potential = []
        n_filtered = 0
        for i, j in enumerate(potential):
            if j == potential[-1]:
                slope = slope_between(potential[i], potential[i-1])
            elif j == potential[0]:
                slope = slope_between(potential[i+1], potential[i])
            else:
                dl = rectangles[potential[i]].diagonal() - rectangles[potential[i-1]].diagonal()
                dr = rectangles[potential[i+1]].diagonal() - rectangles[potential[i]].diagonal()
                slope = (slope_between(potential[i], potential[i-1]) * dl +
                         slope_between(potential[i+1], potential[i]) * dr) / (dl + dr)
            if rectangles[j].func_val - slope * rectangles[j].diagonal() \
               <= best_value - self.epsilon * np.abs(best_value):
                n_filtered += 1
                final_potential.append(j)
        return final_potential

    def _simple_optimise(
        self,
        num_iters: int,
        initial_guess: np.ndarray,
        bounds: list[tuple[float, float]],
        objective: Callable[[np.ndarray], float],
        callback: Callable[[Any], None],
    ) -> None:
        """Optimise the objective function.

        Parameters
        ----------
        num_iters : int
            Number of iterations for the optimisation.
        initial_guess : np.ndarray
            Initial guess for the optimisation.
        bounds : list[tuple[float, float]]
            Bounds for the optimisation variables.
        objective : Callable[[np.ndarray], float]
            Objective function to be minimised.
        callback : Callable[[Any], None]
            Callback function for reporting the optimiser progress and checking for termination.
            This function is called at the end of each iteration and will raise StopIteration if
            the optimisation should be stopped. Keyword arguments are reported from the optimiser.
        """
        # Initialise starting cube
        self.K = 0
        n_dim = len(initial_guess)
        lbounds = np.array([b[0] for b in bounds])
        ubounds = np.array([b[1] for b in bounds])
        center = (ubounds + lbounds) / 2
        cube_size = ubounds - lbounds
        best_value = objective(center)
        rectangles = [Rectangle(cube_size, center, best_value)]

        # Iterations loop
        for i in range(num_iters):
            # Select and subdivide potentially optimal rectangles
            potentially_optimal = self.__potential_optimisers(rectangles, best_value)
            for j in potentially_optimal:
                self.__divide_rectangle(n_dim, rectangles, j, objective)

            # Progress report and termination check
            callback()
