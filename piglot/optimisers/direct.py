"""DIRECT optimiser module."""
from typing import Tuple, Callable, Optional
import copy
import numpy as np
from piglot.objective import Objective
from piglot.optimiser import ScalarOptimiser


class Rectangle:
    """Rectangle class for using with DIRECT.

    Methods
    -------
    diagonal(self):
        Returns the distance between the center and the furthest vertex.
    """
    def __init__(self, size, center, func_val):
        """Constructor for the Rectangle class.

        Parameters
        ----------
        size : array
            Dimensions of the rectangle.
        center : array
            Coordinates of the centre of the rectable.
        func_val : float
            Function value at the centre of the rectangle.
        """
        self.size = size
        self.center = center
        self.func_val = func_val

    def diagonal(self):
        """Returns the distance between the center and the furthest vertex.

        Returns
        -------
        float
            Distance between the center and the furthest vertex
        """
        return np.linalg.norm(self.size / 2)


class DIRECT(ScalarOptimiser):
    """
    DIRECT method for optimisation.

    Reference:
    https://doi.org/10.1007/BF00941892

    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(self, objective: Objective, epsilon=0):
        """Constructs all necessary attributes for the DIRECT optimiser.

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
        epsilon : float, optional
            Model parameter, refer to documentation, by default 0.
        """
        super().__init__('DIRECT', objective)
        self.epsilon = epsilon
        self.K = 0

    def __divide_rectangle(self, n_dim, rectangles, j, func):
        """Method for rectangle division.

        Parameters
        ----------
        n_dim : integer
            Number of dimensions of the hyperrectangle.
        rectangles : array
            Array of current rectangles. This will be modified.
        j : integer
            Index of the rectangle to subdivide
        func : callable
            Function to call on new rectangle centres.

        Returns
        -------
        best_point : array
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
        def slope(d1, d2):
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
            self.K = max(self.K, max([slope((p1, fp1), (r.center, r.func_val))
                                      for r in rectangles]))
            self.K = max(self.K, max([slope((p2, fp2), (r.center, r.func_val))
                                      for r in rectangles]))
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

    def __potential_optimisers(self, rectangles, best_value):
        """Builds the set of potential optimisers.

        Parameters
        ----------
        rectangles : array
            Array of rectangles.
        best_value : float
            Current best value.

        Returns
        -------
        array
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
        def slope_between(x, y):
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

    def _scalar_optimise(
        self,
        objective: Callable[[np.ndarray, Optional[bool]], float],
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Abstract method for optimising the objective.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to optimise.
        n_dim : int
            Number of parameters to optimise.
        n_iter : int
            Maximum number of iterations.
        bound : np.ndarray
            Array where first and second columns correspond to lower and upper bounds, respectively.
        init_shot : np.ndarray
            Initial shot for the optimisation problem.

        Returns
        -------
        float
            Best observed objective value.
        np.ndarray
            Observed optimum of the objective.
        """
        # Initialise starting cube
        self.K = 0
        center = (bound[:, 1] + bound[:, 0]) / 2
        cube_size = bound[:, 1] - bound[:, 0]
        best_value = objective(center)
        rectangles = [Rectangle(cube_size, center, best_value)]

        # Check if this solution is converged
        if self._progress_check(0, best_value, center):
            return center, best_value

        # Iterations loop
        for i in range(0, n_iter):
            # Select potentially optimal rectangles
            potentially_optimal = self.__potential_optimisers(rectangles, best_value)

            # Subdivide each potentialy optimal rectangle
            iter_value = np.inf
            iter_solution = None
            for j in potentially_optimal:
                new_solution, new_value = self.__divide_rectangle(n_dim, rectangles, j, objective)
                best_value = min(best_value, new_value)
                if new_value < iter_value:
                    iter_value = new_value
                    iter_solution = new_solution

            # Update progress and check convergence
            if self._progress_check(i+1, iter_value, iter_solution):
                break

        # Return best value
        i_best = np.argmin([r.func_val for r in rectangles])
        return rectangles[i_best].center, rectangles[i_best].func_val
