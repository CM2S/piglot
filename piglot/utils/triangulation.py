"""Module for robust triangulation utilities."""
from typing import List
import math
import numpy as np
from scipy.spatial import Delaunay, KDTree  # pylint: disable=E0611


class SimplexNotFoundError(Exception):
    """Exception raised when a simplex containing a point is not found."""


class Triangulation:
    """Robust triangulation class for arbitrary points in an n-dimensional space."""

    def __init__(self, points: np.ndarray, tol: float = 1e-10):
        self.points = points.copy()
        if points.shape[0] < points.shape[1] + 1:
            raise ValueError('At least n_dim+1 points are required to form a simplex.')

        # Sanitise the input points using PCA:
        # In an n-dimensional space, if all the points are coplanar, the triangulation will fail.
        # We should reduce the dimensionality of the points to m-dimensional space, with m < n.
        if points.shape[1] > 1:
            std_points = points - np.mean(points, axis=0)
            eig_vals, eig_vecs = np.linalg.eigh(np.cov(std_points.T))
            eig_vals /= np.sum(eig_vals)
            self.transformed_points = std_points @ eig_vecs[:, eig_vals > tol]
        else:
            self.transformed_points = points.copy()

        # With the transformed points, we can compute the triangulation
        # Note that the transformation is a rotation, so the simplices will be the same
        # (although the dimensionality may be reduced)
        # Desambiguate whether the points are 1D or higher-dimensional
        if self.transformed_points.shape[1] == 1:
            sorted_idx = np.argsort(self.transformed_points[:, 0])
            simplices = np.array([sorted_idx[:-1], sorted_idx[1:]]).T
        else:
            simplices = Delaunay(self.transformed_points, qhull_options="Qc").simplices

        # Sanitise the triangulation: remove simplices with zero volume
        volumes = np.array([
            self.__simplex_volume(self.transformed_points[simplex])
            for simplex in simplices
        ])
        simplex_mask = (volumes / np.mean(volumes)) > tol
        self.simplices = simplices[simplex_mask, :]

        # Neighbour list (lazily initialised)
        self._neighbors = None

        # K-dtree for fast point queries (lazily initialised)
        self._kdtree = None

    def __build_neighbours(self) -> None:
        """Build the neighbour relationships between points in the triangulation."""
        neighbors = [set() for _ in range(self.points.shape[0])]
        for simplex in self.simplices:
            for i, idx in enumerate(simplex):
                for j in range(i + 1, len(simplex)):
                    neighbors[idx].add(simplex[j])
                    neighbors[simplex[j]].add(idx)
        self._neighbors = [list(a) for a in neighbors]

    def point_neighbours(self, point_idx: int, depth: int = 1) -> List[int]:
        """Get the point neighbours of a point.

        Parameters
        ----------
        point_idx : int
            Index of the point.
        depth : int, optional
            Depth of the neighbours to return (default is 1, meaning direct neighbours).

        Returns
        -------
        List[int]
            List of indices of the neighbouring points.
        """
        if self._neighbors is None:
            self.__build_neighbours()

        if depth <= 1:
            return self._neighbors[point_idx]

        # Recursive search for neighbours up to the specified depth
        neighbours = set(self._neighbors[point_idx])
        for neighbour in self._neighbors[point_idx]:
            neighbours.update(self.point_neighbours(neighbour, depth=depth - 1))
        return list(neighbours)

    def vertex_to_simplices(self, vertex_idx: int) -> List[int]:
        """Get the simplices that contain a given vertex.

        Parameters
        ----------
        vertex_idx : int
            Index of the vertex.

        Returns
        -------
        List[int]
            List of indices of the simplices that contain the vertex.
        """
        return np.where(self.simplices == vertex_idx)[0].tolist()

    @staticmethod
    def __simplex_volume(vertices: np.ndarray) -> float:
        """Calculate the volume of a simplex given its vertices.

        Parameters
        ----------
        vertices : np.ndarray
            Vertices of the simplex (n_dim + 1 x n_dim).

        Returns
        -------
        float
            Volume of the simplex.
        """
        vertices_matrix = vertices[1:, :] - vertices[0, :]
        return np.abs(np.linalg.det(vertices_matrix.T)) / math.factorial(vertices.shape[1])

    def simplex_volume(self, simplex_idx: int) -> float:
        """Calculate the volume of a simplex.

        Parameters
        ----------
        simplex_idx : int
            Index of the simplex.

        Returns
        -------
        float
            Volume of the simplex.
        """
        idx = self.simplices[simplex_idx]
        vertices = self.transformed_points[idx, :]
        return self.__simplex_volume(vertices)

    def find_simplex(self, point: np.ndarray) -> int:
        """Find the simplex that contains a given point.

        Parameters
        ----------
        point : np.ndarray
            Point to find the simplex for (n_dim,).

        Returns
        -------
        int
            Index of the simplex that contains the point.
        """
        # Lazy initialisation of the k-d tree
        if self._kdtree is None:
            self._kdtree = KDTree(self.points)

        # Search the nearest point and associated simplices
        _, point_idx = self._kdtree.query(point)
        for simplex_idx in self.vertex_to_simplices(point_idx):
            simplex_points = self.points[self.simplices[simplex_idx], :]
            if self.point_inside_simplex(simplex_points, point):
                return simplex_idx

        # If the simplices around the closest point do not contain the point,
        # we can try to find the simplex by checking the neighbours recursively
        depth = 1
        searched_points = [point_idx]
        while len(searched_points) < self.points.shape[0]:
            new_neighbours = [
                neighbour for neighbour in self.point_neighbours(point_idx, depth)
                if neighbour not in searched_points
            ]
            if len(new_neighbours) == 0:
                break
            for neighbour in new_neighbours:
                for simplex_idx in self.vertex_to_simplices(neighbour):
                    simplex_points = self.points[self.simplices[simplex_idx], :]
                    if self.point_inside_simplex(simplex_points, point):
                        return simplex_idx
                searched_points.append(neighbour)
            depth += 1
        raise SimplexNotFoundError(f'Point {point} is not inside any simplex.')

    def interpolation_matrix(self, query_points: np.ndarray) -> np.ndarray:
        """Calculate the interpolation matrix for a set of query points.

        Parameters
        ----------
        query_points : np.ndarray
            Points to calculate the interpolation matrix for (n_query x n_dim).

        Returns
        -------
        np.ndarray
            Interpolation matrix (n_query x n_ref).
        """
        matrix = np.zeros((query_points.shape[0], self.points.shape[0]))
        for i, point in enumerate(query_points):
            simplex_idx = self.find_simplex(point)
            lambdas = self.barycentric_coordinates(
                self.points[self.simplices[simplex_idx], :],
                point.reshape(1, -1),
            )
            matrix[i, self.simplices[simplex_idx]] = lambdas.flatten()
        return matrix

    @staticmethod
    def barycentric_coordinates(
        simplex_points: np.ndarray,
        query_points: np.ndarray,
    ) -> np.ndarray:
        """Calculate barycentric coordinates for a set of points with respect to a simplex.

        Parameters
        ----------
        simplex_points : np.ndarray
            Points of the simplex (n_dim + 1 x n_dim).
        query_points : np.ndarray
            Points to calculate barycentric coordinates for (n_query x n_dim).

        Returns
        -------
        np.ndarray
            Barycentric coordinates for the query points (n_query x n_dim + 1).
        """
        # Edge matrix and distances from the first vertex
        n_dim = simplex_points.shape[1]
        n_query = query_points.shape[0]
        edge_matrix = (simplex_points[1:, :] - simplex_points[0, :]).T
        distances = (query_points - simplex_points[0, :]).T

        # Barycentric coordinates
        alpha = np.linalg.solve(edge_matrix, distances)
        lambdas = np.empty((n_dim + 1, n_query))
        lambdas[1:, :] = alpha
        lambdas[0, :] = 1 - np.sum(alpha, axis=0)
        return lambdas

    @staticmethod
    def interpolate(
        simplex_points: np.ndarray,
        simplex_values: np.ndarray,
        query_points: np.ndarray,
    ) -> np.ndarray:
        """Interpolate values for new points based on the coordinates and values of a simplex.

        Parameters
        ----------
        simplex_points : np.ndarray
            Original points (n_dim + 1 x n_dim).
        simplex_values : np.ndarray
            Values at the original points (n_dim + 1 x n_values).
        query_points : np.ndarray
            Points to interpolate values for (n_query x n_dim).

        Returns
        -------
        np.ndarray
            Interpolated values for the new points (n_query x n_values).
        """
        lambdas = Triangulation.barycentric_coordinates(simplex_points, query_points)
        return lambdas.T @ simplex_values

    @staticmethod
    def point_inside_simplex(
        simplex_points: np.ndarray,
        query_point: np.ndarray,
    ) -> bool:
        """Check if a point is inside a simplex.

        Parameters
        ----------
        simplex_points : np.ndarray
            Points of the simplex (n_dim + 1 x n_dim).
        query_point : np.ndarray
            Point to check (n_dim,).

        Returns
        -------
        bool
            True if the point is inside the simplex, False otherwise.
        """
        lambdas = Triangulation.barycentric_coordinates(simplex_points, query_point.reshape(1, -1))
        return np.all(lambdas >= 0)
