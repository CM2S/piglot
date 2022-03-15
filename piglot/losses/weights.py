"""Weights distribution module."""
from abc import ABC, abstractmethod
from scipy.stats import norm
import numpy as np


class Weights(ABC):
    """
    Interface for implementing different weights distributions.

    Methods
    -------
    get_weights(x):
        get the weights value for x
    """
    @abstractmethod
    def get_weights(self, x):
        """
        Compute the weights value for x with a given distribution

        Parameters
        ----------
        x : array
            where to compute the distribution

        Returns
        -------
        weights
        """


class UniformWeights(Weights):
    """
    Uniform weights distribution.

    Attributes
    ----------
    base_val : float
        base value of the uniform distribution

    Methods
    -------
    get_weights(x):
        get the weights value for x
    """
    def __init__(self, base_val=1.0):
        """
        Constructs all the necessary attributes for the uniform distribution

        Parameters
        ----------
        base_val : float
            base value
        """
        self.base_val = base_val

    def get_weights(self, x):
        """
        Compute the weights value for x with the uniform distribution

        Parameters
        ----------
        x : array
            where to compute the distribution

        Returns
        -------
        weights
        """
        return self.base_val * np.ones_like(x)


class MultiModalWeights(Weights):
    """
    Multi normal weights distribution.

    Attributes
    ----------
    base_val : float
        base value of the uniform distribution to add to the normal distributions

    Methods
    -------
    add_mode(alpha, mu, sigma):
        add normal distribution
    get_weights(x):
        get the weights value for x
    """
    def __init__(self, base_val=1.0):
        """
        Constructs all the necessary attributes for the multi normal distribution

        Parameters
        ----------
        base_val : float
            base value
        """
        self.base_val = base_val
        self.modes = []

    def add_mode(self, alpha, mu, sigma):
        """
        Adds normal distribution center in mu with standard deviation sigma

        Parameters
        ----------
        alpha : float
            value of the normal distribution on its center
        mu : float
            center position of the normal distribution
        sigma : float
            standard deviation of the normal distribution

        Returns
        -------
        normal distribution
        """
        self.modes.append((alpha / norm.pdf(mu, loc=mu, scale=sigma), mu, sigma))

    def get_weights(self, x):
        """
        Compute the weights value for x with a multi modal distribution

        Parameters
        ----------
        x : array
            where to compute the distribution

        Returns
        -------
        weights
        """
        result = self.base_val * np.ones_like(x)
        for mode in self.modes:
            result += mode[0] * norm.pdf(x, loc=mode[1], scale=mode[2])
        return result
