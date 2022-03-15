"""Benchmark test functions module."""
import math
from abc import ABC, abstractmethod
import numpy as np
from piglot.parameter import Parameter


class BenchmarkLoss(ABC):
    """
    Interface for implementing different test functions.

    Methods
    -------
    _init_parameters(n_dim, bounds, init_shot):
        Initialize parameters
    loss(X):
        Computes the loss value
    """
    def __init__(self):
        self.func_calls = 0

    def _init_parameters(self, n_dim, bounds, init_shot):
        """
        Initiates parameters

        Parameters
        ----------
        n_dim : integer
            number of parameters
        bounds : array
            lower and upper bounds, first column lower, second upper
        init_shot : array
            initial shot for the parameters

        Returns
        -------
        List of parameters defined by the class Parameter()
        """
        self.parameters = [Parameter(str(i), init_shot[i], bounds[i][0], bounds[i][1])
                                                                      for i in range(n_dim)]

    @abstractmethod
    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """


class SixHumpCamelbackFunction(BenchmarkLoss):
    """The function has six local minima, two of which are global. (Many local minima)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the SixHumpCamelback function
        """
        super(SixHumpCamelbackFunction, self).__init__()
        self._init_parameters(2, [[-3, 3], [-2, 2]], [-2.9, -1.9])
        self.fglobal = -1.0316
        self.xglobal = [[-0.0898, 0.7126], [0.0898, -0.7126]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
        return func


class TwoDimensionalShubertFunction(BenchmarkLoss):
    """The Shubert function has several local minima and many global minima.
    (Many local minima)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the TwoDimensionalShubert function
        """
        super(TwoDimensionalShubertFunction, self).__init__()
        self._init_parameters(2, [[-10, 10], [-10, 10]], [-7, -7])
        self.fglobal = -186.7309
        self.xglobal = [[5.4826, -1.42513], [-0.80032, 4.85805], [4.85805, -0.80032],
                        [-7.08350, -7.70831], [-0.80032, -7.70831], [5.48286, -7.70831],
                        [-7.70831, -7.708350], [-1.42513, -7.08350], [4.85805, -7.08350],
                        [-7.08350, -1.4251], [-0.80032, -1.42513], [-7.70831, -0.80032],
                        [-1.42513, -0.80032], [-7.08350, 4.85805], [5.48286, 4.85805],
                        [-7.70831, 5.48286], [-1.42513, 5.48286], [4.85805, 5.48286]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        t1 = sum([i* np.cos((i+1) * x1 + i) for i in range(1, 6)])
        t2 = sum([i* np.cos((i+1) * x2 + i) for i in range(1, 6)])
        func = t1 * t2
        return func


class BraninFunction(BenchmarkLoss):
    """The Branin, or Branin-Hoo, function has three global minima."""

    def __init__(self, a=1, b=5.1/(4*(math.pi)**2), c=5/math.pi, r=6, s=10,
                 t=1/(8*(math.pi))):
        """
        Constructs all the necessary attributes for the Branin function

        Parameters
        ----------
        a, b, c, r, s, t : float
            function parameters
        """
        super(BraninFunction, self).__init__()
        self._init_parameters(2, [[-5, 10], [0, 15]], [-1, 7])
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t
        self.fglobal = 0.397887
        self.xglobal = [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = self.a*(x2-self.b*x1**2+self.c*x1-self.r)**2 \
             + self.s*(1-self.t)*np.cos(x1) + self.s
        return func


class GoldsteinPriceFunction(BenchmarkLoss):
    """The Goldstein-Price function has several local minima. """

    def __init__(self):
        """
        Constructs all the necessary attributes for the GoldsteinPriceFunction function
        """
        super(GoldsteinPriceFunction, self).__init__()
        self._init_parameters(2, [[-2, 2], [-2, 2]], [-1.7, 1.7])
        self.fglobal = 3
        self.xglobal = [[0, -1]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = (1+((x1+x2+1)**2)*(19-14*x1+3*x1**2-14*x2+6*x1*x2+3*x2**2))*(30 +
                               ((2*x1-3*x2)**2)*(18-32*x1+12*x1**2+48*x2-36*x1*x2+27*x2**2))
        return func


class SphereFunction(BenchmarkLoss):
    """The Sphere function has d local minima except for the global one. It is continuous,
    convex and unimodal. (Bowl-Shaped)"""

    def __init__(self, n_dim):
        """
        Constructs all the necessary attributes for the Sphere function

        Parameters
        ----------
        n_dim : integer
            number of unknown parameters
        """
        super(SphereFunction, self).__init__()
        self.n_dim = n_dim
        bound = [[-5.12, 5.12] for i in range(self.n_dim)]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(self.n_dim, bound, init_shot)
        self.fglobal = 0
        self.xglobal = np.zeros(self.n_dim)

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        func = 0
        self.func_calls += 1
        for i, value in enumerate(X):
            xi = self.parameters[i].denormalise(value)
            func += xi**2
        return func


class Schaffer2Function(BenchmarkLoss):
    """(Many Local Minima)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the Schaffer2 function
        """
        super(Schaffer2Function, self).__init__()
        bound = [[-100, 100], [-100, 100]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = 0.0
        self.xglobal = [[0, 0]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = 0.5 + ((np.sin(x1**2-x2**2))**2 - 0.5)/((1 + 0.001*(x1**2+x2**2))**2)
        return func


class SchwefelFunction(BenchmarkLoss):
    """The Schwefel function is complex, with many local minima. (Many Local Minima)"""

    def __init__(self, n_dim):
        """
        Constructs all the necessary attributes for the Schwefel function

        Parameters
        ----------
        n_dim : integer
            number of unknown parameters
        """
        super(SchwefelFunction, self).__init__()
        self.n_dim = n_dim
        bound = [[-500, 500] for i in range(self.n_dim)]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(self.n_dim, bound, init_shot)
        self.fglobal = 0
        self.xglobal = np.zeros(self.n_dim) + 420.9687

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        func = 418.9829*self.n_dim
        self.func_calls += 1
        for i, value in enumerate(X):
            xi = self.parameters[i].denormalise(value)
            func += -xi*np.sin(np.sqrt(abs(xi)))
        return func


class BoothFunction(BenchmarkLoss):
    """(Plate-Shaped)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the Booth function
        """
        super(BoothFunction, self).__init__()
        bound = [[-10, 10], [-10, 10]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = 0.0
        self.xglobal = [[1, 3]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = (x1+2*x2-7)**2 + (2*x1+x2-5)**2
        return func


class MatyasFunction(BenchmarkLoss):
    """The Matyas function has no local minima except the global one. (Plate-Shaped)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the Matyas function
        """
        super(MatyasFunction, self).__init__()
        bound = [[-10, 10], [-10, 10]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = 0.0
        self.xglobal = [[0, 0]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = 0.26*(x1**2+x2**2) - 0.48*x1*x2
        return func


class McCormickFunction(BenchmarkLoss):
    """(Plate-Shaped)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the McCormick function
        """
        super(McCormickFunction, self).__init__()
        bound = [[-1.5, 4], [-3, 4]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = -1.9133
        self.xglobal = [[-0.54719, -1.54719]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = np.sin(x1+x2) + (x1-x2)**2 - 1.5*x1 + 2.5*x2 + 1
        return func


class EasomFunction(BenchmarkLoss):
    """The Easom function has several local minima. It is unimodal, and the global minimum
    has a small area relative to the search space. (Steep Ridges/Drops)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the Easom function
        """
        super(EasomFunction, self).__init__()
        bound = [[-50, 50], [-50, 50]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = -1
        self.xglobal = [[math.pi, math.pi]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = -np.cos(x1)*np.cos(x2)*np.exp(-(x1-math.pi)**2-(x2-math.pi)**2)
        return func


class SomeOfDifPowFunction(BenchmarkLoss):
    """The Sum of Different Powers function is unimodal. (Bowl-Shaped)"""

    def __init__(self, n_dim):
        """
        Constructs all the necessary attributes for the SomeOfDifPow function

        Parameters
        ----------
        n_dim : integer
            number of unknown parameters
        """
        super(SomeOfDifPowFunction, self).__init__()
        self.n_dim = n_dim
        bound = [[-3, 1] for i in range(self.n_dim)]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(self.n_dim, bound, init_shot)
        self.fglobal = 0
        self.xglobal = np.zeros(self.n_dim)

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        func = 0
        self.func_calls += 1
        for i, value in enumerate(X):
            xi = self.parameters[i].denormalise(value)
            func += (abs(xi))**(i+1)
        return func


class BukinFunction(BenchmarkLoss):
    """The sixth Bukin function has many local minima, all of which lie in a ridge.
    (Many local minima)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the Bukin function
        """
        super(BukinFunction, self).__init__()
        bound = [[-15, -5], [-3, 3]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = 0
        self.xglobal = [[-10, 1]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = 100*np.sqrt(abs(x2-0.01*x1**2)) + 0.01*abs(x1+10)
        return func


class CrossInTrayFunction(BenchmarkLoss):
    """The Cross-in-Tray function has multiple global minima. (Many local minima)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the CrossInTray function
        """
        super(CrossInTrayFunction, self).__init__()
        bound = [[-10, 10], [-10, 10]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = -2.06261
        self.xglobal = [[1.3491, -1.3491], [1.3491, 1.3491],
                        [-1.3491 ,1.3491], [-1.3491, -1.3491]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = -0.0001*((abs(np.sin(x1)*np.sin(x2)*np.exp(abs(100 -
                                              (np.sqrt(x1**2+x2**2)/math.pi)))) + 1)**0.1)
        return func


class DropWaveFunction(BenchmarkLoss):
    """The Drop-Wave function is multimodal and highly complex. (Many local minima)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the DropWave function
        """
        super(DropWaveFunction, self).__init__()
        bound = [[-5.12, 5.12], [-5.12, 5.12]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = -1
        self.xglobal = [[0, 0]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = - (1+np.cos(12*np.sqrt(x1**2+x2**2))) / (0.5*(x1**2+x2**2) + 2)
        return func


class EggHolderFunction(BenchmarkLoss):
    """The Eggholder function is a difficult function to optimize, because of the large
    number of local minima. (Many local minima)"""

    def __init__(self):
        """
        Constructs all the necessary attributes for the EggHolder function
        """
        super(EggHolderFunction, self).__init__()
        bound = [[-512, 512], [-512, 512]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = -959.6407
        self.xglobal = [[512, 404.2319]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = -(x2+47)*np.sin(np.sqrt(abs(x2+(x1/2)+47)))-x1*np.sin(np.sqrt(abs(x1-(x2+47))))
        return func


class BealeFunction(BenchmarkLoss):
    """The Beale function is multimodal, with sharp peaks at the corners of the domain."""

    def __init__(self):
        """
        Constructs all the necessary attributes for the Beale function
        """
        super(BealeFunction, self).__init__()
        bound = [[-4.5, 4.5], [-4.5, 4.5]]
        init_shot = [np.random.uniform(i[0], i[1]) for i in bound]
        self._init_parameters(2, bound, init_shot)
        self.fglobal = 0
        self.xglobal = [[3, 0.5]]

    def loss(self, X):
        """
        Compute loss function value

        Parameters
        ----------
        X : array
            unknown parameters

        Returns
        -------
        loss : float
            loss function value
        """
        x1 = self.parameters[0].denormalise(X[0])
        x2 = self.parameters[1].denormalise(X[1])
        self.func_calls += 1
        func = (1.5-x1+x1*x2)**2 + (2.25-x1+x1*(x2**2))**2 + (2.625-x1+x1*(x2**3))**2
        return func
