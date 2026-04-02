"""Wrapper for scipy optimisers in piglot."""
from typing import Callable, Any, Literal, Optional, TypeVar
import warnings
import numpy as np
from scipy.optimize import minimize, differential_evolution, direct
from piglot.settings import Settings
from piglot.objective import Objective
from piglot.optimiser import SimpleOptimiser
from piglot.utils.readable import ReadableModel


T = TypeVar("T", bound="ScipyOptimiser")


DifferentialEvolutionStrategies = Literal[
    "best1bin",
    "best1exp",
    "rand1bin",
    "rand1exp",
    "rand2bin",
    "rand2exp",
    "randtobest1bin",
    "randtobest1exp",
    "currenttobest1bin",
    "currenttobest1exp",
    "best2exp",
    "best2bin",
]


class DifferentialEvolutionOptions(ReadableModel):
    """Options for the differential evolution optimiser."""
    strategy: DifferentialEvolutionStrategies = 'best1bin'
    popsize: int = 15
    tol: float = 0.01
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    polish: bool = True
    init: Literal['latinhypercube', 'sobol', 'halton', 'random'] = 'latinhypercube'
    atol: float = 0.0
    updating: Literal['immediate', 'deferred'] = 'immediate'


class DIRECTOptions(ReadableModel):
    """Options for the DIRECT optimiser."""
    eps: float = 1e-4
    maxfun: Optional[int] = None
    locally_biased: bool = True
    f_min: float = -np.inf
    f_min_rtol: float = 1e-4
    vol_tol: float = 1e-16
    len_tol: float = 1e-4


class ScipyOptimiser(SimpleOptimiser):
    """Wrapper for scipy optimisers in piglot."""

    AVAILABLE_OPTIMIZE_METHODS: list[str] = [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]

    FINITE_DIFFERENCE_METHODS: list[str] = [
        "CG",
        "BFGS",
        "L-BFGS-B",
        'trust-constr',
    ]

    GRADIENT_METHODS: list[str] = [
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]

    AVAILABLE_METHODS: list[str] = [
        *AVAILABLE_OPTIMIZE_METHODS,
        "differential_evolution",
        "direct",
    ]

    def __init__(
        self,
        settings: Settings,
        objective: Objective,
        method: str,
        diff_evo_options: DifferentialEvolutionOptions = DifferentialEvolutionOptions(),
        direct_options: DIRECTOptions = DIRECTOptions(),
    ) -> None:
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Unknown optimisation method: {method}. "
                f"Available methods: {', '.join(self.AVAILABLE_METHODS)}."
            )
        if method in self.GRADIENT_METHODS:
            raise ValueError(f"The selected optimisation method '{method}' requires gradients.")
        if method in self.FINITE_DIFFERENCE_METHODS:
            warnings.warn(
                f"The selected optimisation method '{method}' requires gradients, which will be "
                "computed with finite differences."
            )
        super().__init__(settings, objective, normalise_params=False)
        self.method = method
        self.diff_evo_options = diff_evo_options
        self.direct_options = direct_options

    def name(self) -> str:
        """Name of the optimiser.

        Returns
        -------
        str
            Name of the optimiser.
        """
        return f"scipy ({self.method})"

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
        if self.method in self.AVAILABLE_OPTIMIZE_METHODS:
            minimize(
                objective, initial_guess, method=self.method, bounds=bounds, callback=callback
            )
        elif self.method == "differential_evolution":
            differential_evolution(
                objective,
                bounds=bounds,
                x0=initial_guess,
                callback=lambda *args, **kwargs: callback(),
                seed=self.settings.seed,
                **self.diff_evo_options.__dict__
            )
        elif self.method == "direct":
            direct(
                objective,
                bounds=bounds,
                maxiter=num_iters,
                callback=lambda *args, **kwargs: callback(),
                **self.direct_options.__dict__
            )

    @classmethod
    def read(cls: type[T], config: dict[str, Any], settings: Settings, objective: Objective) -> T:
        """Read an optimiser from the given configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the optimiser.
        settings : Settings
            Settings for the optimiser.
        objective : Objective
            Objective to optimise.

        Returns
        -------
        T
            The created optimiser instance.
        """
        if "method" not in config:
            raise ValueError("Missing 'method' key in the configuration of the scipy optimiser.")
        method = config.pop("method")
        options = {}
        if method == "differential_evolution":
            options['diff_evo_options'] = DifferentialEvolutionOptions.read(config)
        elif method == "direct":
            options['direct_options'] = DIRECTOptions.read(config)
        return cls(settings, objective, method, **options)
