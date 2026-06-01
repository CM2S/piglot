"""Module for simple objective functions based on individual objectives."""
from typing import Any, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
from piglot.parameter import ParameterValues
from piglot.settings import Settings
from piglot.objective import (
    Objective,
    Scalarisation,
    ObjectiveResult,
    IndividualObjective,
    IndividualObjectiveResult,
)
from piglot.utils.scalarisations import read_scalarisation


IndividualT = TypeVar('IndividualT', bound='SimpleIndividualObjective')
SimpleObjT = TypeVar('SimpleObjT', bound='SimpleObjective')


class SimpleIndividualObjective(IndividualObjective, ABC):
    """Self-contained individual objective function."""

    @abstractmethod
    def evaluate(self, params: ParameterValues, concurrent: bool) -> IndividualObjectiveResult:
        """Evaluate the objective for a set of parameters.

        Parameters
        ----------
        params : ParameterValues
            Named set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others.

        Returns
        -------
        IndividualObjectiveResult
            Objective value.
        """

    @classmethod
    @abstractmethod
    def read(
        cls: type[IndividualT], name: str, config: dict[str, Any], settings: Settings
    ) -> IndividualT:
        """Read the individual objective from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of this objective.
        config : dict[str, Any]
            Configuration dictionary.
        settings : Settings
            Settings for this problem.

        Returns
        -------
        IndividualT
            Individual objective function.
        """


class SimpleObjective(Objective, Generic[IndividualT], ABC):
    """Objective function based on simple individual objectives."""

    def __init__(
        self,
        settings: Settings,
        objectives: list[IndividualT],
        scalarisation: Optional[Scalarisation] = None,
    ) -> None:
        super().__init__(settings, objectives, scalarisation=scalarisation)
        # Update type hints for the individual objectives
        self.objectives: list[IndividualT]

    def _objective(
        self, params: ParameterValues, concurrent: bool = False
    ) -> list[IndividualObjectiveResult]:
        """Abstract method for objective computation.

        Parameters
        ----------
        params : ParameterValues
            Named set of parameters to evaluate the objective for.
        concurrent : bool, optional
            Whether this call may be concurrent to others, by default False.

        Returns
        -------
        list[IndividualObjectiveResult]
            List of individual objective results.
        """
        return [obj.evaluate(params, concurrent) for obj in self.objectives]

    @classmethod
    def read(
        cls: type[SimpleObjT], config: dict[str, Any], settings: Settings
    ) -> SimpleObjT:
        """Read the objective from the configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary.
        settings : Settings
            Settings for this problem.

        Returns
        -------
        SimpleObjT
            Objective function to optimise for.
        """
        obj_cls: type[IndividualT] = cls.individual_objective_type()

        # Check for the type of the objective
        if 'objectives' in config:
            # We have multiple objectives, so read them as a list and also parse scalarisations
            objectives = [
                obj_cls.read(name, obj_config, settings)
                for name, obj_config in config['objectives'].items()
            ]
            return cls(
                settings,
                objectives=objectives,
                scalarisation=(
                    read_scalarisation(config['scalarisation'], objectives)
                    if 'scalarisation' in config else None
                ),
            )

        # We have a single objective: read it directly and discard any scalarisation
        name = config.get('name', 'Objective')
        return cls(settings, objectives=[obj_cls.read(name, config, settings)], scalarisation=None)

    @classmethod
    @abstractmethod
    def individual_objective_type(cls) -> type[IndividualT]:
        """Return the type of individual objective this objective is based on.

        Returns
        -------
        type[IndividualT]
            Type of individual objective this objective is based on.
        """
