"""Module with utilities for reading dataclasses from configuration dictionaries."""
from typing import TypeVar, Any, Union, Literal, Callable, Generic, _AnnotatedAlias
from abc import ABC, abstractmethod
from dataclasses import Field, MISSING
import warnings


T = TypeVar('T')


class ReadingError(ValueError):
    """Custom error for reading issues."""


class ReadableConstraint(ABC, Generic[T]):
    """Base class for type annotated constraints."""

    def __init__(self, name: str) -> None:
        """Constructor for the readable constraint.

        Parameters
        ----------
        name : str
            Name of the constraint, used for error messages.
        """
        self.name = name

    @abstractmethod
    def check(self, value: T) -> bool:
        """Check if the value satisfies the constraint.

        Parameters
        ----------
        value : T
            Value to check.

        Returns
        -------
        bool
            True if the value satisfies the constraint, False otherwise.
        """


class LambdaConstraint(ReadableConstraint, Generic[T]):
    """Constraint defined by a lambda function."""

    def __init__(self, name: str, func: Callable[[T], bool]) -> None:
        """Constructor for the lambda constraint.

        Parameters
        ----------
        name : str
            Name of the constraint, used for error messages.
        func : Callable[[T], bool]
            Lambda function defining the constraint.
        """
        super().__init__(name)
        self.func = func

    def check(self, value: T) -> bool:
        """Check if the value satisfies the constraint.

        Parameters
        ----------
        value : T
            Value to check.

        Returns
        -------
        bool
            True if the value satisfies the constraint, False otherwise.
        """
        return self.func(value)


def greater_than(value: T) -> LambdaConstraint[T]:
    """Constraint for values greater than a given value."""
    return LambdaConstraint(f"greater_than({value})", lambda x: x > value)


def greater_than_or_equal(value: T) -> LambdaConstraint[T]:
    """Constraint for values greater than or equal to a given value."""
    return LambdaConstraint(f"greater_than_or_equal({value})", lambda x: x >= value)


def less_than(value: T) -> LambdaConstraint[T]:
    """Constraint for values less than a given value."""
    return LambdaConstraint(f"less_than({value})", lambda x: x < value)


def less_than_or_equal(value: T) -> LambdaConstraint[T]:
    """Constraint for values less than or equal to a given value."""
    return LambdaConstraint(f"less_than_or_equal({value})", lambda x: x <= value)


def between(lower: T, upper: T) -> LambdaConstraint[T]:
    """Constraint for values between two given values."""
    return LambdaConstraint(f"between({lower}, {upper})", lambda x: lower < x < upper)


def between_inclusive(lower: T, upper: T) -> LambdaConstraint[T]:
    """Constraint for values between two given values, inclusive."""
    return LambdaConstraint(f"between_inclusive({lower}, {upper})", lambda x: lower <= x <= upper)


def generic_type_factory(key: str, cls: type[T], value: Any) -> Any:
    """Factory for type conversion functions for generic types.

    Parameters
    ----------
    key : str
        Key associated with the value, used for error messages.
    cls : type[T]
        Generic type to convert to.
    value : Any
        Value to convert.

    Returns
    -------
    Any
        Instance of the given generic type created from the value.
    """
    # Tuples
    if cls.__origin__ is tuple:
        # Check for ellipsis
        if cls.__args__[-1] is Ellipsis:
            if len(cls.__args__) != 2:
                raise ReadingError(
                    f"Generic type '{cls}' with '...' must have exactly two arguments for "
                    f"parameter '{key}'."
                )
            return tuple(type_factory(key, cls.__args__[0], v) for v in value)
        # Check for correct number of arguments
        if len(cls.__args__) != len(value):
            raise ReadingError(
                f"Value has wrong number of elements for generic type '{cls}' for "
                f"parameter '{key}'. Expected {len(cls.__args__)}, got {len(value)}."
            )
        return cls.__origin__(type_factory(key, arg, v) for arg, v in zip(cls.__args__, value))

    # Lists and sets
    if cls.__origin__ is list or cls.__origin__ is set:
        if len(cls.__args__) != 1:
            raise ReadingError(
                f"Generic type '{cls}' must have exactly one argument for parameter '{key}'."
            )
        return cls.__origin__(type_factory(key, cls.__args__[0], v) for v in value)

    # Dicts
    if cls.__origin__ is dict:
        if len(cls.__args__) != 2:
            raise ReadingError(
                f"Generic type '{cls}' must have exactly two arguments for parameter '{key}'."
            )
        return cls.__origin__(
            (type_factory(key, cls.__args__[0], k), type_factory(key, cls.__args__[1], v))
            for k, v in value.items()
        )

    raise ReadingError(f"Unsupported generic type '{cls}' for parameter '{key}'.")


def typing_construct_factory(key: str, cls: type[T], value: Any) -> Any:
    """Factory for type conversion functions for typing constructs.

    Parameters
    ----------
    key : str
        Key associated with the value, used for error messages.
    cls : type[T]
        Typing construct to convert to.
    value : Any
        Value to convert.

    Returns
    -------
    Any
        Instance of the given typing construct created from the value.
    """
    # Handle generics
    if hasattr(cls, '__class_getitem__') and hasattr(cls, '__args__'):
        return generic_type_factory(key, cls, value)

    # Handle annotated types
    if isinstance(cls, _AnnotatedAlias):
        converted_value = type_factory(key, cls.__args__[0], value)
        for annotation in cls.__metadata__:
            if isinstance(annotation, ReadableConstraint):
                if not annotation.check(converted_value):
                    raise ReadingError(
                        f"Value '{value}' does not satisfy constraint '{annotation.name}' "
                        f"for annotated parameter '{key}'."
                    )
        return converted_value

    # Literals
    if cls.__origin__ is Literal:
        if value not in cls.__args__:
            raise ReadingError(
                f"Value '{value}' is not a valid literal for '{cls.__args__}' in "
                f"parameter '{key}'."
            )
        return value

    # Unions
    if cls.__origin__ is Union:
        for arg in cls.__args__:
            try:
                return type_factory(key, arg, value)
            except Exception:  # pylint: disable=broad-except
                continue
        raise ReadingError(
            f"Value '{value}' cannot be converted to any of the types in '{cls}' for "
            f"parameter '{key}'."
        )

    raise ReadingError(f"Unsupported typing construct '{cls}' for parameter '{key}'.")


def type_factory(key: str, cls: type[T], value: Any) -> Any:
    """Factory for type conversion functions.

    Parameters
    ----------
    key : str
        Key associated with the value, used for error messages.
    cls : type[T]
        Type to convert to.
    value : Any
        Value to convert.

    Returns
    -------
    Any
        Instance of the given type created from the value.
    """
    # Handle missing type
    if cls is Ellipsis or cls is None or cls is Any or cls is object:
        warnings.warn(
            f"Parameter '{key}' has no type. Returning the value '{value}' as is "
            f"(type: {getattr(type(value), '__name__', str(type(value)))})."
        )
        return value

    # Handle typing constructs
    if hasattr(cls, '__origin__'):
        return typing_construct_factory(key, cls, value)

    # Handle NewTypes
    if hasattr(cls, '__supertype__'):
        return type_factory(key, cls.__supertype__, value)

    # Handle ReadableMixin
    if isinstance(cls, type) and issubclass(cls, ReadableMixin):
        return cls.read(value)

    # Fallback to direct construction
    try:
        return cls(value)
    except Exception as e:
        name = getattr(cls, '__name__', str(cls))
        raise ReadingError(f"Error constructing parameter '{key}' with type '{name}': {e}") from e


class ReadableMixin:
    """Mixin for dataclasses that can be read from a configuration dictionary."""

    @classmethod
    def read(cls: type[T], config: dict[str, Any]) -> T:
        """Read an instance of the class from a configuration dictionary.

        Parameters
        ----------
        cls : type[T]
            Class to create an instance of.
        config : dict
            Configuration dictionary for the instance.

        Returns
        -------
        T
            The created instance.
        """
        parsed_config: dict[str, Any] = {}
        entries: dict[str, Field] = cls.__dataclass_fields__  # pylint: disable=no-member
        # Populate the configuration dictionary
        for key, field in entries.items():
            if key in config:
                parsed_config[key] = type_factory(key, field.type, config[key])
            elif field.default is MISSING:
                raise ReadingError(f"Missing required field '{key}' for reading '{cls.__name__}'.")
        return cls(**parsed_config)
