"""Module with utilities for reading data classes from configuration dictionaries."""
from typing import TypeVar, Any, cast, get_type_hints
import inspect
from pydantic import BaseModel, ValidationError, create_model


ReadableModelT = TypeVar('ReadableModelT', bound='ReadableModel')
ConstructorModelT = TypeVar('ConstructorModelT', bound='ReadableFromConstructorMixin')


class ReadingError(ValueError):
    """Custom error for reading issues."""


def validate_model(cls: type[BaseModel], config: dict[str, Any]) -> BaseModel:
    """Validate a configuration dictionary against a Pydantic model.

    Parameters
    ----------
    cls : type[BaseModel]
        The Pydantic model class to validate against.
    config : dict
        The configuration dictionary to validate.

    Returns
    -------
    BaseModel
        An instance of the model if validation is successful.
    """
    try:
        return cls.model_validate(config)
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            loc = ' -> '.join(str(loc) for loc in error['loc'])
            error_messages.append(f"{loc}: {error['msg']} (Received value: '{error['input']}')")
        error_message = f"Error reading {cls.__name__}:\n  " + "\n  ".join(error_messages)
        raise ReadingError(error_message) from e


class ReadableModel(BaseModel):
    """Data class that can be read from a configuration dictionary.

    This creates a pydantic model that can be validated and parsed from a dictionary, providing
    informative error messages when the input data does not conform to the expected structure.

    This should be used for data that requires input sanitisation and validation.
    """

    @classmethod
    def read(cls: type[ReadableModelT], config: dict[str, Any]) -> ReadableModelT:
        """Read an instance of the class from a configuration dictionary.

        Parameters
        ----------
        cls : type[ReadableModelT]
            Class to create an instance of.
        config : dict
            Configuration dictionary for the instance.

        Returns
        -------
        ReadableModelT
            The created instance.
        """
        return cls(**validate_model(cls, config).model_dump())


class ReadableFromConstructorMixin:
    """Mixin that adds a `read(config)` constructor validator.

    It builds a dynamic Pydantic model from `cls.__init__` annotations/defaults.
    """

    @classmethod
    def read(cls: type[ConstructorModelT], config: dict[str, Any]) -> ConstructorModelT:
        """Read an instance of the class from a configuration dictionary.

        Parameters
        ----------
        cls : type[ConstructorModelT]
            Class to create an instance of.
        config : dict[str, Any]
            Configuration dictionary for the instance.

        Returns
        -------
        ConstructorModelT
            An instance of the class.
        """
        # Check if the config model is already created and cached on the class
        model: type[BaseModel] = getattr(cls, "__config_model__", None)

        # If needed, inspect the __init__ method to create a Pydantic model for validation
        if model is None:
            sig = inspect.signature(cls.__init__)
            hints = get_type_hints(cls.__init__)
            fields: dict[str, tuple[Any, Any]] = {}

            # Inject the fields from the __init__ method into the Pydantic model
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                annotation = hints.get(name, Any)
                default = param.default if param.default is not inspect._empty else ...
                fields[name] = (annotation, default)

            # Create and cache the model on the class for future use
            model = create_model(f"{cls.__name__}Config", **fields)
            setattr(cls, "__config_model__", model)

        return cls(**validate_model(model, config).model_dump())


def readable_from_constructor(cls: type[ConstructorModelT]) -> type[ConstructorModelT]:
    """Class decorator that adds a `read(config)` constructor validator.

    It builds a dynamic Pydantic model from `cls.__init__` annotations/defaults.
    """

    new_cls = type(
        cls.__name__,
        (ReadableFromConstructorMixin, cls),
        {
            "__module__": cls.__module__,
            "__doc__": cls.__doc__,
        },
    )
    return cast(type, new_cls)
