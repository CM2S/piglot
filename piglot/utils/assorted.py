"""Assorted utilities."""
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Optional, Type, TypeVar, Any, Union, Iterable, Iterator
import os
import copy
import contextlib
import importlib
import importlib.util
from concurrent import futures
import numpy as np
from scipy.stats import t
import torch


def str_to_numeric(data: str) -> Union[int, float, str]:
    """Tries to convert a string to a numeric value.

    Parameters
    ----------
    data : str
        String to convert.

    Returns
    -------
    Union[int, float, str]
        Converted value.
    """
    try:
        data = float(data)
    except (TypeError, ValueError):
        return data
    if int(data) == data:
        return int(data)
    return data


def pretty_time(elapsed_sec: float) -> str:
    """Return a human-readable representation of a given elapsed time

    Parameters
    ----------
    elapsed_sec : float
        Elapsed time, in seconds

    Returns
    -------
    str
        Pretty elapsed time string
    """
    mults = {
        'y': 60*60*24*365,
        'd': 60*60*24,
        'h': 60*60,
        'm': 60,
        's': 1,
    }
    time_str = ''
    for suffix, factor in mults.items():
        count = elapsed_sec // factor
        if count > 0:
            time_str += str(int(elapsed_sec / factor)) + suffix
        elapsed_sec %= factor
    if time_str == '':
        time_str = f'{elapsed_sec:.5f}s'
    return time_str


def missing_method(name, package):
    """Class generator for missing packages.

    Parameters
    ----------
    name : str
        Name of the missing method.
    package : str
        Name of the package to install.
    """
    def err_func(name, package):
        """Raise an error for this missing method.

        Parameters
        ----------
        name : str
            Name of the missing method.
        package : str
            Name of the package to install.

        Raises
        ------
        ImportError
            Every time it is called.
        """
        raise ImportError(f"{name} is not available. You need to install package {package}!")

    return type(
        f'Missing_{package}',
        (),
        {
            'name': name,
            'package': package,
            '__init__': (lambda *args, **kwargs: err_func(name, package))
        },
    )


@contextlib.contextmanager
def change_cwd(path: str):
    """Context manager to temporarily change the current working directory.

    Adapted from https://stackoverflow.com/a/75049063

    Parameters
    ----------
    path : str
        New working directory.
    """
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


T = TypeVar('T')
U = TypeVar('U')


def read_custom_module(config: Dict[str, Any], cls: Type[T]) -> Type[T]:
    """Read a custom module from a configuration spec.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration of the custom module.
    cls : Type
        Base class of the module to load.

    Returns
    -------
    Type
        Custom module type read from the script.
    """
    # Sanitise the configuration
    if 'script' not in config:
        raise ValueError(f"Missing 'script' field for reading the script of type `{cls.__name__}`.")
    if 'class' not in config:
        raise ValueError(f"Missing 'class' field for reading the script of type `{cls.__name__}`.")
    # Load the module
    module_name = f'piglot_{os.path.basename(config["script"]).replace(".", "_")}'
    spec = importlib.util.spec_from_file_location(module_name, config['script'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_class = getattr(module, config['class'])
    # Sanitise the class
    if not issubclass(module_class, cls):
        raise ValueError(
            f"Custom class '{module_class.__name__}' is not a subclass of '{cls.__name__}'."
        )
    return module_class


class TorchContainer:
    """Mixin for objects containing torch tensors."""

    def to(self: T, device: torch.device, dtype: torch.dtype = None) -> T:
        """Move the object to a given device/dtype.

        Parameters
        ----------
        device : torch.device
            Device to move the object to.
        dtype : torch.dtype, optional
            Dtype to move the object to, by default None.

        Returns
        -------
        T
            The object in the new device/dtype.
        """
        new_object = copy.deepcopy(self)
        for name, attr in new_object.__dict__.items():
            if isinstance(attr, (torch._C._TensorBase, TorchContainer)):  # pylint: disable=W0212
                setattr(new_object, name, attr.to(device, dtype=dtype))
        return new_object


def parallel_map(func: Callable[[T], U], iterable: Iterable[T], num_workers: int) -> Iterator[U]:
    """Apply a function to each element in an iterable in parallel.

    Parameters
    ----------
    func : Callable[[T], U]
        Function to apply to each element.
    iterable : Iterable[T]
        Iterable of elements to process.
    num_workers : int
        Number of parallel workers.

    Returns
    -------
    Iterator[U]
        Iterator of results from applying the function.
    """
    if num_workers == 1:
        return (func(x) for x in iterable)
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        return executor.map(func, iterable)


class InlineFileManager:
    """Context manager for creating and managing inline temporary files."""
    INLINE_URI = 'inline://'

    def __init__(self, inline_files: dict[str, str]) -> None:
        self.inline_files = inline_files
        self.temp_dir: Optional[TemporaryDirectory] = None

    def __enter__(self) -> 'InlineFileManager':
        if len(self.inline_files) > 0:
            if self.temp_dir is not None:
                raise RuntimeError("InlineFileManager is already in use.")
            self.temp_dir = TemporaryDirectory(prefix='piglot-inline-')
            for filename, content in self.inline_files.items():
                file_path = os.path.join(self.temp_dir.name, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None

    def __update_node(self, node: Any) -> Any:
        if isinstance(node, dict):
            return {self.__update_node(k): self.__update_node(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self.__update_node(x) for x in node]
        if isinstance(node, tuple):
            return tuple(self.__update_node(x) for x in node)
        if isinstance(node, str):
            if node.startswith(self.INLINE_URI):
                filename = node[len(self.INLINE_URI):]
                if filename in self.inline_files:
                    return os.path.join(self.temp_dir.name, filename)
        return node

    def update_config(self, config: Any) -> Any:
        """Update the configuration with the inline file paths.

        Parameters
        ----------
        config : Any
            Configuration to update.

        Returns
        -------
        Any
            Updated configuration.
        """
        # Nothing to update if there are no inline files
        if len(self.inline_files) == 0:
            return config
    
        # Sanity check
        if self.temp_dir is None:
            raise RuntimeError("InlineFileManager is not in use.")
        
        # Update the configuration with the inline file paths
        return self.__update_node(config)


def convert_simple_spec(data: Union[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert a simple specification to a dictionary with the detailed format.

    Parameters
    ----------
    data : Union[str, dict[str, Any]]
        The simple specification, either as a string or a dictionary.

    Returns
    -------
    dict[str, Any]
        The detailed specification as a dictionary.
    """
    if isinstance(data, str):
        return {"name": data}
    return data
