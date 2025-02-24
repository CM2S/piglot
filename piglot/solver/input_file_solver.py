"""Module for input file-based solvers."""
from __future__ import annotations
from typing import Dict, Any, List, TypeVar, Type, Callable, Tuple
from abc import ABC, abstractmethod
import os
import re
import time
import shutil
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import OutputResult, CaseResult
from piglot.solver.multi_case_solver import Case, MultiCaseSolver
from piglot.utils.assorted import read_custom_module


T = TypeVar('T', bound='OutputField')
V = TypeVar('V', bound='InputFileCase')


def write_parameters(
    param_value: Dict[str, float],
    source: str,
    dest: str,
    regex: Callable[[str], str] = lambda param: r'\<' + param + r'\>',
) -> None:
    """Write the set of parameters to the input file.

    Parameters
    ----------
    param_value : Dict[str, float]
        Collection of parameters and their values.
    source : str
        Source input file, to be copied to the destination.
    dest : str
        Destination input file.
    regex : Callable[[str], str], optional
        Function to generate the regex for the parameter substitution.
        By default, uses the regex to replace "<param_name>" with the value.
    """
    with open(source, 'r', encoding='utf8') as fin:
        with open(dest, 'w', encoding='utf8') as fout:
            for line in fin:
                for parameter, value in param_value.items():
                    line = re.sub(regex(parameter), str(value), line)
                fout.write(line)


class InputData:
    """Class for input file-based input data."""

    def __init__(self, tmp_dir: str, input_file: str, dependencies: List[str]) -> None:
        self.tmp_dir = tmp_dir
        self.input_file = input_file
        self.dependencies = dependencies


class InputDataGenerator(ABC):
    """Base class for input data generators for input file-based solvers."""

    @abstractmethod
    def generate(self, parameters: ParameterSet, values: np.ndarray, tmp_dir: str) -> InputData:
        """Generate the input data for the given set of parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        values : np.ndarray
            Current parameters to evaluate.
        tmp_dir : str
            Temporary directory to run the problem.
        """


class DefaultInputDataGenerator(InputDataGenerator):
    """Default input data generator for input file-based solvers."""

    def __init__(
        self,
        input_file: str,
        substitution_dependencies: List[str] = None,
        copy_dependencies: List[str] = None,
    ) -> None:
        self.input_file = input_file
        self.substitution_dependencies = substitution_dependencies or []
        self.copy_dependencies = copy_dependencies or []

    def generate(self, parameters: ParameterSet, values: np.ndarray, tmp_dir: str) -> InputData:
        """Generate the input data for the given set of parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        values : np.ndarray
            Current parameters to evaluate.
        tmp_dir : str
            Temporary directory to run the problem.

        Returns
        -------
        InputData
            Input data for this problem.
        """
        param_dict = parameters.to_dict(values)
        # Replace parameters in the input file
        gen_input_file = os.path.join(tmp_dir, self.input_file)
        write_parameters(param_dict, self.input_file, gen_input_file)
        # Replace parameters in the dependencies
        dependencies = []
        for dep in self.substitution_dependencies:
            output_file = os.path.join(tmp_dir, dep)
            write_parameters(param_dict, dep, output_file)
            dependencies.append(os.path.basename(output_file))
        # Copy dependencies
        for dep in self.copy_dependencies:
            output_file = os.path.join(tmp_dir, dep)
            shutil.copy(dep, output_file)
            dependencies.append(os.path.basename(output_file))
        return InputData(tmp_dir, os.path.basename(gen_input_file), dependencies)


class OutputField(ABC):
    """Generic class for output fields."""

    @abstractmethod
    def check(self, input_data: InputData) -> None:
        """Check for validity in the input data before reading.

        Parameters
        ----------
        input_data : InputData
            Input data to check for.
        """

    @abstractmethod
    def get(self, input_data: InputData) -> OutputResult:
        """Read the output data from the simulation.

        Parameters
        ----------
        input_data : InputData
            Input data to check for.

        Returns
        -------
        OutputResult
            Output result for this field.
        """

    @classmethod
    @abstractmethod
    def read(cls: Type[T], config: Dict[str, Any]) -> T:
        """Read the output field from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        OutputField
            Output field to use for this problem.
        """


class ScriptOutputField(OutputField):
    """Class for script-bsaed output fields."""

    def check(self, input_data: InputData) -> None:
        """Check for validity in the input data before reading.

        Parameters
        ----------
        input_data : InputData
            Input data to check for.
        """

    @staticmethod
    def read(config: Dict[str, Any]) -> ScriptOutputField:
        """Read the output field from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        ScriptOutputField
            Output field to use for this problem.
        """
        raise RuntimeError("Cannot read the configuration for a script-based output field.")


class InputFileCase(Case, ABC):
    """Base case class for input file-based solvers."""

    def __init__(
        self,
        name: str,
        fields: Dict[str, OutputField],
        generator: InputDataGenerator,
    ) -> None:
        self.case_name = name
        self.fields = fields
        self.generator = generator

    def name(self) -> str:
        """Return the name of the case.

        Returns
        -------
        str
            Name of the case.
        """
        return self.case_name

    def get_fields(self) -> List[str]:
        """Get the fields to output for this case.

        Returns
        -------
        List[str]
            Fields to output for this case.
        """
        return list(self.fields.keys())

    @abstractmethod
    def _run_case(self, input_data: InputData, tmp_dir: str) -> bool:
        """Run the case for the given set of parameters.

        Parameters
        ----------
        input_data : InputData
            Input data for this problem.
        tmp_dir : str
            Temporary directory to run the problem.

        Returns
        -------
        bool
            Whether the case ran successfully or not.
        """

    def run(
        self,
        parameters: ParameterSet,
        values: np.ndarray,
        tmp_dir: str,
    ) -> CaseResult:
        """Run the case for the given set of parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        values : np.ndarray
            Current parameters to evaluate.
        tmp_dir : str
            Temporary directory to run the problem.

        Returns
        -------
        CaseResult
            Result of the case.
        """
        # Isolate the input data into a new directory and generate the input data
        tmp_dir = os.path.join(tmp_dir, self.name())
        os.makedirs(tmp_dir, exist_ok=True)
        param_hash = parameters.hash(values)
        input_data = self.generator.generate(parameters, values, tmp_dir)
        # Ensure the temporary directory is consistent
        if input_data.tmp_dir != tmp_dir:
            raise ValueError(
                f'Input data temporary directory "{input_data.tmp_dir}" does not match '
                f'the expected temporary directory "{tmp_dir}".'
            )
        # Ensure the input file has been generated
        input_file = os.path.join(input_data.tmp_dir, input_data.input_file)
        if not os.path.exists(input_file):
            raise RuntimeError(
                f'Input file "{input_data.input_file}" does not exist '
                f'in the temporary directory "{input_data.tmp_dir}".'
            )
        # Sanitise the input data and output fields
        for field in self.fields.values():
            field.check(input_data)
        # Run and time the case (no high precision timing to track start time)
        begin_time = time.time()
        success = self._run_case(input_data, tmp_dir)
        elapsed_time = time.time() - begin_time
        # Read and return the fields
        responses = {name: field.get(input_data) for name, field in self.fields.items()}
        return CaseResult(begin_time, elapsed_time, values, success, param_hash, responses)

    @classmethod
    @abstractmethod
    def get_supported_fields(cls) -> Dict[str, Type[OutputField]]:
        """Get the supported fields for this input file type.

        Returns
        -------
        Dict[str, Type[OutputField]]
            Names and supported fields for this input file type.
        """

    @classmethod
    def get_dependencies(cls, input_file: str) -> Tuple[List[str], List[str]]:
        """Get the dependencies for a given input file.

        Override this method to provide custom dependencies.

        Parameters
        ----------
        input_file : str
            Input file to check for dependencies.

        Returns
        -------
        Tuple[List[str], List[str]]
            Substitution and copy dependencies for this input file.
        """
        if not os.path.exists(input_file):
            raise ValueError(f'Input file "{input_file}" does not exist.')
        return [], []

    @classmethod
    def read(
        cls: Type[V],
        name: str,
        config: Dict[str, Any],
    ) -> V:
        """Read the case from the configuration dictionary.

        Parameters
        ----------
        name : str
            Name of the case.
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        InputFileCase
            Case to use for this problem.
        """
        # Sanitise fields
        if 'fields' not in config:
            raise ValueError(f'No fields defined for case "{name}".')
        config_fields = config.pop('fields')
        # Read each field
        fields: Dict[str, OutputField] = {}
        supported_fields = cls.get_supported_fields()
        for field_name, field_config in config_fields.items():
            # Sanitise name
            if 'name' not in field_config:
                raise ValueError(f'No name defined for field "{field_name}" of case "{name}".')
            field_type = field_config['name']
            # Check if we are using a script
            if field_type == 'script':
                fields[field_name] = read_custom_module(field_config, ScriptOutputField)()
            else:
                # Check if supported
                if field_type not in supported_fields:
                    raise ValueError(f'Field "{field_type}" not supported for case "{name}".')
                fields[field_name] = supported_fields[field_type].read(field_config)
        # Check if we are using a custom generator
        if 'generator' in config:
            generator = read_custom_module(config.pop('generator'), InputDataGenerator)()
            # Ensure we don't have dependencies with a custom generator
            if 'substitution_dependencies' in config or 'copy_dependencies' in config:
                raise ValueError('Dependencies not supported with custom input data generators.')
        else:
            # Try to find the dependencies for this case
            substitution_deps, copy_deps = cls.get_dependencies(name)
            # Override the dependencies if they are defined in the config
            substitution_deps = config.pop('substitution_dependencies', substitution_deps)
            copy_deps = config.pop('copy_dependencies', copy_deps)
            # Build the generator
            generator = DefaultInputDataGenerator(name, substitution_deps, copy_deps)
        return cls(name, fields, generator, **config)


class InputFileSolver(MultiCaseSolver):
    """Base class for input file-based solvers."""

    @classmethod
    @abstractmethod
    def get_case_class(cls) -> Type[InputFileCase]:
        """Get the case class to use for this solver.

        Returns
        -------
        Type[InputFileCase]
            InputFileCase class to use for this solver.
        """
