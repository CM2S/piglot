"""Module for Links solver."""
from typing import Dict, Type, List, Any
import os
import re
import sys
import subprocess
from piglot.solver.input_file_solver import (
    InputDataGenerator,
    InputFileCase,
    InputData,
    InputFileSolver,
    OutputField,
)
from piglot.solver.abaqus.fields import FieldsOutput


class AbaqusCase(InputFileCase):
    """Class for Abaqus cases."""

    def __init__(
        self,
        name: str,
        fields: Dict[str, OutputField],
        generator: InputDataGenerator,
        abaqus: str,
        step_name: str = None,
        instance_name: str = None,
        job_name: str = None,
        extra_args: str = None,
    ) -> None:
        super().__init__(name, fields, generator)
        self.abaqus_bin = abaqus
        self.step_name = step_name
        self.instance_name = instance_name
        self.job_name = job_name
        self.extra_args = extra_args

    @staticmethod
    def __sanitize_field(field_name: str, field_list: List[str], keyword: str) -> str:
        """Sanitize the field name (jobs, instances and steps) in Abaqus input file.

        Parameters
        ----------
        field_name : str
            Field name to sanitize.
        field_list : List[str]
            A list of fields present in the file.
        keyword : str
            Keyrword to use in the error message. (job, instance, step)

        Returns
        -------
        str
            The field name and field list.

        Raises
        ------
        ValueError
            If the field list is empty.
        ValueError
            If the field name is not in the field list.
        ValueError
            If the field name is not specified and there are multiple fields in the list.
        """
        if len(field_list) == 0:
            raise ValueError(f"No {keyword}s found in the file.")
        if field_name is not None:
            if field_name not in field_list:
                raise ValueError(f"The {keyword} name '{field_name}' not found in the file.")
        if field_name is None:
            if len(field_list) > 1:
                raise ValueError(
                    f"Multiple {keyword}s found in the file. Please specify the {keyword} name."
                )
            return field_list[0]
        return field_name

    def check(self, input_data: InputData) -> None:
        """Check if the input data is valid according to the given parameters.

        Parameters
        ----------
        parameters : ParameterSet
            Parameter set for this problem.
        """
        input_file = os.path.join(input_data.tmp_dir, input_data.input_file)

        with open(input_file, 'r', encoding='utf-8') as file:
            data = file.read()

            job_list = re.findall(r'\*\* Job name: ([^M]+)', data)
            job_list = [job.strip() for job in job_list]
            self.job_name = self.__sanitize_field(self.job_name, job_list, "job")
            print(self.job_name)

            instance_list = re.findall(r'\*Instance, name=([^,]+)', data)
            self.instance_name = self.__sanitize_field(self.instance_name,
                                                       instance_list,
                                                       "instance")

            step_list = re.findall(r'\*Step, name=([^,]+)', data)
            self.step_name = self.__sanitize_field(self.step_name, step_list, "step")

    def _post_proc_variables(self, input_data: InputData) -> Dict[str, Any]:
        """Generate the post-processing variables.

        Parameters
        ----------
        input_data : AbaqusInputData
            Input data for the simulation.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the post processing variables.
        """
        variables = {
            'input_file': input_data.input_file,
            'job_name': self.job_name,
            'step_name': self.step_name,
            'instance_name': self.instance_name,
        }

        # Accessing set_name, field, and x_field from the fields dictionary
        field = next(iter(self.fields.values()), None)
        if field:
            variables['set_name'] = getattr(field, 'set_name', None)
            variables['field'] = getattr(field, 'field', None)
            variables['x_field'] = getattr(field, 'x_field', None)

        return variables

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
        # Call the check method before running the case
        self.check(input_data)

        extra_args: List[str] = self.extra_args.split() if self.extra_args else []

        run_inp = subprocess.run(
            [
                self.abaqus_bin,
                f"job={self.job_name}",
                f"input={input_data.input_file}",
                'interactive',
                'ask_delete=OFF',
            ] + extra_args,
            cwd=tmp_dir,
            shell=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False
        )
        variables = self._post_proc_variables(input_data)
        python_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reader.py')

        run_odb = subprocess.run(
            [
                self.abaqus_bin,
                'viewer',
                f"noGUI={python_script}",
                "--",
                f"input_file={variables['input_file']}",
                "--",
                f"job_name={variables['job_name']}",
                "--",
                f"step_name={variables['step_name']}",
                "--",
                f"instance_name={variables['instance_name']}",
                "--",
                f"set_name={variables['set_name']}",
                "--",
                f"field={variables['field']}",
                "--",
                f"x_field={variables['x_field']}"
            ],
            cwd=tmp_dir,
            shell=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        if run_inp.returncode != 0 or run_odb.returncode != 0:
            return False
        return True

    @classmethod
    def get_supported_fields(cls) -> Dict[str, Type[OutputField]]:
        """Get the supported fields for this input file type.

        Returns
        -------
        Dict[str, Type[OutputField]]
            Names and supported fields for this input file type.
        """
        return {
            'FieldsOutput': FieldsOutput,
        }


class AbaqusSolver(InputFileSolver):
    """ABAQUS solver class."""

    @classmethod
    def get_case_class(cls) -> Type[InputFileCase]:
        """Get the case class to use for this solver.

        Returns
        -------
        Type[InputFileCase]
            InputFileCase class to use for this solver.
        """
        return AbaqusCase
