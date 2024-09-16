from __future__ import annotations
from typing import Dict, Any
import time
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver.solver import (
    Solver,
    Case,
    CaseResult,
    OutputResult,
    InputData,
    OutputField,
)


class SampleInputData(InputData):

    def __init__(self) -> None:
        self.param_values = None

    def prepare(
        self,
        values: np.ndarray,
        parameters: ParameterSet,
        tmp_dir: str = None,
    ) -> SampleInputData:
        # We just store the parameter values, which we will need later
        result = SampleInputData()
        result.param_values = parameters.denormalise(values)
        return result

    def name(self) -> str:
        # Here we just use a dummy name
        # In a real case this should be derived from the actual data
        return "sample_input_data"


class SampleOutputField(OutputField):

    def get(self, input_data: SampleInputData) -> OutputResult:
        # Our outputs are just the parameter values
        return OutputResult(
            np.arange(len(input_data.param_values)),
            np.array(input_data.param_values),
        )

    def check(self, input_data: SampleInputData) -> None:
        # Nothing to check in this case
        pass

    @staticmethod
    def read(config: Dict[str, Any]) -> SampleOutputField:
        # In this simple case, we don't need to read anything
        pass


class SampleSolver(Solver):

    def __init__(
        self,
        parameters: ParameterSet,
        output_dir: str,
    ) -> None:
        super().__init__(
            [
                Case(
                    SampleInputData(),
                    {
                        'sample_field':  # Note the field name
                        SampleOutputField(),
                    }
                ),
            ],
            parameters,
            output_dir,
        )

    def _solve(
        self,
        values: np.ndarray,
        concurrent: bool,
    ) -> Dict[Case, CaseResult]:
        # In this simple case, we don't need to run any simulations
        # We just return the parameter values
        result = {}
        for case in self.cases:
            input_data = case.input_data.prepare(values, self.parameters)
            responses = {
                name: field.get(input_data)
                for name, field in case.fields.items()
            }
            result[case] = CaseResult(
                begin_time=time.time(),
                run_time=0.0,
                values=values,
                success=True,
                param_hash=self.parameters.hash(values),
                responses=responses,
            )
        return result

    def get_current_response(self) -> Dict[str, OutputResult]:
        # For simplicity, we won't implement this method in this example
        pass

    @staticmethod
    def read(
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> Solver:
        # Unlike the other read methods, we do need to return an instance here
        return SampleSolver(parameters, output_dir)
