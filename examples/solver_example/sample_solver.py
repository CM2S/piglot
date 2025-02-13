import numpy as np
from piglot.solver.script_solver import ScriptSolverCallable, OutputResult


class SampleSolver(ScriptSolverCallable):

    @staticmethod
    def get_output_fields() -> list[str]:
        return ["sine"]

    @staticmethod
    def solve(values: dict[str, float]) -> dict[str, OutputResult]:
        time = np.linspace(0, 1, 32)
        sine = np.sin(2 * np.pi * time * values["freq"])
        return {
            "sine": OutputResult(time, sine),
        }
