from piglot.solver.solver import OutputResult
from piglot.solver.input_file_solver import InputData, ScriptOutputField


class BadOutputField(ScriptOutputField):

    def __init__(self):
        super().__init__()
        # Fake a read to trigger the error
        self.read(None)

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
