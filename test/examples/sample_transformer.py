"""Sample response transformer module."""
from piglot.solver.solver import OutputResult
from piglot.utils.response_transformer import ResponseTransformer


class SampleTransformer(ResponseTransformer):
    """Sample response transformer."""

    def transform(self, response: OutputResult) -> OutputResult:
        """Transform the input data.

        Parameters
        ----------
        response : OutputResult
            Time and data points of the response.

        Returns
        -------
        OutputResult
            Transformed time and data points of the response.
        """
        return OutputResult(
            response.time,
            response.data * 2,
        )
