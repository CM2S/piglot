"""Module for tabular data utilities in piglot."""
from typing import Any, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TabularColumn(ABC):
    """Container for a single column in a table."""

    name: str
    width: int
    align: str = ">"

    @abstractmethod
    def format_value(self, value: Any) -> str:
        """Format a value according to the column's settings.

        Parameters
        ----------
        value : Any
            The value to format.

        Returns
        -------
        str
            The formatted value as a string.
        """

    def format_header(self) -> str:
        """Format the column header according to the column's settings.

        Returns
        -------
        str
            The formatted header as a string.
        """
        return f"{self.name:{self.align}{self.width}s}"


@dataclass
class TabularIntColumn(TabularColumn):
    """Container for an integer column in a table."""

    def format_value(self, value: int) -> str:
        """Format an integer value according to the column's settings.

        Parameters
        ----------
        value : int
            The value to format.

        Returns
        -------
        str
            The formatted value as a string.
        """
        return f"{int(value):{self.align}{self.width}d}"


@dataclass
class TabularFloatColumn(TabularColumn):
    """Container for a float column in a table."""

    precision: int = 8
    notation: Literal["e", "f", "g"] = "f"

    def format_value(self, value: float) -> str:
        """Format a float value according to the column's settings.

        Parameters
        ----------
        value : float
            The value to format.

        Returns
        -------
        str
            The formatted value as a string.
        """
        return f"{float(value):{self.align}{self.width}.{self.precision}{self.notation}}"


class TabularStringColumn(TabularColumn):
    """Container for a string column in a table."""

    def format_value(self, value: str) -> str:
        """Format a string value according to the column's settings.

        Parameters
        ----------
        value : str
            The value to format.

        Returns
        -------
        str
            The formatted value as a string.
        """
        return f"{str(value):{self.align}{self.width}s}"


class Table:
    """Container for a table of data."""

    def __init__(self, columns: list[TabularColumn], sep: str = '\t') -> None:
        self.columns = columns
        self.rows: list[list[Any]] = []
        self.sep = sep
        # Adjust column widths based on the header
        for column in self.columns:
            column.width = max(column.width, len(column.name))

    def header(self) -> str:
        """Generate the header row of the table.

        Returns
        -------
        str
            The formatted header row as a string.
        """
        return self.sep.join(column.format_header() for column in self.columns)

    def format_row(self, row: list[Any]) -> str:
        """Format a single row of data according to the column settings.

        Parameters
        ----------
        row : list[Any]
            The row of data to format.

        Returns
        -------
        str
            The formatted row as a string.
        """
        if len(row) != len(self.columns):
            raise ValueError(
                f"Row length {len(row)} does not match number of columns {len(self.columns)}"
            )
        return self.sep.join(
            column.format_value(value) for column, value in zip(self.columns, row)
        )


class TabularFile:
    """Manager for a tabular file, which can be used to write tabular data during the run."""

    def __init__(
        self, path: str, columns: list[TabularColumn], sep: str = '\t'
    ) -> None:
        self.path = path
        self.table = Table(columns, sep=sep)

    def prepare(self) -> None:
        """Prepare the file for writing by writing the header."""
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(self.table.header() + '\n')

    def write_row(self, row: list[Any]) -> None:
        """Write a row of data to the file.

        Parameters
        ----------
        row : list[Any]
            The data to write as a list of values corresponding to the columns.
        """
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(self.table.format_row(row) + '\n')
