"""Module for losses associated with LINKS calls."""
from __future__ import annotations
import os
import re
import time
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool as Pool
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from yaml import safe_dump_all, safe_load_all
from piglot.parameter import ParameterSet
from piglot.optimisers.optimiser import pretty_time, reverse_pretty_time
from piglot.objective import SingleObjective, SingleCompositeObjective, MSEComposition
from piglot.objective import DynamicPlotter
from piglot.losses.loss import Loss
from piglot.utils.reduce_response import reduce_response


def write_parameters(param_value, source, dest):
    """Write the set of parameters to the input file.

    Parameters
    ----------
    param_value : dict
        Collection of parameters and their values.
    source : str
        Source input file, to be copied to the destination.
    dest : str
        Destination input file.
    """
    with open(source, 'r') as fin:
        with open(dest, 'w') as fout:
            for line in fin:
                out = line
                for parameter, value in param_value.items():
                    # Replace full expression
                    regex = r'\<' + parameter + '\(.*?\>'
                    out = re.sub(regex, str(value), out)
                    # Replace short expression
                    regex = r'\<' + parameter + '\>'
                    out = re.sub(regex, str(value), out)
                fout.write(out)


def extract_parameters(input_file):
    """Extract a ParameterSet from an input file.

    Parameters
    ----------
    input_file : str
        Input file path.

    Returns
    -------
    ParameterSet
        Set of parameters discovered in the input file.

    Raises
    ------
    RuntimeError
        When a repeated pattern is found.
    RuntimeError
        When a pattern is referenced but never defined.
    """
    parameters = ParameterSet()
    full_parameters = []
    short_parameters = []
    with open(input_file, 'r') as file:
        for line in file:
            full_expression = re.findall(r"\<.*?\(.*?\>", line)
            short_expression = re.findall(r"\<\w+\>", line)
            for a in full_expression:
                full_parameters.append(a)
            for a in short_expression:
                short_parameters.append(a)

    for value in full_parameters:
        pattern = value[value.find("<")+1:value.find("(")]
        if pattern in [a.name for a in parameters]:
            raise RuntimeError(f"Repeated pattern {pattern} in file!")
        init = float(value[value.find("(")+1:value.find(",")])
        low_bound = float(value[value.find(",")+1:value.rfind(",")])
        up_bound = float(value[value.rfind(",")+1:value.rfind(")")])
        parameters.add(pattern, init, low_bound, up_bound)

    for value in short_parameters:
        pattern = value[value.find("<")+1:value.find(">")]
        if pattern not in [a.name for a in parameters]:
            raise RuntimeError(f"Pattern {pattern} referenced but not defined!")

    return parameters


def get_case_name(input_file):
    """Extracts the name of a given case.

    Parameters
    ----------
    input_file : str
        Path for the input file.

    Returns
    -------
    str
        Name of the case.
    """
    filename = os.path.basename(input_file)
    return os.path.splitext(filename)[0]


def has_keyword(input_file, keyword):
    """Checks whether an input file contains a given keyword.

    Parameters
    ----------
    input_file : str
        Path for the input file.
    keyword : str
        Keyword to locate.

    Returns
    -------
    bool
        Whether the input file contains the keyword or not.
    """
    with open(input_file, 'r') as file:
        for line in file:
            if line.lstrip().startswith(keyword):
                return True
    return False


def find_keyword(file, keyword):
    """Finds the first line where a keyword is defined.

    Parameters
    ----------
    file : str
        Path for the input file.
    keyword : str
        Keyword to locate.

    Returns
    -------
    integer
        Line number of the keyword.

    Raises
    ------
    RuntimeError
        If the keyword is not found.
    """
    for line in file:
        if line.lstrip().startswith(keyword):
            return line
    raise RuntimeError(f"Keyword {keyword} not found!")


class OutputField(ABC):
    """Generic class for output fields.

    Methods
    -------
    check(input_file):
        Checks for validity in the input file before reading. This needs to be called prior
        to any reading on the file.
    get(input_file):
        Reads the input file and returns the requested fields.
    """

    def __init__(self, loss: Loss) -> None:
        self.loss = loss

    @abstractmethod
    def check(self, input_file: str) -> None:
        """Checks for validity in the input file before reading.

        This needs to be called prior to any reading on the file.

        Parameters
        ----------
        input_file : str
            Path to the input file.
        """

    @abstractmethod
    def get(self, input_file: str) -> np.ndarray:
        """Reads an output from the simulation.

        Parameters
        ----------
        input_file : str
            Path to the input file.
        """

    @abstractmethod
    def name(self, field_idx: int = None) -> str:
        """Return the name of the current field.

        Parameters
        ----------
        field_idx : int, optional
            Index of the field to output, by default None

        Returns
        -------
        str
            Field name
        """

    @classmethod
    def read(cls, input_file: str, *args, **kwargs) -> np.ndarray:
        """Direct reading of the results of the given input file.

        Parameters
        ----------
        input_file : str
            Input file path.

        Returns
        -------
        array
            2D array with the requested fields.
        """
        reader = cls(*args, **kwargs)
        reader.check(input_file)
        return reader.get(input_file)


class Reaction(OutputField):
    """Reaction outputs reader."""

    def __init__(self, dim, group=1, loss: Loss=None):
        """Constructor for reaction reader

        Parameters
        ----------
        dim : integer
            Direction to read.
        group : int, optional
            Node group to read, by default 1
        """
        super().__init__(loss)
        dim_dict = {'x': 1, 'y': 2, 'z': 3}
        self.dim = dim_dict[dim]
        self.dim_name = dim
        self.group = group

    def check(self, input_file: str) -> None:
        """Sanity checks on the input file.

        This checks if:
        - we are reading the reactions of a macroscopic analysis;
        - the file has NODE_GROUPS outputs.

        Parameters
        ----------
        input_file : str
            Path to the input file.

        Raises
        ------
        RuntimeError
            If not reading a macroscopic analysis file.
        RuntimeError
            If reaction output is not requested in the input file.
        """
        # Is macroscopic file?
        if not input_file.endswith('.dat'):
            raise RuntimeError("Reactions only available for macroscopic simulations!")
        # Has node groups keyword?
        if not has_keyword(input_file, "NODE_GROUPS"):
            raise RuntimeError("Reactions requested on an input file without NODE_GROUPS!")
        # TODO: check group number and dimensions

    def get(self, input_file: str) -> np.ndarray:
        """Reads reactions from a Links analysis.

        Parameters
        ----------
        input_file : str
            Path to the input file

        Returns
        -------
        array
            2D array with load factor in the first column and reactions in the second.
        """
        casename = get_case_name(input_file)
        output_dir = os.path.splitext(input_file)[0]
        reac_filename = os.path.join(output_dir, f'{casename}.reac')
        # Ensure the file exists
        if not os.path.exists(reac_filename):
            return np.empty((0, 2))
        data = np.genfromtxt(reac_filename)
        return (data[data[:,0] == self.group, 1:])[:,[0, self.dim]]

    def name(self, field_idx: int = None) -> str:
        """Return the name of the current field.

        Parameters
        ----------
        field_idx : int, optional
            Index of the field to output, by default None

        Returns
        -------
        str
            Field name
        """
        return f"Reaction: {self.dim_name}"


class OutFile(OutputField):
    """Links .out file reader."""

    def __init__(self, fields, i_elem=None, i_gauss=None, x_field="LoadFactor", loss: Loss=None):
        """Constructor for .out file reader

        Parameters
        ----------
        fields : list or str or int
            Field(s) to read from the output file.
            Can be a single column index or name, or a list of either.
        i_elem : integer, optional
            For GP outputs, from which element to read, by default None
        i_gauss : integer, optional
            From GP outputs, from which GP to read, by default None
        x_field : str, optional
            Field to use as index in the resulting DataFrame, by default "LoadFactor"

        Raises
        ------
        RuntimeError
            If element and GP numbers are not consistent.
        """
        super().__init__(loss)
        # Ensure fields is a list
        if isinstance(fields, str) or isinstance(fields, int):
            self.fields = [fields]
        else:
            self.fields = fields
        # Homogenised .out or GP output?
        if i_elem is None and i_gauss is None:
            # Homogenised: no suffix
            self.suffix = ''
        else:
            # GP output: needs both element and GP numbers
            if (i_elem is None) or (i_gauss is None):
                raise RuntimeError("Need to pass both element and Gauss point numbers!")
            self.suffix = f'_ELEM_{i_elem}_GP_{i_gauss}'
            self.i_elem = i_elem
            self.i_gauss = i_gauss
        self.x_field = x_field
        self.separator = 16

    def check(self, input_file: str) -> None:
        """Sanity checks on the input file.

        This checks for:
        - homogenised outputs only from microscopic simulations;
        - for GP outputs, ensure the input file lists the element and GP number requested;
        - whether the .out file is in single or double precision.

        Parameters
        ----------
        input_file : str
            Path to the input file.

        Raises
        ------
        RuntimeError
            If requesting homogenised outputs from macroscopic analyses.
        RuntimeError
            If GP outputs have not been specified in the input file.
        RuntimeError
            If the requested element and GP has not been specified in the input file.
        """
        # Check if appropriate scale
        extension = os.path.splitext(input_file)[1]
        if extension == ".dat" and self.suffix == '':
            raise RuntimeError("Cannot extract homogenised .out from macroscopic simulations!")
        # For GP outputs, check if the output has been requsted
        if self.suffix != '':
            if not has_keyword(input_file, "GAUSS_POINTS_OUTPUT"):
                raise RuntimeError("Did not find valid GP output request in the input file!")
            # Check number of GP outputs and if ours is a valid one
            with open(input_file, 'r') as file:
                line = find_keyword(file, "GAUSS_POINTS_OUTPUT")
                n_points = int(line.split()[1])
                found = False
                for i in range(0, n_points):
                    line_split = file.readline().split()
                    i_elem = int(line_split[0])
                    i_gaus = int(line_split[1])
                    if self.i_elem == i_elem and self.i_gauss == i_gaus:
                        found = True
                        break
                if not found:
                    raise RuntimeError("The specified GP output {0} {1} was not found!"\
                                    .format(self.i_elem, self.i_gauss))
        # Check if single or double precision output
        self.separator = 24 if has_keyword(input_file, "DOUBLE_PRECISION_OUTPUT") else 16

    def get(self, input_file: str) -> np.ndarray:
        """Get a parameter from the .out file.

        Parameters
        ----------
        input_file : str
                Input file to read.

        Returns
        -------
        DataFrame
                DataFrame with the requested fields.
        """
        casename = get_case_name(input_file)
        output_dir = os.path.splitext(input_file)[0]
        filename = os.path.join(output_dir, f'{casename}{self.suffix}.out')
        from_gp = self.suffix != ''
        df_columns = [self.x_field] + self.fields
        # Ensure the file exists
        if not os.path.exists(filename):
            return np.empty((0, len(df_columns)))
        # Read the first line of the file to find the total number of columns
        with open(filename, 'r', encoding='utf8') as file:
            # Consume the first line if from a GP output (contains GP coordinates)
            if from_gp:
                file.readline()
            line_len = len(file.readline())
        n_columns = int(line_len / self.separator)
        # Fixed-width read (with a skip on the first line for GP outputs)
        header = 1 if from_gp else 0
        df = pd.read_fwf(filename, widths=n_columns*[self.separator], header=header)
        # Extract indices for named columns
        columns = df.columns.tolist()
        int_columns = [columns.index(a) if isinstance(a, str) else a for a in df_columns]
        # Return the given quantity as the x-variable
        return df.iloc[:,int_columns].to_numpy()

    def name(self, field_idx: int = None) -> str:
        """Return the name of the current field.

        Parameters
        ----------
        field_idx : int, optional
            Index of the field to output, by default None

        Returns
        -------
        str
            Field name
        """
        field = self.fields[field_idx if field_idx else 0]
        if isinstance(field, str):
            return f"OutFile: {field}"
        return f"OutFile: column {field + 1}"


class Reference:
    """Container for reference solutions"""

    def __init__(
            self,
            filename: str,
            x_col: int=1,
            y_col: int=2,
            skip_header: int=0,
            x_scale: float=1.0,
            y_scale: float=1.0,
            x_offset: float=0.0,
            y_offset: float=0.0,
            filter_tol: float=0.0,
            show: bool=False,
        ):
        self.data = np.genfromtxt(filename, skip_header=skip_header)[:,[x_col - 1, y_col - 1]]
        self.data[:,0] = x_offset + x_scale * self.data[:,0]
        self.data[:,1] = y_offset + y_scale * self.data[:,1]
        self.orig_data = np.copy(self.data)
        self.filter_tol = filter_tol
        self.show = show

    def prepare(self) -> None:
        """Prepare the reference data"""
        if self.has_filtering():
            print("Filtering reference ...", end='')
            num, error, (x, y) = reduce_response(self.data[:,0], self.data[:,1], self.filter_tol)
            self.data = np.array([x, y]).T
            print(f" done (from {self.orig_data.shape[0]} to {num} points, error = {error:.2e})")
            if self.show:
                _, ax = plt.subplots()
                ax.plot(self.orig_data[:,0], self.orig_data[:,1], label="Reference")
                ax.plot(self.data[:,0], self.data[:,1], c='r', ls='dashed')
                ax.scatter(self.data[:,0], self.data[:,1], c='r', label="Resampled")
                ax.legend()
                plt.show()

    def num_fields(self) -> int:
        """Get the number of reference fields

        Returns
        -------
        int
            Number of reference fields
        """
        return self.data.shape[1] - 1

    def has_filtering(self) -> bool:
        """Check if the reference has filtering

        Returns
        -------
        bool
            Whether the reference has filtering
        """
        return self.filter_tol > 0.0

    def get_time(self) -> np.ndarray:
        """Get the time column of the reference

        Returns
        -------
        np.ndarray
            Time column
        """
        return self.data[:, 0]

    def get_data(self, field_idx: int=0) -> np.ndarray:
        """Get the data column of the reference

        Parameters
        ----------
        field_idx : int
            Index of the field to output

        Returns
        -------
        np.ndarray
            Data column
        """
        return self.data[:, field_idx + 1]

    def get_orig_time(self) -> np.ndarray:
        """Get the original time column of the reference

        Returns
        -------
        np.ndarray
            Original time column
        """
        return self.orig_data[:, 0]

    def get_orig_data(self, field_idx: int=0) -> np.ndarray:
        """Get the original data column of the reference

        Parameters
        ----------
        field_idx : int
            Index of the field to output

        Returns
        -------
        np.ndarray
            Original data column
        """
        return self.orig_data[:, field_idx + 1]


class LinksCase:
    """Container with the required fields for each case to be run with Links."""

    def __init__(
            self,
            filename: str,
            fields: Dict[OutputField, Reference],
            loss: Loss = None,
            weight: float=1.0,
        ):
        """Constructor for the container.

        Parameters
        ----------
        filename : str
            Input file path.
        fields : Dict[OutputField, Reference]
            Pairs of fields to read from results and their reference solutions.
        loss : Loss
            Loss function to use when comparing predictions and references.
        weight : float, optional
            Relative weight of this case (defaults to 1.0).
        """
        self.filename = filename
        self.fields = fields
        self.loss = loss
        self.weight = weight

    def prepare(self) -> None:
        """Prepare the case for running."""
        for field, reference in self.fields.items():
            # Prepare the field
            field.check(self.filename)
            # Assign the default loss to fields without one
            if field.loss is None and self.loss is not None:
                field.loss = self.loss
            # Prepare the reference
            reference.prepare()

    def get_field(self, name: str) -> OutputField:
        """Get the field with the given name

        Parameters
        ----------
        name : str
            Name of the field to get

        Returns
        -------
        OutputField
            Field with the given name
        """
        for field in self.fields.keys():
            if field.name() == name:
                return field
        return None

    def get_reference(self, name: str) -> Reference:
        """Get the reference with the given name

        Parameters
        ----------
        name : str
            Name of the reference to get

        Returns
        -------
        Reference
            Reference with the given name
        """
        return self.fields[self.get_field(name)]


@dataclass
class LinksCaseResult:
    """Container with the results from a given case run with Links."""
    begin_time: float
    end_time: float
    values: np.ndarray
    failed: bool
    responses: Dict[str, np.ndarray]

    @staticmethod
    def read(filename: str) -> LinksCaseResult:
        """Read a case result file

        Parameters
        ----------
        filename : str
            Path to the case result file

        Returns
        -------
        LinksCaseResult
            Result instance
        """
        with open(filename, 'r', encoding='utf8') as file:
            metadata, responses_raw = safe_load_all(file)
        responses = {name: (np.array([a[0] for a in data]), np.array([a[1] for a in data]))
                     for name, data in responses_raw.items()}
        return LinksCaseResult(
            0.0,
            reverse_pretty_time(metadata["run_time"]),
            np.array(metadata["parameters"].values()),
            not metadata["success"] == "true",
            responses,
        )


class CurrentPlot(DynamicPlotter):
    """Container for dynamically-updating plots."""

    def __init__(self, case: LinksCase, tmp_dir: str):
        """Constructor for dynamically-updating plots

        Parameters
        ----------
        case : LinksCase
            Case to plot
        tmp_dir : str
            Path of the temporary directory
        """
        self.case = case
        n_fields = len(case.fields)
        n_cols = min(max(1, n_fields), 2)
        n_rows = int(np.ceil(n_fields / 2))
        self.fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
        self.axes = [a for b in axes for a in b]
        self.path = os.path.join(tmp_dir, case.filename)
        # Make initial plot
        self.pred = {}
        for i, (field, reference) in enumerate(self.case.fields.items()):
            name = field.name()
            data = field.get(self.path)
            self.axes[i].plot(reference.get_time(), reference.get_data(),
                              label='Reference', ls='dashed', c='black', marker='x')
            self.pred[name], = self.axes[i].plot(data[:, 0], data[:, 1],
                                                 label='Prediction', c='red')
            self.axes[i].set_title(name)
            self.axes[i].grid()
            self.axes[i].legend()
        self.fig.suptitle(case.filename)
        plt.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self) -> None:
        """Update the plot with the most recent data"""
        for i, field in enumerate(self.case.fields.keys()):
            name = field.name()
            try:
                data = field.get(self.path)
                self.pred[name].set_xdata(data[:, 0])
                self.pred[name].set_ydata(data[:, 1])
                self.axes[i].relim()
                self.axes[i].autoscale_view()
            except (FileNotFoundError, IndexError):
                pass
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class LinksSolver:
    """Main loss class for Links problems."""

    def __init__(
            self,
            cases: List[LinksCase],
            parameters: ParameterSet,
            links_bin: str,
            parallel: int,
            tmp_dir: str,
            output_dir: str,
        ) -> None:
        """Constructor for the Links_Loss class.

        Parameters
        ----------
        cases : list
            List of LinksCases to be run.
        parameters : ParameterSet
            Parameter set for this problem.
        links_bin : str
            Path to the Links binary
        parallel : int
            Number of concurrent calls to Links in the multi-objective case, by default 1.
        tmp_dir : str
            Path to temporary directory to run the analyses, by default 'tmp'.
        output_dir : str
            Path to the output directory, if not passed history storing is disabled.
        """
        self.cases = cases
        self.parameters = parameters
        self.links_bin = links_bin
        self.parallel = parallel
        self.output_dir = output_dir
        self.begin_time = time.time()
        self.tmp_dir = os.path.join(output_dir, tmp_dir)
        self.cases_dir = os.path.join(output_dir, "cases")
        self.cases_hist = os.path.join(output_dir, "cases_hist")
        # Check if there is a reference response reduction
        self.use_loss_orig = any([reference.has_filtering()
                                  for case in self.cases
                                  for reference in case.fields.values()])
        


    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        os.makedirs(self.cases_dir, exist_ok=True)
        if os.path.isdir(self.cases_hist):
            shutil.rmtree(self.cases_hist)
        os.mkdir(self.cases_hist)
        # Build headers for case log files
        for case in self.cases:
            case_dir = os.path.join(self.cases_dir, case.filename)
            with open(case_dir, 'w', encoding='utf8') as file:
                file.write(f"{'Start Time /s':>15}\t")
                file.write(f"{'Run Time /s':>15}\t")
                file.write(f"{'Loss':>15}\t")
                if self.use_loss_orig:
                    file.write(f"{'Loss (orig.)':>15}\t")
                file.write(f"{'Success':>10}\t")
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Hash":>64}\n')
        # Prepare each case for optimisation
        for case in self.cases:
            case.prepare()


    def _run_case(self, values: np.ndarray, case: LinksCase, tmp_dir: str) -> LinksCaseResult:
        """Run a single case wth Links.

        Parameters
        ----------
        values: np.ndarray
            Current parameter values
        case : LinksCase
            Case to run.
        tmp_dir: str
            Temporary directory to run the simulation

        Returns
        -------
        LinksCaseResult
            Results for this case
        """
        # Copy input file replacing parameters by passed value
        filename = os.path.basename(case.filename)
        input_file = os.path.join(tmp_dir, filename)
        case_name = get_case_name(input_file)
        write_parameters(self.parameters.to_dict(values), case.filename, input_file)
        # Run LINKS (we don't use high precision timers here to keep track of the start time)
        begin_time = time.time()
        process_result = subprocess.run(
            [self.links_bin, input_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        end_time = time.time()
        # Check if simulation completed
        screen_file = os.path.join(os.path.splitext(input_file)[0], f'{case_name}.screen')
        failed_case = (process_result.returncode != 0 or
                       not has_keyword(screen_file, "Program L I N K S successfully completed."))
        # Read results from output directories
        responses = {field.name(): field.get(input_file) for field in case.fields.keys()}
        return LinksCaseResult(begin_time, end_time, values, failed_case, responses)


    def write_history_entry(
            self,
            case: LinksCase,
            result: LinksCaseResult,
            loss: float,
            loss_orig: float,
        ) -> None:
        """Write this case's history entry

        Parameters
        ----------
        case : LinksCase
            Case to write
        result : LinksCaseResult
            Result for this case
        loss : float
            Loss value
        loss_orig : float
            Loss value in the original reference
        """
        # Build case metadata
        param_hash = self.parameters.hash(result.values)
        cases_hist = {
            "filename": case.filename,
            "parameters": {p: float(v) for p, v in self.parameters.to_dict(result.values).items()},
            "param_hash" : str(param_hash),
            "loss": float(loss),
            "success": not result.failed,
            "start_time": time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(result.begin_time)),
            "run_time": pretty_time(result.end_time - result.begin_time),
        }
        if self.use_loss_orig:
            cases_hist['loss_orig'] = float(loss_orig)
        # Encode response to write
        responses = {}
        for field, prediction in result.responses.items():
            reference = case.get_reference(field)
            for i in range(reference.num_fields()):
                responses[field] = list(zip([float(a) for a in prediction[:, 0]],
                                            [float(a) for a in prediction[:, i + 1]]))
        # Write out the case file
        output_case_hist = os.path.join(self.cases_hist, f'{case.filename}-{param_hash}')
        with open(output_case_hist, 'w', encoding='utf8') as file:
            safe_dump_all((cases_hist, responses), file)
        # Add record to case log file
        with open(os.path.join(self.cases_dir, case.filename), 'a', encoding='utf8') as file:
            file.write(f'{result.begin_time - self.begin_time:>15.8e}\t')
            file.write(f'{result.end_time - result.begin_time:>15.8e}\t')
            file.write(f'{loss:>15.8e}\t')
            if self.use_loss_orig:
                file.write(f'{loss_orig:>15.8e}\t')
            file.write(f'{not result.failed:>10}\t')
            for i, param in enumerate(self.parameters):
                file.write(f"{param.denormalise(result.values[i]):>15.6f}\t")
            file.write(f'{param_hash}\n')


    def run(self, values: np.ndarray, parallel: bool) -> Dict[LinksCase, LinksCaseResult]:
        """Run stored problems with Links.

        Parameters
        ----------
        values : array
            Current parameters to evaluate.
        parallel : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        float
            Loss value for this set of parameters.
        """
        # Ensure tmp directory is clean
        tmp_dir = f'{self.tmp_dir}_{self.parameters.hash(values)}' if parallel else self.tmp_dir
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        # Run cases (in parallel if specified)
        def run_case(case: LinksCase) -> LinksCaseResult:
            return self._run_case(values, case, tmp_dir)
        if self.parallel > 1:
            with Pool(self.parallel) as pool:
                results = pool.map(run_case, self.cases)
        else:
            results = map(run_case, self.cases)
        # Ensure we actually resolve the map
        results = list(results)
        # Cleanup temporary directories
        if parallel:
            shutil.rmtree(tmp_dir)
        # Build output dict
        return dict(zip(self.cases, results))


    def plot_case(self, case_hash: str) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        figures = []
        for case in self.cases:
            # Load responses for this case
            filename = os.path.join(self.cases_hist, f'{case.filename}-{case_hash}')
            with open(filename, 'r', encoding='utf8') as file:
                _, responses_raw = safe_load_all(file)
            responses = {name: np.array(data) for name, data in responses_raw.items()}
            # Build figure, index axes and plot response
            fig, axes_raw = plt.subplots(len(responses_raw), squeeze=False)
            axes = {field.name(): axes_raw[i,0] for i, field in enumerate(case.fields.keys())}
            for field, reference in case.fields.items():
                name = field.name()
                axis = axes[name]
                axis.plot(reference.get_time(), reference.get_data(),
                          label='Reference', ls='dashed', c='black', marker='x')
                axis.plot(responses[name][:,0], responses[name][:,1], c='red', label='Prediction')
                axis.set_title(name)
                axis.grid()
                axis.legend()
            figures.append(fig)
        return figures


    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of a updatable plots
        """
        return [CurrentPlot(case, self.tmp_dir) for case in self.cases]


class LinksLoss(SingleObjective):
    """Main loss class for scalar single-objective Links problems."""

    def __init__(
        self,
        cases: List[LinksCase],
        parameters: ParameterSet,
        links_bin: str,
        output_dir: str,
        parallel: int=1,
        tmp_dir: str='tmp',
    ) -> None:
        super().__init__(parameters, output_dir=output_dir)
        self.solver = LinksSolver(cases, parameters, links_bin, parallel, tmp_dir, output_dir)

    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        super().prepare()
        self.solver.prepare()

    def _objective(self, values: np.ndarray, parallel: bool=False) -> float:
        """Objective function for a problem with Links.

        Parameters
        ----------
        values : np.ndarray
            Current parameters to evaluate the loss.
        parallel : bool
            Whether this run may be concurrent to another one (so use unique file names)

        Returns
        -------
        float
            Loss value for this set of parameters.
        """
        # Call solver for all cases
        results = self.solver.run(values, parallel)
        # Compute scalar loss for each case and write to output files
        losses = {}
        for case, result in results.items():
            case_losses = []
            case_losses_orig = []
            for field_name, prediction in result.responses.items():
                field = case.get_field(field_name)
                reference = case.get_reference(field_name)
                for i in range(reference.num_fields()):
                    loss = field.loss(reference.get_time(), prediction[:,0],
                                      reference.get_data(i), prediction[:,i+1])
                    loss_orig = field.loss(reference.get_orig_time(), prediction[:,0],
                                           reference.get_orig_data(i), prediction[:,i+1])
                    if not np.isfinite(loss):
                        loss = field.loss.max_value(reference.get_time(), reference.get_data(i))
                    if not np.isfinite(loss_orig):
                        loss_orig = field.loss.max_value(reference.get_orig_time(),
                                                         reference.get_orig_data(i))
                    case_losses.append(loss)
                    case_losses_orig.append(loss_orig)
            case_loss = np.mean(case_losses)
            case_loss_orig = np.mean(case_losses_orig)
            self.solver.write_history_entry(case, result, case_loss, case_loss_orig)
            losses[case] = case_loss
        # Accumulate final loss with weighting
        return np.mean([case.weight * loss for case, loss in losses.items()])

    def plot_case(self, case_hash: str) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        return self.solver.plot_case(case_hash)

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of a updatable plots
        """
        return self.solver.plot_current()


class CompositeLinksLoss(SingleCompositeObjective):
    """Main loss class for scalar single-objective Links problems."""

    def __init__(
        self,
        cases: List[LinksCase],
        parameters: ParameterSet,
        links_bin: str,
        output_dir: str,
        parallel: int=1,
        tmp_dir: str='tmp',
    ) -> None:
        super().__init__(parameters, MSEComposition(), output_dir=output_dir)
        self.solver = LinksSolver(cases, parameters, links_bin, parallel, tmp_dir, output_dir)

    def prepare(self) -> None:
        """Prepare output directories for the optimsation"""
        super().prepare()
        self.solver.prepare()

    def _inner_objective(self, values: np.ndarray, parallel: bool=False) -> np.ndarray:
        """Objective function for a problem with Links.

        Parameters
        ----------
        values : np.ndarray
            Current parameters to evaluate the loss.
        parallel : bool
            Whether this run may be concurrent to another one (so use unique file names)

        Returns
        -------
        float
            Loss value for this set of parameters.
        """
        # Call solver for all cases
        results = self.solver.run(values, parallel)
        # Compute scalar loss for each case and write to output files
        losses = {}
        for case, result in results.items():
            case_losses = np.array([])
            case_losses_orig = np.array([])
            for field_name, prediction in result.responses.items():
                field = case.get_field(field_name)
                reference = case.get_reference(field_name)
                for i in range(reference.num_fields()):
                    loss = field.loss(reference.get_time(), prediction[:,0],
                                      reference.get_data(i), prediction[:,i+1])
                    loss_orig = field.loss(reference.get_orig_time(), prediction[:,0],
                                           reference.get_orig_data(i), prediction[:,i+1])
                    case_losses = np.append(case_losses, loss)
                    case_losses_org = np.append(case_losses_orig, loss_orig)
            self.solver.write_history_entry(
                case,
                result,
                self.composition.composition(case_losses),
                self.composition.composition(case_losses_org),
            )
            losses[case] = case_losses
        # Accumulate final loss with weighting
        return np.concatenate([case.weight * loss for case, loss in losses.items()])

    def plot_case(self, case_hash: str) -> List[Figure]:
        """Plot a given function call given the parameter hash

        Parameters
        ----------
        case_hash : str, optional
            Parameter hash for the case to plot

        Returns
        -------
        List[Figure]
            List of figures with the plot
        """
        return self.solver.plot_case(case_hash)

    def plot_current(self) -> List[DynamicPlotter]:
        """Plot the currently running function call

        Returns
        -------
        List[DynamicPlotter]
            List of instances of a updatable plots
        """
        return self.solver.plot_current()
