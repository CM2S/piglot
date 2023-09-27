"""Module for losses associated with LINKS calls."""
import os
import re
import time
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool as Pool
from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
from yaml import safe_dump_all
from piglot.parameter import ParameterSet
from piglot.optimisers.optimiser import pretty_time
from piglot.objective import SingleObjective, SingleCompositeObjective, MSEComposition
from piglot.losses.loss import Loss


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
    def check(self, input_file: str):
        """Checks for validity in the input file before reading.

        This needs to be called prior to any reading on the file.

        Parameters
        ----------
        input_file : str
            Path to the input file.
        """

    @abstractmethod
    def get(self, input_file: str):
        """Reads an output from the simulation.

        Parameters
        ----------
        input_file : str
            Path to the input file.
        """

    @abstractmethod
    def name(self, field_idx: int = None):
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
    def read(cls, input_file: str, *args, **kwargs):
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

    def check(self, input_file: str):
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

    def get(self, input_file: str):
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

    def name(self, field_idx: int = None):
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

    def check(self, input_file: str):
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

    def get(self, input_file: str):
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

    def name(self, field_idx: int = None):
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


class LinksCase:
    """Container with the required fields for each case to be run with Links."""

    def __init__(
            self,
            filename: str,
            fields: Dict[OutputField, np.ndarray],
            loss: Loss = None,
            weight: float=1.0,
        ):
        """Constructor for the container.

        Parameters
        ----------
        filename : str
            Input file path.
        fields : dict
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
        for field in self.fields.keys():
            field.check(filename)
            # Assign the default loss to fields without one
            if field.loss is None and loss is not None:
                field.loss = loss


@dataclass
class LinksCaseResult:
    """Container with the results from a given case run with Links."""
    begin_time: float
    end_time: float
    values: np.ndarray
    failed: bool
    responses: Dict[OutputField, Tuple[np.ndarray, np.ndarray]]


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
        # Prepare required output directories
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
                file.write(f"{'Success':>10}\t")
                for param in self.parameters:
                    file.write(f"{param.name:>15}\t")
                file.write(f'{"Hash":>64}\n')


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
        responses = {field: (ref, field.get(input_file)) for field, ref in case.fields.items()}
        return LinksCaseResult(begin_time, end_time, values, failed_case, responses)


    def write_history_entry(self, case: LinksCase, result: LinksCaseResult, loss: float) -> None:
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
        # Encode response to write
        responses = {}
        for field, (reference, prediction) in result.responses.items():
            # The first column of the reference is the time
            num_fields = reference.shape[1] - 1
            for i in range(0, num_fields):
                responses[field.name(i)] = list(zip([float(a) for a in prediction[:, 0]],
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
            for field, (reference, prediction) in result.responses.items():
                for i in range(1, reference.shape[1]):
                    loss = field.loss(reference[:,0], prediction[:,0],
                                      reference[:,i], prediction[:,i])
                    if not np.isfinite(loss):
                        loss = field.loss.max_value(reference[:,0], reference[:,i])
                    case_losses.append(loss)
            case_loss = np.mean(case_losses)
            self.solver.write_history_entry(case, result, case_loss)
            losses[case] = case_loss
        # Accumulate final loss with weighting
        return np.mean([case.weight * loss for case, loss in losses.items()])



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
            for field, (reference, prediction) in result.responses.items():
                for i in range(1, reference.shape[1]):
                    loss = field.loss(reference[:,0], prediction[:,0],
                                      reference[:,i], prediction[:,i])
                    case_losses = np.append(case_losses, loss)
            self.solver.write_history_entry(case, result, self.composition.composition(case_losses))
            losses[case] = case_losses
        # Accumulate final loss with weighting
        return np.concatenate([case.weight * loss for case, loss in losses.items()])
