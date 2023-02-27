"""Module for losses associated with LINKS calls."""
import os
import re
import time
import shutil
import subprocess
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import pandas as pd
from yaml import safe_dump_all
from piglot.parameter import ParameterSet
from piglot.optimisers.optimiser import pretty_time


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
    Exception
        When a repeated pattern is found.
    Exception
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
            raise Exception("Repeated pattern {0} in file!".format(pattern))
        init = float(value[value.find("(")+1:value.find(",")])
        low_bound = float(value[value.find(",")+1:value.rfind(",")])
        up_bound = float(value[value.rfind(",")+1:value.rfind(")")])
        parameters.add(pattern, init, low_bound, up_bound)

    for value in short_parameters:
        pattern = value[value.find("<")+1:value.find(">")]
        if pattern not in [a.name for a in parameters]:
            raise Exception("Pattern {0} referenced but not defined!".format(pattern))

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
    Exception
        If the keyword is not found.
    """
    for line in file:
        if line.lstrip().startswith(keyword):
            return line
    raise Exception("Keyword {0} not found!".format(keyword))


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

    def __init__(self, dim, group=1):
        """Constructor for reaction reader

        Parameters
        ----------
        dim : integer
            Direction to read.
        group : int, optional
            Node group to read, by default 1
        """
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
        Exception
            If not reading a macroscopic analysis file.
        Exception
            If reaction output is not requested in the input file.
        """
        # Is macroscopic file?
        if not input_file.endswith('.dat'):
            raise Exception("Reactions only available for macroscopic simulations!")
        # Has node groups keyword?
        if not has_keyword(input_file, "NODE_GROUPS"):
            raise Exception("Reactions requested on an input file without NODE_GROUPS!")
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
        reac_filename = os.path.join(output_dir, '{0}.reac'.format(casename))
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

    def __init__(self, fields, i_elem=None, i_gauss=None, x_field="LoadFactor"):
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
        Exception
            If element and GP numbers are not consistent.
        """
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
                raise Exception("Need to pass both element and Gauss point numbers!")
            self.suffix = '_ELEM_{0}_GP_{1}'.format(i_elem, i_gauss)
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
        Exception
            If requesting homogenised outputs from macroscopic analyses.
        Exception
            If GP outputs have not been specified in the input file.
        Exception
            If the requested element and GP has not been specified in the input file.
        """
        # Check if appropriate scale
        extension = os.path.splitext(input_file)[1]
        if extension == ".dat" and self.suffix == '':
            raise Exception("Cannot extract homogenised .out from macroscopic simulations!")
        # For GP outputs, check if the output has been requsted
        if self.suffix != '':
            if not has_keyword(input_file, "GAUSS_POINTS_OUTPUT"):
                raise Exception("Did not find valid GP output request in the input file!")
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
                    raise Exception("The specified GP output {0} {1} was not found!"\
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
        reac_filename = os.path.join(output_dir, '{0}{1}.out'.format(casename, self.suffix))
        from_gp = self.suffix != ''
        # Read the first line of the file to find the total number of columns
        with open(reac_filename, 'r') as file:
            # Consume the first line if from a GP output (contains GP coordinates)
            if from_gp:
                file.readline()
            line_len = len(file.readline())
        n_columns = int(line_len / self.separator)
        # Fixed-width read (with a skip on the first line for GP outputs)
        header = 1 if from_gp else 0
        df = pd.read_fwf(reac_filename, widths=n_columns*[self.separator], header=header)
        # Extract indices for named columns
        columns = df.columns.tolist()
        df_columns = [self.x_field] + self.fields
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

    def __init__(self, filename, fields: dict, loss):
        """Constructor for the container.

        Parameters
        ----------
        filename : str
            Input file path.
        fields : dict
            Pairs of fields to read from results and their reference solutions.
        loss : Loss
            Loss function to use when comparing predictions and references.
        """
        self.filename = filename
        self.fields = fields
        self.loss = loss
        for field in self.fields.keys():
            field.check(filename)


class LinksLoss:
    """Main loss class for Links problems."""

    def __init__(self, cases, parameters, links_bin, n_concurrent=1, tmp_dir='tmp', output_dir=None):
        """Constructor for the Links_Loss class.

        Parameters
        ----------
        cases : list
            List of Links_Case's to be run.
        parameters : ParameterSet
            Parameter set for this problem.
        links_bin : str
            Path to the Links binary
        n_concurrent : int, optional
            Number of concurrent calls to Links in the multi-objective case, by default 1.
        tmp_dir : str, optional
            Path to temporary directory to run the analyses, by default 'tmp'.
        output_dir : str, optional
            Path to the output directory, if not passed history storing is disabled.
        """
        self.cases = cases
        self.parameters = parameters
        self.links_bin = links_bin
        self.n_concurrent = n_concurrent
        self.tmp_dir = tmp_dir
        self.func_calls = 0
        self.begin_time = time.time()
        self.output_dir = output_dir
        self.cases_dir = os.path.join(output_dir, "cases") if output_dir else None
        self.cases_hist = os.path.join(output_dir, "cases_hist") if output_dir else None
        self.func_calls_file = os.path.join(output_dir, "func_calls") if output_dir else None
        if output_dir:
            os.makedirs(self.cases_dir, exist_ok=True)
            if os.path.isdir(self.cases_hist):
                shutil.rmtree(self.cases_hist)
            os.mkdir(self.cases_hist)
            # Build headers for case files
            for case in self.cases:
                with open(os.path.join(self.cases_dir, case.filename), 'w') as file:
                    file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}\t{"Loss":>15}\t{"Success":>10}')
                    for param in self.parameters:
                        file.write(f"\t{param.name:>15}")
                    file.write(f'\t{"Hash":>64}\n')
            # Build header for function calls file
            with open(os.path.join(self.func_calls_file), 'w') as file:
                file.write(f'{"Start Time /s":>15}\t{"Run Time /s":>15}\t{"Loss":>15}')
                for param in self.parameters:
                    file.write(f"\t{param.name:>15}")
                file.write(f'\t{"Hash":>64}\n')


    def _run_case(self, case: LinksCase):
        """Run a single case wth Links.

        Parameters
        ----------
        case : Links_Case
            Case to run.

        Returns
        -------
        float
            Loss value for this case.
        """
        # Copy input file replacing parameters by passed value
        filename = os.path.basename(case.filename)
        input_file = os.path.join(self.tmp_dir, filename)
        case_name = get_case_name(input_file)
        write_parameters(self.parameters.to_dict(self.X), case.filename, input_file)
        # Run LINKS
        begin_time = time.time()
        subprocess.run([self.links_bin, input_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end_time = time.time()
        # Check if simulation completed
        screen_file = os.path.join(os.path.splitext(input_file)[0], '{0}.screen'.format(case_name))
        failed_case = not has_keyword(screen_file, "Program L I N K S successfully completed.")
        # Post-process results
        responses = {}
        case_loss = 0.0
        for field, reference in case.fields.items():
            ref_x = reference[:,0]
            ref_y = np.squeeze(reference[:,1:])
            # If the case failed, return the maximum possible loss value
            if failed_case:
                case_loss += case.loss.max_value(ref_x, ref_y)
            else:
                # Compute loss for this case: check if single or multiple dimensions on y
                field_data = field.get(input_file)
                field_x = field_data[:,0]
                field_y = np.squeeze(field_data[:,1:])
                if len(ref_y.shape) == 1:
                    case_loss += case.loss(ref_x, field_x, ref_y, field_y)
                    responses[field.name()] = list(zip([float(a) for a  in field_x],
                                                       [float(a) for a  in field_y]))
                else:
                    for i in range(0, ref_y.shape[1]):
                        case_loss += case.loss(ref_x, field_x, ref_y[:,i], field_y[:,i])
                        responses[field.name(i)] = list(zip([float(a) for a  in field_x],
                                                            [float(a) for a  in field_y[:,i]]))
        final_loss = case_loss / len(case.fields)
        # Final touches on case history and file writing
        if self.cases_hist:
            cases_hist = {
                "filename": case.filename,
                "parameters": {p: float(v) for p, v in self.parameters.to_dict(self.X).items()},
                "loss": float(final_loss),
                "success": not failed_case,
                "start_time": time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(begin_time)),
                "run_time": pretty_time(end_time - begin_time),
            }
            with open(os.path.join(self.cases_hist, f'{case.filename}-{self.param_hash}'), 'w') as file:
                safe_dump_all((cases_hist, responses), file)
            with open(os.path.join(self.cases_dir, case.filename), 'a') as file:
                file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                file.write(f'{end_time - begin_time:>15.8e}\t')
                file.write(f'{final_loss:>15.8e}\t')
                file.write(f'{not failed_case:>10}')
                for i, param in enumerate(self.parameters):
                    file.write(f"\t{param.denormalise(self.X[i]):>15.6f}")
                file.write(f'\t{self.param_hash}\n')
        return final_loss


    def loss(self, X):
        """Public loss function for a problem with Links.

        Parameters
        ----------
        X : array
            Current parameters to evaluate the loss.

        Returns
        -------
        float
            Loss value for this set of parameters.
        """
        # Ensure tmp directory is clean
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)
        self.func_calls += 1
        # Set current attribute vector
        self.X = X
        self.param_hash = self.parameters.hash(self.X)
        # Run cases (in parallel if specified)
        begin_time = time.time()
        if self.n_concurrent > 1:
            with Pool(self.n_concurrent) as pool:
                losses = pool.map(self._run_case, self.cases)
        else:
            losses = map(self._run_case, self.cases)
        end_time = time.time()
        # Compute total loss
        final_loss = sum(losses) / len(self.cases)
        # Update function call history file
        if self.output_dir:
            with open(os.path.join(self.func_calls_file), 'a') as file:
                file.write(f'{begin_time - self.begin_time:>15.8e}\t')
                file.write(f'{end_time - begin_time:>15.8e}\t')
                file.write(f'{final_loss:>15.8e}')
                for i, param in enumerate(self.parameters):
                    file.write(f"\t{param.denormalise(self.X[i]):>15.6f}")
                file.write(f'\t{self.param_hash}\n')
        return final_loss
