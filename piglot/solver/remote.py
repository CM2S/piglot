"""Module for solvers over remote machines via SSH."""
from __future__ import annotations
from typing import List, Dict, Any
import os
import sys
import shutil
import subprocess
import tempfile
import numpy as np
from piglot.parameter import ParameterSet
from piglot.solver import read_solver
from piglot.solver.solver import OutputResult, SingleCaseSolver
from piglot.utils.yaml_parser import parse_solver_file


class RemoteSolver(SingleCaseSolver):
    """Remote solver via SSH."""

    def __init__(
        self,
        host: str,
        ssh_options: str,
        remote_file: str,
        piglot_solve: str,
        parameters: ParameterSet,
        output_dir: str,
        tmp_dir: str,
    ) -> None:
        self.host = host
        self.ssh_options = ssh_options
        self.remote_file = remote_file
        self.piglot_solve = piglot_solve

        # Check if we can connect to the remote host
        proc = subprocess.run(
            ['ssh'] + self.ssh_options.split() + [self.host, 'true'],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to connect to the remote host {self.host}. "
                f"SSH command output: {proc.stderr.decode().strip()}"
            )

        # Check if piglot-solve is installed on the remote host
        proc = subprocess.run(
            ['ssh'] + self.ssh_options.split() + [self.host, 'which', self.piglot_solve],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"piglot-solve is not installed on the remote host {self.host}."
            )

        # Read the solver file from the remote host to figure out the output fields
        with tempfile.TemporaryDirectory() as dummy_dir:
            # Copy the remote file to a temporary directory
            if not self.__copy_from_remote([self.remote_file], dummy_dir):
                raise RuntimeError(
                    f"Failed to copy remote configuration file {self.host}:{self.remote_file}."
                )
            # Read the solver file
            config = parse_solver_file(os.path.join(dummy_dir, os.path.basename(self.remote_file)))
            solver = read_solver(config['solver'], parameters, output_dir)
            output_fields = solver.get_output_fields()

        super().__init__(output_fields, parameters, output_dir, tmp_dir, 'none')

    def __copy_from_remote(self, remote_paths: List[str], local_path: str) -> bool:
        """Copy files from the remote host to the local path.

        Parameters
        ----------
        remote_paths : List[str]
            List of remote paths to copy.
        local_path : str
            Local path to copy the files to.

        Returns
        -------
        bool
            Whether the copy was successful.
        """
        options = (
            ['scp']
            + self.ssh_options.split()
            + [f"{self.host}:{path}" for path in remote_paths]
            + [local_path]
        )
        result = subprocess.run(options, capture_output=True, check=False)
        return result.returncode == 0

    def __run_remote(self, params: List[str], tmp_dir: str) -> Dict[str, OutputResult]:
        """Run the remote solver with the given parameters.

        Parameters
        ----------
        params : List[str]
            List of parameter names to extract.
        tmp_dir : str
            Path to the temporary directory.

        Returns
        -------
        Dict[str, OutputResult]
            Extracted output results.
        """
        options = (
            ['ssh']
            + self.ssh_options.split()
            + [self.host, self.piglot_solve, self.remote_file, '--wait_reply', '--parameters']
            + params
        )
        proc = subprocess.Popen(
            options,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )

        # Search for the beginning of the output
        for line in iter(proc.stdout.readline, b''):
            line = line.decode('utf-8').strip()
            if line.startswith(f"Output results for {self.remote_file}:"):
                num_results = int(line.split(':')[-1].split()[0].strip())
                break
        else:
            raise RuntimeError(
                f"Failed to find output results in the remote solver output for {self.remote_file}."
            )

        # Read each output result location
        result_paths = {}
        for _ in range(num_results):
            line = proc.stdout.readline().decode('utf-8').strip()
            if not line:
                raise RuntimeError("Unexpected end of output while reading results.")
            field_name, field_file = line.split(':', 1)
            result_paths[field_name.strip()] = field_file.strip()

        # Sanitise the termination message
        end_msg = proc.stdout.readline().decode('utf-8').strip()
        if end_msg != f"{num_results} results dumped":
            raise RuntimeError(
                f"Unexpected termination message from remote solver: '{end_msg}'."
            )

        # Copy the output files from the remote host and read them
        if not self.__copy_from_remote(list(result_paths.values()), tmp_dir):
            raise RuntimeError(
                f"Failed to copy output files from the remote host {self.host}."
            )
        results = {
            name: OutputResult.read(
                os.path.join(tmp_dir, os.path.basename(path)),
            ) for name, path in result_paths.items()
        }

        # Send a termination message to the remote process
        proc.communicate(input=b'\n')
        if proc.wait() != 0:
            raise RuntimeError(
                f"Remote solver {self.remote_file} failed with exit code {proc.returncode}."
            )
        return results

    def _solve(self, values: np.ndarray, concurrent: bool) -> Dict[str, OutputResult]:
        """Internal solver for the prescribed problems.

        Parameters
        ----------
        values : array
            Current parameters to evaluate.
        concurrent : bool
            Whether this run may be concurrent to another one (so use unique file names).

        Returns
        -------
        Dict[str, OutputResult]
            Evaluated results for each output field.
        """
        # Resolve tmp directory: use unique directory if concurrent
        tmp_dir = f'{self.tmp_dir}_{self.parameters.hash(values)}' if concurrent else self.tmp_dir
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

        # Run the solver
        param_dict = self.parameters.to_dict(values)
        results = self.__run_remote([str(f) for f in param_dict.values()], tmp_dir)

        # Sanitise output fields
        for field in self.output_fields:
            if field not in results:
                raise ValueError(f"Missing output field '{field}'.")
        for field in results:
            if field not in self.output_fields:
                raise ValueError(f"Unknown output field '{field}'.")

        # Cleanup concurrent temporary directory before returning
        if concurrent:
            shutil.rmtree(tmp_dir)
        return results

    @classmethod
    def read(
        cls,
        config: Dict[str, Any],
        parameters: ParameterSet,
        output_dir: str,
    ) -> RemoteSolver:
        """Read the solver from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.
        parameters : ParameterSet
            Parameter set for this problem.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        RemoteSolver
            Solver to use for this problem.
        """
        # Mandatory fields
        if 'host' not in config:
            raise ValueError("Missing 'host' in the configuration.")
        if 'remote_file' not in config:
            raise ValueError("Missing 'remote_file' in the configuration.")
        host = config['host']
        remote_file = config['remote_file']
        # Optional fields
        ssh_options = config.pop('ssh_options', '')
        piglot_solve = config.pop('piglot_solve', 'piglot-solve')
        tmp_dir = os.path.join(output_dir, config.pop('tmp_dir', 'tmp'))
        return cls(
            host=host,
            ssh_options=ssh_options,
            remote_file=remote_file,
            piglot_solve=piglot_solve,
            parameters=parameters,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
        )
