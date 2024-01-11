import os
import os.path
import shutil
import numpy as np
import sympy
import pandas as pd
from yaml import safe_load
from yaml.parser import ParserError
from yaml.scanner import ScannerError
import piglot
from piglot.parameter import ParameterSet, DualParameterSet
from piglot.optimisers.optimiser import StoppingCriteria


def optional_dict(d, field, default, conv):
    return conv(d[field]) if field in d else default


def str_to_numeric(data):
    try:
        data = float(data)
    except (TypeError, ValueError):
        return data
    if int(data) == data:
        return int(data)
    return data


def parse_parameters(config):
    params_conf = config["parameters"]
    parameters = DualParameterSet() if "output_parameters" in config else ParameterSet()
    for name, spec in params_conf.items():
        int_spec = [float(s) for s in spec]
        parameters.add(name, *int_spec)
    if "output_parameters" in config:
        symbs = sympy.symbols(list(params_conf.keys()))
        for name, spec in config["output_parameters"].items():
            parameters.add_output(name, sympy.lambdify(symbs, spec))
    # Fetch initial shot from another run
    if "init_shot_from" in config:
        with open(config["init_shot_from"], 'r') as f:
            source = safe_load(f)
        func_calls_file = os.path.join(source["output"], "func_calls")
        df = pd.read_table(func_calls_file)
        df.columns = df.columns.str.strip()
        min_series = df.iloc[df["Loss"].idxmin()]
        for param in parameters:
            if param.name in min_series.index:
                param.inital_value = min_series[param.name]
    return parameters


def parse_stop_criteria(config):
    def param_or_none(name, convert_type):
        return convert_type(config[name]) if name in config else None
    return StoppingCriteria(
        conv_tol=param_or_none("conv_tol", float),
        max_func_calls=param_or_none("max_func_calls", int),
        max_iters_no_improv=param_or_none("max_iters_no_improv", int),
        max_timeout=param_or_none("max_timeout", float),
    )


def parse_config_file(config_file):
    """Parses the YAML configuration file.

    Parameters
    ----------
    file : TextIOWrapper
        Configuration file object.

    Returns
    -------
    dict
        Dictionary with the YAML data.

    Raises
    ------
    RuntimeError
        When the YAML parsing fails.
    """
    try:
        with open(config_file, 'r', encoding='utf8') as file:
            config = safe_load(file)
    except (ParserError, ScannerError) as exc:
        raise RuntimeError("Failed to parse the config file: YAML syntax seems invalid.") from exc
    # Check required terms
    if 'iters' not in config:
        raise RuntimeError("Missing number of iterations from the config file")
    if 'objective' not in config:
        raise RuntimeError("Missing objective from the config file")
    if 'optimiser' not in config:
        raise RuntimeError("Missing optimiser from the config file")
    if 'parameters' not in config:
        raise RuntimeError("Missing parameters from the config file")
    # Add missing optional items
    if 'output' not in config:
        config['output'] = os.path.splitext(config_file)[0]
    if 'quiet' not in config:
        config["quiet"] = False
    elif config['quiet']:
        config["quiet"] = True
    return config
