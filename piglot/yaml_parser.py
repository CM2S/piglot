import os
import numpy as np
import sympy
import pandas as pd
from yaml import safe_load
from yaml.parser import ParserError
from yaml.scanner import ScannerError
import piglot
from piglot.losses import MixedLoss, Range, Minimum, Maximum, Slope
from piglot.parameter import ParameterSet, DualParameterSet
from piglot.optimisers.optimiser import StoppingCriteria
from piglot.objective import AnalyticalObjective
from piglot.links import LinksCase, Reaction, OutFile, LinksLoss, CompositeLinksLoss



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



def parse_loss_filter(file, filt_data):
    # Parse the simple specification
    if isinstance(filt_data, str):
        name = filt_data
        kwargs = {}
    else:
        # Normal specification
        if not isinstance(filt_data, dict):
            raise RuntimeError(f"Failed to parse the loss filter for case {file}")
        if not 'name' in filt_data:
            raise RuntimeError(f"Loss filter name not found for file {file}")
        name = filt_data.pop("name")
        kwargs = {args: str_to_numeric(data) for args, data in filt_data.items()}
    # Known filters
    filters = {"range": Range, "minimum": Minimum, "maximum": Maximum}
    if name not in filters:
        raise RuntimeError(f"Unknown loss filter name for file {file}")
    return filters[name](**kwargs)
        


def parse_loss_modifier(file, mod_data):
    # Parse the simple specification
    if isinstance(mod_data, str):
        name = mod_data
        kwargs = {}
    else:
        # Normal specification
        if not isinstance(mod_data, dict):
            raise RuntimeError(f"Failed to parse the loss modifier for case {file}")
        if not 'name' in mod_data:
            raise RuntimeError(f"Loss modifier name not found for file {file}")
        name = mod_data.pop("name")
        kwargs = {args: str_to_numeric(data) for args, data in mod_data.items()}
    # Known modifiers
    modifiers = {"slope": Slope}
    # TODO: weightings require more work
    if name not in modifiers:
        raise RuntimeError(f"Unknown loss modifier name for file {file}")
    return modifiers[name](**kwargs)



def parse_loss(file, loss_data):
    # Parse the simple specification: only the loss name is given
    if isinstance(loss_data, str):
        return piglot.loss(loss_data)
    # Parse the detailed specification
    if not isinstance(loss_data, dict):
        raise RuntimeError(f"Failed to parse the loss field for case {file}")
    if not 'name' in loss_data:
        raise RuntimeError(f"Loss name not found for file {file}")
    # Build the loss
    name = loss_data.pop("name")
    # Particular case for mixed losses
    if name == "mixed":
        mixed_loss = MixedLoss()
        if "losses" not in loss_data:
            raise RuntimeError(f"Missing losses field in mixed loss of file {file}")
        # Recursively parse nested losses
        losses = loss_data.pop("losses")
        for loss in losses:
            # Extract mixture ratio
            ratio = 1.0 / len(losses)
            if isinstance(loss, dict) and "ratio" in loss:
                ratio = float(loss.pop("ratio"))
            mixed_loss.add_loss(parse_loss(file, loss), ratio)
        return mixed_loss
    # Extract filters and modifiers from the fields
    filters = []
    modifiers = []
    if "filters" in loss_data:
        filt_data = loss_data.pop("filters")
        if isinstance(filt_data, str):
            filters = [parse_loss_filter(file, filt_data)]
        else:
            filters = [parse_loss_filter(file, data) for data in filt_data]
    if "modifiers" in loss_data:
        mod_data = loss_data.pop("modifiers")
        if isinstance(mod_data, str):
            modifiers = [parse_loss_modifier(file, mod_data)]
        else:
            modifiers = [parse_loss_modifier(file, data) for data in mod_data]
    return piglot.loss(name, filters=filters, modifiers=modifiers, **loss_data)



def parse_reference_file(file, index, reference):
    # Parse the simple specification: only the path to the file with 2 columns is given
    if isinstance(reference, str):
        return np.genfromtxt(reference)[:, 0:2]
    # Parse the detailed specification
    if not isinstance(reference, dict):
        raise RuntimeError(f"Failed to parse the reference for field {index + 1} of case {file}")
    if not 'file' in reference:
        raise RuntimeError(f"Reference file not given for field {index + 1} of case {file}")
    # Parse optional arguments
    x_col = optional_dict(reference, "x_col", 1, int) - 1
    y_col = optional_dict(reference, "y_col", 2, int) - 1
    x_scale = optional_dict(reference, "x_scale", 1.0, float)
    y_scale = optional_dict(reference, "y_scale", 1.0, float)
    x_offset = optional_dict(reference, "x_offset", 0.0, float)
    y_offset = optional_dict(reference, "y_offset", 0.0, float)
    # Load the data and perform the given transformation
    data = np.genfromtxt(reference["file"])[:, [x_col, y_col]]
    data[:, 0] = x_offset + x_scale * data[:, 0]
    data[:, 1] = y_offset + y_scale * data[:, 1]
    return data



def parse_field(file, index, fields):
    # Initial sanity checks
    if not isinstance(fields, dict):
        raise RuntimeError(f"Failed to parse the field {index + 1} of case {file}")
    if not 'type' in fields:
        raise RuntimeError(f"Output type not given for field {index + 1} of case {file}")
    if not 'reference' in fields:
        raise RuntimeError(f"Reference file not given for field {index + 1} of case {file}")
    if fields["type"] not in ["Reaction", "OutFile"]:
        raise RuntimeError(f"Invalid output type {fields['type']} for field {index + 1} of case {file}")
    # Extract type and reference
    if fields["type"] == "Reaction":
        if not 'field' in fields:
            raise RuntimeError(f"Missing reaction dimension for field {index + 1} of case {file}")
        group = optional_dict(fields, "group", 1, int)
        field = Reaction(fields["field"], group=group)
    elif fields["type"] == "OutFile":
        if not 'field' in fields:
            raise RuntimeError(f"Missing out file field for field {index + 1} of case {file}")
        i_elem = optional_dict(fields, "elem", None, int)
        i_gauss = optional_dict(fields, "gauss", None, int)
        x_field = optional_dict(fields, "x_field", "LoadFactor", str)
        field = OutFile(str_to_numeric(fields["field"]), i_elem=i_elem, i_gauss=i_gauss, x_field=str_to_numeric(x_field))
    reference = parse_reference_file(file, index, fields["reference"])
    return field, reference



def parse_case(file, case):
    # Initial sanity checks
    if not os.path.exists(file):
        raise RuntimeError(f"Input file {file} not found")
    if not 'loss' in case:
        raise RuntimeError(f"Loss keyword not found for case {file}")
    if not 'fields' in case:
        raise RuntimeError(f"Fields keyword not found for case {file}")
    if len(case["fields"]) < 1:
        raise RuntimeError(f"Need at least one field for case {file}")
    # Parse remaining items
    loss = parse_loss(file, case["loss"])
    fields = {}
    for index, field in enumerate(case["fields"]):
        key, value = parse_field(file, index, field)
        fields[key] = value
    # Build case
    return LinksCase(file, fields, loss)



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



def parse_optimiser(opt_config):
    # Parse the simple specification: optimiser name
    if isinstance(opt_config, str):
        return piglot.optimiser(opt_config)
    # Parse the detailed specification
    if not isinstance(opt_config, dict):
        raise RuntimeError("Failed to parse the optimiser")
    if not 'name' in opt_config:
        raise RuntimeError("Missing optimiser name")
    name = opt_config.pop("name")
    kwargs = {n: str_to_numeric(v) for n, v in opt_config.items()}
    return piglot.optimiser(name, **kwargs)



def parse_analytical_objective(objective_conf, parameters, output_dir):
    # Check for mandatory arguments
    if not 'expression' in objective_conf:
        raise RuntimeError("Missing Links binary location")
    return AnalyticalObjective(parameters, objective_conf['expression'], output_dir=output_dir)



def parse_test_function_objective(objective_conf, parameters, output_dir):
    # Check for mandatory arguments
    if not 'function' in objective_conf:
        raise RuntimeError("Missing test function")
    function = objective_conf.pop('function')
    return SyntheticObjective(parameters, function, output_dir, **objective_conf)



def parse_links_objective(objective_conf, parameters, output_dir):
    # Manually parse cases
    if not 'cases' in objective_conf:
        raise RuntimeError("Missing Links cases")
    cases_conf = objective_conf.pop("cases")
    cases = [parse_case(file, case) for file, case in cases_conf.items()]
    # Check for mandatory arguments
    if not 'links' in objective_conf:
        raise RuntimeError("Missing Links binary location")
    links_bin = objective_conf.pop("links")
    # Sanitise output directory, just in case it is passed
    if 'output_dir' in objective_conf:
        objective_conf.pop('output_dir')
    # Build Links objective instance with remaining arguments
    kwargs = {n: str_to_numeric(v) for n, v in objective_conf.items()}
    return LinksLoss(cases, parameters, links_bin, output_dir=output_dir, **kwargs)


def parse_links_cf_objective(objective_conf, parameters, output_dir):
    # Manually parse cases
    if not 'cases' in objective_conf:
        raise RuntimeError("Missing Links cases")
    cases_conf = objective_conf.pop("cases")
    cases = [parse_case(file, case) for file, case in cases_conf.items()]
    # Check for mandatory arguments
    if not 'links' in objective_conf:
        raise RuntimeError("Missing Links binary location")
    links_bin = objective_conf.pop("links")
    # Sanitise output directory, just in case it is passed
    if 'output_dir' in objective_conf:
        objective_conf.pop('output_dir')
    # Build Links objective instance with remaining arguments
    kwargs = {n: str_to_numeric(v) for n, v in objective_conf.items()}
    return CompositeLinksLoss(cases, parameters, links_bin, output_dir=output_dir, **kwargs)



def parse_objective(config, parameters, output_dir):
    if not 'name' in config:
        raise RuntimeError("Missing objective name")
    name = config.pop('name')
    # Delegate the objective
    objectives = {
        'analytical': parse_analytical_objective,
        'test_function': parse_test_function_objective,
        'links': parse_links_objective,
        'links_cf': parse_links_cf_objective,
    }
    if name not in objectives:
        raise RuntimeError(f"Unknown objective {name}. Must be one of {list(objectives.keys())}")
    return objectives[name](config, parameters, os.path.join(output_dir, name))



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
