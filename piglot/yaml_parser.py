import os
import numpy as np
import piglot
from piglot.losses import MixedLoss
from piglot.parameter import ParameterSet
from piglot.links import LinksCase, Reaction, OutFile, extract_parameters



def optional_dict(d, field, default, conv):
    return conv(d[field]) if field in d else default



def str_to_numeric(data):
    if isinstance(data, list):
        return [str_to_numeric(a) for a in data]
    try:
        data = float(data)
    except ValueError:
        return data
    if int(data) == data:
        return int(data)
    return data



def parse_loss(file, loss_data):
    # Parse the simple specification: only the loss name is given
    if isinstance(loss_data, str):
        return piglot.loss(loss_data)
    # Parse the detailed specification
    if not isinstance(loss_data, dict):
        raise Exception(f"Failed to parse the loss field for case {file}")
    if not 'name' in loss_data:
        raise Exception(f"Loss name not found for file {file}")
    # Build the loss
    name = loss_data.pop("name")
    # Particular case for mixed losses
    if name == "mixed":
        loss = MixedLoss()
        # TODO: build the mixed loss
        return loss
    # Extract filters and modifiers from the fields
    filters = []
    if "filters" in loss_data:
        filt_data = loss_data.pop("filters")
    modifiers = []
    if "modifiers" in loss_data:
        mod_data = loss_data.pop("modifiers")
    # TODO: manually parse filters and modifiers
    return piglot.loss(name, filters=filters, modifiers=modifiers, **loss_data)



def parse_reference_file(file, index, reference):
    # Parse the simple specification: only the path to the file with 2 columns is given
    if isinstance(reference, str):
        return np.genfromtxt(reference)[:, 0:2]
    # Parse the detailed specification
    if not isinstance(reference, dict):
        raise Exception(f"Failed to parse the reference for field {index + 1} of case {file}")
    if not 'file' in reference:
        raise Exception(f"Reference file not given for field {index + 1} of case {file}")
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
        raise Exception(f"Failed to parse the field {index + 1} of case {file}")
    if not 'type' in fields:
        raise Exception(f"Output type not given for field {index + 1} of case {file}")
    if not 'reference' in fields:
        raise Exception(f"Reference file not given for field {index + 1} of case {file}")
    if fields["type"] not in ["Reaction", "OutFile"]:
        raise Exception(f"Invalid output type {fields['type']} for field {index + 1} of case {file}")
    # Extract type and reference
    if fields["type"] == "Reaction":
        if not 'field' in fields:
            raise Exception(f"Missing reaction dimension for field {index + 1} of case {file}")
        group = optional_dict(fields, "group", 1, int)
        field = Reaction(fields["field"], group=group)
    elif fields["type"] == "OutFile":
        if not 'field' in fields:
            raise Exception(f"Missing out file field for field {index + 1} of case {file}")
        i_elem = optional_dict(fields, "elem", None, int)
        i_gauss = optional_dict(fields, "gauss", None, int)
        x_field = optional_dict(fields, "x_field", "LoadFactor", str)
        field = OutFile(fields["field"], i_elem=i_elem, i_gauss=i_gauss, x_field=x_field)
    reference = parse_reference_file(file, index, fields["reference"])
    return field, reference



def parse_case(file, case):
    # Initial sanity checks
    if not os.path.exists(file):
        raise Exception(f"Input file {file} not found")
    if not 'loss' in case:
        raise Exception(f"Loss keyword not found for case {file}")
    if not 'fields' in case:
        raise Exception(f"Fields keyword not found for case {file}")
    if len(case["fields"]) < 1:
        raise Exception(f"Need at least one field for case {file}")
    # Parse remaining items
    loss = parse_loss(file, case["loss"])
    fields = {}
    for index, field in enumerate(case["fields"]):
        key, value = parse_field(file, index, field)
        fields[key] = value
    # Build case
    return LinksCase(file, fields, loss)



def parse_parameters(config):
    if "parameters" in config:
        params_conf = config["parameters"]
        parameters = ParameterSet()
        for name, spec in params_conf.items():
            int_spec = [float(s) for s in spec]
            parameters.add(name, *int_spec)
    else:
        if len(config["cases"]) != 1:
            raise Exception("Cannot find a suitable input file to extract parameters from!")
        parameters = extract_parameters(list(config["cases"].keys())[0])
    return parameters



def parse_optimiser(opt_config):
    # Parse the simple specification: optimiser name
    if isinstance(opt_config, str):
        return piglot.optimiser(opt_config)
    # Parse the detailed specification
    if not isinstance(opt_config, dict):
        raise Exception(f"Failed to parse the optimiser")
    if not 'name' in opt_config:
        raise Exception(f"Missing optimiser name")
    name = opt_config.pop("name")
    kwargs = {n: str_to_numeric(v) for n, v in opt_config.items()}
    return piglot.optimiser(name, **kwargs)
