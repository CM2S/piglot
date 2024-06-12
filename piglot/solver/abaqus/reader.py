"""Module to extract the nodal data from the output database (.odb) file
Note: This script has older python syntax because it is used in Abaqus, which uses Python 2.7.
"""
import re
import os
import sys
import codecs
import numpy as np
from odbAccess import *


def input_variables():
    """Reads the input file.

    Returns
    -------
    variables
        Variables to use in this problem
    """
    args = sys.argv
    variables = {}

    variable_names = [
        'input_file',
        'job_name',
        'step_name',
        'instance_name',
        'set_name',
        'field',
        'x_field',
    ]

    for var_name in variable_names:
        var_list = [a for a in args if a.startswith(var_name + "=")]
        variables[var_name] = var_list[0].replace(var_name + '=', '')

    return variables


def file_name_func(set_name, variable_name, inp_name):
    """Defines the txt file name of the extracted data.

    Parameters
    ----------
    set_name : str
        name of the set
    variable_name : str
        name of the field variable
    inp_name : str
        name of the input file name

    Returns
    -------
    file_name
        name of the txt file
    """
    file_name = "{}_{}_{}.txt".format(os.path.splitext(inp_name)[0], set_name, variable_name)
    return file_name


def field_location(i, variables_array, output_variable, location):
    """It gets the node data of the specified node set. Create a variable that refers to the output 
    variable of the node set. If the field is S or E it extrapolates the data to the nodes, if the 
    field is U or RF the data is already on the nodes so it doesn't need extrapolation.

    Parameters
    ----------
    i : int
        It is an int number that represents the index of the variables_array.
    variables_array : list
        It is a list that contains the field variables (S, U, RF, E or LE)
    output_variable : str
        Its the output variable (S, U, RF, E or LE) of the nodes.
    location : str
        Location of the current node set.

    Returns
    -------
    location_output_variable
        Location of the output variable.
    """
    variable = variables_array[i]
    if variable in ['S', 'E', 'LE']:
        location_output_variable = output_variable.getSubset(region=location,
                                                             position=ELEMENT_NODAL)
    else:
        location_output_variable = output_variable.getSubset(region=location)
    return location_output_variable


def get_nlgeom_setting(inp_name):
    """It verifies if the 'nlgeom' setting is set to 'YES' in the input file.

    Parameters
    ----------
    inp_name : str
        The name of the input file.

    Returns
    -------
    int
        It returns 1 if the 'nlgeom' setting is set to 'YES' in the input file, otherwise it 
        returns 0.

    Raises
    ------
    ValueError
        Raises an error if the 'nlgeom' setting is not found in the input file.
    """
    with codecs.open(inp_name, 'r', encoding='utf-8') as input_file:
        file_content = input_file.read()
        match = re.search(r'\*Step.*nlgeom=(\w+)', file_content)
        if match:
            nlgeom_setting = match.group(1)
            return 1 if nlgeom_setting.upper() == 'YES' else 0

        raise ValueError("'nlgeom' setting not found in the input file.")

def check_nlgeom(nlgeom, field, x_field):
    """Checks if the user is trying to extract the logaritmic strain ('LE') when the 'nlgeom' is 
    OFF.

    Parameters
    ----------
    nlgeom : int
        It is an int number that represents the 'nlgeom' setting. If it is 1 the 'nlgeom' is ON,
        if it is 0 it is OFF.
    field : str
        Name of the y-axis field variable.
    x_field : str
        Name of the x-axis field variable.

    Raises
    ------
    ValueError
        Raises an error if the user is trying to extract the logaritmic strain ('LE') when the 
        'nlgeom' is OFF.
    """
    if nlgeom == 0 and (x_field == 'LE' or field == 'LE'):
        raise ValueError("'LE' is not allowed when nlgeom is OFF, use 'E' instead.")

def get_node_sets(instance_name, odb):
    """Gets the node sets of the instance. If the instance_name is None it gets the node sets of the
    assembly. If the instance_name is not None it gets the node sets of the instance specified by
    the user.

    Parameters
    ----------
    instance_name : str
        Name of the instance.
    odb : Odb
        An instance of the Odb class from the Abaqus scripting interface, representing the output
        database.

    Returns
    -------
    list
        List of the node sets of the instance.
    """
    if instance_name is not None:
        return odb.rootAssembly.instances[instance_name].nodeSets.items()

    return odb.rootAssembly.nodeSets.items()

def write_output_file(i, variables_array, variable, step, location, file_name):
    """Writes the output file with the nodal data of the specified node set.

    Parameters
    ----------
    i : int
        It is an int number that represents the index of the variables_array.
    variables_array : list
        It is a list that contains two field variables (S, U, RF, E or LE).
    step : str
        It is a string that represents the step of the output database.
    location : str
        It is a string that represents the location of the node set.
    file_name : str
        It is a string that represents the name of the output file.
    """
    with codecs.open(file_name, 'w', encoding='utf-8') as output_file:
        output_variable = step.frames[0].fieldOutputs[variable]
        location_output_variable = field_location(i, variables_array, output_variable, location)
        component_labels = output_variable.componentLabels
        # Write the column headers dynamically based on the number of nodes and output
        # variable components
        header = "Frame " + " ".join("%s_%d" % (label, v.nodeLabel)
                                        for v in location_output_variable.values
                                        for label in component_labels) + "\n"
        output_file.write(header)
        for frame in step.frames:
            output_variable = frame.fieldOutputs[variable]
            location_output_variable = field_location(i,
                                                        variables_array,
                                                        output_variable,
                                                        location)
            output_file.write("%d " % frame.frameId)
            for v in location_output_variable.values:
                output_file.write(" ".join("%.9f" % value for value in v.data))
                output_file.write(" ")
            output_file.write("\n")

def find_case_insensitive_key(key_name, keys_list):
    """
    Find the original key name in a list of keys, ignoring case sensitivity.
    
    Parameters
    ----------
    key_name : str
        The name of the key to find, case-insensitively.
    keys_list : list
        A list of keys (strings) to search through.
    
    Returns
    -------
    str
        The original key name from the list that matches the provided key_name, ignoring case.
    
    Raises
    ------
    ValueError
        If the key_name is not found in the keys_list, ignoring case.
    """
    keys_list_upper = [key.upper() for key in keys_list]
    key_name_upper = key_name.upper()
    if key_name_upper not in keys_list_upper:
        raise ValueError("{} not found.".format(key_name))
    return keys_list[keys_list_upper.index(key_name_upper)]

def main():
    """Main function to extract the nodal data from the output database (.odb) file.
    """
    variables = input_variables()

    nlgeom = get_nlgeom_setting(variables["input_file"])
    check_nlgeom(nlgeom, variables["field"], variables["x_field"])

    variables_array = np.array([variables["field"], variables["x_field"]])

    # Open the output database
    odb_name = variables["job_name"] + ".odb"
    odb = openOdb(path=odb_name)

    # Create a variable that refers to the respective step
    step = odb.steps[find_case_insensitive_key(variables["step_name"], list(odb.steps.keys()))]
    instance_name = find_case_insensitive_key(
        variables["instance_name"],
        list(odb.rootAssembly.instances.keys()),
    )
    nodeset_name = find_case_insensitive_key(
        variables["set_name"],
        list(odb.rootAssembly.instances[instance_name].nodeSets.keys()),
    )

    node_sets = get_node_sets(instance_name, odb)

    for i, var in enumerate(variables_array):
        for set_name, location in node_sets:
            if set_name == str(nodeset_name):
                file_name = file_name_func(set_name, var, variables["input_file"])
                write_output_file(i, variables_array, var, step, location, file_name)

    odb.close()


# Run the main function
if __name__ == "__main__":
    main()
