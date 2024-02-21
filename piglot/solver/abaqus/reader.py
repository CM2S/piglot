####################################################################################################
'''
This script extracts the nodal data from the output database (.odb) file.

Tiago Pires, October 2023, Initial coding
'''
####################################################################################################

# Import the necessary modules
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

    input_file_list = [a for a in args if a.startswith("input_file=")]
    job_name_list = [a for a in args if a.startswith("job_name=")]
    step_name_list = [a for a in args if a.startswith("step_name=")]
    instance_name_list = [a for a in args if a.startswith("instance_name=")]

    # Checks if the input_file, job_name, step_name and instance_name are not empty
    input_file = input_file_list[0].replace('input_file=', '') \
        if input_file_list else None
    job_name = job_name_list[0].replace('job_name=', '') \
        if job_name_list else None
    step_name = step_name_list[0].replace('step_name=', '') \
        if step_name_list else None
    instance_name = instance_name_list[0].replace('instance_name=', '') \
        if instance_name_list else None

    variables['input_file'] = input_file
    variables['job_name'] = job_name
    variables['step_name'] = step_name
    variables['instance_name'] = instance_name

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


def field_location(i, output_variable, location):
    """It gets the node data of the specified node set.

    Parameters
    ----------
    i : int
        It is an int number, 0 or 1 in the case of the stresses and strains.
    output_variable : str
        Its the output variable (S, U, RF, E or LE) of the nodes.
    location : str
        Location of the current node set.

    Returns
    -------
    location_output_variable
        Location of the output variable.
    """

    if i in (0, 1):
        location_output_variable = output_variable.getSubset(region=location,
                                                             position=ELEMENT_NODAL)
    else:
        location_output_variable = output_variable.getSubset(region=location)

    return location_output_variable


def main():
    """Main function of the reader.py
    """

    variables = input_variables()

    # Data defined by the user
    job_name = variables["job_name"]  # Replace with the actual job name
    # Replace with the actual step name
    step_name = variables["step_name"]

    # Read the input file to check if the nlgeom setting is on or off
    inp_name = variables["input_file"]

    with codecs.open(inp_name, 'r', encoding='utf-8') as input_file:
        file_content = input_file.read()

        # Use a regular expression to find the nlgeom setting
        match = re.search(r'\*Step.*nlgeom=(\w+)', file_content)

        # Check if the match is found and extract the value
        if match:
            nlgeom_setting = match.group(1)
            nlgeom = 1 if nlgeom_setting.upper() == 'YES' else 0
        else:
            print("nlgeom setting not found in the input file.")
            sys.exit(1)  # Stop the script with an exit code

    if nlgeom == 0:
        variables_array = np.array(["S", "E", "U", "RF"])
    else:
        variables_array = np.array(["S", "LE", "U", "RF"])

    # Open the output database
    odb_name = job_name + ".odb"
    odb = openOdb(path=odb_name)

    # Create a variable that refers to the first step.
    step = odb.steps[step_name]

    for i, var in enumerate(variables_array):

        header_variable = "%s_%d"
        variable = var

        for set_name, location in odb.rootAssembly.nodeSets.items():

            file_name = file_name_func(set_name, var, inp_name)

            # Create a text file to save the output data
            with codecs.open(file_name, 'w', encoding='utf-8') as output_file:

                output_variable = step.frames[0].fieldOutputs[variable]

                # Create a variable that refers to the output variable of the node set. If the
                # field is S or E it extrapolates the data to the nodes, if the field is U or RF
                # the data is already on the nodes so it doesn't need to be specified.
                location_output_variable = field_location(i, output_variable, location)

                # Get the component labels
                component_labels = output_variable.componentLabels

                # Write the column headers dynamically based on the number of nodes and
                # output variable components
                header = "Frame " + " ".join(header_variable % (label, v.nodeLabel) for v in
                                             location_output_variable.values for label in
                                             component_labels) + "\n"
                output_file.write(header)

                for frame in step.frames:
                    output_variable = frame.fieldOutputs[variable]

                    # Create a variable that refers to the output_variable of the node
                    # set in the current frame.
                    location_output_variable = field_location(i, output_variable, location)

                    output_file.write("%d " % frame.frameId)
                    for v in location_output_variable.values:
                        output_file.write(" ".join("%.9f" % value for value in v.data))
                        output_file.write(" ")
                    output_file.write("\n")

    odb.close()


# Run the main function
if __name__ == "__main__":

    main()
