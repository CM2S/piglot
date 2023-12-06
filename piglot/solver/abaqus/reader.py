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


# Function to run the input python script


def read_input():
    args = [a for a in sys.argv if a.startswith("input_file=")][0].replace('input_file=', '')
    variables = {}
    with codecs.open(args, "r", encoding='utf-8') as file:
        for line in file:
            if "=" in line:
                var_name, var_value = line.split("=")
                var_name = var_name.strip()
                var_value = var_value.strip()

                if var_value.startswith("'") and var_value.endswith("'"):
                    var_value = str(var_value[1:-1])
                else:
                    var_value = int(var_value)

                variables[var_name] = var_value
    return variables


# Function to define the file name


def file_name_func(set_name, variable_name, inp_name):
    file_name = "{}_{}_{}.txt".format(os.path.splitext(inp_name)[0], set_name, variable_name)
    return file_name


# Main function


def main():

    variables = read_input()

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

                # Create a variable that refers to the output variable of the node set.
                # Se forem tensoes ou deformacoes extrapola para os nos, se forem deslocamentos ou reacoes fica so nos nos
                if i in (0, 1):
                    location_output_variable = output_variable.getSubset(
                        region=location, position=ELEMENT_NODAL)
                else:
                    location_output_variable = output_variable.getSubset(
                        region=location)

                # Get the component labels
                component_labels = output_variable.componentLabels

                # Write the column headers dynamically based on the number of nodes and output variable
                # components
                header = "Frame " + " ".join(header_variable % (label, v.nodeLabel)
                                            for v in location_output_variable.values for label in
                                            component_labels) + "\n"
                output_file.write(header)

                for frame in step.frames:
                    output_variable = frame.fieldOutputs[variable]

                    # Create a variable that refers to the output_variable of the node
                    # set in the current frame.
                    if i in (0, 1):
                        location_output_variable = output_variable.getSubset(
                            region=location, position=ELEMENT_NODAL)
                    else:
                        location_output_variable = output_variable.getSubset(
                            region=location)

                    output_file.write("%d " % frame.frameId)
                    for v in location_output_variable.values:
                        output_file.write(" ".join("%.9f" %
                                        value for value in v.data))
                        output_file.write(" ")
                    output_file.write("\n")

    odb.close()


# Run the main function
if __name__ == "__main__":

    main()
