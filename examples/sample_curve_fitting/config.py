import os
import shutil
from piglot.parameter import ParameterSet
from piglot.solver.solver import Case
from piglot.solver.curve.solver import CurveSolver
from piglot.solver.curve.fields import CurveInputData, Curve
from piglot.objectives.fitting import Reference, MSE
from piglot.objectives.fitting import FittingObjective, FittingSolver
from piglot.optimisers.botorch.bayes import BayesianBoTorch

# Set up output and temporary directories
output_dir = 'config'
tmp_dir = os.path.join(output_dir, 'tmp')
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Set up optimisation parameters
parameters = ParameterSet()
parameters.add('a', 1.0, 0.0, 4.0)

# Set up the reference
reference = Reference('reference_curve.txt', ['case_1'], output_dir)

# Set up the solver to use
input_data = CurveInputData('case_1', '<a> * x ** 2', 'x', (-5.0, 5.0), 100)
case_1 = Case(input_data, {'case_1': Curve()})
solver = CurveSolver([case_1], parameters, output_dir, tmp_dir=tmp_dir)

# Set up the fitting objective
references = {reference: ['case_1']}
fitting_solver = FittingSolver(solver, references)
objective = FittingObjective(parameters, fitting_solver, output_dir, MSE())

# Set up the optimiser and run optimisation
optimiser = BayesianBoTorch(objective)
value, params = optimiser.optimise(10, parameters, output_dir)
print(f"Optimal value: {value}")
print(f"Optimal parameters: {params}")
