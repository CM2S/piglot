<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/media/logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/media/logo.svg">
    <img alt="Hashnode logo" src="docs/source/media/logo.svg" width="250">
  </picture>
</div>

A Python package for the optimisation of numerical responses.


## Introduction

Welcome to `piglot`, a Python tool taylored for the automated optimisation of responses from numerical solvers.
We aim at providing a simple and user-friendly interface which is also easily extendable, allowing intergration with other solvers within the community.
Whether you're working on structural analysis, material modelling, fluid dynamics, control systems or astrophysics (to name a few) using, for instance, finite element analysis, spectral methods or Monte Carlo methods, `piglot` provides a versatile solution for solving inverse problems.
The primary emphasis is on derivative-free optimisation, ensuring compatibility with black-box solvers in scenarios where gradient information is not available, and cases where the function evaluations may be noisy.

* **Integration with solvers:** We provide an extensible interface for coupling with physics solvers. As long as your solver can return a time-response for the fields you are interested, you can optimise it with `piglot`.
* **Optimisation algorithms:** Off the shelf, there are several optimisers included in the package. Among them, we highlight our fully-fledged Bayesian optimisation (based on [BoTorch](https://botorch.org/)) that supports optimising stochastic and composite objectives and is highly customisable. Additional methods can also be easily implemented within `piglot`.
* **Visualisation tools:** You can use the builtin tool `piglot-plot` to visualise the results of the optimisation. There are native plotting utilities for the optimised responses, the parameter history, objective history and, for supported solvers, live plotting of the currently running case. Also, an animation of the optimisation process can be exported.


## Getting started

We provide some examples to get you started with `piglot`.
There are two modes of operation available: running using the given `piglot` and `piglot-plot` tools and configuration files, or building the optimisation problem in a Python script.

### Using configuration files

We use YAML configuration files to specify the optimisation problem to solve.
This is the simplest form of using `piglot` and is the recommended approach unless you have a strong motive to use Python scripts (described [here](#using-python-scripts)).
A simple analytical curve fitting problem is included to showcase how to use configuration files.
In this case, we fit a quadratic expression of the type $f(x) = a x^2$.
As a reference, a numerically generated reference from the expression $f(x) = 2 x^2$ is used (provided in the `examples/sample_curve_fitting/reference_curve.txt` file).
We want to find the value for $a$ that better fits our reference (it should be 2).
The configuration file for this example is:
```yaml
iters: 10

optimiser: botorch

parameters:
  a: [1, 0, 4]

objective:
  name: fitting
  solver:
    name: curve
    cases:
      'case_1':
        expression: <a> * x ** 2
        parametric: x
        bounds: [-5, 5]
        points: 100
  references:
    'reference_curve.txt':
      prediction: ['case_1']
```
You can find this file in `examples/sample_curve_fitting/config.yaml`
We run 10 iterations using the `botorch` optimiser (our interface for Bayesian optimisation), and set the parameter `a` for optimisation with bounds `[0,4]` and initial value 1.
Our optimisation objective is the fitting of an analytical curve, with the expression `<a> * x ** 2`.
The notation `<a>` indicates that this parameter should be optimised.
We also define a parameterisation using the variable $x$, where we sample the function between `[-5,5]` with 100 points.
Finally, we compare this generated response (with the label `case_1`) with our reference, given from the file `reference_curve.txt`

To run this example, open a terminal inside the `piglot` repository, enter the `examples/sample_curve_fitting` directory and run piglot with the given configuration file
```bash
cd examples/sample_curve_fitting
piglot config.yaml
```
You should see an output similar to
```
BoTorch: 100%|██████████████████████████████████████████████████████| 10/10 [00:00<00:00, 17.66it/s, Loss: 8.8505e-08]
Completed 10 iterations in 0.56614s
Best loss:  8.85050592e-08
Best parameters
- a:     1.999508
```
As you can see, piglot correctly identifies the `a` parameter close to the expected value of 2, and the error of the fitting is in the order of $10^{-8}$.
In addition to these outputs, `piglot` creates an output directory, with the same name of the configuration file (minus the extension), where it stores the optimisation data.

To visualise the optimisation results, use the `piglot-plot` utility.
In the same directory, run
```bash
piglot-plot best config.yaml
```
Which will display the best observed value for the optimisation problem.
You should see the following output in the terminal
```
Best run:
Start Time /s    0.587397
Run Time /s      0.004439
a                1.999508
Name: 18, dtype: object
Hash: 2313718f75bc0445aa71df7d6d4e50ba82ad593d65f3762efdcbed01af338e30
Objective:  8.85050592e-08
```
The script will also plot the best observed response, and its comparison with the reference response: 
![Best case plot](docs/source/simple_example/best.svg)

Now, try running (this may take some time)
```bash
piglot-plot animation config.yaml
```
This generates an animation for all the function evaluations that have been made throughout the optimisation procedure.
You can find the `.gif` file(s) inside the output directory, which should give something like:
![Best case plot](docs/source/simple_example/animation.gif)


### Using Python scripts

Another way of using `piglot` is via its package and Python modules.
This approach may offer increase flexibility in the setup of the optimisation problem, at the cost of increased complexity and verbosity.
A sample script equivalent to the configuration file for the problem described in [the previous section](#using-configuration-files) is provided in `examples/sample_curve_fitting/config.py`, given by:
```python
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
```
Run with
```bash
python config.py
```
Example output
```
BoTorch: 100%|██████████████████████████████████████████████████████| 10/10 [00:00<00:00, 16.75it/s, Loss: 8.9167e-08]
Completed 10 iterations in 0.59692s
Best loss:  8.91673999e-08
Best parameters
- a:     1.999506
Optimal value: 8.916739991036405e-08
Optimal parameters: [1.99950592]
```


### Installation

You can install `piglot` by only installing the main scripts or as a standard Python package.
If you only intend to use the `piglot` and `piglot-plot` binaries, we strongly recommend the first option, as it avoids having to manage the dependencies in your Python environment.
However, if you wish to use the tools in the `piglot` package, you may have to resort to the second option.
Currently, we only support Python 3.9 onwards.

#### Option 1: Install binaries

This option is recomended for end-users that only need to interact with the provided `piglot` and `piglot-plot` scripts.
We use [`pipx`](https://github.com/pypa/pipx) to install the package in an isolated environment with the required dependencies (we recommend reading the pipx documentation to check the advantages of using this approach).
  1. Install `pipx` in your system using the instructions [here](https://github.com/pypa/pipx#install-pipx)
  2. Clone the `piglot` repository.
  3. Open your favourite terminal, change directory to the recently cloned `piglot` repository and run: `pipx install .`
  4. Confirm the package is correctly installed by calling the `piglot` and `piglot-plot` executables.


#### Option 2: Install package

We recommend this option for users aiming to use the `piglot` package directly.
Note that this option also provides the `piglot` and `piglot-plot` scripts, but requires manually handling the installation environment.
  1. Clone the `piglot` repository.
  2. Open your favourite terminal, change directory to the recently cloned `piglot` repository and run: `pip install .`
  3. Confirm the package is correctly installed by calling the `piglot` and `piglot-plot` executables.

