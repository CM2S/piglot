<div align="center">
  <img width="250" src="docs/source/logo.svg" alt="piglot logo" />
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

