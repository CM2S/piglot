piglot
======

.. image:: media/logo.svg
   :class: only-light
   :width: 300
   :align: center

|

.. image:: https://img.shields.io/badge/GitHub-black?logo=GitHub
   :target: https://github.com/CM2S/piglot
   :alt: GitHub

.. image:: https://github.com/CM2S/piglot/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/CM2S/piglot/actions/workflows/test.yaml
   :alt: Unit and integration testing

.. image:: https://img.shields.io/pypi/v/piglot
   :target: https://pypi.org/project/piglot/
   :alt: PyPI - Version

.. image:: https://img.shields.io/github/license/CM2S/piglot
   :target: https://github.com/CM2S/piglot/blob/main/LICENSE
   :alt: GitHub License

.. image:: https://img.shields.io/codefactor/grade/github/CM2S/piglot
   :target: https://www.codefactor.io/repository/github/cm2s/piglot
   :alt: CodeFactor Grade

Welcome to :code:`piglot`, a Python tool taylored for the automated optimisation of responses from numerical solvers.
We aim at providing a simple and user-friendly interface which is also easily extendable, allowing intergration with other solvers within the community.
Whether you're working on structural analysis, material modelling, fluid dynamics, control systems or astrophysics (to name a few) using, for instance, finite element analysis, spectral methods or Monte Carlo methods, :code:`piglot` provides a versatile solution for solving inverse problems.
The primary emphasis is on derivative-free optimisation, ensuring compatibility with black-box solvers in scenarios where gradient information is not available, and cases where the function evaluations may be noisy.
We highlight:

- **Integration with solvers:** We provide an extensible interface for coupling with physics solvers. As long as your solver can return a time-response for the fields you are interested, you can optimise it with :code:`piglot`.
- **Optimisation algorithms:** Off the shelf, there are several optimisers included in the package. Among them, we highlight our fully-fledged Bayesian optimisation (based on `BoTorch <https://botorch.org/>`_) that supports optimising stochastic and composite objectives and is highly customisable. Additional methods can also be easily implemented within :code:`piglot`.
- **Visualisation tools:** You can use the builtin tool :code:`piglot-plot` to visualise the results of the optimisation. There are native plotting utilities for the optimised responses, the parameter history, objective history and, for supported solvers, live plotting of the currently running case. Also, an animation of the optimisation process can be exported.

Feel free to explore, contribute, and optimize with :code:`piglot`!


Installation
------------

You can install :code:`piglot` by only installing the main scripts or as a standard Python package.
If you only intend to use the :code:`piglot` and :code:`piglot-plot` binaries, we strongly recommend the first option, as it avoids having to manage the dependencies in your Python environment.
However, if you wish to use the tools in the :code:`piglot` package, you may have to resort to the second option.
Currently, we require Python 3.9 onwards.

Option 1: Install binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is recommended for end-users that only need to interact with the provided :code:`piglot` and :code:`piglot-plot` scripts.
We use `pipx <https://github.com/pypa/pipx>`_ to install the package in an isolated environment with the required dependencies (we recommend reading the pipx documentation to check the advantages of using this approach).
  1. Install :code:`pipx` in your system using the instructions `here <https://github.com/pypa/pipx#install-pipx>`_;
  2. In your favourite terminal, run: :code:`pipx install piglot`;
  3. Confirm the package is correctly installed by calling the :code:`piglot` and :code:`piglot-plot` executables.


Option 2: Install package
^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend this option for users aiming to use the :code:`piglot` package directly.
Note that this option also provides the :code:`piglot` and :code:`piglot-plot` scripts, but requires manual handling of the installation environment.
  1. In your favourite terminal, run: :code:`pip install piglot`;
  2. Confirm the package is correctly installed by calling the :code:`piglot` and :code:`piglot-plot` executables.

Installing additional optimisers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also support some optional external optimisers, which are not automatically installed along with :code:`piglot` to reduce the number of dependencies and the installation cost.
You can either install them along with :code:`piglot`, or manually using your package manager.
Their detection is done at runtime and, if not installed, an error will be raised.
Currently, the following optional optimisers are supported:
- :code:`lipo` - LIPO optimiser
- :code:`geneticalgorithm` - Genetic algorithm
- :code:`pyswarms` - Particle swarm optimiser

These can be installed directly from PyPI (with the package names above).
If you wish to install :code:`piglot` with one of these optimisers (which may be required when using a :code:`pipx` install), you can run the following commands:
- :code:`pip install piglot[lipo]` for the LIPO optimiser
- :code:`pip install piglot[genetic]` for the Genetic algorithm
- :code:`pip install piglot[pso]` for the Particle swarm optimiser optimiser

To simultaneously install more than one optimiser, for instance, the LIPO and the Particle swarm optimisers, run :code:`pip install piglot[lipo,pso]`.
If you wish to install all optimisers at once, you can run :code:`pip install piglot[full]`.


.. toctree::
   :maxdepth: 3
   :caption: Examples
   :hidden:

   examples/fitting
   examples/design
   examples/analytical


.. toctree::
   :maxdepth: 1
   :caption: Input file templates
   :hidden:

   templates/input_file
   templates/optimisers
   templates/solvers


.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   Reference <_autosummary/piglot>
