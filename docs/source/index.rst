piglot
======

.. image:: media/logo.svg
   :class: only-light
   :width: 400
   :align: center

.. image:: media/logo_dark.svg
   :class: only-dark
   :width: 400
   :align: center

|

Welcome to :code:`piglot`, a Python tool taylored for the automated optimisation of responses from numerical solvers.
We aim at providing a simple and user-friendly interface which is also easily extendable, allowing intergration with other solvers within the community.
Whether you're working on structural analysis, material modelling, fluid dynamics, control systems or astrophysics (to name a few) using, for instance, finite element analysis, spectral methods or Monte Carlo methods, :code:`piglot` provides a versatile solution for solving inverse problems.
The primary emphasis is on derivative-free optimisation, ensuring compatibility with black-box solvers in scenarios where gradient information is not available, and cases where the function evaluations may be noisy.

- **Integration with solvers:** We provide an extensible interface for coupling with physics solvers. As long as your solver can return a time-response for the fields you are interested, you can optimise it with :code:`piglot`.
- **Optimisation algorithms:** Off the shelf, there are several optimisers included in the package. Among them, we highlight our fully-fledged Bayesian optimisation (based on `BoTorch <https://botorch.org/>`_) that supports optimising stochastic and composite objectives and is highly customisable. Additional methods can also be easily implemented within :code:`piglot`.
- **Visualisation tools:** You can use the builtin tool :code:`piglot-plot` to visualise the results of the optimisation. There are native plotting utilities for the optimised responses, the parameter history, objective history and, for supported solvers, live plotting of the currently running case. Also, an animation of the optimisation process can be exported.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   examples
   templates
