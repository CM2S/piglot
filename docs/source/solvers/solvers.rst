Adding support for new solvers
==============================

If you wish to use ``piglot`` with a new solver, there is some programming required.
In this guide, we provide the main steps and building blocks needed to connect ``piglot`` to your solver.
Additionally, we strongly encourage developers to contribute their solver interfaces to this project according to our `guidelines <https://github.com/CM2S/piglot/blob/main/CONTRIBUTING.md>`__.

What is a solver for ``piglot``?
--------------------------------

At its core, a solver can be seen as a function that, for a given set of parameters, returns a set of discrete responses.
While this is an overly simplistic view of the job of an interface, from an optimisation perspective, this is exactly the behaviour we are expecting.
Note that the solver is responsible for evaluating the **numerical response** for a given set of parameters; ``piglot`` is then responsible for the calculation of the objective function and its optimisation.
We do not place any restrictions on the solver, even experimental campaigns can be used as solvers for ``piglot``.

In a more realistic scope, we require some additional aspects to fully integrate a solver with ``piglot``.
We provide several wrappers to ease the development of a solver interface.
Particularly, two main categories of solvers are supported:

- **Interface-based solvers:** These require changing ``piglot``'s codebase to fully couple the toolbox with the solver.
  This is the most flexible approach, as it allows for full control over the solver's behaviour.
  However, it requires additional coding effort and is not recommended for one-shot cases; it is more appropriate when the solver is used frequently with varying cases and inputs.
  For this class of solvers, we welcome contributions to the project.
- **Script-based solvers:** These do not require changing ``piglot``'s codebase.
  In practice, a ``script`` solver is implemented that reads a Python script and runs it with the parameters to optimise.
  For simple and one-shot cases, this is the recommended approach.

In the remainder of the guide, both approaches are discussed.
If you are unsure which approach to take, we recommend starting with the script-based solver, as it is simpler to implement and requires no changes to the ``piglot`` codebase.
However, if you plan to use the solver frequently or contribute to the project with a new solver, we recommend the interface-based solver.
The higher upfront cost is offset by the flexibility and ease of use in the long term.


Script-based solvers
--------------------

The easiest approach to use a custom solver in ``piglot`` is to implement a script-based solver.
To this end, a simple class derived from ``ScriptSolverCallable`` must be implemented, with the following methods:

* ``get_output_fields()`` - returns a list of the output fields that the solver will return.
* ``solve()`` - receives a dictionary with the mapping between the parameter names and their values, and returns the list of output results.
  This is the main method for running the solver for a given set of parameters.
  The results should contain all the responses listed in ``get_output_fields()``.

The following example shows a curve fitting problem using a custom solver that fits a sine curve to a set of points.
The design variable is the frequency of the sine wave, with a single parameter ``freq``.
The solver is specified in the ``sample_solver.py`` file, given by

.. literalinclude:: ../../../examples/solver_example/sample_solver.py
  :language: python

The associated input file is:

.. literalinclude:: ../../../examples/solver_example/config.yaml
  :language: yaml

Where the reference curve in the file ``sine.txt`` is:

.. literalinclude:: ../../../examples/solver_example/sine.txt
  :language: text

Thus, simple Python functions can be used as solvers for ``piglot``, which can be useful for simple cases or for prototyping.
There is no restriction on the nature of the results returned by the solver, as long as they can be converted to a numerical value.
With this in mind, simple interfaces around complex solvers can also be devised using this approach.


Interface-based solvers
-----------------------

The base class for any solver in ``piglot`` is the ``Solver`` class, which introduces many methods required by the toolbox during optimisation and post-processing.
In the vast majority of the cases, you don't need to explicitly derive from this class, as we provide a set of helper classes that simplify the implementation of a solver interface:

::

   Solver
   ├── SingleCaseSolver
   └── MultiCaseSolver
       └── InputFileSolver

The ``SingleCaseSolver`` class is used when the solver only runs a single analysis to extract the responses of interest.
On the other hand, the ``MultiCaseSolver`` class is used when the solver may have to run multiple simulations for a given set of parameters, and automatically handles the case management, the multiple output responses and supports parallel evaluations.
In general, we recommend using the multi-case variant, as it is more flexible and can be used for both single and multi-case problems.
Finally, if your solver works directly with input and output files, we strongly encourage exploring the ``InputFileSolver`` class, which has built-in support for preparing input files, running the solver and reading the output files.


Single-case solvers
~~~~~~~~~~~~~~~~~~~

This is the simplest type of solver interface, where the solver is limited to a single analysis for a given set of parameters.
As previously mentioned, unless you have a strong motive to use this class, we recommend using the multi-case solver instead.
An example of a single-case solver is the interface for script-based solvers, which expects a single function that returns the responses for a given set of parameters.
The class ``SingleCaseSolver`` requires the following methods to be implemented:

- ``_solve()`` - the main method that receives the current parameter values to evaluate the responses for and returns a mapping between the output field and the output result.
  This is the main function of the solver interface, which is called once for each function evaluation.
  This method is expected to run the solver for the given set of parameters and read the output fields.
  Additionally, a flag signalling whether this evaluation may be concurrent to others is also passed --- this is useful to ensure the solver behaves correctly in parallel evaluations.
- ``read()`` - receives the configuration dictionary, the parameter set and the output directory, and returns a ``SingleCaseSolver`` instance.
  This is required to parse the YAML file into a class instance.

Note that, generally, the constructor of the class should also be overridden to set the field names that the solver will return.
To exemplify this, a simple implementation of a single-case solver that returns a response with the numerical values of the parameters is shown below.
Additionally, note that this example does not connect with any solver outside ``piglot``; it is merely illustrative of the process of implementing a solver interface.

.. literalinclude:: ../../../examples/solver_example/single.py
  :language: python


Multi-case solvers
~~~~~~~~~~~~~~~~~~

The wrapper for multi-case solvers is the ``MultiCaseSolver`` class.
Unlike the single-case solver, the multi-case variant assumes that a single evaluation may require running multiple simulations or analyses.
With this in mind, the focus is shifted towards how to run each individual case and extract its responses of interest.
Therefore, the problem specification now follows the hierarchy:

::

   solver
   ├── options
   └── cases
       ├── case_1
       |   └── case 1 options
       └── case_2
           └── case 2 options

The ``MultiCaseSolver`` class implements the logic for running all the cases and combining the responses into a single output.
This approach automatically handles the creation of temporary directories, the possibility of parallel evaluation of the cases and the writing of output progress files.
Therefore, the main aspect of the interface is the definition of the evaluation of each case.
This is done by defining a case class that derives from the ``Case`` class.
In total, two classes must be implemented, with the following methods for each:

- A case class, derived from ``Case`` - represents a single case for the solver.
  Requires the following methods:
   * ``name()`` - returns the name of the case.
   * ``get_fields()`` - returns the output fields that this case will produce.
   * ``run()`` - receives the parameter set, their values and a temporary directory, and returns the results for this case.
     This is the main method for running the solver for a given set of parameters.
     The results should contain all the responses listed in ``get_fields()``.
   * ``read()`` - receives the assigned name to the case and the configuration dictionary, and returns the instance of the case.
     Used for parsing the case in the YAML file into a suitable class instance.
- A solver class, derived from ``MultiCaseSolver`` - represents the solver and its cases.
  Requires the following methods:
   * ``get_case_class()`` - returns the **type** of the case class to be used --- the class derived from ``Case``.

Note the minimal implementation required for the class derived from ``MultiCaseSolver``; this is intended, as most of the logic is already implemented in the base class, and the specificities of the solver are handled by the case class.
As an example, the reading of the solver from the YAML file is already implemented in the base class, and only the reading of each individual case need be implemented.

.. note::
   For convenience, the default implementation of the ``MultiCaseSolver`` class allows passing unused solver options directly to each case.
   Some solver options are consumed (namely the ``tmp_dir``, ``output_dir`` and ``parallel`` options), but any other options are passed directly to all the cases in the configuration file.
   For example, the following YAML file:

   .. code-block:: yaml

       solver:
         name: your_fancy_solver
         parallel: 2
         foo: bar
         case_1:
           case_option: 2
         case_2:
           case_option: 3
   
   Is equivalent to:

   .. code-block:: yaml

       solver:
         name: your_fancy_solver
         parallel: 2
         case_1:
           case_option: 2
           foo: bar
         case_2:
           case_option: 3
           foo: bar
  
   Since ``foo`` is not consumed by the solver, it is appended to the cases' options.
   Note, however, that the ``parallel`` option is consumed by the solver and is not passed to the cases.
   This is useful for specifying common options to all cases.
   Beware of the naming of the options to avoid conflicts with the solver or case options.

Similarly to the previous case, an example of an interface that returns a response with the numerical values of the parameters multiplied by an user-specified multiplier is shown below.

.. literalinclude:: ../../../examples/solver_example/multi.py
  :language: python

A sample YAML configuration file is shown below.
This file registers two cases: (i) ``case_1`` multiplies the parameters by 2 and writes the output to a field named ``sample_case``; (ii) ``case_2`` multiplies the parameters by 3 and writes the output to a field named ``case_2`` (note that the implementation uses the case name as the default output name, if not specified).

.. code-block:: yaml

    solver:
      name: solver_name
      case_1:
        multiplier: 2
        output_name: sample_case
      case_2:
        multiplier: 3


Input file-based solvers
~~~~~~~~~~~~~~~~~~~~~~~~

Many solvers work directly with input and output files.
Since this is a very common use scenario, we provide a wrapper class that simplifies the implementation of such solvers.
Towards this end, the utilities in the class ``InputFileSolver`` are designed to handle the preparation of input files, the running of the solver and the reading of the output files.
This class inherits from the ``MultiCaseSolver`` class, so multiple cases can be run in parallel.
With this approach, for a given set of parameter values, the following steps occur for each case:

1. An ``InputDataGenerator`` receives the parameters and generates the appropriate input file for the solver, along with any required dependencies that may be needed.
   These are stored in an instance of the ``InputData`` class.
2. The solver is run with the generated input file, using the procedure defined in an ``InputFileCase`` class.
3. For each registered output field, an ``OutputField`` class reads the output file and returns the result for the field of interest.

The hierarchical structure of the solver is as follows:

::

   solver
   ├── options
   └── cases
       ├── input_file_1.dat
       |   ├── case 1 options
       |   └── fields
       |       ├── field_1
       |       |   ├── name: stresses
       |       |   └── field 1 options
       |       └── field_2
       |           ├── name: strains
       |           └── field 2 options
       └── input_file_2.dat
           ├── case 2 options
           └── fields
               └── field_3
                   ├── name: forces
                   └── field 3 options

In this case, two input files are considered: ``input_file_1.dat`` and ``input_file_2.dat``.
For the first, two output fields are registered: (i) ``fields_1`` that extracts the stresses from the output results and (ii) ``fields_2`` that extracts the strains.
The second input file has a single output field, ``fields_3``, that extracts the forces.
Each case is stored in an instance of the ``InputFileCase`` class, which is responsible for running the solver for the given case.
The ``stresses``, ``strains`` and ``forces`` fields are the names associated with an ``OutputField`` class that reads the output file and returns the result for the field of interest.
From this perspective, the following methods and classes must be implemented:

* One or more classes derived from ``OutputField`` - these specify the kinds of outputs that can be extracted from the solver.
  Requires implementing the following methods:
   * ``check()`` - receives the input data (via an instance of the ``InputData`` class) and checks if the output field is valid and consistent with it.
   * ``get()`` - receives the input data (via an instance of the ``InputData`` class) and returns the output result for the field of interest.
     This is the main method for reading from the output of the numerical solver.
   * ``read()`` - receives the configuration dictionary (from the YAML file) and returns the instance of the output field.
     Used for parsing the YAML file into a class instance.
* A class derived from ``InputFileCase`` - represents a single case for the solver.
  Requires the following methods:
   * ``_run_case()`` - receives the input data (via an instance of the ``InputData`` class) and the temporary directory, and runs the solver for the given case.
     This method should return a bool whether the analysis was successful or not.
   * ``get_supported_fields()`` - returns a dictionary mapping the name of the supported output fields with their respective ``OutputField`` class **type**.
     This list defines the supported types of outputs that the interface can read from the solver, and should return the types of the classes implemented in the previous point.
* A class derived from ``InputFileSolver`` - represents the solver and its cases.
  Requires the following method:
   * ``get_case_class()`` - returns the **type** of the case class to be used --- the class derived from ``InputFileCase``.

Additionally, if required, override any default constructor to pass additional options to each instance.
The default ``read()`` method of ``InputFileSolver`` is already implemented, and automatically passes extra options to each case instance.

.. note::
   A generator class must be used for the generation of the input data.
   By default, the ``DefaultInputDataGenerator`` class is used, which copies the input file and its dependencies to the temporary directory that is used for the solver, and then does the parameter substitution of the placeholders in the input file (and specified dependencies).
   Naturally, this behaviour can be customised by deriving from the ``InputDataGenerator`` class and adapting the case read methods to use the new generator.

.. note::
   For existing solvers, a custom input file generator can also be used.
   Each case supports the ``generator`` option, which can be used to specify the generator class to be used for the case.
   As an example, consider the following YAML configuration file:

   .. code-block:: yaml
   
      solver:
        name: links
        links: LINKS
        cases:
          case_1:
            generator:
              script: custom_generator.py
              class: CustomGenerator
            fields:
              forces:
                name: Reaction
                field: x

   In this case, the ``CustomGenerator`` class from the ``custom_generator.py`` file is used to generate the input data for the case.
   This class should return a valid instance of the ``InputData`` class, which is then used to run the solver for the case.

The following example shows a simple implementation of an input file-based solver that reads the parameters from an input file and copies them to an output file.
Note that the parameters in the input file are substituted by the default input data generator.

.. literalinclude:: ../../../examples/solver_example/input.py
   :language: python
