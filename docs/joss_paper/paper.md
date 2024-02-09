---
title: 'piglot: an Open-source Package for Derivative-free Optimisation of Numerical Responses'
tags:
  - Python
  - computational mechanics
  - inverse problems
  - derivative-free optimisation
  - Bayesian optimisation
  - parameter identification
authors:
  - name: R. P. Cardoso Coelho
    orcid: 0000-0001-9989-964X
    affiliation: "1, 2"
  - name: A. Francisca Carvalho Alves
    orcid: 0000-0003-1214-5453
    affiliation: "1, 2"
  - name: T. M. Nogueira Pires
    orcid: 0009-0000-1518-2845
    affiliation: "1"
  - name: F. M. Andrade Pires
    orcid: 0000-0002-4802-6360
    corresponding: true
    affiliation: "1, 2"
affiliations:
 - name: Faculty of Engineering, University of Porto, Porto, Portugal
   index: 1
 - name: Institute of Science and Innovation in Mechanical and Industrial Engineering, Porto, Portugal
   index: 2
date: 7 February 2024
bibliography: references.bib

---

# Summary
`piglot` is an open-source Python tool taylored for the automated optimisation of responses stemming from numerical solvers. With this tool we aim at providing a simple and user-friendly interface which is also easily extendable, allowing intergration with other solvers within the community. `piglot` provides a versatile solution for solving inverse problems on several research areas, such as structural analysis, material modelling, fluid dynamics, control systems or astrophysics, using, for instance, finite element analysis, spectral methods or Monte Carlo methods. The primary emphasis is on derivative-free optimisation, ensuring compatibility with black-box solvers in scenarios where gradient information is not available, and cases where the function evaluations may be noisy.

![Logo of `piglot`. \label{fig:piglot_logo}](../source/media/logo.svg){width=35%}

# Statement of need

The increasingly growing interest in computational analysis for engineering problems has been driving the development of more accurate, robust and efficient methods and models.
With the advent of this technology, the application of the so-called inverse problems has been gaining traction over   the last years, where one seeks optimised parameters, geometries, configurations or models for numerical problems arising in engineering.
In this context, in the past years, some packages have been developed to automate the identification of parameters [@nevergrad,@optuna_2019].
These packages, however, do not provide an interface for different solvers, so the optimisation of numerical responses is not readily available.

In this work, we present `piglot` an open-source Python package for automated optimisation of numerical responses, such as responses stemming from finite element simulations.
In particular, focus is placed on derivative-free optimisation, to allow compatibility with black-solvers where gradient information may be unavailable.
In this context an extensible interface for coupling with physics solvers is provided.
As long as the solver can return a time-response for the fields of interest, it is possible to optimise it with `piglot`.
Notwithstanding, some solvers are already provided, namely a solver for fitting analytical functions, a solver for the in-house finite element code `Links`, a solver for the finite element software `Abaqus`, and a solver for the clustering-based reduced-order model `CRATE` package [@Ferreira2023].

For the optimisation itself, several optimisation methods are implemented and available, such as DIRECT, LIPO, Bayesian optimisation, among others.
Particularly, a significant effort has been employed into Bayesian optimisation algorithms, backed with an open-source implementation [@balandatBoTorchFrameworkEfficient2020] and allowing for single- and multi-objective optimisation of both noise-free and stochastic objectives.
Furthermore, a novel composite Bayesian optimisation strategy is available for curve-fitting problems, which, in our tests, severely outperforms classical optimisation approaches [@Coelho2023optm].

The package also provides a builtin tool `piglot-plot` to visualise the results of the optimisation.
There are native plotting utilities for the optimised responses, the parameter history, objective history and, for supported solvers, live plotting of the currently running case.
The package also includes full documentation for a clear installation and usage, supporting a simple framework for new developments. 
With this in mind, a thorough automated testing is incorporated, ensuring the compliance of new developments.

In \autoref{fig:piglot_example} a scheme of the workflow of `piglot` is illustrated.
There are two modes of initialisation available: using `.yaml` configuration files, or building the optimisation problem in a Python script. 
The use of configuration files is the simplest and recommended approach of using `piglot`.
During the optimisation there is a continuous exchange of information between the physics solvers, `piglot`, and the optimisers.
Whereas the optimisers are responsible for providing a candidate solution for the parameters, $\boldsymbol{\theta}$, based on the loss function value, $J(\boldsymbol{\theta})$, the physics solvers receive the parameters, $\boldsymbol{\theta}$, and compute the numerical response, $\boldsymbol{\sigma}$, accordingly.
The results of the optimisation can be then visualise using the `piglot-plot` tool.


![Schematic illustration of `piglot`. \label{fig:piglot_example}](piglot.svg){width=100%}


The `piglot` package has been successfully used for the identification of constitutive parameters for classical elasto-plastic models from multi-scale simulations, crystal plasticity models with mechanically-induced martensitic transformations [@cardosocoelhoMultiscaleModelCombining2023] and models for amorphous polymers [@ALVES2023112488].
Moreover, this tool has also demonstrated its potential in the material design of different microstructures, such as particulate PC/ABS polymer blends.


With this package, we aim to provide a simple and effective tool for general optimisation of numerical responses, which can be easily extended for other solvers in the community.


# Acknowledgements

R. P. Cardoso Coelho and A. Francisca Carvalho Alves gratefully acknowledge the support provided by Fundação para a Ciência e a Tecnologia (FCT) through the scholarships with references 2020.07159.BD and 2020.07279.BD, respectively.
This research has also been supported by Instituto de Ciência e Inovação em Engenharia Mecânica e Engenharia Industrial (INEGI).

# References
