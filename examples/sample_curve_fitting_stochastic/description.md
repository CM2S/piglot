## sample_curve_fitting_stochastic example

A simple analytical curve fitting problem with noise in the input data is included to demonstrate how to use `piglot` with variance.
In this case, we aim to fit two quadratic expressions of the type $f(x) = a x^2$ and $f(x) = 2a x^2$, using as a reference response, a numerically generated reference from the expression $f(x) = 2 x^2$ (provided in the `examples/sample_curve_fitting_stochastic/reference_curve.txt` file).
We want to find the value for $a$ that better fits our reference. In this case, the optimal solution is no longer $a=2$, as two distinct functions are used to compute the generated response.

We run 10 iterations using the `botorch` optimiser (our interface for Bayesian optimisation), and set the parameter `a` for optimisation with bounds `[0,4]` and initial value 1.
The notation `<a>` indicates that this parameter should be optimised.
We also define a parameterisation using the variable $x$, where we sample the function between `[-5,5]` with 100 points.


DESCRIBE STOCHASTIC

The configuration file (`examples/sample_curve_fitting_stochastic/config.yaml`) for this example is:
```yaml
iters: 10

optimiser: botorch

parameters:
  a: [1, 0, 4]

objective:
  name: fitting
  stochastic: True
  composite: False
  solver:
    name: curve
    cases:
      'case_1':
        expression: <a> * x ** 2
        parametric: x
        bounds: [-5, 5]
        points: 100
      'case_2':
        expression: 2* <a> * x ** 2
        parametric: x
        bounds: [-5, 5]
        points: 100
  references:
    'reference_curve.txt':
      prediction: ['case_1', 'case_2']
```
The stochastic strategy is activated by setting ```stochastic: True```, and by adding a new generated response with the label `case_2`, given by the expression $f(x) = 2a x^2$.

To run this example, open a terminal inside the `piglot` repository, enter the `examples/sample_curve_fitting_stochastic` directory and run piglot with the given configuration file
```bash
cd examples/sample_curve_fitting_stochastic
piglot config.yaml
```
You should see an output similar to
```
BoTorch: 100%|████████████████████████████████████████| 10/10 [00:00<00:00, 14.42it/s, Loss: 1.7315e-01]
Completed 10 iterations in 0.69363s
Best loss:  1.73146009e-01
Best parameters
- a:     1.198191
```
Piglot identifies the `a` parameter as 1.2, and the error of the fitting is in the order of $10^{-1}$.
For this case, the use of the composite Bayesian strategy (as decribed in [here](examples/sample_curve_fitting_composite/description.md)) significantly improves the quality of the fitting.

To visualise the optimisation results, use the `piglot-plot` utility.
In the same directory, run
```bash
piglot-plot best config.yaml
```
Which will display the best observed value for the optimisation problem.
You should see the following output in the terminal
```
Best run:
Start Time /s    0.683277
Run Time /s      0.014079
Variance         0.005519
a                1.198191
Name: 18, dtype: object
Hash: f07c094fdbbaa637387a31cdeeb946783f4b3aeefe99639995d7e6539cf48475
Objective:  1.73146009e-01
```
The script plots the best observed responses, and its comparison with the reference response. Moreover, the average, the median, the standard deviation and the 95\% mean condidence intervals are also provided.
![Best case plot](../../docs/source/simple_stochastic_example/best_0.svg)
![Best case plot](../../docs/source/simple_stochastic_example/best_1.svg)
![Best case plot](../../docs/source/simple_stochastic_example/best_2.svg)


Try running the same example with the composite strategy (by simply setting ```composite: True```). With the composite Bayesian optimisation the error of the fitting is reduced to $10^{-5}$.


