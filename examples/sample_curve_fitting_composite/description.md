## sample_curve_fitting_composite example

A simple analytical curve fitting problem using a composite strategy is included to demonstrate how to use `piglot` in the composite setting.
In this case, we aim to fit a quadratic expression of the type $f(x) = a x^2$, using as a reference, a numerically generated reference from the expression $f(x) = 2 x^2$ (provided in the `examples/sample_curve_fitting/reference_curve.txt` file).
We want to find the value for $a$ that better fits our reference (it should be 2).

We run 10 iterations using the `botorch` optimiser (our interface for Bayesian optimisation), and set the parameter `a` for optimisation with bounds `[0,4]` and initial value 1.
Our optimisation objective is the fitting of an analytical curve, with the expression `<a> * x ** 2`.
The notation `<a>` indicates that this parameter should be optimised.
We also define a parameterisation using the variable $x$, where we sample the function between `[-5,5]` with 100 points.

The particularity of this example is that a composite strategy is used to fit the response.
The advantages of this composite Bayesian optimisation are demonstrated in [Coelho et al.]([docs/source/simple_example/best.svg](https://dx.doi.org/10.2139/ssrn.4674421)) and are two-fold: (i) more accurate posteriors for the loss function and (ii) reduced loss of information during the computation of the reduction function.

In short, in the composite setting the loss function $\mathcal{L}\left(\bm{\theta}\right)$ is written as
$
    \mathcal{L}\left(\bm{\theta}\right)
    =
    \hat{\mathcal{L}}\left(\bm{e}\left(\bm{\theta}\right)\right)
    =
    \dfrac{1}{N}
    \sum_{i=1}^{N}
    \left[e_i\left(\bm{\theta}\right)\right]^2,
$
where $\bm{e}$ is a vector containing the pointwise errors at every reference point, $\hat{\mathcal{L}}\left(\bullet\right)$ is the scalar reduction function applied (NMSE in this case) and $N$ is the number of reference points.
Thus, the problem can be stated as the minimisation of a composite function $\hat{\mathcal{L}}\left(\bm{e}\left(\bm{\theta}\right)\right)$, where $\hat{\mathcal{L}}\left(\bullet\right)$ is known and $\bm{e}\left(\bm{\theta}\right)$ is unknown.

Consider the minimisation of $\mathcal{L}\left(\bm{\theta}\right) = \hat{\mathcal{L}}\left(\bm{e}\left(\bm{\theta}\right)\right)$.
Within this setting, we start by replacing the single-output Gaussian Process on the loss $\mathcal{L}\left(\bm{\theta}\right)$ with a multi-output Gaussian Process for the error of each reference point $e_i$, that is,
$
    e_i\left(\bm{\theta}\right)
    \sim
    \mathcal{GP}
    \left(
        \mu_i\left(\bm{\theta}\right),
        k_i\left(\bm{\theta},\bm{\theta}'\right)
    \right).
$
Naturally, this requires feeding the optimiser with the entire response instead of a single scalar value.
At this stage, each GP is assumed independent and the correlations between the outputs are not considered; that is, each value $e_i$ is assumed as independent and uniquely defined by the set of parameters $\bm{\theta}$.

The configuration file (`examples/sample_curve_fitting_composite/config.yaml`) for this example is:
```yaml
iters: 10

optimiser: botorch

parameters:
  a: [1, 0, 4]

objective:
  name: fitting
  composite: True
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
The composite strategy is activated by setting ```composite: True```.

To run this example, open a terminal inside the `piglot` repository, enter the `examples/sample_curve_fitting_composite` directory and run piglot with the given configuration file
```bash
cd examples/sample_curve_fitting_composite
piglot config.yaml
```
You should see an output similar to
```
BoTorch: 100%|████████████████████████████████████████| 10/10 [00:01<00:00,  7.94it/s, Loss: 5.6009e-08]
Completed 10 iterations in 1s
Best loss:  5.60089334e-08
Best parameters
- a:     1.999685
```
It is observed that piglot correctly identifies the `a` parameter close to the expected value of 2, and the error of the fitting is in the order of $10^{-8}$.
As this example is quite simple, there is no great advantage of using the composite strategy, as the simple Bayesian optimisation is already able of finding accurate solutions within few function evaluations.


To visualise the optimisation results, use the `piglot-plot` utility.
In the same directory, run for instance
```bash
piglot-plot best config.yaml
```