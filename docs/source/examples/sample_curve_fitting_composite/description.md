# Composite setting

In curve fitting problems with a reference response, we can exploit the function composition of the objective function to drastically improve the optimisation.
This technique has been widely explored in Bayesian optimisation (as proposed in [Astudillo and Frazier (2019)](https://doi.org/10.48550/arXiv.1906.01537)) and, for the curve fitting problem, in [Cardoso Coelho et al. (2023)](https://dx.doi.org/10.2139/ssrn.4674421).
In `piglot`, this strategy is available out of the box and can be easily enabled.
We now display an example of the application of this technique, based on the [simple curve fitting example](../sample_curve_fitting/description.md) (please check out that example before diving into the composite setting).
In this example, we are heavily relying on Bayesian optimisation (if you are unfamiliar with the topic, we highly recommend checking this [tutorial](https://doi.org/10.48550/arXiv.1807.02811) before proceeding).

**Note:** Composite optimisation is not supported by most optimisers.
Currently, only Bayesian optimisation with BoTorch supports the full version of the composite objective.

## Background

We start by defining our objective function (or *loss* in the curve fitting scenario) $\mathcal{L}\left(\boldsymbol{\theta}\right)$ as

$
    \mathcal{L}\left(\boldsymbol{\theta}\right)
    =
    \hat{\mathcal{L}}\left(\boldsymbol{e}\left(\boldsymbol{\theta}\right)\right)
    =
    \dfrac{1}{N}
    \sum_{i=1}^{N}
    \left[e_i\left(\boldsymbol{\theta}\right)\right]^2
$

where $\boldsymbol{e}$ is a vector containing the pointwise errors at every reference point, $\hat{\mathcal{L}}\left(\bullet\right)$ is the scalar reduction function applied (MSE in this case) and $N$ is the number of reference points.
In other words, we are minimising the average squared error between each point of the reference and the prediction.
Thus, the problem can be stated as the minimisation of a composite function $\hat{\mathcal{L}}\left(\boldsymbol{e}\left(\boldsymbol{\theta}\right)\right)$, where $\hat{\mathcal{L}}\left(\bullet\right)$ is known (and we can compute gradients) and $\boldsymbol{e}\left(\boldsymbol{\theta}\right)$ is "unknown" (comes from our black-box solver).

When using Bayesian optimisation, we build a Gaussian process (GP) regression model of our observations.
This surrogate model is used to choose the next potential points to evaluate.
In the simple curve fitting example, the model is built on the loss function values, that is:

$
    \mathcal{L}\left(\boldsymbol{\theta}\right)
    \sim
    \mathcal{GP}
    \left(
        \mu_i\left(\boldsymbol{\theta}\right),
        k_i\left(\boldsymbol{\theta},\boldsymbol{\theta}'\right)
    \right)
$

According to this, the *objective function value* is assumed to follow a Gaussian distribution with known mean and variance for each point $\boldsymbol{\theta}$ in the parameter space.
We then use this GP to build and optimise our acquisition functions of choice, as usual in Bayesian optimisation.

However, in the composite setting, we know that $\mathcal{L}\left(\boldsymbol{\theta}\right) = \hat{\mathcal{L}}\left(\boldsymbol{e}\left(\boldsymbol{\theta}\right)\right)$.
Therefore, we can instead build a surrogate model for the error at each reference point $e_i\left(\boldsymbol{\theta}\right)$:

$
    e_i\left(\boldsymbol{\theta}\right)
    \sim
    \mathcal{GP}
    \left(
        \mu_i\left(\boldsymbol{\theta}\right),
        k_i\left(\boldsymbol{\theta},\boldsymbol{\theta}'\right)
    \right)
$

Thus, we are saying that, for a given $\boldsymbol{\theta}$, the *prediction error at each reference point* follows a Gaussian distribution with known mean and variance.
However, we need a posterior probablity distribution for the values of $\mathcal{L}\left(\boldsymbol{\theta}\right)$ to use our Bayesian optimisation tools, which we cannot derive for generic functions $\hat{\mathcal{L}}\left(\bullet\right)$!
The solution to this problem is Monte Carlo sampling - we draw samples from the posteriors of $e_i\left(\boldsymbol{\theta}\right)$ and then evaluate them through $\hat{\mathcal{L}}\left(\bullet\right)$.
The resulting sample values should follow the posterior distribution for $\mathcal{L}\left(\boldsymbol{\theta}\right)$ and we can use them to compute approximate acquisition function values.
We leverage BoTorch's quasi-Monte Carlo acquisition functions for this task, and you can read more details on the entire procedure in [Cardoso Coelho et al. (2023)](https://dx.doi.org/10.2139/ssrn.4674421).

## Application

Putting all the mathematical details aside, it is extremely simple to use the composite setting in `piglot`.
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
The composite strategy is activated by setting `composite: True`.
Running this example, you should see an output similar to
```
BoTorch: 100%|█████████████████████████████| 10/10 [00:01<00:00,  7.94it/s, Loss: 5.6009e-08]
Completed 10 iterations in 1s
Best loss:  5.60089334e-08
Best parameters
- a:     1.999685
```
It is observed that `piglot` correctly identifies the `a` parameter close to the expected value of 2, and the error of the fitting is in the order of $10^{-8}$.
While this objective function is sufficiently simple that the simple and composite strategies behave similarly, for problems with multiple parameters and complex objective functions, it is well recognised that the composite setting drastically outperforms the direct optimisation of the objective function (check [Cardoso Coelho et al. (2023)](https://dx.doi.org/10.2139/ssrn.4674421) for examples).
However, note that the computational cost per iteration is significantly higher in the composite setting and, crucially, it is proportional to the number of points in the reference response.
Check the tutorial on the [reduction of reference points](../reference_reduction_composite/description.md) for additional details on this topic and for strategies to mitigate this issue.
