# Linear Regression

Widely used method for predicting a real-valued output (also called the
**dependent variable** or **target**) $y\in\mathbb{R}$, given a vector of
real-valued inputs (also called **independent variables, explanatory
variables,** or **covariates**) $\mathbf{x} \in \mathbb{R}^D$.

**Key Property** of this model is that the expected value of the output is
assumed to be a linear function of the input,
$\mathbb{E}[y|\mathbf{x}] = \mathbf{w}^\top \mathbf{x}$ making the model:

- easy to interpret
- easy to fit to data

Linear regression models the conditional distribution of the output as a
Gaussian whose mean is a linear function of the inputs:

$$p(y \mid x, \theta) = \mathcal{N}\!\left(y \mid w_0 + w^\top x,\; \sigma^2\right)$$

- **Parameters:** $\theta = (w_0,\, w,\, \sigma^2)$
- **Notation:** statistics texts often write $w_0, w$ as $\beta_0, \beta$.

## Weights / regression coefficients

The vector $w_{1:D}$ holds the **weights** (a.k.a. regression coefficients).

- Each coefficient $w_d$ = the expected change in the output $y$ when feature
  $x_d$ increases by **one unit**, holding everything else fixed.

**Example** — predicting income $y$ from age $x_1$ and education level $x_2$:

| Coefficient | Interpretation                                                    |
| ----------- | ----------------------------------------------------------------- |
| $w_1$       | expected income gain per additional year of age (more experience) |
| $w_2$       | expected income gain per one-level rise in education              |

## Bias / offset term

- $w_0$ is the **offset** (a.k.a. bias or intercept) — the predicted output when
  all inputs are 0.
- It captures the **unconditional mean** of the response and serves as a
  baseline:

$$w_0 = \mathbb{E}[y]$$

## The absorption trick

Prepend a constant 1 to the input vector, $x = [1, x_1, \dots, x_D]$. Then $w_0$
folds into the weight vector $w$, and the model is just $w^\top x$ — no separate
intercept term to track.

## Simple vs. multiple

- **Simple linear regression** — one input ($D = 1$). Reduces to a line:
  $$f(x; w) = ax + b, \qquad a = w_1\ (\text{slope}),\quad b = w_0\ (\text{intercept})$$
- **Multiple linear regression** — multi-dimensional input, $x \in \mathbb{R}^D$
  with $D > 1$.

## Multi-output regression

When the output is a vector of $J$ targets, model each one with its own weight
vector $w_j$ and noise variance $\sigma_j^2$, treating them as conditionally
independent given $x$:

$$p(y \mid x, W) = \prod_{j=1}^{J} \mathcal{N}\!\left(y_j \mid w_j^\top x,\; \sigma_j^2\right)$$

## Nonlinear feature transforms (basis functions)

A straight line rarely fits real data well. Fix this by passing the inputs
through a **feature extractor** $\phi(\cdot)$ before the linear part:

$$p(y \mid x, \theta) = \mathcal{N}\!\left(y \mid w^\top \phi(x),\; \sigma^2\right)$$

**Key point:** as long as the parameters of $\phi$ are _fixed_ (not learned),
the model is still **linear in the parameters** $w$ — even though it's nonlinear
in the inputs $x$. This is what keeps least-squares / closed-form estimation
usable while letting the model capture curves, interactions, etc.

> The "linear" in linear regression refers to linearity in $w$, not in $x$.

## Least Squares Estimation

To fit a linear regression model to data, minimize the negative log likelihood
on the training set.

Fit the model by **minimizing the negative log likelihood (NLL)** over the
training set. Plugging the Gaussian into the NLL gives:

$$\text{NLL}(w, \sigma^2) = -\sum_{n=1}^{N} \log\left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{1}{2\sigma^2}(y_n - w^\top x_n)^2\right) \right]$$

$$= \frac{1}{2\sigma^2}\sum_{n=1}^{N}(y_n - \hat{y}_n)^2 \;+\; \frac{N}{2}\log(2\pi\sigma^2)$$

where the predicted response is $\hat{y}_n \triangleq w^\top x_n$.

**Finding the MLE.** The estimate is the stationary point where
$\nabla_{w,\sigma}\,\text{NLL}(w,\sigma^2) = 0$. A convenient order: optimize
over $w$ first, then solve for the optimal $\sigma$. (These notes focus on the
weights $w$.)

**Reduction to RSS.** Holding $\sigma^2$ fixed, the only $w$-dependent term is
the sum of squared errors, so minimizing the NLL over $w$ is the same (up to
constants) as minimizing the **residual sum of squares**:

$$\text{RSS}(w) = \frac{1}{2}\sum_{n=1}^{N}(y_n - w^\top x_n)^2 = \frac{1}{2}\lVert Xw - y \rVert_2^2 = \frac{1}{2}(Xw - y)^\top (Xw - y)$$

This is why Gaussian-noise MLE and ordinary least squares coincide.
