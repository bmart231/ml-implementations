# Linear Regression — Learning Guide

A 7-step progression. Each step teaches something the previous one hid. Do them in order, ideally in a single Jupyter notebook with markdown cells explaining your thinking.

---

## 1. Closed-form (Normal Equation)

**Goal:** See that linear regression has an exact analytical solution — no iteration needed.

**Math:** `w = (XᵀX)⁻¹ Xᵀy`

**Steps:**
1. Generate a small synthetic dataset: `y = 3x + 5 + noise` with ~50 points.
2. Add a bias column of 1s to `X` so the intercept is learned as part of `w`.
3. Compute `w` using `np.linalg.inv(X.T @ X) @ X.T @ y`.
4. Plot the data and your fitted line.
5. Try a dataset where two features are nearly identical — watch `XᵀX` become near-singular and the solution blow up.

**What you learn:** Why iterative methods exist (singular matrices, huge datasets).

---

## 2. Gradient Descent from Scratch

**Goal:** Build the foundation for *every* model that follows (NNs, logistic regression, etc.).

**Math:** Minimize MSE = `(1/n) Σ (y_pred - y)²`. Update rule: `w -= lr * ∂MSE/∂w`.

**Steps:**
1. Initialize `w` to zeros or small random values.
2. Loop for N epochs:
   - Compute predictions: `y_pred = X @ w`
   - Compute gradient: `(2/n) * X.T @ (y_pred - y)`
   - Update: `w -= lr * gradient`
   - Record the loss
3. Plot loss vs. epoch — should decrease smoothly.
4. Compare final `w` to the closed-form solution from Step 1. They should match.
5. Crank the learning rate up until it diverges. Note what happens.

**What you learn:** The single most important algorithm in ML. How learning rate controls convergence.

---

## 3. Stochastic / Mini-batch Gradient Descent

**Goal:** Understand how real ML training works on large data.

**Steps:**
1. Modify Step 2 to update `w` using one random sample at a time (SGD).
2. Then try mini-batches of 16 or 32 samples.
3. Plot the loss curve — it'll be noisier than full-batch GD but converges faster on large datasets.
4. Time all three (full / mini-batch / SGD) on a dataset of 100k synthetic points.

**What you learn:** Why nobody uses full-batch GD in practice. Tradeoff between noise and speed.

---

## 4. Multivariate + Feature Scaling

**Goal:** See why preprocessing matters.

**Steps:**
1. Use a real dataset (e.g., scikit-learn's diabetes or California housing).
2. Run gradient descent without scaling. Note how slow it is — features on very different scales make the loss surface stretched.
3. Standardize features: `(x - mean) / std`.
4. Run gradient descent again. Should converge dramatically faster.
5. Plot the loss curves side by side.

**What you learn:** Preprocessing is not optional. Gradient descent loves spherical loss surfaces.

---

## 5. Regularization: Ridge and Lasso

**Goal:** Bridge to understanding overfitting and feature selection.

**Math:**
- Ridge loss: `MSE + λ‖w‖²`
- Lasso loss: `MSE + λ‖w‖₁`

**Steps:**
1. Generate data with many correlated features (e.g., 20 features, only 3 actually matter).
2. Fit plain linear regression — coefficients will be unstable.
3. Add the Ridge penalty to your gradient. Watch all coefficients shrink toward zero.
4. Add the Lasso penalty (subgradient is `sign(w)`). Watch some coefficients become exactly zero.
5. Sweep `λ` and plot coefficient values vs. `λ` — the classic "regularization path" plot.

**What you learn:** Regularization controls model complexity. Ridge shrinks; Lasso selects.

---

## 6. Polynomial Regression

**Goal:** See that "linear" means linear in the *parameters*, not in x.

**Steps:**
1. Generate non-linear data: `y = sin(x) + noise`.
2. Build polynomial features: `X_poly = [1, x, x², x³, …, x^d]`.
3. Run your existing linear regression on `X_poly`.
4. Plot fits for `d = 1, 3, 9, 15`. Observe underfitting → good fit → overfitting.
5. Add Ridge regularization to tame the high-degree overfit.

**What you learn:** A "linear" model can fit highly non-linear data via feature engineering.

---

## 7. Diagnostics

**Goal:** Learn to evaluate a regression model the right way.

**Steps:**
1. Compute R², MSE, MAE manually from your predictions.
2. Plot residuals (`y - y_pred`) vs. predictions. They should look like random noise — patterns mean your model is missing structure.
3. Plot residuals vs. each feature. Same idea.
4. Compare your final weights to `sklearn.linear_model.LinearRegression`. They should match to several decimal places.

**What you learn:** Metrics alone lie. Diagnostic plots tell you *why* a model is wrong.

---

## Suggested File Layout

```
linear_regression/
├── README.md          # this file
├── notebook.ipynb     # all 7 steps with explanations
└── utils.py           # shared helpers (data generation, plotting)
```

By the end, you'll have a mental model that carries into every model in the rest of the repo.
