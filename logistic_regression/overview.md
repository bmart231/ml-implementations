# Logistic Regression — Learning Guide

A progression in the same spirit as your linear regression guide: each step
teaches something the previous one hid. Do them in order, in a single Jupyter
notebook with markdown cells explaining your thinking.

**One contrast to hold onto from the start:** linear regression opened with a
_closed form_ and then justified iteration. Logistic regression has **no**
closed form — the sigmoid makes the gradient nonlinear in `w`. Iteration isn't
an optional add-on here; it's the only way in. Everything else (the
`(pred − target)·x` gradient, SGD, regularization, diagnostics) carries straight
over from your linear regression work.

This guide is a companion to your theory notes — section references like
(§1.2.1) point back to the relevant part of `ML_Implementations.pdf`.

---

## 1. The Sigmoid and the Forward Pass

**Goal:** Understand the model _before_ training it — a linear logit squashed
into a probability.

**Math:** `a = wᵀx + b` (the logit / log-odds), `µ = σ(a) = 1 / (1 + e⁻ᵃ)`.
Decision rule: `ŷ = 𝟙(a > 0)`.

**Steps:**

1. Implement `sigmoid(a)`. Plot it from −10 to 10. Note the saturation: in the
   tails the slope goes to zero — remember this, it bites gradient descent
   later.
2. Generate 2D linearly separable data (two Gaussian blobs).
3. Pick `w` and `b` _by hand_. Compute logits `a`, then `µ = σ(a)`.
4. Plot the data colored by class, draw the boundary `wᵀx + b = 0`, and overlay
   probability contours (`µ = 0.25, 0.5, 0.75`).
5. Scale `w` up by 5× keeping its _direction_ fixed. Watch the contours bunch
   up: larger `‖w‖` ⇒ steeper sigmoid ⇒ more confident predictions (§1.1.1).

**What you learn:** The boundary is a hyperplane. `w`'s _direction_ orients it;
its _magnitude_ sets confidence. The saturated tails are a warning for later.

---

## 2. Cross-Entropy Loss — and Why There's No Closed Form

**Goal:** See why classification needs its own loss, and why you can't just
invert a matrix.

**Math:** `NLL(w) = −(1/N) Σ [ yₙ log µₙ + (1 − yₙ) log(1 − µₙ) ]` (your eq.
1–4).

**Steps:**

1. Implement the NLL. Add a tiny `ε` inside the logs to avoid `log(0)`.
2. For a single point with `y = 1`, plot loss vs predicted `µ`. Confirm the
   `−log q` shape: ≈ 0 when confident-and-correct, → ∞ when confident-and-wrong
   (the cross-entropy figure in your notes).
3. Try to solve `∇NLL = 0` by hand. You land on `Σ(µₙ − yₙ)xₙ = 0` with `µ`
   _nonlinear_ in `w` — no algebraic solution. Contrast: the normal equation
   existed only because linear regression's gradient was _linear_ in `w`.
4. (Optional, illuminating) Swap CE for MSE on top of the sigmoid and plot the
   loss surface over two weights. It's **non-convex** — bumpy with flat
   plateaus. CE is convex. This is _why_ CE.

**What you learn:** MSE + sigmoid is non-convex and saturates; cross-entropy is
the convex, well-behaved choice. No normal equation exists — iteration is
mandatory.

---

## 3. Gradient Descent from the Hand-Derived Gradient

**Goal:** Build the workhorse and verify your calculus.

**Math:** `∇NLL = (1/N) Σ (µₙ − yₙ) xₙ = (1/N) Xᵀ(µ − y)` (your eq. 10). Update:
`w ← w − η ∇NLL`.

**Steps:**

1. Derive the gradient yourself following §1.2.2 (sigmoid derivative → chain
   rule). The cancellation down to `(µ − y)x` is worth doing once by hand.
2. Implement full-batch GD. Record the loss each epoch.
3. **Gradient check:** compare your analytic gradient to a finite-difference
   estimate `(L(w+ε) − L(w−ε)) / 2ε`, component by component. They should agree
   to ~1e-6. Never trust a gradient you haven't checked.
4. Plot loss vs epoch (smooth decrease), and snapshot the decision boundary as
   it rotates into place.
5. Notice the gradient form is _identical_ to linear regression's — the only
   change is `µ = σ(a)` in place of `ŷ = a`.

**What you learn:** The `(prediction − target)·input` gradient is universal.
Convexity (the Hessian argument in §1.2.2) guarantees GD reaches the global
optimum.

---

## 4. Stochastic and Mini-batch Gradient Descent

**Goal:** Train the way real systems do.

**Steps:**

1. SGD: update on one random sample, `w ← w − η(µₙ − yₙ)xₙ` (§1.3).
2. Mini-batch: average the gradient over 16–32 samples.
3. Plot all three loss curves (full / mini-batch / SGD). SGD is noisy;
   mini-batch is the practical sweet spot.
4. Add a learning-rate schedule (e.g. `ηₜ = η₀ / (1 + decay·t)`). Your notes
   flag that SGD needs a _decaying_ rate to converge — verify it by running
   constant vs decayed `η` and watching the tail of the loss curve.

**What you learn:** The noise-vs-speed tradeoff, and why a learning-rate
schedule matters for convergence on the convex objective.

---

## 5. Second-Order: Newton's Method / IRLS

**Goal:** Use curvature to converge in a handful of steps — something linear
regression never needed.

**Math:** `H = (1/N) Xᵀ S X` with `S = diag(µₙ(1 − µₙ))`. Newton update:
`w ← w − H⁻¹ g` (your eq. 11–15).

**Steps:**

1. Implement the Hessian. Confirm it's positive definite on your data (the
   `‖S^½ X v‖² > 0` argument, §1.2.2).
2. Implement the Newton / IRLS update (`η = 1`). Run it on the Step 3 data.
3. Plot iterations-to-converge: Newton in ~5–10 vs GD in hundreds — same
   optimum, far fewer steps.
4. Recognize the IRLS form: each step is a weighted least-squares solve on the
   working response `zₜ` (eq. 15), with weights `S` recomputed every iteration.
5. Push the data toward separable and watch `S` entries → 0 as `µ → {0,1}`; the
   Hessian goes near-singular (the "practical caveat" in §1.2.2). This sets up
   the next step.

**What you learn:** Curvature buys speed — but on (near-)separable data the
Hessian degenerates, so second-order isn't free.

---

## 6. Regularization, the Separability Blow-up, and Standardization

**Goal:** Fix the pathology where MLE weights run off to infinity.

**Math:** `PNLL = NLL + λ‖w‖²`; `∇PNLL = g + 2λw`; `∇²PNLL = H + 2λI` (§1.3.2).

**Steps:**

1. Make the two blobs _perfectly_ separable. Run plain GD for many epochs and
   watch `‖w‖` grow without bound — the boundary keeps sharpening, the loss → 0
   but the weights never settle.
2. Add the L2 penalty to the gradient and Hessian. `‖w‖` now stabilizes and the
   boundary stops over-sharpening.
3. Sweep `λ` across several orders of magnitude; plot `‖w‖` and the boundary's
   confidence vs `λ`. (Recall `λ = 1/C`, where `C` is the prior variance —
   larger `C` ⇔ weaker regularization.)
4. **Standardization (§1.3.3):** give two features wildly different scales (e.g.
   multiply one by 1000). Show GD crawls. Standardize via `(x − µ̂)/σ̂`; show it
   converges far faster _and_ makes the isotropic Gaussian prior a sensible
   assumption (it penalizes every weight equally, which only makes sense on
   comparable scales).

**What you learn:** Separable data makes unregularized MLE diverge; L2 (a
zero-mean Gaussian prior, i.e. MAP) keeps weights finite. Standardization
improves the conditioning of `H = (1/N)XᵀSX` _and_ justifies the isotropic
prior.

---

---

## 8. Diagnostics and Evaluation

**Goal:** Judge a classifier honestly — accuracy alone lies.

**Steps:**

1. Compute accuracy, precision, recall, F1, and the confusion matrix from your
   predictions by hand.
2. Make the classes imbalanced (e.g. 95/5). Show that an "always predict
   majority" baseline scores 95% accuracy — why accuracy is a trap.
3. Plot the ROC curve and compute AUC by sweeping the threshold on `µ`; also
   plot the precision–recall curve (more informative under imbalance).
4. **Calibration:** bin predictions by `µ` and plot empirical frequency vs
   predicted probability (a reliability diagram). A well-calibrated model sits
   on the diagonal. Note where the plug-in estimate is overconfident — this
   foreshadows Step 10.
5. Fit `sklearn.linear_model.LogisticRegression` and confirm your weights match
   to several decimals. Mind that sklearn parameterizes by `C = 1/λ` and applies
   L2 by default.

**What you learn:** Metrics depend on the threshold and the base rate; ROC / PR
/ calibration tell the real story that a single accuracy number hides.

---

## Extensions — Matching §1.4–1.5 of Your Notes

These go beyond the linear regression guide and round out the rest of your
write-up.

### 9. Robust Logistic Regression

**Goal:** Stop a few mislabeled points from wrecking the fit.

**Math:** Mixture likelihood `p(y|x) = π·Ber(y|0.5) + (1−π)·Ber(y|σ(wᵀx))`
(§1.4.1).

**Steps:**

1. Take 1D two-class data (e.g. a slice of Iris). Fit standard LR; note the
   boundary.
2. Contaminate: add a handful of class-1 points at absurd feature values (your
   contaminated-Iris setup). Refit standard LR — the boundary lurches and
   posterior uncertainty inflates (Fig. 10.10a).
3. Implement the mixture likelihood (the `π` component absorbs outliers) and fit
   by SGD. Show the boundary stays close to the clean fit (Fig. 10.10b).
4. (Stretch) Implement the bi-tempered loss (§1.4.2): the tempered log `logₜ`
   with `t₁ < 1` bounds the loss for _far_ outliers; the tempered softmax with
   `t₂ > 1` puts heavier tails on _near-boundary_ outliers (solve the
   normalization `λ(a)` with the binary-search iteration, Alg. 10.2). Reproduce
   the `t₁ = 0.2, t₂ = 4` "both kinds of noise" case.

**What you learn:** Convex losses are fragile to label noise; a junk-mixture
component or tempered losses bound the damage each outlier can do.

### 10. Bayesian Logistic Regression

**Goal:** Get uncertainty over the boundary, not just a point estimate.

**Math:** Laplace: `p(w|D) ≈ N(w | ŵ, H⁻¹)`. Predictive:
`p(y=1|x,D) = ∫ σ(wᵀx) p(w|D) dw` (§1.5).

**Steps:**

1. Reuse your MAP estimate `ŵ` (Step 6) and the inverse Hessian `H⁻¹` (Step 5) —
   the Laplace approximation comes almost for free from quantities you already
   have.
2. Sample `wₛ ∼ N(ŵ, H⁻¹)`; over a grid of inputs compute the Monte Carlo
   predictive `(1/S) Σ σ(wₛᵀx)`.
3. Plot the mean predictive plus a credible band. Compare to the plug-in
   `σ(ŵᵀx)`: the Bayesian version moderates toward 0.5 where data is scarce
   (§1.5.2).
4. Tie it back: the credible band _is_ the spread of the sampled sigmoids — the
   same mechanism behind the bands in the robust-regression figure.

**What you learn:** A point estimate is overconfident off-data; Laplace + Monte
Carlo turns the Hessian you already computed into calibrated uncertainty.

---

## Suggested File Layout

```
logistic_regression/
├── README.md          # this file
├── notebook.ipynb     # steps 1–8 with markdown explanations
├── extensions.ipynb   # steps 9–10 (robust + Bayesian)
└── utils.py           # shared helpers (sigmoid, data gen, plotting, gradient check)
```

By the end you'll have implemented every section of your logistic-regression
notes — from the sigmoid through Bayesian credible bands — and you'll have
reused the gradient-descent core from your linear regression work, now with a
convex classification loss bolted on top.
