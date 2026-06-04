# k-Nearest Neighbors — Learning Guide

An 8-step progression, in the same spirit as the linear and logistic regression
guides: each step teaches something the previous one hid. Do them in order,
ideally in a single Jupyter notebook with markdown cells explaining your
thinking.

This one is a different _kind_ of model, and that's the whole point. Linear and
logistic regression were **parametric**: you trained a fixed set of weights and
then threw the data away. KNN is **non-parametric, lazy, and instance-based** —
there is no training step, the training set _is_ the model, and all the work
happens at prediction time. Most of these steps are unpacking the consequences
of that single shift.

---

## 1. The Whole Algorithm: Brute-Force KNN from Scratch

**Goal:** Implement the entire algorithm in ~15 lines, and sit with the fact
that there's no training.

**Math:** To classify a query `z`, compute the distance to every training point,
keep the `k` closest, and predict the majority label among them. Euclidean
distance:

```
d(z, x) = √ Σᵈ (zᵈ - xᵈ)²
```

**Steps:**

1. "Fit" = just store `X_train, y_train`. Zero parameters, zero optimization.
   Notice how strange that feels after logistic regression.
2. Implement `predict`: for each query, compute Euclidean distances to all
   training points, `argsort` them, take the top-`k` indices, and majority-vote
   (`np.bincount` or `Counter`).
3. Test on 2D `make_moons` (or 2-class Iris). Plot the decision boundary with a
   meshgrid + `contourf`.
4. **Vectorize** the distance computation — for query matrix `Z` and train
   matrix `X`, all squared distances at once: `‖Z‖² + ‖X‖² - 2 Z Xᵀ`. Time it
   against the loop; it's orders of magnitude faster.
5. Compare predictions against `sklearn.neighbors.KNeighborsClassifier`. They
   should match exactly.

**What you learn:** KNN is "lazy" — it defers all computation to inference. The
model is literally the training set; the cost just moved from training time to
prediction time.

---

## 2. Choosing k: The Bias-Variance Knob

**Goal:** See how the single hyperparameter `k` trades overfitting against
underfitting.

**Steps:**

1. On noisy `make_moons`, plot the decision boundary for `k = 1, 5, 25, 101`.
2. At `k = 1`: zero training error and a jagged boundary that carves little
   islands around noisy points — pure memorization, maximum variance.
3. At large `k`: a smooth, nearly linear boundary; at `k = N` it just predicts
   the majority class everywhere — maximum bias.
4. Plot training and validation error vs. `k` — the classic U-shaped validation
   curve. Note the effective model complexity scales like `N/k` (small `k` ⇒
   many effective parameters).
5. Pick `k` by cross-validation. For binary problems, prefer **odd** `k` to
   avoid tie votes.

**What you learn:** `k` is your bias-variance dial — `k = 1` is maximum
variance, `k = N` is maximum bias. Cross-validation finds the sweet spot.

---

## 3. Distance Metrics & Feature Scaling (the distance _is_ the model)

**Goal:** Realize that for KNN, the choice of distance function is the entire
model — and feature scale silently controls it.

**Math:**

- Manhattan (L1): `Σᵈ |zᵈ - xᵈ|`
- Minkowski (Lp): `(Σᵈ |zᵈ - xᵈ|ᵖ)^(1/p)` — generalizes L1 (`p=1`) and L2
  (`p=2`)
- Cosine distance: `1 - (z·x) / (‖z‖‖x‖)` — direction, not magnitude

**Steps:**

1. Take a dataset where one feature lives in `[0,1]` and another in `[0, 10000]`
   (e.g. multiply one column by `1e4`). Run KNN — the large-scale feature
   dominates every distance and the small one is effectively ignored.
2. Standardize features (`(x - µ̂)/σ̂`, **training stats only** — the leakage rule
   you already know). Re-run; accuracy jumps. This is the single most common KNN
   bug.
3. Swap in Manhattan and Minkowski distances; compare the boundaries. Note L1 is
   more robust to a single wild dimension.
4. Try cosine distance on text/embedding-style data (or L2-normalized vectors) —
   for when _direction_ matters more than magnitude.

**What you learn:** KNN has no weights to learn, so the distance metric and
feature scaling carry _all_ of the modeling. Forgetting to standardize is the
number-one way KNN silently fails.

---

## 4. The Curse of Dimensionality

**Goal:** Understand the deepest reason KNN — and much of ML — struggles in high
dimensions.

**Math:** To capture a fraction `r` of the data inside an axis-aligned cube in
`D` dimensions, the cube's edge length must be `r^(1/D)`. In 10 dimensions,
capturing just 1% of points needs edge `0.01^(0.1) ≈ 0.63` — 63% of each axis's
range. "Neighbors" are no longer local.

**Steps:**

1. Sample `N` uniform points in the `D`-dim unit cube. For some query points,
   compute `(max_dist - min_dist) / min_dist` as `D = 2, 10, 100, 1000`. Watch
   it collapse toward 0 — nearest and farthest points become nearly equidistant.
2. Plot the distribution of pairwise distances for growing `D` — it concentrates
   into a thin shell.
3. Conclude: in high `D`, "nearest" stops being meaningful, so KNN degenerates
   into noise.
4. Mitigations to try: PCA before KNN, feature selection, or learned embeddings.
   Re-run KNN on PCA-reduced features and measure the lift.

**What you learn:** As `D` grows, volume concentrates in a shell and all
distances converge. This is _why_ KNN wants low-dimensional, dense data — and
why good embeddings are so valuable.

---

## 5. KNN Regression & Distance Weighting

**Goal:** See that KNN isn't just a classifier, and that _how_ you combine the
neighbors changes everything.

**Math:**

- Regression (uniform): `ŷ(z) = (1/k) Σ yₙ` over the `k` nearest neighbors.
- Distance-weighted: `ŷ(z) = Σ wₙ yₙ / Σ wₙ`, with `wₙ = 1/d(z, xₙ)` or a
  Gaussian kernel `wₙ = exp(-d²/2h²)`.

**Steps:**

1. Generate 1D data `y = sin(x) + noise`. Implement KNN regression (average of
   the `k` nearest targets). Plot the fit for `k = 1, 5, 25`.
2. Notice the fit is a piecewise-constant **step function** — discontinuous and
   ugly.
3. Switch to **distance-weighted** averaging so closer neighbors count more. The
   fit smooths out.
4. Use a Gaussian kernel weight with bandwidth `h`. Recognize this as the
   **Nadaraya–Watson kernel regression** estimator. The connection: fixed-`k`
   KNN ≈ a _variable-bandwidth_ kernel; fixed-`h` kernel ≈ _variable-k_.
5. Look at the edges of the data — KNN can't extrapolate, it just repeats its
   boundary neighbors.

**What you learn:** KNN does regression by averaging and classification by
voting — same idea. Distance weighting turns the jagged step function into
smooth kernel regression, KNN's continuous cousin.

---

## 6. Making It Fast: KD-Trees, Ball Trees & Approximate NN

**Goal:** Confront that brute-force KNN is `O(ND)` per query, and learn the
structures that fix it.

**Steps:**

1. Time brute-force KNN as `N` grows toward `10⁵`–`10⁶`. It scales linearly per
   query — unusable at scale.
2. Build a **KD-tree** (`algorithm='kd_tree'`): a binary tree splitting on
   alternating axes, giving roughly `O(log N)` queries in low dimensions by
   pruning whole branches.
3. Crank `D` up and watch the KD-tree degrade back toward brute force around
   `D ≳ 20` — the curse of dimensionality again (pruning fails once everything
   is equidistant).
4. Try a **Ball tree** (`algorithm='ball_tree'`): partitions into nested
   hyperspheres and copes somewhat better in higher `D`.
5. Read about **approximate nearest neighbors** — LSH, and HNSW graphs (the
   engine behind FAISS and modern vector databases). Trade a sliver of accuracy
   for huge speedups; this is how embedding search and RAG retrieval actually
   run at scale.

**What you learn:** Exact KNN doesn't scale, so you either index it (KD/Ball
trees, good in low `D`) or approximate it (HNSW/LSH, the only real option in
high `D`) — and the curse of dimensionality even sabotages the indexing.

---

## 7. The Probabilistic & Theoretical View

**Goal:** Understand _why_ KNN works, and connect it to density estimation and a
famous error bound.

**Math:**

- As a classifier, KNN estimates `p(y=c | z) ≈ kᶜ / k` — the fraction of the `k`
  neighbors in class `c`.
- This is a local density estimate: `p(z | y=c) ∝ kᶜ / (Nᶜ · V)`, where `V` is
  the volume of the ball containing the `k` neighbors. The bandwidth is
  _adaptive_ — it's whatever radius encloses `k` points.

**Steps:**

1. Instead of a hard vote, output the class probabilities `kᶜ/k` and plot them
   as a probability surface. Compare to logistic regression's smooth sigmoid
   surface.
2. Reframe KNN as a non-parametric density estimator with an adaptive bandwidth,
   and contrast it with a fixed-bandwidth kernel density estimate.
3. Read the **Cover–Hart (1967)** result: as `N → ∞`, the 1-NN error rate is
   bounded by `Bayes ≤ err₁ₙₙ ≤ 2·Bayes`. Sit with how strong that is — a
   trivial rule loses at most a factor of 2 against the optimal classifier,
   asymptotically.
4. Empirically: generate data with a known Bayes boundary, estimate the 1-NN
   error for growing `N`, and check it tracks toward the bound.

**What you learn:** KNN is local density estimation in disguise; its vote
fractions are probability estimates; and a single nearest neighbor already
captures (asymptotically) at least half the achievable information.

---

## 8. Evaluation & Where KNN Belongs

**Goal:** Evaluate KNN honestly and build a sharp sense of when to reach for it.

**Steps:**

1. Reuse your classification eval suite (accuracy, confusion matrix, ROC/AUC,
   calibration, log-loss). Note that KNN's probabilities `kᶜ/k` take only `k+1`
   distinct values, so calibration is chunky — larger `k` or distance weighting
   smooths it.
2. Run a head-to-head against logistic regression: on `make_moons` (KNN wins —
   nonlinear boundary, no feature engineering) and on a clean, linearly
   separable, higher-`D` set (logistic wins — fast, extrapolates, scale-robust).
3. Write a decision checklist. **Reach for KNN when:** the boundary is
   irregular/nonlinear, dimensionality is low-to-moderate, data is dense and
   plentiful, you want a zero-training baseline, or "show me similar examples"
   _is_ the goal. **Avoid it when:** `D` is large, data is sparse, inference
   latency or memory is tight, or you need an interpretable model.
4. Re-list the footguns one last time: scale your features, pick `k` by CV,
   handle ties, and budget for slow inference.

**What you learn:** KNN is the dead-simple, strong baseline that thrives on
low-dimensional nonlinear problems with dense data, and falls apart in high
dimensions or under latency constraints. Knowing _when_ it's the right tool is
most of the skill.

---

## Suggested File Layout

```
k_nearest_neighbors/
├── README.md          # this file
├── notebook.ipynb     # all 8 steps with explanations
└── utils.py           # distance functions, brute-force + vectorized KNN, plotting, metrics
```

KNN is your first non-parametric model, and it's the cleanest illustration of
three ideas you'll meet everywhere: the bias-variance tradeoff (Step 2), the
curse of dimensionality (Step 4), and "the distance metric is the model" (Step
3). It also quietly underpins a surprising amount of modern ML — kernel methods,
Gaussian processes, and the entire world of embedding similarity search (vector
databases, retrieval, RAG). The "find the most similar examples" instinct you
build here is the same one running underneath those systems.
