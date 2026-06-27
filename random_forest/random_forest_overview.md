# Random Forest — Roadmap

> Topic 7 of the list, first in the Ensembles track. Builds directly on Decision
> Trees: a random forest is the answer to a single tree's central flaw (high
> variance). The natural next step after this is gradient boosting.

## 1. Core idea and motivation

- A random forest is an **ensemble of decision trees** whose predictions are
  combined (averaged for regression, voted for classification).
- The point, straight from the decision trees topic: a single deep tree is **low
  bias but high variance** — it overfits and is unstable. A forest **averages
  many trees to cancel that variance** while keeping the low bias.
- Two ingredients make it work: **bagging** (resampling) + **random feature
  subsampling** (decorrelation). Everything else is detail.

## 2. Bagging and the bootstrap

- **Bootstrap**: sample `n` rows _with replacement_ from the training set, so
  each tree sees a slightly different dataset.
- **Aggregating**: train one tree per bootstrap sample, then combine — the
  "bagging" = **B**ootstrap **AGG**regat**ing**.
- The variance-reduction math (the heart of it): for `B` trees each with
  variance `σ²` and pairwise correlation `ρ`,
  $$\text{Var}(\text{average}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.$$ More
  trees shrink the second term toward 0; **bias is unchanged** by averaging.

## 3. The decorrelation trick: random feature subsampling

- This is what turns _bagged trees_ into a _random forest_.
- At **each split**, the tree may only consider a **random subset of `m`
  features** (not all of them) — controlled by `max_features`.
- Why it matters: it **lowers `ρ`** (trees stop all picking the same dominant
  feature), which lowers the `ρσ²` variance **floor** from step 2. Decorrelated
  trees average better.

## 4. Out-of-bag (OOB) error

- Each bootstrap sample leaves out roughly **37%** of rows (the `1/e ≈ 0.368`
  result). Those are the tree's **out-of-bag** samples.
- For each row, aggregate predictions only from the trees that _didn't_ train on
  it → a **free, built-in validation estimate** with no separate hold-out needed
  (`oob_score=True`).
- Caveat for time-series (TickerTracker): OOB assumes exchangeable rows, so it
  can be **optimistic on temporally correlated data** — still prefer a
  chronological hold-out there.

## 5. Aggregating predictions

- **Regression**: average the per-tree predictions.
- **Classification**: sklearn averages the per-tree **class probabilities**
  (soft voting), then takes the argmax — not a raw majority vote of hard labels.
- Probability estimates from a forest are smoother and better-calibrated than a
  single tree's leaf proportions.

## 6. Hyperparameters and tuning

- **`n_estimators`** — number of trees. More is (almost) always safer; accuracy
  plateaus and you trade compute for diminishing returns. _Not_ a source of
  overfitting.
- **`max_features`** — the key knob. Defaults: `sqrt(p)` for classification,
  `p/3` for regression. Smaller → more decorrelation (lower variance, slightly
  higher bias).
- **Tree-size controls** (`max_depth`, `min_samples_leaf`, `max_leaf_nodes`) —
  forests usually grow trees **deep** (variance is handled by averaging), but
  capping leaf size can still help on noisy data.
- **`bootstrap` / `max_samples`** — whether to resample and how big each sample
  is.

## 7. Feature importance in forests

- **MDI** (mean decrease in impurity), **averaged over all trees** — more stable
  than a single tree's, but **inherits the same biases** from the decision trees
  topic: inflates **high-cardinality** features and **splits credit among
  correlated** ones.
- **Permutation importance** on a hold-out set is the more honest cross-check
  (model-agnostic, generalization-based).
- For TickerTracker: averaging over trees helps, but still verify rankings with
  permutation importance on a **chronological** hold-out.

## 8. Properties and failure modes

- **Strengths**: strong out-of-the-box accuracy with little tuning, robust to
  noise/outliers, no feature scaling, captures interactions for free,
  **embarrassingly parallel** (`n_jobs`), built-in OOB validation, hard to
  overfit by adding trees.
- **Weaknesses**: **less interpretable** than a single tree (no readable
  flowchart), **large memory / slower inference**, can't **extrapolate** beyond
  the training target range (predictions are bounded by training leaf values —
  matters for regression), and still **high-cardinality bias** in importances.

## 9. Variants and the bridge to boosting

- **Extremely Randomized Trees (ExtraTrees)**: also randomize the _split
  thresholds_, not just the feature subset → faster, even more decorrelation,
  sometimes lower variance.
- **Random forest vs boosting** (the hand-off to the next topic):
  - RF builds trees **in parallel, independently**, to reduce **variance**.
  - Boosting builds trees **sequentially**, each correcting the last, to reduce
    **bias**.
  - RF is easier to tune and harder to overfit; boosting often reaches higher
    accuracy but needs more care. This contrast sets up gradient boosting and
    XGBoost.

---

## Implementation milestone

Build a random forest on top of the decision tree you wrote earlier:

- A **bootstrap sampler** (sample `n` indices with replacement).
- A loop training `B` trees, each with **per-split random feature subsampling**.
- An **aggregator** (average / soft-vote) for prediction.
- An **OOB score** computed from each tree's left-out rows.
- Check it against sklearn's `RandomForestClassifier` / `RandomForestRegressor`,
  and compare its variance to a single tree's (reuse the bias–variance
  decomposition from the decision trees notes).
