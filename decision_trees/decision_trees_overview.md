# Decision Trees — Roadmap

> Topic 5 of the Foundations track. The conceptual foundation for everything in
> the Ensembles track (random forests, gradient boosting, XGBoost).

## 1. Core idea and geometry

- A tree recursively partitions feature space into **axis-aligned boxes**; the
  prediction is **constant within each box**.
- Contrast with what you've already covered: the smooth boundary of logistic
  regression, the distance-based regions of KNN.

## 2. The two flavors

- **Classification trees** — leaf predicts the majority class / class
  distribution.
- **Regression trees** — leaf predicts the mean of the targets.
- Same algorithm; different leaf outputs and split criteria.

## 3. How a split is chosen (the heart of it)

- **Greedy, top-down recursive binary splitting**: at each node, scan every
  feature and every candidate threshold, pick the split that most reduces
  impurity.
- Impurity measures:
  - Classification: **Gini** and **entropy** (information gain).
  - Regression: **variance / SSE reduction**.
- Understand _why_ Gini and entropy are used for growing rather than plain
  misclassification error (more sensitive to changes in class proportions).

## 4. Why a fully grown tree overfits

- Grown to purity, a tree memorizes the training set: **low bias, very high
  variance**.
- The central tension of the whole topic. Tie back to the bias–variance picture.

## 5. Controlling complexity

- Hyperparameters that constrain growth: `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `min_impurity_decrease`, `max_leaf_nodes`.
- **Pre-pruning** (stop early) vs **post-pruning** (grow then cut back).

## 6. Cost-complexity pruning (CART)

- The principled post-pruning method: grow the full tree, then prune the weakest
  links by penalizing leaf count with a parameter **α**.
- Connects directly to the regularization ideas seen elsewhere (e.g. LASSO).

## 7. Practical handling

- Categorical features, missing values and **surrogate splits**.
- Algorithm families at a high level: **CART** (binary splits, Gini) vs **ID3 /
  C4.5**.
- CART is what scikit-learn implements — your reference point.

## 8. Properties and failure modes

- **Strengths**: interpretable, no feature scaling needed, captures interactions
  and nonlinearity for free.
- **Weaknesses**: instability (small data change → very different tree), greedy
  splitting is only locally optimal, axis-aligned limitation, bias toward
  high-cardinality features.

## 9. Feature importance

- How it's computed: total **weighted impurity decrease** attributable to each
  feature.
- Why it can mislead: inflates high-cardinality and correlated features.
- You'll lean on this in TickerTracker — know its caveats before trusting it.

## 10. The bridge to ensembles

- A single tree's high variance is exactly what **bagging** averages away (→
  random forests) and what **boosting** attacks sequentially (→ gradient
  boosting, XGBoost).
- This is _why_ the Ensembles track exists.

---

## Implementation milestone

Build a tree from scratch to cement the concepts:

- A recursive **node class**.
- A **best-split search** by impurity.
- A **stopping rule**.
- Check it against sklearn's `DecisionTreeClassifier` / `DecisionTreeRegressor`.
