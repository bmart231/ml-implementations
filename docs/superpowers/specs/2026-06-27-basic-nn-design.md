# basic_nn.ipynb — Design Spec
*2026-06-27*

## Goal

Build a comprehensive, from-scratch neural network basics notebook in `neural_network/basic_nn.ipynb` using only NumPy (plus matplotlib for plots). The notebook teaches NN fundamentals by interleaving full mathematical derivations with corresponding NumPy implementations and small experiments.

## Decisions

| Question | Choice | Reason |
|---|---|---|
| Backprop depth | Full chain rule derivation | User wants to understand every partial derivative |
| Problem types | Regression warmup → binary classification | Regression (MSE) is simpler to derive; builds intuition for classification |
| Architecture | Two hidden layers | Enough to see backprop generalize; not so deep it obscures the pattern |
| Structure | Interleaved build-up (math then code, section by section) | Matches existing notebook style (linear_regression.ipynb, decision_trees.ipynb) |

## Notebook Structure

### Section 1 — The Single Neuron
- Intuition: weighted sum + nonlinearity
- Math: `z = W·x + b`, `a = σ(z)`
- NumPy: implement a single neuron forward pass
- Plot: what a neuron with different weights computes over a 1D input

### Section 2 — Activation Functions
- Sigmoid: equation, derivative `σ'(z) = σ(z)(1 - σ(z))`
- ReLU: equation, derivative (subgradient at 0)
- Tanh: equation, derivative
- Side-by-side plot of all three functions and their derivatives
- Note: derivatives are needed verbatim in the backprop derivation later

### Section 3 — Loss Functions
- MSE: `L = (1/n) Σ (y - ŷ)²`, gradient `dL/dŷ = -2/n (y - ŷ)`
- Binary cross-entropy: `L = -[y log(ŷ) + (1-y) log(1-ŷ)]`, gradient `dL/dŷ = -(y/ŷ) + (1-y)/(1-ŷ)`
- NumPy implementations of both
- Plot: loss as a function of prediction for a fixed true label

### Section 4 — Forward Pass: Regression (2 hidden layers)
- Architecture: `X → [W1,b1] → Z1 → ReLU → A1 → [W2,b2] → Z2 → ReLU → A2 → [W3,b3] → Z3 → ŷ`
- Full matrix equations written out explicitly for each layer
- NumPy implementation with random weight init (He initialization)
- Sanity check: shapes match, output is a scalar per sample

### Section 5 — Backpropagation: Full Chain Rule Derivation
- Chain rule refresher (one markdown cell)
- Output layer: `dL/dZ3`, `dL/dW3`, `dL/db3`
- Hidden layer 2: `dL/dA2`, `dL/dZ2`, `dL/dW2`, `dL/db2`
- Hidden layer 1: `dL/dA1`, `dL/dZ1`, `dL/dW1`, `dL/db1`
- Every partial derivative written in full before the NumPy code
- Numerical gradient check: verify `∂L/∂W ≈ (L(W+ε) - L(W-ε)) / 2ε` for a small network

### Section 6 — Training Loop: Regression
- Weight update rule: `W -= lr * dW`, `b -= lr * db`
- Full training loop (epochs, forward, backward, update)
- Synthetic dataset: `y = sin(x) + noise`
- Plots: loss curve over epochs, final predictions vs true curve

### Section 7 — Classification: Binary
- Architecture change: swap output layer activation to sigmoid, loss to cross-entropy
- Re-derive output layer gradient: combined sigmoid + cross-entropy gradient simplifies to `dL/dZ3 = ŷ - y`
- Show the simplification explicitly (this is a key "aha" moment)
- Hidden layers: same backprop as regression
- Training loop on synthetic 2D classification dataset (e.g. two Gaussian blobs)
- Plots: loss curve, decision boundary colored by predicted probability

### Section 8 — What's Next
- Multiclass: softmax output, categorical cross-entropy (mention, don't implement)
- Mini-batch SGD: why full-batch is slow, how batching works
- Regularization: L2 weight decay, dropout
- What CNNs / RNNs add on top of this foundation

## Style Conventions

- Follow existing notebook style: LaTeX math in markdown cells, commented NumPy code cells
- Each section starts with a markdown cell stating what the section covers and why
- No sklearn — all implementations from scratch with NumPy
- Matplotlib for all plots; no seaborn
- Code cells are self-contained within their section (no hidden state dependencies across sections)

## Out of Scope

- Multiclass implementation (mentioned in Section 8 only)
- Mini-batch SGD implementation
- Regularization implementation
- Any deep learning framework (PyTorch, TensorFlow, JAX)
