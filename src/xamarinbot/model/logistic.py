"""Pure-Python, L2-regularized logistic regression (Roadmap Phase 5: "Build
a logistic-regression baseline ... Use regularization and walk-forward
validation.")

Deliberately dependency-free (no numpy/scikit-learn) to match this
project's minimal-dependency stance elsewhere - dataset sizes here (a few
thousand rows, under a dozen features) don't need a vectorized
implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_Z_CLIP = 35.0  # keeps exp() arguments in a safe range


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-_Z_CLIP, min(_Z_CLIP, z))))


@dataclass(frozen=True)
class LogisticModel:
    feature_set_name: str
    column_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def predict_proba(self, x: list[float]) -> float:
        x_std = [(xi - m) / s for xi, m, s in zip(x, self.means, self.stds)]
        z = self.bias + sum(w * xi for w, xi in zip(self.weights, x_std))
        return _sigmoid(z)


def _fit_standardization(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(X)
    d = len(X[0])
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return means, stds


def fit_logistic_regression(
    X: list[list[float]],
    y: list[int],
    feature_set_name: str,
    column_names: tuple[str, ...],
    l2: float = 1.0,
    lr: float = 0.3,
    n_iters: int = 500,
) -> LogisticModel:
    """Batch gradient descent on the L2-regularized log-loss. Features are
    standardized (fit on this call's X only - callers must pass the
    training split, never validation/test, to avoid leakage)."""
    if not X:
        raise ValueError("cannot fit a model on zero examples")
    n = len(X)
    d = len(X[0])
    means, stds = _fit_standardization(X)
    Xs = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X]

    weights = [0.0] * d
    bias = 0.0
    # Proximal gradient descent: an explicit gradient step on the L2 term
    # (weights[j] -= lr * l2 * weights[j]) is only stable while
    # lr * l2 < 2 - anything past that oscillates with growing amplitude
    # and diverges to +-inf within a handful of iterations for no obvious
    # reason at the call site. The proximal/shrinkage update below
    # (`weights *= 1 / (1 + 2*lr*l2)` after the ordinary data-loss step) is
    # the exact solution to the L2 sub-problem and is unconditionally
    # stable for every l2 >= 0, so a caller's regularization choice can
    # never blow up the optimizer.
    shrink = 1.0 / (1.0 + 2.0 * lr * l2) if l2 > 0 else 1.0
    for _ in range(n_iters):
        grad_w = [0.0] * d
        grad_b = 0.0
        for xi, yi in zip(Xs, y):
            z = bias + sum(w * x for w, x in zip(weights, xi))
            err = _sigmoid(z) - yi
            for j in range(d):
                grad_w[j] += err * xi[j]
            grad_b += err
        for j in range(d):
            weights[j] = (weights[j] - lr * grad_w[j] / n) * shrink
        bias -= lr * grad_b / n

    return LogisticModel(
        feature_set_name=feature_set_name,
        column_names=column_names,
        weights=tuple(weights),
        bias=bias,
        means=tuple(means),
        stds=tuple(stds),
    )
