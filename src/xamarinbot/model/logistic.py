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


def _fit_standardization(
    X: list[list[float]], w: list[float] | None = None
) -> tuple[list[float], list[float]]:
    """Feature standardization, optionally WEIGHTED (Gate A.0 item 6).

        mu_j     = sum_i w_i x_ij / sum_i w_i
        sigma^2_j = sum_i w_i (x_ij - mu_j)^2 / sum_i w_i

    Weighting matters here as much as in the fit itself: an unweighted mean
    over rows lets a busy round's 80 decision points pull the centering twice
    as far as a quiet round's 40, so the model would be standardized against
    a distribution that over-represents high-event-rate market conditions.
    """
    d = len(X[0])
    if w is None:
        w = [1.0] * len(X)
    sw = sum(w)
    if sw <= 0:
        raise ValueError("total sample weight must be positive")
    means = [sum(wi * row[j] for wi, row in zip(w, X)) / sw for j in range(d)]
    stds = []
    for j in range(d):
        var = sum(wi * (row[j] - means[j]) ** 2 for wi, row in zip(w, X)) / sw
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
    sample_weight: list[float] | None = None,
) -> LogisticModel:
    """Batch gradient descent on the L2-regularized log-loss. Features are
    standardized (fit on this call's X only - callers must pass the
    training split, never validation/test, to avoid leakage).

    `sample_weight` (Gate A.0 item 6) makes each ROUND contribute equal total
    weight rather than each decision row. The target is one settlement
    outcome per round while `q` is evaluated at many intra-round decision
    times, so unweighted rows are pseudo-replicates: a round with twice as
    many valid decision points would exert twice the influence on the fit
    despite carrying exactly one independent observation of the outcome.

    Weights enter both the gradient and the normalization, so the update is
    a weighted mean rather than a plain one - scaling every weight by a
    constant leaves the fit unchanged.
    """
    if not X:
        raise ValueError("cannot fit a model on zero examples")
    n = len(X)
    d = len(X[0])
    if sample_weight is None:
        sample_weight = [1.0] * n
    elif len(sample_weight) != n:
        raise ValueError(
            f"sample_weight has {len(sample_weight)} entries for {n} examples"
        )
    total_w = sum(sample_weight)
    if total_w <= 0:
        raise ValueError("total sample weight must be positive")
    means, stds = _fit_standardization(X, sample_weight)
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
        for xi, yi, wi in zip(Xs, y, sample_weight):
            z = bias + sum(w * x for w, x in zip(weights, xi))
            err = wi * (_sigmoid(z) - yi)
            for j in range(d):
                grad_w[j] += err * xi[j]
            grad_b += err
        # Normalizing by TOTAL WEIGHT rather than row count keeps the step
        # size independent of how the weights happen to be scaled.
        for j in range(d):
            weights[j] = (weights[j] - lr * grad_w[j] / total_w) * shrink
        bias -= lr * grad_b / total_w

    return LogisticModel(
        feature_set_name=feature_set_name,
        column_names=column_names,
        weights=tuple(weights),
        bias=bias,
        means=tuple(means),
        stds=tuple(stds),
    )
