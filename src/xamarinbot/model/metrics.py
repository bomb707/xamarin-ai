"""Probability-quality metrics (Roadmap Phase 5 verification: "Brier/log-
loss. Calibration plots. Out-of-sample settlement accuracy. Economic edge
conditional on q-price.")

"Report q calibration by time bucket, gap bucket, price bucket and
volatility regime" - `calibration_by_group` is the shared machinery for all
four groupings (see scripts/run_model_training_demo.py for the key
functions used for each).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Hashable

from xamarinbot.model.dataset import Example


def brier_score(q: list[float], y: list[int]) -> float:
    """Brier = mean((q_i - y_i)^2), Strategy doc SS9."""
    return sum((qi - yi) ** 2 for qi, yi in zip(q, y)) / len(y)


def log_loss(q: list[float], y: list[int], eps: float = 1e-12) -> float:
    total = 0.0
    for qi, yi in zip(q, y):
        p = min(1.0 - eps, max(eps, qi))
        total += -(yi * math.log(p) + (1 - yi) * math.log(1.0 - p))
    return total / len(y)


def settlement_accuracy(q: list[float], y: list[int], threshold: float = 0.5) -> float:
    """Out-of-sample settlement accuracy: does argmax(q) match the outcome."""
    correct = sum(1 for qi, yi in zip(q, y) if (qi >= threshold) == bool(yi))
    return correct / len(y)


@dataclass(frozen=True)
class CalibrationBucket:
    n: int
    mean_predicted: float
    empirical_rate: float

    @property
    def gap(self) -> float:
        """mean_predicted - empirical_rate; 0 is perfect calibration."""
        return self.mean_predicted - self.empirical_rate


def calibration_table(q: list[float], y: list[int], n_bins: int = 10) -> dict[int, CalibrationBucket]:
    """Equal-width [0,1] bin calibration table ("calibration plot" data)."""
    bins: dict[int, list[tuple[float, int]]] = {}
    for qi, yi in zip(q, y):
        b = min(n_bins - 1, max(0, int(qi * n_bins)))
        bins.setdefault(b, []).append((qi, yi))
    return {
        b: CalibrationBucket(
            n=len(items),
            mean_predicted=sum(x[0] for x in items) / len(items),
            empirical_rate=sum(x[1] for x in items) / len(items),
        )
        for b, items in sorted(bins.items())
    }


def calibration_by_group(
    examples: list[Example], q: list[float], key_fn: Callable[[Example], Hashable]
) -> dict[Hashable, CalibrationBucket]:
    """Groups by an arbitrary key (time bucket, gap bucket, price bucket,
    volatility regime, ...) and reports calibration per group."""
    groups: dict[Hashable, list[tuple[float, int]]] = {}
    for ex, qi in zip(examples, q):
        groups.setdefault(key_fn(ex), []).append((qi, ex.y))
    return {
        key: CalibrationBucket(
            n=len(items),
            mean_predicted=sum(x[0] for x in items) / len(items),
            empirical_rate=sum(x[1] for x in items) / len(items),
        )
        for key, items in groups.items()
    }


def economic_edge(q: list[float], price: list[float]) -> float:
    """Mean(q - price): Strategy doc SS2.1 Pi_one_share = q - c. A positive
    average edge means the model's fair value tends to exceed the price it
    would need to pay - necessary but not sufficient for profitability
    once execution costs (Phase 7) are included."""
    return sum(qi - pi for qi, pi in zip(q, price)) / len(price)
