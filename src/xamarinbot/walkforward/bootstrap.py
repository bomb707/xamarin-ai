"""Bootstrap confidence intervals (Roadmap Phase 11 verification, named
explicitly: "Bootstrap confidence intervals on PnL/EV metrics.")

Reuses Phase 2's `seeded_random` for reproducibility - the same
"reproducible random seeds" utility used for stochastic fill models,
applied here to resampling instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.events.replay import seeded_random


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    n_resamples: int
    confidence: float
    n_samples: int


def bootstrap_ci(values: list[float], n_resamples: int = 1000, confidence: float = 0.95, seed_key: str = "bootstrap") -> BootstrapResult:
    n = len(values)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, n_resamples, confidence, 0)
    if n == 1:
        return BootstrapResult(values[0], values[0], values[0], n_resamples, confidence, 1)

    rng = seeded_random(seed_key, "bootstrap")
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()

    alpha = 1.0 - confidence
    lower_idx = max(0, min(n_resamples - 1, int(alpha / 2 * n_resamples)))
    upper_idx = max(0, min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1))

    return BootstrapResult(
        point_estimate=sum(values) / n,
        lower=means[lower_idx],
        upper=means[upper_idx],
        n_resamples=n_resamples,
        confidence=confidence,
        n_samples=n,
    )
