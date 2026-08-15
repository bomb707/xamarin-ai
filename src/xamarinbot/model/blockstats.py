"""Serial-correlation-aware statistics for round-level metrics (Gate A.0 item 7).

The claim being retracted
-------------------------
`run_continuous_capture.py --status` used to say consecutive five-minute
rounds are "chronologically independent". They are non-overlapping, which is
not the same thing. BTC volatility, spreads, book depth and the direction of
the underlying all persist across five-minute boundaries; two adjacent rounds
share a regime even though they share no data.

Treating serially correlated rounds as IID understates variance, so a
confidence interval computed that way is narrower than the evidence supports
- exactly the error that makes a marginal strategy look significant.

What replaces it
----------------
* `autocorrelation` / `ljung_box_q` to measure whether persistence is
  actually present rather than assuming either way.
* `moving_block_bootstrap` to resample contiguous BLOCKS of rounds, which
  preserves within-block dependence.
* `effective_sample_size` to report how many independent observations the
  correlated series is really worth.
* `block_length_sensitivity` to show the answer across several reasonable
  block lengths, because a single block length is itself a choice.

Block length must never be tuned to maximize significance. Report the
sensitivity; do not pick the winner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from xamarinbot.events.replay import seeded_random


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def autocorrelation(xs: list[float], max_lag: int = 10) -> list[float]:
    """Sample autocorrelation r_k for k = 1..max_lag.

    Uses the standard biased estimator (dividing by n, not n-k), which is
    what block-bootstrap and effective-sample-size formulas assume and which
    is better behaved at long lags on short series.
    """
    n = len(xs)
    if n < 2:
        return []
    mu = mean(xs)
    denom = sum((x - mu) ** 2 for x in xs)
    if denom <= 1e-15:
        return [0.0] * min(max_lag, n - 1)
    out = []
    for k in range(1, min(max_lag, n - 1) + 1):
        num = sum((xs[i] - mu) * (xs[i + k] - mu) for i in range(n - k))
        out.append(num / denom)
    return out


def ljung_box_q(xs: list[float], lags: int = 10) -> tuple[float, int]:
    """Ljung-Box Q statistic and its degrees of freedom.

    Returned without a p-value on purpose: a chi-squared CDF would mean
    either a new dependency or a hand-rolled approximation, and the honest
    use here is comparative - Q far above `df` indicates persistence worth
    respecting in the block length.
    """
    n = len(xs)
    acf = autocorrelation(xs, lags)
    if not acf:
        return 0.0, 0
    q = n * (n + 2) * sum(r * r / (n - k - 1) for k, r in enumerate(acf) if n - k - 1 > 0)
    return q, len(acf)


def autocovariance(xs: list[float], max_lag: int) -> list[float]:
    """Biased sample autocovariances gamma_k for k = 0..max_lag."""
    n = len(xs)
    mu = mean(xs)
    out = []
    for k in range(0, min(max_lag, n - 1) + 1):
        out.append(sum((xs[i] - mu) * (xs[i + k] - mu) for i in range(n - k)) / n)
    return out


def positive_run_autocorrelation_ess(xs: list[float], max_lag: int = 10) -> float:
    """n_eff = n / (1 + 2 * sum_k r_k), summing the INITIAL POSITIVE RUN of
    ordinary autocorrelations.

    This is the simpler, more optimistic estimator: it stops at the first
    negative r_k, so a series whose autocorrelation alternates in sign (a
    real possibility for round-level PnL, where a large win is often
    followed by mean reversion) truncates after one lag and reports almost
    no penalty.

    Kept as a named diagnostic, NOT as `effective_sample_size` - Gate A.0.1
    item 5: it was previously published under Geyer's name, which it is not.
    """
    n = len(xs)
    if n < 2:
        return float(n)
    total = 0.0
    for r in autocorrelation(xs, max_lag):
        if r <= 0:
            break
        total += r
    return max(1.0, min(float(n), n / (1.0 + 2.0 * total)))


def effective_sample_size(xs: list[float], max_lag: int | None = None) -> float:
    """Geyer's initial-positive-sequence effective sample size.

    Geyer's rule pairs ADJACENT lags before testing for positivity:

        Gamma_m = gamma_(2m) + gamma_(2m+1),      m = 0, 1, 2, ...

    and truncates at the first non-positive `Gamma_m`. The pairing is the
    substance of the method, not a detail: for a reversible chain the paired
    sums are provably positive and decreasing, so the truncation point is a
    property of the process rather than of estimator noise at a single lag.
    Summing raw `r_k` and stopping at the first negative one - what this
    function used to do under Geyer's name - throws that away and stops too
    early on any series with alternating-sign autocorrelation, overstating
    how much independent evidence the series carries.

    The variance of the mean is then

        sigma^2 = -gamma_0 + 2 * sum_{m<=M} Gamma_m
        n_eff   = n * gamma_0 / sigma^2

    Clamped to [1, n]: a negative or explosive denominator on a short noisy
    series is an artifact of the estimator, not a real gain in information.
    """
    n = len(xs)
    if n < 2:
        return float(n)
    lag_cap = (n - 1) if max_lag is None else min(max_lag, n - 1)
    gamma = autocovariance(xs, lag_cap)
    if not gamma or gamma[0] <= 1e-15:
        return float(n)

    sigma2 = -gamma[0]
    prev = None
    m = 0
    while 2 * m + 1 < len(gamma):
        pair = gamma[2 * m] + gamma[2 * m + 1]
        if pair <= 0:
            break
        # Geyer's initial MONOTONE sequence refinement: the paired sums are
        # theoretically non-increasing, so enforcing that suppresses
        # estimator noise in the tail rather than letting it accumulate.
        if prev is not None and pair > prev:
            pair = prev
        sigma2 += 2.0 * pair
        prev = pair
        m += 1

    if sigma2 <= 1e-15:
        return float(n)
    return max(1.0, min(float(n), n * gamma[0] / sigma2))


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lo: float
    hi: float
    block_length: int
    n_resamples: int
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def as_dict(self) -> dict:
        return {
            "point": self.point, "lo": self.lo, "hi": self.hi,
            "block_length": self.block_length, "n_resamples": self.n_resamples,
            "confidence": self.confidence, "excludes_zero": self.excludes_zero,
        }


def moving_block_bootstrap(
    xs: list[float],
    block_length: int,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed_key: str = "blockstats",
    statistic=mean,
) -> BootstrapResult:
    """Moving-block bootstrap percentile interval.

    Resamples contiguous blocks of `block_length` consecutive observations
    with replacement, so dependence WITHIN a block survives resampling. An
    ordinary IID bootstrap destroys exactly the structure that makes the
    naive interval too narrow.

    Deterministic given `seed_key` (via Phase 2's `seeded_random`), so a
    reported interval is reproducible and cannot drift between runs.
    """
    n = len(xs)
    if n == 0:
        raise ValueError("cannot bootstrap an empty series")
    block_length = max(1, min(block_length, n))
    n_blocks = math.ceil(n / block_length)
    starts = n - block_length + 1
    rng = seeded_random(seed_key, f"{block_length}:{n_resamples}:{n}")

    stats = []
    for _ in range(n_resamples):
        sample: list[float] = []
        for _ in range(n_blocks):
            s = rng.randrange(starts)
            sample.extend(xs[s:s + block_length])
        stats.append(statistic(sample[:n]))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = stats[max(0, int(alpha * len(stats)) - 1)]
    hi = stats[min(len(stats) - 1, int((1.0 - alpha) * len(stats)))]
    return BootstrapResult(
        point=statistic(xs), lo=lo, hi=hi, block_length=block_length,
        n_resamples=n_resamples, confidence=confidence,
    )


#: Block lengths to report, in MINUTES of wall clock. Rounds are 5 minutes,
#: so these are 6 / 12 / 24 consecutive rounds. Reported together; never
#: selected by which one gives the desired answer.
DEFAULT_BLOCK_MINUTES = (30, 60, 120)
ROUND_MINUTES = 5


@dataclass
class BlockSensitivity:
    """The same estimate under several reasonable block lengths."""

    series_name: str
    n_rounds: int
    point: float
    autocorr: list[float] = field(default_factory=list)
    ljung_box: tuple[float, int] = (0.0, 0)
    #: Geyer initial-positive-sequence ESS - the headline number.
    n_eff: float = 0.0
    #: The simpler initial-positive-RUN estimator, reported alongside so the
    #: two are visibly different rather than one masquerading as the other.
    n_eff_positive_run: float = 0.0
    by_block: dict[int, BootstrapResult] = field(default_factory=dict)
    suggested_block_rounds: int = 1

    def as_dict(self) -> dict:
        return {
            "series": self.series_name,
            "n_rounds": self.n_rounds,
            "point": self.point,
            "autocorrelation_lag1_5": self.autocorr[:5],
            "ljung_box_q": self.ljung_box[0],
            "ljung_box_df": self.ljung_box[1],
            "effective_sample_size": self.n_eff,
            "effective_sample_size_estimator": "geyer_initial_positive_sequence",
            "positive_run_autocorrelation_ess": self.n_eff_positive_run,
            "suggested_block_rounds": self.suggested_block_rounds,
            "blocks": {
                f"{b * ROUND_MINUTES}min": r.as_dict() for b, r in sorted(self.by_block.items())
            },
        }


def suggest_block_rounds(xs: list[float], max_lag: int = 10) -> int:
    """A data-driven block length: the first lag at which autocorrelation
    stops being positive, i.e. how far dependence actually reaches.

    Offered as a REPORTED diagnostic alongside the fixed 30/60/120-minute
    grid, not as a replacement for it - a block length chosen from the data
    is one more thing that could be tuned, so the fixed grid stays the
    primary evidence.
    """
    for k, r in enumerate(autocorrelation(xs, max_lag), start=1):
        if r <= 0:
            return k
    return min(max_lag, max(1, len(xs) // 4)) or 1


def analyze_series(
    xs: list[float],
    series_name: str,
    block_minutes: tuple[int, ...] = DEFAULT_BLOCK_MINUTES,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> BlockSensitivity:
    """Full item-7 treatment of one round-level metric series.

    `xs` must be ONE value per round, in chronological order - never one per
    intra-round decision point.
    """
    result = BlockSensitivity(
        series_name=series_name,
        n_rounds=len(xs),
        point=mean(xs),
        autocorr=autocorrelation(xs, 10),
        ljung_box=ljung_box_q(xs, min(10, max(1, len(xs) // 2))),
        n_eff=effective_sample_size(xs),
        n_eff_positive_run=positive_run_autocorrelation_ess(xs),
        suggested_block_rounds=suggest_block_rounds(xs),
    )
    for minutes in block_minutes:
        b = max(1, minutes // ROUND_MINUTES)
        result.by_block[b] = moving_block_bootstrap(
            xs, b, n_resamples=n_resamples, confidence=confidence,
            seed_key=f"{series_name}:{minutes}",
        )
    return result
