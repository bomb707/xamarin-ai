"""REAL-data causal example path (Gate A.0 items 4, 5, 6).

Three separate defects in the synthetic-era `build_examples_multi` make it
unusable for real data, and they compound:

**Item 4 - every market event was a decision point.**
`ReplayClock.decision_points()` emits one timestamp per event. On synthetic
data (1 Hz ticks) that is a tidy grid. On real data the CLOB book updates
~130 times a second, so a single round yields tens of thousands of decision
points - and a round that happened to be *busier* would contribute
proportionally more training rows purely because the market was noisy. Event
rate is not information about the outcome; letting it set example count lets
market microstructure vote on the fit.

The replacement is a fixed, preregistered grid inside the tradable window,
derived from the strategy's own execution cadence rather than tuned from
outcomes.

**Item 5 - visibility was source-time only.**
`compute()` filters `event_time <= decision_ts`, which for a real capture is
the EXCHANGE's clock. A live system cannot act on an observation it has not
received yet. A usable observation must satisfy BOTH

    recv_ts   <= t      (we have it)
    source_ts <= t      (it had happened)

so the caller pre-filters on `recv_ts` and `compute()` keeps enforcing its
own invariant on top. Applied identically to TRAIN, CALIBRATE, VALIDATE and
TEST - a gate that differs between splits manufactures the very optimism the
splits exist to detect.

**Item 6 - within-round pseudo-replication.**
There is ONE settlement outcome per round, but `q` is evaluated at many
decision points inside it. Treating those rows as independent observations
lets a single round with many valid decisions dominate the fit, and inflates
every significance estimate. Each round therefore carries total weight 1,
spread evenly over its own valid decision points:

    sum_{t in r} w_{r,t} = 1,     w_{r,t} = 1 / N_r
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.events.store import EventStore
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.model.features import FeatureSet, design_vector
from xamarinbot.provenance import DataProvenance, require_real
from xamarinbot.rounds import RoundLabel

#: The preregistered decision grid, in seconds after the round opens.
#:
#: Start 15s in: the first seconds of a BTC 5-minute round have no realized
#: volatility history and no lead-lag signal, and `FeatureConfig.vol_min_samples`
#: cannot be satisfied yet. Stop 30s before the close: `SETTLEMENT_PROTECTION`
#: begins at 270s and the book routinely goes one-sided once the outcome is
#: effectively decided.
#:
#: 3-second spacing is the strategy's own execution cadence, NOT a value tuned
#: against outcomes. It is declared here, once, as a constant so that changing
#: it is a visible, reviewable act rather than an experiment knob.
DECISION_GRID_START_S = 15.0
DECISION_GRID_END_S = 270.0
DECISION_GRID_STEP_S = 3.0


def decision_grid(
    start_s: float = DECISION_GRID_START_S,
    end_s: float = DECISION_GRID_END_S,
    step_s: float = DECISION_GRID_STEP_S,
) -> tuple[float, ...]:
    """`t = 15, 18, 21, ..., 270` seconds after the round open.

    Deterministic and independent of the data: the same grid for a quiet
    round and a violent one, so event-rate inflation cannot change the
    example count.
    """
    out, t = [], start_s
    while t <= end_s + 1e-9:
        out.append(round(t, 6))
        t += step_s
    return tuple(out)


@dataclass(frozen=True)
class RealExample:
    """One (features, target) row, carrying its round-balanced weight."""

    round_id: str
    #: Seconds after the round open - the grid offset, not a wall clock.
    t: float
    decision_ts: float
    features: FeatureVector
    x: list[float]
    y: int  # 1 if the round settled UP
    #: `1 / N_r` for the N_r valid decision points in this round.
    weight: float


@dataclass
class RealDatasetResult:
    feature_set: str
    examples: list[RealExample] = field(default_factory=list)
    #: Valid decision points per round, before weighting.
    valid_per_round: dict[str, int] = field(default_factory=dict)
    #: Why grid points produced no example, by reason.
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def round_ids(self) -> list[str]:
        return sorted(self.valid_per_round)

    def total_weight(self, round_id: str) -> float:
        return sum(e.weight for e in self.examples if e.round_id == round_id)


def visible_events(events: list, decision_ts: float) -> list:
    """Item 5: receive-time causal visibility.

    Only events this system had ACTUALLY RECEIVED by `decision_ts`.
    `compute()` then applies its own `event_time <= decision_ts` filter, so a
    usable observation must satisfy both gates - which is the real-world
    condition, since `recv_ts >= source_ts` always.
    """
    return [e for e in events if e.recv_ts <= decision_ts]


class TrainingEligibilityError(RuntimeError):
    """A round that is not Gate-A eligible was offered to the REAL training
    builder (Gate A.0.1 item 4)."""


def build_real_examples(
    store: EventStore,
    labels: list[RoundLabel],
    feature_cfg: FeatureConfig,
    feature_set: FeatureSet,
    *,
    eligibility: "dict | None" = None,
    grid: tuple[float, ...] | None = None,
    allow_synthetic: bool = False,
) -> RealDatasetResult:
    """Build a round-balanced, receive-time-causal dataset from a projected
    real capture.

    Gate A.0.1 item 4 - eligibility is enforced HERE, not by the caller.
    `ProjectionResult.label` only emits a target when the independent
    reconstruction is CONFIRMED, so an unverified OUTCOME cannot reach this
    function. But a verified outcome says nothing about the round's market
    data: a CONFIRMED label sitting on a book that drifted from the venue
    produced a perfectly well-formed `RoundLabel`, and a caller who forgot
    to filter would have trained on it.

    So for REAL provenance an `eligibility` map is REQUIRED, and every round
    in `labels` must be `training_eligible` in it. Omitting it raises rather
    than defaulting to "no filtering" - the failure mode this closes is
    precisely a caller who did not think about filtering at all. Callers
    should normally not construct the map themselves: `model/gate_a.py`
    builds the whole path from a raw capture.
    """
    require_real(store.provenance, "real training dataset", allow_synthetic=allow_synthetic)
    if store.provenance.is_real:
        if eligibility is None:
            raise TrainingEligibilityError(
                "REAL training data requires a per-round eligibility map; refusing to "
                "build a dataset with no eligibility gate. Use "
                "xamarinbot.model.gate_a.build_gate_a_dataset(), which derives it from "
                "the capture."
            )
        offered = [l.round_id for l in labels]
        missing = [r for r in offered if r not in eligibility]
        if missing:
            raise TrainingEligibilityError(
                f"no eligibility verdict for round(s) {missing}; an unknown round "
                "cannot be assumed clean"
            )
        ineligible = [
            (r, eligibility[r]) for r in offered
            if not getattr(eligibility[r], "training_eligible", False)
        ]
        if ineligible:
            detail = "; ".join(
                f"{r} ({', '.join(d.value for d in rec.disqualifiers) or 'not eligible'})"
                for r, rec in ineligible
            )
            raise TrainingEligibilityError(
                f"round(s) not training-eligible: {detail}"
            )
    grid = grid if grid is not None else decision_grid()
    result = RealDatasetResult(feature_set=feature_set.name)

    def skip(reason: str) -> None:
        result.skipped[reason] = result.skipped.get(reason, 0) + 1

    for label in labels:
        if not (label.provenance.is_real or allow_synthetic):
            skip("label_not_real")
            continue
        events = store.all_events(label.round_id)
        if not events:
            skip("no_events_for_round")
            continue

        from xamarinbot.events.types import EventType

        configs = [e for e in events if e.event_type is EventType.MARKET_CONFIG]
        if not configs:
            skip("no_market_config")
            continue
        start_ts = min(c.payload["start_ts"] for c in configs)

        y = 1 if label.outcome.value == "UP" else 0
        round_rows: list[RealExample] = []
        for t in grid:
            decision_ts = start_ts + t
            # Item 4: never a pre-round decision point. The grid starts at
            # +15s so this is belt and braces, but it is the invariant that
            # matters, not the constant.
            if t <= 0:
                skip("pre_round_decision_point")
                continue
            fv = compute(
                visible_events(events, decision_ts),
                label.round_id, decision_ts, label.p0, feature_cfg,
            )
            if not isinstance(fv, FeatureVector):
                skip(f"invalid_features:{fv.reason.value}")
                continue
            vec = design_vector(fv, feature_set)
            if vec is None:
                skip("design_vector_unavailable")
                continue
            round_rows.append(RealExample(
                round_id=label.round_id, t=t, decision_ts=decision_ts,
                features=fv, x=vec, y=y, weight=0.0,
            ))

        if not round_rows:
            skip("round_produced_no_valid_decision")
            continue

        # Item 6: this round contributes total weight 1, however many valid
        # decision points it happened to produce.
        n = len(round_rows)
        result.valid_per_round[label.round_id] = n
        for row in round_rows:
            result.examples.append(
                RealExample(
                    round_id=row.round_id, t=row.t, decision_ts=row.decision_ts,
                    features=row.features, x=row.x, y=row.y, weight=1.0 / n,
                )
            )

    result.examples.sort(key=lambda e: (e.decision_ts, e.round_id))
    return result


@dataclass(frozen=True)
class ChronologicalSplit:
    """Round-disjoint, strictly chronological TRAIN -> CALIBRATE -> TEST.

    Item 7: no random shuffle, and the unit of splitting is the ROUND, never
    the decision row. Splitting rows would put decision points from the same
    round on both sides of the boundary - the same outcome, the same book,
    leaking directly across the split.
    """

    train: list[RealExample]
    calibrate: list[RealExample]
    test: list[RealExample]

    def round_ids(self, part: str) -> set[str]:
        return {e.round_id for e in getattr(self, part)}

    @property
    def is_round_disjoint(self) -> bool:
        a, b, c = self.round_ids("train"), self.round_ids("calibrate"), self.round_ids("test")
        return not (a & b) and not (a & c) and not (b & c)


def chronological_split(
    result: RealDatasetResult,
    train_frac: float = 0.6,
    calibrate_frac: float = 0.2,
) -> ChronologicalSplit:
    """Split BY ROUND, in time order. Never shuffles."""
    rounds = sorted(
        result.valid_per_round,
        key=lambda rid: min(e.decision_ts for e in result.examples if e.round_id == rid),
    )
    n = len(rounds)
    n_train = int(n * train_frac)
    n_cal = int(n * calibrate_frac)
    train_ids = set(rounds[:n_train])
    cal_ids = set(rounds[n_train:n_train + n_cal])
    test_ids = set(rounds[n_train + n_cal:])
    return ChronologicalSplit(
        train=[e for e in result.examples if e.round_id in train_ids],
        calibrate=[e for e in result.examples if e.round_id in cal_ids],
        test=[e for e in result.examples if e.round_id in test_ids],
    )
