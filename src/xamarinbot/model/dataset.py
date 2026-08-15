"""Builds (X, y) training examples from causal replay (Roadmap Phase 5 /
Phase 4 integration). Each example pairs a causally-computed FeatureVector
with the eventual round settlement outcome - never a future price or
mid-round proxy, per the causal-replay invariant established in Phase 2/4.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.model.features import FeatureSet, design_vector
from xamarinbot.rounds import RoundLabel


@dataclass(frozen=True)
class Example:
    round_id: str
    decision_ts: float
    features: FeatureVector
    x: list[float]
    y: int  # 1 if UP, 0 if DOWN


def build_examples_multi(
    store: EventStore,
    results: list[RoundLabel],
    feature_cfg: FeatureConfig,
    feature_sets: list[FeatureSet],
    heartbeat_s: float = 5.0,
) -> dict[str, list[Example]]:
    """Computes each causal FeatureVector once per (round, decision_ts) and
    derives a design vector for every requested feature_set from it, rather
    than recomputing the (expensive, O(events)) FeatureVector once per
    feature set - the naive approach is a 3x-plus slowdown for no benefit
    since all feature sets read from the same underlying FeatureVector.

    decision_ts is a globally increasing absolute timestamp across the
    whole synthetic dataset (each round's start_ts is offset past the
    previous round's end), so sorting examples by decision_ts alone gives a
    valid chronological order for walk-forward splitting without needing a
    separate per-round anchor.
    """
    out: dict[str, list[Example]] = {fs.name: [] for fs in feature_sets}
    for result in results:
        events = store.all_events(result.round_id)
        clock = ReplayClock(store, result.round_id)
        y = 1 if result.outcome.value == "UP" else 0
        for decision_ts in clock.decision_points(heartbeat=heartbeat_s):
            fv = compute(events, result.round_id, decision_ts, result.p0, feature_cfg)
            if not isinstance(fv, FeatureVector):
                continue
            for fs in feature_sets:
                vec = design_vector(fv, fs)
                if vec is None:
                    continue
                out[fs.name].append(Example(round_id=result.round_id, decision_ts=decision_ts, features=fv, x=vec, y=y))
    return out
