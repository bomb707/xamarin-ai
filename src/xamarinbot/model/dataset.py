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
from xamarinbot.synthetic.rounds import SyntheticRoundResult


@dataclass(frozen=True)
class Example:
    round_id: str
    decision_ts: float
    features: FeatureVector
    x: list[float]
    y: int  # 1 if UP, 0 if DOWN


def build_examples(
    store: EventStore,
    results: list[SyntheticRoundResult],
    feature_cfg: FeatureConfig,
    feature_set: FeatureSet,
    heartbeat_s: float = 5.0,
) -> list[Example]:
    """decision_ts is a globally increasing absolute timestamp across the
    whole synthetic dataset (each round's start_ts is offset past the
    previous round's end), so sorting examples by decision_ts alone gives a
    valid chronological order for walk-forward splitting without needing a
    separate per-round anchor."""
    examples: list[Example] = []
    for result in results:
        events = store.all_events(result.round_id)
        clock = ReplayClock(store, result.round_id)
        y = 1 if result.outcome.value == "UP" else 0
        for decision_ts in clock.decision_points(heartbeat=heartbeat_s):
            fv = compute(events, result.round_id, decision_ts, result.p0, feature_cfg)
            if not isinstance(fv, FeatureVector):
                continue
            vec = design_vector(fv, feature_set)
            if vec is None:
                continue
            examples.append(Example(round_id=result.round_id, decision_ts=decision_ts, features=fv, x=vec, y=y))
    return examples
