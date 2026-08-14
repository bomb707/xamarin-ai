"""Live-vs-replay parity (Roadmap Phase 12 deliverable: "Compare live
shadow decisions with offline replay for the same timestamps." /
"Live-vs-replay parity report.").

Isolates exactly one variable: whether an event has *actually arrived*
(`recv_ts`, the shadow's gate) versus whether its *source* timestamp has
merely passed (`event_time`, offline replay's gate - see
`shadow/runner.py`'s docstring). Both sides evaluate candidates against an
identical, frozen `PortfolioState()` - never advanced by fills - the same
convention Phase 8-10's own demo scripts use ("portfolio deliberately not
advanced here - this demo measures decision-quality/latency, not a full
backtest") so a fill-timing difference downstream can never masquerade as
a causal-gating mismatch upstream.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.feeds.mock import MockBookFeed, MockFeedCursor
from xamarinbot.model.features import FeatureSet, design_vector
from xamarinbot.model.calibrated import CalibratedModel
from xamarinbot.model.logistic import LogisticModel
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.regime.matrix import ActionPermissionMatrix, classify_seed_action
from xamarinbot.shadow.types import ShadowDecisionRecord


@dataclass(frozen=True)
class ParityMismatch:
    round_id: str
    decision_ts: float
    offline_action_id: str
    shadow_action_id: str


@dataclass(frozen=True)
class ParityReport:
    round_id: str
    n_compared: int
    mismatches: tuple[ParityMismatch, ...]

    @property
    def n_mismatches(self) -> int:
        return len(self.mismatches)

    @property
    def parity_rate(self) -> float:
        return 1.0 - (self.n_mismatches / self.n_compared) if self.n_compared else 1.0


def _offline_decisions_at(
    decision_timestamps: set[float], store: EventStore, round_id: str, p0: float,
    feature_cfg: FeatureConfig, fee_config: FeeConfig, exec_cfg: ExecutionConfig,
    one_step_cfg: OneStepConfig, model: LogisticModel | CalibratedModel | None, feature_set: FeatureSet | None,
) -> dict[float, str]:
    events = store.all_events(round_id)
    clock = ReplayClock(store, round_id)
    cursor = MockFeedCursor(store, round_id, preloaded=events)  # default time_attr="event_time"
    book_feed = MockBookFeed(cursor)
    regime_clf = RegimeClassifier(round_id=round_id)
    one_step = OneStepController(one_step_cfg, exec_cfg, fee_config)
    frozen_portfolio = PortfolioState()

    market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    tick_size = market_config["tick_size"]

    result: dict[float, str] = {}
    for decision_ts in clock.decision_points(heartbeat=None):
        if decision_ts not in decision_timestamps:
            continue
        cursor.advance_to(decision_ts)
        fv = compute(events, round_id, decision_ts, p0, feature_cfg)
        if not isinstance(fv, FeatureVector):
            continue
        snapshot = regime_clf.observe(fv)
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)
        vec = design_vector(fv, feature_set) if (model is not None and feature_set is not None) else None
        q = model.predict_proba(vec) if (model is not None and vec is not None) else 0.5
        permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(snapshot.state))
        decision = one_step.decide(round_id, decision_ts, frozen_portfolio, q, permitted, book_up, book_down, tick_size, True)
        result[decision_ts] = decision.chosen.action_id
    return result


def compare_live_vs_replay(
    shadow_records: tuple[ShadowDecisionRecord, ...], store: EventStore, round_id: str, p0: float,
    feature_cfg: FeatureConfig, fee_config: FeeConfig, exec_cfg: ExecutionConfig,
    one_step_cfg: OneStepConfig, model: LogisticModel | CalibratedModel | None, feature_set: FeatureSet | None,
) -> ParityReport:
    """Compares each shadow decision (already made under the true
    recv_ts-gated live view) against what the offline, event_time-gated
    replay would choose at the identical decision_ts - both against a
    frozen zero portfolio, per the module docstring."""
    non_deadline_records = [r for r in shadow_records if not r.missed_deadline]
    decision_ts_set = {r.decision_ts for r in non_deadline_records}
    offline = _offline_decisions_at(decision_ts_set, store, round_id, p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, model, feature_set)

    mismatches = []
    n_compared = 0
    for r in non_deadline_records:
        offline_action_id = offline.get(r.decision_ts)
        if offline_action_id is None:
            continue  # offline saw an InvalidFeatureState at this ts (shouldn't happen given shadow saw a valid one, but don't crash on it)
        n_compared += 1
        if offline_action_id != r.action_id:
            mismatches.append(ParityMismatch(round_id, r.decision_ts, offline_action_id, r.action_id))

    return ParityReport(round_id=round_id, n_compared=n_compared, mismatches=tuple(mismatches))
