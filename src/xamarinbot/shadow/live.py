"""LIVE real-market shadow service (readiness audit items 4, 10, 12, 13).

What was missing
----------------
`ShadowRunner` is a batch runner: it takes a finished round's `EventStore`,
enumerates every decision point up front via `ReplayClock`, and walks them.
`RealRecorderService` owns the live feeds but never calls the strategy at
all. So the only route from real market data to a decision was

    raw capture -> normalized replay DB -> ReplayCursor -> decision

which is a replay system, not a live bot.

This service closes that: it rides the recorder's own live event stream,
projects each event into a normalized in-memory `EventStore` as it arrives,
and fires the FROZEN strategy clock inside the round.

What it deliberately does NOT do
--------------------------------
Reimplement the strategy. Every piece of decision mathematics is the shared
module `ShadowRunner` uses - `features.engine.compute`, `RegimeClassifier`,
`design_vector`, `OneStepController.decide`, `TradingSession` - and this
module is orchestration only. `tests/test_live_shadow.py` asserts parity
against `ShadowRunner` over the same data so the two cannot silently drift.

PAPER ONLY. This module imports no authenticated client, holds no key, and
signs nothing. The only "dispatch" mutates `TradingSession`. Enforced
structurally by `tests/test_import_boundaries.py`.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.session import TradingSession
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.market.constraints import MarketConstraints, reconcile_execution_state
from xamarinbot.model.calibrated import CalibratedModel
from xamarinbot.model.features import FeatureSet, design_vector
from xamarinbot.model.logistic import LogisticModel
from xamarinbot.optimizer.candidates import wait_candidate
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.freshness import FreshnessPolicy
from xamarinbot.realtime.raw_events import Topic
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.regime.config import RegimeConfig
from xamarinbot.regime.matrix import ActionPermissionMatrix, classify_seed_action
from xamarinbot.replay.feeds import ReplayBookFeed, ReplayCursor, market_config_from_payload
from xamarinbot.shadow.journal import ShadowJournal
from xamarinbot.shadow.live_projection import LiveProjector
from xamarinbot.shadow.manifest import (
    NO_REAL_MODEL,
    build_manifest,
    decision_grid,
    strategy_v0_configs,
)
from xamarinbot.shadow.types import DecisionBlockReason


@dataclass
class LiveRoundShadow:
    """One round's live strategy state."""

    round_id: str
    metadata: object
    store: EventStore
    projector: LiveProjector
    session: TradingSession
    regime: RegimeClassifier
    p0: float | None = None
    fired: set = field(default_factory=set)
    decisions: int = 0
    blocked: dict = field(default_factory=dict)
    missed_deadlines: int = 0
    settled: bool = False
    #: Raw recorder sequence range this round's decisions were made from,
    #: for provenance linkage (audit item 13).
    raw_seq_first: int | None = None
    raw_seq_last: int | None = None

    def note_block(self, reason: str) -> None:
        self.blocked[reason] = self.blocked.get(reason, 0) + 1


class LiveShadowService:
    """Rides `RealRecorderService`'s live feeds and runs Strategy V0 on
    paper, on the frozen strategy clock."""

    def __init__(
        self,
        recorder_service,
        journal: ShadowJournal,
        *,
        feature_cfg: FeatureConfig | None = None,
        one_step_cfg: OneStepConfig | None = None,
        regime_cfg: RegimeConfig | None = None,
        exec_cfg: ExecutionConfig | None = None,
        model: LogisticModel | CalibratedModel | None = None,
        feature_set: FeatureSet | None = None,
        model_version: str = NO_REAL_MODEL,
        decision_deadline_ms: float = 50.0,
        freshness_policy: FreshnessPolicy | None = None,
        log=print,
    ):
        self.svc = recorder_service
        self.journal = journal
        self.feature_cfg = feature_cfg or FeatureConfig()
        self.one_step_cfg = one_step_cfg or strategy_v0_configs()
        self.regime_cfg = regime_cfg or RegimeConfig()
        self.exec_cfg = exec_cfg or ExecutionConfig()
        self.model = model
        self.feature_set = feature_set
        self.model_version = model_version
        self.decision_deadline_ms = decision_deadline_ms
        self.freshness_policy = freshness_policy
        self._log = log
        self.grid = decision_grid()
        self.manifest = build_manifest(
            self.feature_cfg, self.one_step_cfg, self.regime_cfg, self.exec_cfg,
            self.feature_set, self.model_version,
        )
        self.rounds: dict[str, LiveRoundShadow] = {}
        #: Live events arrive on the WebSocket READER threads, while the
        #: stores and decisions live on the service-loop thread. SQLite
        #: connections are not shareable across threads, so events are
        #: buffered here and drained on the decision thread.
        #:
        #: This does not weaken causality: `recv_wall_timestamp_ns` is
        #: stamped when the bytes arrived, not when they are drained, and
        #: every decision gate is on that stamp. `deque.append`/`popleft`
        #: are atomic under the GIL, so no lock is needed.
        self._inbox: deque = deque()

        # Ride the recorder's own stream. Both hooks run on the recorder's
        # thread and must never raise into it.
        recorder_service.on_live_event = self._on_live_event
        recorder_service.on_tick = self._on_tick
        recorder_service.on_round_finalized = self._on_finalized

        self.journal.write_manifest(self.manifest)

    # ------------------------------------------------------- round set-up

    def _settlement_topic(self, metadata) -> Topic:
        from xamarinbot.realtime.label import topic_for_basis
        from xamarinbot.realtime.rtds import _TOPIC_MAP

        wire = topic_for_basis(metadata.settlement_kind, metadata.twap_window_s)
        return _TOPIC_MAP[wire]

    def ensure_round(self, capture) -> LiveRoundShadow | None:
        m = capture.metadata
        if m.round_id in self.rounds:
            return self.rounds[m.round_id]
        try:
            settlement_topic = self._settlement_topic(m)
        except Exception as exc:
            self._log(f"[live-shadow] {m.round_id}: unsupported settlement rule ({exc})")
            return None

        store = EventStore(":memory:", provenance=DataProvenance.REAL_LIVE)
        projector = LiveProjector(store, m.round_id, settlement_topic)
        projector.emit_market_config(m, int(time.time() * 1e9))

        # The round's executable parameters come from the MARKET_CONFIG we
        # just emitted, i.e. from the market's own metadata - never from a
        # static strategy config. One construction path, shared with the
        # per-decision re-read below.
        opening = store.all_events(m.round_id)
        constraints = self._constraints_from(opening)
        # Deliberately NOT passing `self.exec_cfg`: on REAL data the market
        # is the single source of truth for the fee schedule and taker delay,
        # and `reconcile_execution_state` refuses a caller value that
        # contradicts it. The strategy's declared ExecutionConfig is recorded
        # in the manifest; the per-round ACTUAL values come from the venue.
        fee_cfg, exec_cfg = reconcile_execution_state(constraints)
        session = TradingSession(
            round_id=m.round_id,
            portfolio=PortfolioState(),
            fee_config=fee_cfg,
            exec_cfg=exec_cfg,
            cfg=self.one_step_cfg,
            constraints=constraints,
        )
        shadow = LiveRoundShadow(
            round_id=m.round_id, metadata=m, store=store, projector=projector,
            session=session, regime=RegimeClassifier(round_id=m.round_id),
        )
        self.rounds[m.round_id] = shadow
        self.journal.write_round_opened(shadow, self.manifest, self.svc.store.db_path)
        return shadow

    # --------------------------------------------------------- live feeds

    def _on_live_event(self, event) -> None:
        """Every raw event, on the reader thread. Buffer only - see `_inbox`."""
        self._inbox.append(event)

    def _drain_inbox(self) -> int:
        """Project buffered events into their rounds, on the decision thread."""
        n = 0
        while True:
            try:
                event = self._inbox.popleft()
            except IndexError:
                break
            n += 1
            for shadow in self.rounds.values():
                if shadow.settled:
                    continue
                if shadow.projector.apply(event):
                    if shadow.raw_seq_first is None:
                        shadow.raw_seq_first = event.recorder_sequence
                    shadow.raw_seq_last = event.recorder_sequence
        return n

    def _on_tick(self, captures, now: float) -> None:
        from xamarinbot.realtime.lifecycle import RoundState

        for capture in captures:
            if capture.lifecycle.state in (RoundState.DISCOVERED,):
                continue
            self.ensure_round(capture)
        # Drain AFTER every active round exists, so an event is never
        # discarded merely because its round had not been created yet.
        self._drain_inbox()
        for capture in captures:
            shadow = self.rounds.get(capture.metadata.round_id)
            if shadow is None or shadow.settled:
                continue
            self._fire_due_decisions(shadow, now)

    # ---------------------------------------------------- strategy clock

    def _fire_due_decisions(self, shadow: LiveRoundShadow, now: float) -> None:
        """Fire every grid point whose wall time has passed and which has
        not yet fired.

        The FROZEN clock, not the market's. A grid point that comes due
        while the process is busy still fires (late, and recorded as such)
        rather than being skipped - a silently dropped decision point is
        indistinguishable from a WAIT in the record.
        """
        start_ts = shadow.metadata.start_ts
        for t in self.grid:
            if t in shadow.fired:
                continue
            due = start_ts + t
            if due > now:
                break
            shadow.fired.add(t)
            lateness_ms = (now - due) * 1000.0
            try:
                self._decide(shadow, due, t, lateness_ms)
            except Exception as exc:  # noqa: BLE001
                # One bad decision must never kill the service (audit 15).
                shadow.note_block(f"decision_error:{type(exc).__name__}")
                self.journal.write_decision_error(shadow, due, t, exc)

    def _p0(self, shadow: LiveRoundShadow) -> float | None:
        """The opening settlement reference, from the declared basis.

        Never defaulted: with no observation at or before the open there is
        no `p0`, and every feature that rests on it is undefined.
        """
        if shadow.p0 is not None:
            return shadow.p0
        start = shadow.metadata.start_ts
        best = None
        for e in shadow.store.all_events(shadow.round_id):
            if e.event_type is not EventType.TWAP:
                continue
            if e.event_time <= start and (best is None or e.event_time > best.event_time):
                best = e
        if best is not None:
            shadow.p0 = float(best.payload["value"])
        return shadow.p0

    def _constraints_from(self, causal: list) -> MarketConstraints:
        configs = [e for e in causal if e.event_type is EventType.MARKET_CONFIG]
        if not configs:
            raise LookupError("no MARKET_CONFIG visible yet")
        latest = max(configs, key=lambda e: (e.event_time, e.sequence))
        return MarketConstraints.from_market_config(
            market_config_from_payload(latest.payload),
            provenance=DataProvenance.REAL_LIVE,
            source="live",
        )

    # ------------------------------------------------------- one decision

    def _decide(self, shadow: LiveRoundShadow, decision_ts: float, t: float,
                lateness_ms: float) -> None:
        t0 = time.perf_counter()
        events = shadow.store.all_events(shadow.round_id)
        # The live causal gate: only what has ACTUALLY ARRIVED. `compute`
        # additionally re-filters on event_time, so a usable observation
        # must satisfy both clocks.
        causal = [e for e in events if e.recv_ts <= decision_ts]

        cursor = ReplayCursor(shadow.store, shadow.round_id, preloaded=events,
                              time_attr="recv_ts")
        cursor.advance_to(decision_ts)
        book_feed = ReplayBookFeed(cursor)

        def book_at(round_id, side, at_ts):
            c = ReplayCursor(shadow.store, round_id, preloaded=events)
            c.advance_to(at_ts)
            return ReplayBookFeed(c).get_snapshot(round_id, side)

        # Exchange truth first: a pending taker's fill is decided by the real
        # matching engine at its own matched_ts, regardless of our freshness.
        shadow.session.resolve_ready_takers(decision_ts, book_at)

        p0 = self._p0(shadow)
        if p0 is None:
            shadow.note_block("NO_P0")
            self.journal.write_decision(
                shadow, decision_ts, t, None, None, None, None,
                blocked_reason="NO_P0", elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                lateness_ms=lateness_ms, manifest=self.manifest,
            )
            return

        fv = compute(causal, shadow.round_id, decision_ts, p0, self.feature_cfg)
        if not isinstance(fv, FeatureVector):
            shadow.note_block(f"INVALID_FEATURES:{fv.reason.value}")
            self.journal.write_decision(
                shadow, decision_ts, t, None, None, None, None,
                blocked_reason=DecisionBlockReason.INVALID_FEATURES.value,
                invalid_reason=fv.reason.value,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                lateness_ms=lateness_ms, manifest=self.manifest,
            )
            return

        snapshot = shadow.regime.observe(fv)
        book_up = book_feed.get_snapshot(shadow.round_id, Side.UP)
        book_down = book_feed.get_snapshot(shadow.round_id, Side.DOWN)
        constraints = self._constraints_from(causal)
        shadow.session.constraints = constraints

        vec = (design_vector(fv, self.feature_set)
               if (self.model is not None and self.feature_set is not None) else None)
        if self.model is None or vec is None:
            # No fabricated q. On REAL data "we have no probability estimate"
            # must never become "the probability is 50%".
            shadow.note_block(DecisionBlockReason.MODEL_UNAVAILABLE.value)
            self.journal.write_decision(
                shadow, decision_ts, t, fv, None, snapshot, None,
                blocked_reason=DecisionBlockReason.MODEL_UNAVAILABLE.value,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                lateness_ms=lateness_ms, manifest=self.manifest,
                constraints=constraints,
            )
            shadow.decisions += 1
            return

        q = self.model.predict_proba(vec)
        permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(snapshot.state))
        shadow.session.review_open_orders(
            decision_ts, snapshot.state, q, book_up, book_down, fv.tau, True,
        )
        decision = shadow.session and OneStepController(self.one_step_cfg).decide(
            shadow.round_id, decision_ts, shadow.session.portfolio, q, permitted,
            book_up, book_down, constraints, True,
            tau=fv.tau, sigma=fv.realized_vol, risk_view=shadow.session.risk_view(),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        missed = elapsed_ms > self.decision_deadline_ms
        if missed:
            shadow.missed_deadlines += 1
        chosen = decision.chosen if not missed else wait_candidate("wait", shadow.session.portfolio)

        self.journal.write_decision(
            shadow, decision_ts, t, fv, q, snapshot, decision,
            chosen=chosen, elapsed_ms=elapsed_ms, lateness_ms=lateness_ms,
            missed_deadline=missed, manifest=self.manifest, constraints=constraints,
        )
        shadow.session.dispatch(chosen, decision_ts, snapshot.state, q, book_up, book_down)
        shadow.decisions += 1

    # ---------------------------------------------------------- settlement

    def _on_finalized(self, capture) -> None:
        shadow = self.rounds.get(capture.metadata.round_id)
        if shadow is None or shadow.settled:
            return
        shadow.settled = True
        self.journal.write_settlement(shadow, capture, self.manifest)
        self._log(
            f"[live-shadow] {shadow.round_id} settled: {shadow.decisions} decisions, "
            f"blocked {dict(shadow.blocked)}"
        )

    # ------------------------------------------------------------ summary

    def summary(self) -> dict:
        return {
            "strategy_version": self.manifest.strategy_version,
            "config_hash": self.manifest.config_hash,
            "model_version": self.model_version,
            "rounds": {
                rid: {
                    "decisions": s.decisions,
                    "grid_points_fired": len(s.fired),
                    "blocked": dict(s.blocked),
                    "missed_deadlines": s.missed_deadlines,
                    "settled": s.settled,
                    "projected": dict(s.projector.counts),
                    "raw_seq_range": [s.raw_seq_first, s.raw_seq_last],
                }
                for rid, s in self.rounds.items()
            },
        }
