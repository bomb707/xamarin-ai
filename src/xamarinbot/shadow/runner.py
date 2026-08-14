"""ShadowRunner (Roadmap Phase 12 deliverable: "Shadow service.").

"Run production feeds and controller continuously while submitting no real
orders. Record desired order, hypothetical submit time, expected fill and
real subsequent book evolution. Run a parallel paper executor with
realistic delay/queue assumptions."

Feeds here are the same pluggable Phase 1 interfaces used everywhere else
(mock/replay today; real adapters once credentials are confirmed - see
docs/PHASE_STATUS.md's "Live adapter confidence"), so this runner is
already the code that would run against a live stream, not a separate
implementation to be swapped in later.

The one genuinely new piece versus every prior phase's replay loop: this
runner gates causal visibility on `recv_ts` (`time_attr="recv_ts"` on
`MockFeedCursor`), not `event_time`. Every earlier phase's replay/backtest
code treats an event as "known" once its *source* timestamp has passed
(`event_time <= decision_ts`, preferring source_ts) - correct for
reproducible backtesting, but mildly optimistic relative to when a live
system actually received the bytes over the wire (`recv_ts`, always
`>= source_ts`). A real "shadow" system literally cannot act on data it
hasn't received yet, so this is the one place in the codebase that must
use the stricter gate. `shadow/parity.py` measures how often that
difference actually changes a decision.

No real order is ever submitted here - fills are entirely paper, via
Phase 7's `ExecutionSimulator`, exactly like Phases 8-11's backtests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.simulator import ExecutionSimulator, TakerOrderQueue
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.feeds.mock import MockBookFeed, MockFeedCursor
from xamarinbot.model.features import FeatureSet, design_vector
from xamarinbot.model.calibrated import CalibratedModel
from xamarinbot.model.logistic import LogisticModel
from xamarinbot.optimizer.candidates import wait_candidate
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import CandidateAction, OrderMode
from xamarinbot.portfolio.state import Fill, FeeConfig, LiquidityRole, PortfolioState, Side, apply_fill
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.regime.matrix import ActionPermissionMatrix, classify_seed_action
from xamarinbot.shadow.config import ShadowConfig
from xamarinbot.shadow.types import ShadowDecisionRecord, ShadowRoundResult


@dataclass(frozen=True)
class FaultInjector:
    """Test/demo-only: simulates a feed outage at specific decision
    timestamps so "24/7 reconnect stability" is an exercised code path,
    not an assumption. Real adapters raise their own connection errors;
    this stands in for that until live credentials exist."""

    disconnect_at: frozenset[float] = frozenset()

    def should_disconnect(self, decision_ts: float) -> bool:
        return decision_ts in self.disconnect_at


def _record_for(round_id: str, decision_ts: float, chosen: CandidateAction, elapsed_ms: float, missed: bool, reconnected: bool) -> ShadowDecisionRecord:
    return ShadowDecisionRecord(
        round_id=round_id, decision_ts=decision_ts, action_id=chosen.action_id, mode=chosen.mode.value,
        side=chosen.side.value if chosen.side is not None else None, price=chosen.price, qty=chosen.qty,
        expected_fill=chosen.expected_fill, delta_ev=chosen.delta_ev, g_after=chosen.g_after,
        decide_elapsed_ms=elapsed_ms, missed_deadline=missed, reconnected=reconnected,
    )


class ShadowRunner:
    def __init__(
        self,
        store: EventStore,
        round_id: str,
        p0: float,
        feature_cfg: FeatureConfig,
        fee_config: FeeConfig,
        exec_cfg: ExecutionConfig,
        one_step_cfg: OneStepConfig,
        model: LogisticModel | CalibratedModel | None,
        feature_set: FeatureSet | None,
        cfg: ShadowConfig,
        fault: FaultInjector | None = None,
    ):
        self.store = store
        self.round_id = round_id
        self.p0 = p0
        self.feature_cfg = feature_cfg
        self.fee_config = fee_config
        self.exec_cfg = exec_cfg
        self.one_step_cfg = one_step_cfg
        self.model = model
        self.feature_set = feature_set
        self.cfg = cfg
        self.fault = fault or FaultInjector()

    def run(self) -> ShadowRoundResult:
        events = self.store.all_events(self.round_id)
        clock = ReplayClock(self.store, self.round_id)
        # time_attr="recv_ts": the true live-arrival gate, not Phase 2's
        # event_time gate - see module docstring.
        live_cursor = MockFeedCursor(self.store, self.round_id, preloaded=events, time_attr="recv_ts")
        book_feed = MockBookFeed(live_cursor)
        # Dedicated cursor/book_feed for fetching the actual causal book at
        # a delayed taker order's matched_ts, only at resolve time (Phase
        # 12B Tranche 1.2 items 1/2) - gated on recv_ts like the main
        # cursor above, since this runner must never act on data it
        # hasn't actually "received" yet, even when resolving a pending
        # order.
        revalidation_cursor = MockFeedCursor(self.store, self.round_id, preloaded=events, time_attr="recv_ts")
        revalidation_book_feed = MockBookFeed(revalidation_cursor)
        regime_clf = RegimeClassifier(round_id=self.round_id)
        one_step = OneStepController(self.one_step_cfg, self.exec_cfg, self.fee_config)
        sim = ExecutionSimulator(self.round_id, self.fee_config, self.exec_cfg)
        queue = TakerOrderQueue(sim)
        portfolio = PortfolioState()
        records: list[ShadowDecisionRecord] = []
        n_reconnects = 0
        n_missed = 0
        order_seq = 0
        pending_reconnect_ack = False

        market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
        tick_size = market_config["tick_size"]

        def _book_at(pending) -> tuple:
            revalidation_cursor.advance_to(pending.matched_ts)
            book = revalidation_book_feed.get_snapshot(self.round_id, pending.side)
            return book.asks if book is not None else ()

        for decision_ts in clock.decision_points(heartbeat=self.cfg.heartbeat_s):
            live_cursor.advance_to(decision_ts)

            for pending, taker_result in queue.resolve_ready(decision_ts, _book_at):
                if taker_result.walk.filled_shares > 0:
                    portfolio = apply_fill(portfolio, Fill(pending.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))

            if self.fault.should_disconnect(decision_ts):
                # Simulated outage: this decision point is skipped entirely
                # (no data, no order) rather than acted on with a stale
                # view - the runner must survive this and keep going, not
                # crash or wedge the loop.
                book_feed.reconnect()
                n_reconnects += 1
                pending_reconnect_ack = True
                continue

            # Only events that have *actually arrived* by decision_ts are
            # visible - compute() additionally re-filters by event_time
            # internally (Phase 4's own guarantee), so this is a strictly
            # tighter, live-correct view, not a replacement for it.
            causal_events = [e for e in events if e.recv_ts <= decision_ts]

            t0 = time.perf_counter()
            fv = compute(causal_events, self.round_id, decision_ts, self.p0, self.feature_cfg)
            if not isinstance(fv, FeatureVector):
                continue
            snapshot = regime_clf.observe(fv)
            book_up = book_feed.get_snapshot(self.round_id, Side.UP)
            book_down = book_feed.get_snapshot(self.round_id, Side.DOWN)
            vec = design_vector(fv, self.feature_set) if (self.model is not None and self.feature_set is not None) else None
            q = self.model.predict_proba(vec) if (self.model is not None and vec is not None) else 0.5
            permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(snapshot.state))

            decision = one_step.decide(self.round_id, decision_ts, portfolio, q, permitted, book_up, book_down, tick_size, True)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            missed = elapsed_ms > self.cfg.decision_deadline_ms
            chosen = decision.chosen if not missed else wait_candidate("wait", portfolio)
            if missed:
                n_missed += 1

            records.append(_record_for(self.round_id, decision_ts, chosen, elapsed_ms, missed, pending_reconnect_ack))
            pending_reconnect_ack = False

            if chosen.mode is OrderMode.FAK and chosen.qty > 0 and not queue.has_pending:
                # Phase 12B audit items 13/E/L, Tranche 1.2 items 1/2/5:
                # real submit->(delay)->resolve lifecycle via the shared
                # TakerOrderQueue, not a direct pre-evaluation-walk-to-Fill
                # shortcut, and never resolved/mutated before matched_ts.
                # `not queue.has_pending` is the conservative item 5
                # admission gate: at most one PENDING_DELAY taker at a time.
                order_seq += 1
                asks = book_up.asks if chosen.side is Side.UP else book_down.asks
                limit_price = chosen.max_execution_price if chosen.max_execution_price is not None else chosen.price
                pending = queue.try_submit(f"{self.round_id}-o{order_seq}", chosen.side, chosen.qty, limit_price, asks, decision_ts)
                if pending is not None and not pending.was_delayed:
                    taker_result = sim.resolve_taker(pending)
                    if taker_result.walk.filled_shares > 0:
                        portfolio = apply_fill(portfolio, Fill(chosen.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))
            elif chosen.mode is OrderMode.POST_ONLY:
                order_seq += 1
                order_state = sim.submit_maker_order(f"{self.round_id}-o{order_seq}", chosen.side, chosen.qty, chosen.price, decision_ts)
                draw = sim.draw_maker_fill(order_state, 0.0, 0.0, chosen.ttl_s or self.one_step_cfg.maker_horizon_s)
                if draw.filled:
                    portfolio = apply_fill(portfolio, Fill(chosen.side, chosen.price, chosen.qty, LiquidityRole.MAKER, 0.0))

        return ShadowRoundResult(
            round_id=self.round_id, records=tuple(records), final_portfolio=portfolio,
            n_reconnects=n_reconnects, n_missed_deadlines=n_missed,
        )
