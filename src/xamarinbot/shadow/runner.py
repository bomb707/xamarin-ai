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
from xamarinbot.execution.session import TradingSession
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
from xamarinbot.optimizer.types import CandidateAction
from xamarinbot.portfolio.state import FeeConfig, Side
from xamarinbot.realtime.freshness import (
    FeedKind,
    FreshnessPolicy,
    FreshnessReport,
    evaluate_freshness,
)
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.regime.matrix import ActionPermissionMatrix, classify_seed_action
from xamarinbot.shadow.config import ShadowConfig
from xamarinbot.shadow.types import ShadowDecisionRecord, ShadowRoundResult

#: Which `FeedKind` each legacy (normalized) event type reports freshness
#: for. The Phase-2 event vocabulary predates the real recorder and has one
#: `SPOT` type where the live service now has two genuinely distinct feeds
#: (Chainlink reference and Binance); a replay therefore cannot claim to
#: know both. `SPOT` is mapped to BINANCE only - the synthetic generator's
#: spot series is a leading price, not an oracle observation - and the
#: replay policy below correspondingly does not require a Chainlink
#: reference that no replay event can supply. Attributing one synthetic
#: series to both feeds would have manufactured a fresh settlement
#: reference out of nothing, which is the same class of pretence item 10
#: is about.
_LEGACY_FEED_FOR_EVENT: dict[EventType, FeedKind] = {
    EventType.BOOK_SNAPSHOT: FeedKind.BOOK,
    EventType.BOOK_DELTA: FeedKind.BOOK,
    EventType.TWAP: FeedKind.CHAINLINK_TWAP_60,
    EventType.SPOT: FeedKind.BINANCE,
}

#: Freshness budgets for a REPLAY of the Phase-2 event store.
#:
#: These are derived from the publication cadence of the data being
#: replayed, exactly as the live budgets in `realtime/freshness.py` are
#: derived from the measured live rates - not tuned against outcomes. The
#: synthetic generator (`synthetic/rounds.py`) declares
#: `tick_interval_s=1.0` for its spot/TWAP series and
#: `resnapshot_interval_s=5.0` for full book snapshots, and the observed
#: worst-case ages over a generated round match those exactly (book 5.0s,
#: TWAP 1.0s, spot 1.0s). Each budget is that cadence plus one interval of
#: margin, so a feed has to genuinely skip a publication to read as stale.
#:
#: A live recorder run uses the default `FreshnessPolicy` instead, whose
#: much tighter book budget reflects a ~130 message/second stream.
REPLAY_FRESHNESS_POLICY = FreshnessPolicy(
    max_age_s={
        FeedKind.BOOK: 6.0,
        FeedKind.CHAINLINK_TWAP_60: 2.0,
        FeedKind.BINANCE: 2.0,
    },
    required=frozenset({FeedKind.BOOK, FeedKind.CHAINLINK_TWAP_60, FeedKind.BINANCE}),
)


def freshness_from_events(
    causal_events: list,
    decision_ts: float,
    policy: FreshnessPolicy,
) -> FreshnessReport:
    """Latest SOURCE timestamp per feed among the events that have actually
    ARRIVED by `decision_ts`, turned into a freshness report.

    The two timestamps do different jobs here and both are needed:
    visibility is gated on `recv_ts` (a live system cannot act on bytes it
    has not received), but AGE is measured from `source_ts` (a feed that
    stopped publishing an hour ago is stale no matter how promptly its last
    message arrived). Using `recv_ts` for the age too - the tempting
    simplification - would report a dead feed as perfectly fresh.
    """
    latest: dict[FeedKind, float | None] = {}
    for e in causal_events:
        kind = _LEGACY_FEED_FOR_EVENT.get(e.event_type)
        if kind is None:
            continue
        source_ts = e.source_ts if e.source_ts is not None else e.recv_ts
        current = latest.get(kind)
        if current is None or source_ts > current:
            latest[kind] = source_ts
    return evaluate_freshness(decision_ts, latest, policy)


@dataclass(frozen=True)
class FaultInjector:
    """Test/demo-only: simulates a feed outage at specific decision
    timestamps so "24/7 reconnect stability" is an exercised code path,
    not an assumption. Real adapters raise their own connection errors;
    this stands in for that until live credentials exist."""

    disconnect_at: frozenset[float] = frozenset()

    def should_disconnect(self, decision_ts: float) -> bool:
        return decision_ts in self.disconnect_at


def _record_for(
    round_id: str,
    decision_ts: float,
    chosen: CandidateAction,
    elapsed_ms: float,
    missed: bool,
    reconnected: bool,
    *,
    is_fresh: bool = True,
    freshness_reason: str | None = None,
    invalid_reason: str | None = None,
    suppressed_by_freshness: bool = False,
) -> ShadowDecisionRecord:
    return ShadowDecisionRecord(
        round_id=round_id, decision_ts=decision_ts, action_id=chosen.action_id, mode=chosen.mode.value,
        side=chosen.side.value if chosen.side is not None else None, price=chosen.price, qty=chosen.qty,
        expected_fill=chosen.expected_fill, delta_ev=chosen.delta_ev, g_after=chosen.g_after,
        decide_elapsed_ms=elapsed_ms, missed_deadline=missed, reconnected=reconnected,
        is_fresh=is_fresh, freshness_reason=freshness_reason,
        invalid_reason=invalid_reason, suppressed_by_freshness=suppressed_by_freshness,
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
        freshness_policy: FreshnessPolicy | None = None,
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
        # Phase 12C item 10. Defaults to the replay-cadence policy because
        # this constructor's `store` is a Phase-2 event store; a live
        # service passes the tighter real policy explicitly.
        self.freshness_policy = freshness_policy or REPLAY_FRESHNESS_POLICY

    def run(self) -> ShadowRoundResult:
        events = self.store.all_events(self.round_id)
        clock = ReplayClock(self.store, self.round_id)
        # time_attr="recv_ts": the true live-arrival gate, not Phase 2's
        # event_time gate - see module docstring.
        live_cursor = MockFeedCursor(self.store, self.round_id, preloaded=events, time_attr="recv_ts")
        book_feed = MockBookFeed(live_cursor)
        # Dedicated cursor/book_feed for fetching the actual causal book at
        # a delayed taker order's matched_ts, only at resolve time (Phase
        # 12B Tranche 1.2 items 1/2, corrected in Tranche 2A item 5).
        #
        # Deliberately time_attr="event_time" (the default), NOT recv_ts,
        # unlike the strategy's own decision-time cursor above. These are
        # two genuinely different clocks/views:
        #   strategy_view (decisions)   -> recv_ts   (this system's own wire-arrival gate)
        #   execution_truth (fills)     -> event_time/source_ts (what the real EXCHANGE's book actually was)
        # A delayed taker's fill at matched_ts is determined by the real
        # exchange matching engine against the book as it truly stood at
        # matched_ts (by source/exchange event ordering) - that has
        # nothing to do with when *this* system happened to receive the
        # update over its own wire. Gating the resolution book on recv_ts
        # would make the simulated fill depend on this system's own
        # network/processing latency, which is backwards: a book update
        # that reached the real exchange before matched_ts determines the
        # fill even if it reaches this system's feed handler later.
        revalidation_cursor = MockFeedCursor(self.store, self.round_id, preloaded=events)
        revalidation_book_feed = MockBookFeed(revalidation_cursor)
        regime_clf = RegimeClassifier(round_id=self.round_id)
        one_step = OneStepController(self.one_step_cfg, self.exec_cfg, self.fee_config)
        # Phase 12B Tranche 2.2 item 3: the same shared execution/session
        # engine the supervisor-enabled walk-forward ablation arm uses -
        # RiskView-gated dispatch, open makers tracked (never an immediate
        # Bernoulli draw at submission), reviewed/canceled/replaced via
        # OrderSupervisor on later decision points, taker fills only ever
        # applied at their own resolved matched_ts. ShadowRunner is no
        # longer a separately-maintained, simplified execution engine.
        session = TradingSession(self.round_id, self.fee_config, self.exec_cfg, self.one_step_cfg)
        records: list[ShadowDecisionRecord] = []
        n_reconnects = 0
        n_missed = 0
        n_freshness_failures = 0
        n_invalid = 0
        n_reviewed_stale = 0
        pending_reconnect_ack = False

        market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
        tick_size = market_config["tick_size"]

        def _book_at(pending) -> tuple:
            revalidation_cursor.advance_to(pending.matched_ts)
            book = revalidation_book_feed.get_snapshot(self.round_id, pending.side)
            return book.asks if book is not None else ()

        # Last values genuinely computed from a valid, fresh observation.
        # Used only to re-evaluate resting orders on a blind decision point:
        # the supervisor's FEED_STALE check fires before any of them is
        # consulted, so they never influence the outcome, but carrying the
        # last known values is still more honest than inventing q=0.5,
        # tau=0 - which would additionally read as a time-compression
        # trigger that the data never showed.
        last_q = 0.5
        last_tau = 0.0

        def _supervise_unsafe_orders(decision_ts: float) -> int:
            """Phase 12C item 10: "review/cancel unsafe resting paper
            orders" whenever a required input is missing or stale.

            The previous code simply `continue`d past such a decision point,
            which quietly asserted that resting orders remained safely
            supervised while the system was, by its own account, blind.
            Passing `is_fresh=False` into the supervisor is what actually
            reaches SS16's FEED_STALE trigger - checked before every other
            cancel predicate - so the orders are genuinely reviewed and
            canceled rather than left resting unexamined.

            The book snapshots handed in are whatever the feed can still
            reconstruct; when there is none, `review_open_orders` skips that
            order, and it is re-examined at the next decision point.
            """
            open_ids = list(session.supervisor.open_order_ids())
            if not open_ids:
                return 0
            state = regime_clf.current_state
            if state is None:
                # No state has ever been classified, so no order can have
                # been placed; nothing to review.
                return 0
            session.review_open_orders(
                decision_ts, state, last_q,
                book_feed.get_snapshot(self.round_id, Side.UP),
                book_feed.get_snapshot(self.round_id, Side.DOWN),
                last_tau, False, tick_size,
            )
            return len(open_ids)

        for decision_ts in clock.decision_points(heartbeat=self.cfg.heartbeat_s):
            live_cursor.advance_to(decision_ts)

            # Exchange truth: a pending taker's fill is determined by the
            # real matching engine, which is unaffected by whether OUR feed
            # is fresh. Resolved before any freshness gating, deliberately.
            session.resolve_ready_takers(decision_ts, _book_at)

            if self.fault.should_disconnect(decision_ts):
                # Simulated outage. No new order can be generated - but the
                # runner must NOT simply skip the decision point while
                # pretending existing resting orders remain safely
                # supervised (item 10). They are reviewed against an
                # explicitly stale view, which is what lets the supervisor
                # cancel them.
                book_feed.reconnect()
                n_reconnects += 1
                n_freshness_failures += 1
                n_reviewed_stale += _supervise_unsafe_orders(decision_ts)
                # `reconnected=False` here deliberately: this record IS the
                # outage, not the recovery from it. `pending_reconnect_ack`
                # stays set so the next decision that actually runs - the
                # first one made on a restored feed - carries the flag,
                # which is what the field has always meant.
                records.append(_record_for(
                    self.round_id, decision_ts, wait_candidate("wait", session.portfolio),
                    0.0, False, False,
                    is_fresh=False, freshness_reason="feed disconnected (simulated outage)",
                    suppressed_by_freshness=True,
                ))
                pending_reconnect_ack = True
                continue

            # Only events that have *actually arrived* by decision_ts are
            # visible - compute() additionally re-filters by event_time
            # internally (Phase 4's own guarantee), so this is a strictly
            # tighter, live-correct view, not a replacement for it.
            causal_events = [e for e in events if e.recv_ts <= decision_ts]

            # Phase 12C item 10: real freshness from real source
            # timestamps, replacing the hardcoded `is_fresh=True`.
            freshness = freshness_from_events(causal_events, decision_ts, self.freshness_policy)
            is_fresh = freshness.is_fresh
            freshness_reason = freshness.reason

            t0 = time.perf_counter()
            fv = compute(causal_events, self.round_id, decision_ts, self.p0, self.feature_cfg)
            if not isinstance(fv, FeatureVector):
                # An explicit InvalidFeatureState. Previously this
                # `continue`d and the decision point vanished from the
                # record entirely; it is now recorded with its reason, and
                # any resting order is reviewed rather than assumed safe.
                n_invalid += 1
                n_freshness_failures += 1
                n_reviewed_stale += _supervise_unsafe_orders(decision_ts)
                records.append(_record_for(
                    self.round_id, decision_ts, wait_candidate("wait", session.portfolio),
                    (time.perf_counter() - t0) * 1000.0, False, pending_reconnect_ack,
                    is_fresh=False,
                    freshness_reason=freshness_reason,
                    invalid_reason=fv.reason.value,
                    suppressed_by_freshness=True,
                ))
                pending_reconnect_ack = False
                continue

            snapshot = regime_clf.observe(fv)
            book_up = book_feed.get_snapshot(self.round_id, Side.UP)
            book_down = book_feed.get_snapshot(self.round_id, Side.DOWN)
            vec = design_vector(fv, self.feature_set) if (self.model is not None and self.feature_set is not None) else None
            q = self.model.predict_proba(vec) if (self.model is not None and vec is not None) else 0.5
            permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(snapshot.state))
            last_q, last_tau = q, fv.tau

            # Review every open maker order (cancel/replace/expire) against
            # this decision point's fresh economics before generating a new
            # candidate - the same ordering the supervisor-enabled ablation
            # arm uses. `is_fresh` is now the REAL value: when it is False
            # this is what triggers SS16's FEED_STALE cancel.
            n_open_before_review = len(list(session.supervisor.open_order_ids()))
            session.review_open_orders(
                decision_ts, snapshot.state, q, book_up, book_down, fv.tau, is_fresh, tick_size
            )

            if not is_fresh:
                # Counted in ORDERS, consistently with
                # `_supervise_unsafe_orders`'s return value - the metric is
                # "how many resting orders were re-examined while the view
                # was stale", not "how many decision points were stale"
                # (which is `n_freshness_failures`).
                n_reviewed_stale += n_open_before_review
                # No new ALPHA while an input is missing or stale (item 10).
                # Recorded explicitly rather than looking like a WAIT the
                # optimizer chose on the economics.
                n_freshness_failures += 1
                records.append(_record_for(
                    self.round_id, decision_ts, wait_candidate("wait", session.portfolio),
                    (time.perf_counter() - t0) * 1000.0, False, pending_reconnect_ack,
                    is_fresh=False, freshness_reason=freshness_reason,
                    suppressed_by_freshness=True,
                ))
                pending_reconnect_ack = False
                continue

            risk_view = session.risk_view()
            # Phase 12C item 9: `tau` and `sigma` are passed EXPLICITLY from
            # the same FeatureVector `q` was computed from. They were
            # previously omitted, so `decide()` fell back to its defaults
            # (`tau=None`, `sigma=0.0`) and dynamic maker mode - when
            # enabled - silently operated as if the remaining time were
            # unknown and realized volatility were zero, while both real
            # values were sitting on `fv`.
            decision = one_step.decide(
                self.round_id, decision_ts, session.portfolio, q, permitted,
                book_up, book_down, tick_size, is_fresh,
                tau=fv.tau, sigma=fv.realized_vol, risk_view=risk_view,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            missed = elapsed_ms > self.cfg.decision_deadline_ms
            chosen = decision.chosen if not missed else wait_candidate("wait", session.portfolio)
            if missed:
                n_missed += 1

            records.append(_record_for(
                self.round_id, decision_ts, chosen, elapsed_ms, missed, pending_reconnect_ack,
                is_fresh=True, freshness_reason=None,
            ))
            pending_reconnect_ack = False

            session.dispatch(chosen, decision_ts, snapshot.state, q, book_up, book_down)

        return ShadowRoundResult(
            round_id=self.round_id, records=tuple(records), final_portfolio=session.portfolio,
            n_reconnects=n_reconnects, n_missed_deadlines=n_missed,
            n_freshness_failures=n_freshness_failures,
            n_invalid_feature_states=n_invalid,
            n_orders_reviewed_while_stale=n_reviewed_stale,
        )
