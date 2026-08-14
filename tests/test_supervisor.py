"""Phase 9 verification: "Rapid regime flip." "Partial fill then cancel."
"Tick-size change with open order." "Feed stale while maker order rests."
Plus predicate correctness, rate limiting, and cancel-regret analytics.
"""
from __future__ import annotations

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.order_state import OrderLifecycleState, OrderState
from xamarinbot.journal.schema import SupervisorDecisionRecord
from xamarinbot.journal.writer import JournalWriter
from xamarinbot.optimizer.candidates import maker_price_grid
from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.portfolio.state import LiquidityRole, Side
from xamarinbot.regime.types import Direction, GapRegime, RegimeState
from xamarinbot.reports.supervisor_report import build_supervisor_report
from xamarinbot.supervisor.config import SupervisorConfig
from xamarinbot.supervisor.predicates import book_displacement, edge_failure, feed_stale, regime_flip, risk_breach, time_compression
from xamarinbot.supervisor.supervisor import OrderSupervisor
from xamarinbot.supervisor.types import CancelReason, SupervisorActionType, TrackedOrder

STATE_A = RegimeState(GapRegime.UPPER_MIDDLE, Direction.UP, Direction.UP)
STATE_B = RegimeState(GapRegime.LOWER_MIDDLE, Direction.DOWN, Direction.DOWN)


def _tracked(order_id="o1", side=Side.UP, price=0.45, qty=20.0, submit_ts=0.0, origin=STATE_A) -> TrackedOrder:
    order_state = OrderState(order_id=order_id, side=side, role=LiquidityRole.MAKER, limit_price=price, requested_shares=qty, submit_ts=submit_ts)
    return TrackedOrder(
        order_state=order_state, origin_regime_state=origin, purpose=OrderPurpose.ALPHA,
        q_at_submit=0.6, fair_value_at_submit=0.6, g_after_if_fill_at_submit=-5.0, ev_at_submit=1.0,
        ttl_s=10.0, submit_ts=submit_ts, last_action_ts=submit_ts,
    )


# --------------------------------------------------------------------------
# Individual predicates
# --------------------------------------------------------------------------


def test_edge_failure_predicate():
    cfg = SupervisorConfig(edge_min=1.0)
    assert edge_failure(0.5, cfg)
    assert not edge_failure(1.5, cfg)


def test_regime_flip_predicate():
    assert regime_flip(STATE_A, STATE_B)
    assert not regime_flip(STATE_A, STATE_A)


def test_risk_breach_predicate():
    cfg = SupervisorConfig(g_min=-10.0)
    assert risk_breach(-20.0, cfg)
    assert not risk_breach(-5.0, cfg)


def test_time_compression_predicate():
    cfg = SupervisorConfig(min_tau_for_passive_s=15.0)
    assert time_compression(10.0, cfg)
    assert not time_compression(20.0, cfg)


def test_feed_stale_predicate():
    assert feed_stale(False)
    assert not feed_stale(True)


def test_book_displacement_predicate_requires_clearing_churn_threshold():
    cfg = SupervisorConfig(churn_threshold=0.5)
    assert not book_displacement(current_optimal_ev=1.2, ev_at_submit=1.0, cfg=cfg)  # improvement too small
    assert book_displacement(current_optimal_ev=2.0, ev_at_submit=1.0, cfg=cfg)


# --------------------------------------------------------------------------
# review_order: priority order and the "everything's fine" HOLD path
# --------------------------------------------------------------------------


def test_review_order_holds_when_thesis_still_valid():
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(submit_ts=0.0)
    tracked.last_action_ts = -1000.0  # clear of rate limiting
    decision = supervisor.review_order(tracked, now_ts=5.0, current_regime_state=STATE_A, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.HOLD
    assert decision.reason is None


def test_feed_stale_takes_priority_over_everything_else():
    """Even with a fine EV/G/regime, stale data alone must cancel."""
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked()
    tracked.last_action_ts = -1000.0
    decision = supervisor.review_order(tracked, now_ts=5.0, current_regime_state=STATE_A, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=False)
    assert decision.action is SupervisorActionType.CANCEL
    assert decision.reason is CancelReason.FEED_STALE


# --------------------------------------------------------------------------
# "Rapid regime flip" (Roadmap Phase 9 verification, named explicitly)
# --------------------------------------------------------------------------


def test_rapid_regime_flip_cancels_immediately():
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(origin=STATE_A)
    supervisor.register(tracked)

    decision = supervisor.review_order(tracked, now_ts=1.0, current_regime_state=STATE_B, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.CANCEL
    assert decision.reason is CancelReason.REGIME_FLIP

    result = supervisor.apply_cancel(decision, now_ts=1.0)
    assert result.accepted
    assert "o1" not in supervisor.open_order_ids()


def test_rapid_successive_flips_are_rate_limited():
    """A second flip arriving faster than min_action_interval_s must not
    trigger a second action - this is exactly the "rapid" part of "rapid
    regime flip": the supervisor must not thrash on every tick."""
    cfg = SupervisorConfig(min_action_interval_s=2.0, g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(origin=STATE_A, submit_ts=0.0)
    tracked.last_action_ts = 0.0
    supervisor.register(tracked)

    # flips again 0.5s later - inside the rate-limit window
    decision = supervisor.review_order(tracked, now_ts=0.5, current_regime_state=STATE_B, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.HOLD
    assert decision.detail == "rate_limited"


# --------------------------------------------------------------------------
# "Partial fill then cancel" (Roadmap Phase 9 verification, named explicitly)
# --------------------------------------------------------------------------


def test_partial_fill_then_cancel_via_supervisor():
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(qty=100.0)
    supervisor.register(tracked)

    tracked.order_state.reconcile_fill(1.0, 40.0)  # partial fill happens first
    assert tracked.order_state.state is OrderLifecycleState.PARTIALLY_FILLED

    decision = supervisor.review_order(tracked, now_ts=2.0, current_regime_state=STATE_B, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.CANCEL

    result = supervisor.apply_cancel(decision, now_ts=2.0)
    assert result.accepted
    assert tracked.order_state.state is OrderLifecycleState.CANCELED
    assert tracked.order_state.filled_shares == 40.0  # partial fill preserved, not rolled back
    assert "o1" not in supervisor.open_order_ids()


def test_fill_beating_cancel_still_removes_order_from_tracking():
    """A fill that completes the order right before the supervisor's
    cancel attempt (a real race) must still result in the order being
    dropped from tracking - it's terminal either way."""
    cfg = SupervisorConfig(min_action_interval_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(qty=20.0)
    supervisor.register(tracked)
    tracked.order_state.reconcile_fill(1.0, 20.0)  # fully filled before cancel arrives
    assert tracked.order_state.state is OrderLifecycleState.FILLED

    decision = supervisor.review_order(tracked, now_ts=2.0, current_regime_state=STATE_B, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    result = supervisor.apply_cancel(decision, now_ts=2.0)
    assert not result.accepted  # already terminal, cancel itself is rejected
    assert "o1" not in supervisor.open_order_ids()  # but tracking still cleans it up


# --------------------------------------------------------------------------
# "Tick-size change with open order" (Roadmap Phase 9 verification, named explicitly)
# --------------------------------------------------------------------------


def test_tick_size_change_with_open_order_replace_uses_new_grid():
    """A REPLACE decision after a tick_size change must generate its new
    price on the *new* grid, not the stale one the order was placed on."""
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0, churn_threshold=0.1)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(price=0.45)
    supervisor.register(tracked)

    decision = supervisor.review_order(tracked, now_ts=1.0, current_regime_state=STATE_A, current_delta_ev=10.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True, current_optimal_ev=15.0)
    assert decision.action is SupervisorActionType.REPLACE

    old_tick_size = 0.01
    new_tick_size = 0.001  # simulates a tick_size_change event (Phase 1)
    old_grid = maker_price_grid(0.44, 0.46, old_tick_size, (0, 1, 2))
    new_grid = maker_price_grid(0.44, 0.46, new_tick_size, (0, 1, 2))
    assert old_grid != new_grid  # sanity: the grids really do differ

    new_price, _ = new_grid[0]
    result = supervisor.apply_replace(decision, now_ts=1.0, new_order_id="o1-r1", new_price=new_price, new_qty=tracked.order_state.remaining_shares)
    assert result.new_order is not None
    assert result.new_order.limit_price == new_price
    assert "o1" not in supervisor.open_order_ids()  # old order gone; caller is responsible for registering the replacement


def test_order_survives_tick_size_change_event_without_error():
    """The order itself doesn't need to change when tick_size changes -
    only future replacement prices are affected. Reviewing an order across
    a market_config update must not error."""
    store = EventStore(":memory:")
    store.append(EventType.MARKET_CONFIG, "r1", recv_ts=0.0, source_ts=0.0, payload=dict(
        market_id="r1", up_token_id="U", down_token_id="D", start_ts=0.0, end_ts=300.0,
        tick_size=0.01, min_order_size=1.0, fee_rate=0.07, taker_delay_ms=0.0, twap_window_seconds=30,
    ))
    store.append(EventType.MARKET_CONFIG, "r1", recv_ts=100.0, source_ts=100.0, payload=dict(
        market_id="r1", up_token_id="U", down_token_id="D", start_ts=0.0, end_ts=300.0,
        tick_size=0.001, min_order_size=1.0, fee_rate=0.07, taker_delay_ms=0.0, twap_window_seconds=30,
    ))
    configs = [e.payload["tick_size"] for e in store.all_events("r1") if e.event_type is EventType.MARKET_CONFIG]
    assert configs == [0.01, 0.001]

    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked()
    supervisor.register(tracked)
    decision = supervisor.review_order(tracked, now_ts=150.0, current_regime_state=STATE_A, current_delta_ev=10.0, current_g_after_if_fill=-1.0, tau=150.0, is_fresh=True)
    assert decision.action is SupervisorActionType.HOLD  # nothing else changed, still a valid thesis


# --------------------------------------------------------------------------
# "Feed stale while maker order rests" (Roadmap Phase 9 verification, named explicitly)
# --------------------------------------------------------------------------


def test_feed_stale_while_maker_order_rests_cancels_it():
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked()
    supervisor.register(tracked)

    decision = supervisor.review_order(tracked, now_ts=3.0, current_regime_state=STATE_A, current_delta_ev=10.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=False)
    assert decision.action is SupervisorActionType.CANCEL
    assert decision.reason is CancelReason.FEED_STALE
    result = supervisor.apply_cancel(decision, now_ts=3.0)
    assert result.accepted
    assert "o1" not in supervisor.open_order_ids()


# --------------------------------------------------------------------------
# Cancel-regret analytics
# --------------------------------------------------------------------------


def test_cancel_regret_detects_a_crossing_market():
    store = EventStore(":memory:")
    store.append(EventType.BOOK_SNAPSHOT, "r1", recv_ts=0.0, source_ts=0.0, payload={"side": "UP", "bids": [], "asks": [[0.50, 100.0]]})
    # market comes down and crosses the canceled order's price (0.45) shortly after
    store.append(EventType.BOOK_DELTA, "r1", recv_ts=5.0, source_ts=5.0, payload={"side": "UP", "book": "asks", "price": 0.50, "size": 0})
    store.append(EventType.BOOK_DELTA, "r1", recv_ts=5.0, source_ts=5.0, payload={"side": "UP", "book": "asks", "price": 0.44, "size": 100.0})

    journal = JournalWriter(":memory:")
    journal.write(SupervisorDecisionRecord(round_id="r1", order_id="o1", decision_ts=1.0, action="CANCEL", reason="REGIME_FLIP", side="UP", price=0.45))

    report = build_supervisor_report(journal, store, regret_lookback_s=10.0)
    assert report.n_cancels_checked_for_regret == 1
    assert report.n_cancel_regret == 1


def test_cancel_regret_no_regret_when_market_never_crosses():
    store = EventStore(":memory:")
    store.append(EventType.BOOK_SNAPSHOT, "r1", recv_ts=0.0, source_ts=0.0, payload={"side": "UP", "bids": [], "asks": [[0.60, 100.0]]})
    store.append(EventType.BOOK_DELTA, "r1", recv_ts=5.0, source_ts=5.0, payload={"side": "UP", "book": "asks", "price": 0.60, "size": 0})
    store.append(EventType.BOOK_DELTA, "r1", recv_ts=5.0, source_ts=5.0, payload={"side": "UP", "book": "asks", "price": 0.65, "size": 100.0})

    journal = JournalWriter(":memory:")
    journal.write(SupervisorDecisionRecord(round_id="r1", order_id="o1", decision_ts=1.0, action="CANCEL", reason="REGIME_FLIP", side="UP", price=0.45))

    report = build_supervisor_report(journal, store, regret_lookback_s=10.0)
    assert report.n_cancel_regret == 0


def test_supervisor_report_action_and_reason_counts():
    journal = JournalWriter(":memory:")
    journal.write(SupervisorDecisionRecord(round_id="r1", order_id="o1", decision_ts=1.0, action="CANCEL", reason="REGIME_FLIP", side="UP", price=0.45))
    journal.write(SupervisorDecisionRecord(round_id="r1", order_id="o2", decision_ts=2.0, action="HOLD", reason=None, side="UP", price=0.45))
    store = EventStore(":memory:")
    report = build_supervisor_report(journal, store)
    assert report.action_counts == {"CANCEL": 1, "HOLD": 1}
    assert report.reason_counts == {"REGIME_FLIP": 1}
