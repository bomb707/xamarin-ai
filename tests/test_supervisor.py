"""Phase 9 verification: "Rapid regime flip." "Partial fill then cancel."
"Tick-size change with open order." "Feed stale while maker order rests."
Plus predicate correctness, rate limiting, and cancel-regret analytics.
"""
from __future__ import annotations

import math

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
from xamarinbot.supervisor.predicates import book_displacement, edge_failure, feed_stale, hold_eligible, regime_flip, risk_breach, time_compression, value_cancel, value_hold, value_replace
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
# Phase 12B Tranche 2E: V_hold/V_cancel/V_replace value functions
# --------------------------------------------------------------------------


def test_value_hold_reduces_to_delta_ev_at_default_zero_penalties():
    cfg = SupervisorConfig()
    v = value_hold(7.0, STATE_A, STATE_B, tau=200.0, cfg=cfg)  # flipped, but penalty=0.0
    assert v == 7.0


def test_value_hold_applies_regime_flip_penalty_only_on_an_actual_flip():
    cfg = SupervisorConfig(regime_flip_penalty=3.0)
    assert value_hold(7.0, STATE_A, STATE_B, tau=200.0, cfg=cfg) == 4.0
    assert value_hold(7.0, STATE_A, STATE_A, tau=200.0, cfg=cfg) == 7.0


def test_value_hold_applies_time_compression_penalty_only_when_compressed():
    cfg = SupervisorConfig(min_tau_for_passive_s=15.0, time_compression_penalty=2.0)
    assert value_hold(7.0, STATE_A, STATE_A, tau=10.0, cfg=cfg) == 5.0  # compressed (tau < 15)
    assert value_hold(7.0, STATE_A, STATE_A, tau=20.0, cfg=cfg) == 7.0  # not compressed


def test_value_cancel_is_only_the_negative_cancel_cost():
    """Phase 12B Tranche 2.1 item 10 regression: V_cancel must not include
    edge_min - canceling realizes no economic value at all, only the
    (negative) cost of executing the cancel itself."""
    cfg = SupervisorConfig(edge_min=1.0, cancel_cost=0.4)
    assert math.isclose(value_cancel(cfg), -0.4)


def test_edge_min_gates_hold_eligibility_not_value_hold_magnitude():
    """Phase 12B Tranche 2.1 item 10: edge_min must not shift value_hold's
    own magnitude (that would wrongly distort the HOLD-vs-REPLACE
    comparison too) - it only gates whether HOLD is eligible at all, via
    `hold_eligible`."""
    cfg = SupervisorConfig(edge_min=1.0)
    v = value_hold(7.0, STATE_A, STATE_A, tau=200.0, cfg=cfg)
    assert v == 7.0  # unaffected by edge_min
    assert hold_eligible(v, cfg)  # 7.0 >= 1.0 - 0.0
    assert not hold_eligible(0.5, cfg)  # 0.5 < 1.0 - 0.0


def test_hold_eligible_threshold_widens_by_hysteresis_margin():
    cfg = SupervisorConfig(edge_min=0.0, hysteresis_margin=1.0)
    assert hold_eligible(-0.5, cfg)  # -0.5 >= 0.0 - 1.0
    assert not hold_eligible(-1.5, cfg)  # -1.5 < 0.0 - 1.0


def test_value_replace_is_none_without_an_evaluated_optimal_tick():
    cfg = SupervisorConfig(churn_threshold=0.5)
    assert value_replace(None, cfg) is None


def test_value_replace_nets_out_the_churn_cost():
    cfg = SupervisorConfig(churn_threshold=0.5)
    assert math.isclose(value_replace(2.0, cfg), 1.5)


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
# Phase 12B Tranche 2.1 item 1: hard safety overrides must fire even
# inside the rate-limit window - a stale feed or risk breach occurring
# 100ms after the previous action must still CANCEL immediately, not wait
# for a 1s min_action_interval_s to clear.
# --------------------------------------------------------------------------


def test_feed_stale_bypasses_rate_limiter_even_100ms_after_last_action():
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=1.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(submit_ts=0.0)
    tracked.last_action_ts = 0.0  # last action just happened
    decision = supervisor.review_order(tracked, now_ts=0.1, current_regime_state=STATE_A, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=False)
    assert decision.action is SupervisorActionType.CANCEL
    assert decision.reason is CancelReason.FEED_STALE
    assert decision.detail != "rate_limited"


def test_risk_breach_bypasses_rate_limiter_even_100ms_after_last_action():
    cfg = SupervisorConfig(g_min=-10.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=1.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(submit_ts=0.0)
    tracked.last_action_ts = 0.0
    decision = supervisor.review_order(tracked, now_ts=0.1, current_regime_state=STATE_A, current_delta_ev=5.0, current_g_after_if_fill=-20.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.CANCEL
    assert decision.reason is CancelReason.RISK_BREACH
    assert decision.detail != "rate_limited"


def test_ordinary_economic_churn_is_still_rate_limited():
    """Confirms item 1 only reordered the two hard-safety triggers ahead
    of the limiter - ordinary economic HOLD/CANCEL/REPLACE churn (a
    regime flip with no safety issue) is still throttled."""
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=0.0, min_tau_for_passive_s=0.0, min_action_interval_s=2.0, regime_flip_penalty=100.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(origin=STATE_A, submit_ts=0.0)
    tracked.last_action_ts = 0.0
    decision = supervisor.review_order(tracked, now_ts=0.5, current_regime_state=STATE_B, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.HOLD
    assert decision.detail == "rate_limited"


# --------------------------------------------------------------------------
# "Rapid regime flip" (Roadmap Phase 9 verification, named explicitly) -
# Phase 12B Tranche 2E: a regime flip is no longer an unconditional cancel
# trigger. It now degrades V_hold by `regime_flip_penalty` (0.0 by
# default); whether that actually flips the argmax to CANCEL depends on
# whether the order's edge can absorb the penalty.
# --------------------------------------------------------------------------


def test_regime_flip_alone_does_not_force_cancellation_when_edge_remains_strong():
    """Phase 12B Tranche 2E regression: the pre-2E supervisor canceled on
    ANY regime flip regardless of economics. With `regime_flip_penalty`
    at its default (0.0), a flip must no longer force a cancel by itself
    when the order's edge is still strongly positive - V_hold must beat
    V_cancel on the merits, not lose automatically to a categorical rule."""
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(origin=STATE_A)
    supervisor.register(tracked)

    decision = supervisor.review_order(tracked, now_ts=1.0, current_regime_state=STATE_B, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.HOLD
    assert decision.reason is None


def test_regime_flip_penalty_can_still_trigger_cancellation_once_it_erodes_the_edge():
    """The flip must remain economically *relevant*, just not an absolute
    veto: once `regime_flip_penalty` is large enough to push V_hold below
    V_cancel (here, below `edge_min`), the argmax picks CANCEL and still
    attributes it to REGIME_FLIP for diagnostics - proving the trigger
    still does real work, now mediated through the economics rather than
    bypassing them."""
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=0.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0, regime_flip_penalty=10.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(origin=STATE_A)
    supervisor.register(tracked)

    # Same delta_ev=5.0 that HOLDs above (test_regime_flip_alone_does_not_...)
    # - only the added penalty changes the outcome.
    decision = supervisor.review_order(tracked, now_ts=1.0, current_regime_state=STATE_B, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.CANCEL
    assert decision.reason is CancelReason.REGIME_FLIP

    result = supervisor.apply_cancel(decision, now_ts=1.0)
    assert result.accepted
    assert "o1" not in supervisor.open_order_ids()


def test_regime_flip_penalty_is_not_applied_when_no_flip_occurred():
    """The penalty must be conditional on an actual flip, not a constant
    drag applied every review regardless of regime state."""
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=0.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0, regime_flip_penalty=10.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(origin=STATE_A)
    supervisor.register(tracked)

    decision = supervisor.review_order(tracked, now_ts=1.0, current_regime_state=STATE_A, current_delta_ev=5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.HOLD


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
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=0.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0)
    supervisor = OrderSupervisor(cfg)
    tracked = _tracked(qty=100.0)
    supervisor.register(tracked)

    tracked.order_state.reconcile_fill(1.0, 40.0)  # partial fill happens first
    assert tracked.order_state.state is OrderLifecycleState.PARTIALLY_FILLED

    # current_delta_ev below edge_min so the economics themselves (not a
    # categorical regime-flip rule) drive the CANCEL this test exercises.
    decision = supervisor.review_order(tracked, now_ts=2.0, current_regime_state=STATE_B, current_delta_ev=-5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
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

    # current_delta_ev below the default edge_min=0.0 so the decision is
    # CANCEL (via the V_hold/V_cancel argmax) and apply_cancel's rejection
    # path is actually exercised.
    decision = supervisor.review_order(tracked, now_ts=2.0, current_regime_state=STATE_B, current_delta_ev=-5.0, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.CANCEL
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
# Phase 12B Tranche 2E acceptance items: "cancellation hysteresis" and
# "hold-vs-replace economic comparison"
# --------------------------------------------------------------------------


def test_hysteresis_margin_prevents_thrashing_on_a_marginal_edge_dip():
    """Without hysteresis, an edge that dips just below edge_min tips the
    argmax straight to CANCEL. `hysteresis_margin` biases V_hold so a
    small, possibly-noisy dip doesn't immediately thrash the order -
    proving the incumbency bonus is a real cancellation-hysteresis
    mechanism, not just documentation."""
    tracked_kwargs = dict(now_ts=1.0, current_regime_state=STATE_A, current_delta_ev=-0.5, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)

    cfg_no_margin = SupervisorConfig(g_min=-1000.0, edge_min=0.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0, hysteresis_margin=0.0)
    supervisor_no_margin = OrderSupervisor(cfg_no_margin)
    tracked = _tracked(origin=STATE_A)
    supervisor_no_margin.register(tracked)
    decision_no_margin = supervisor_no_margin.review_order(tracked, **tracked_kwargs)
    assert decision_no_margin.action is SupervisorActionType.CANCEL

    cfg_with_margin = SupervisorConfig(g_min=-1000.0, edge_min=0.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0, hysteresis_margin=1.0)
    supervisor_with_margin = OrderSupervisor(cfg_with_margin)
    tracked2 = _tracked(origin=STATE_A)
    supervisor_with_margin.register(tracked2)
    decision_with_margin = supervisor_with_margin.review_order(tracked2, **tracked_kwargs)
    assert decision_with_margin.action is SupervisorActionType.HOLD


def test_hysteresis_contract_intentionally_permits_a_raw_value_inferior_hold():
    """Phase 12B Tranche 2.2 item 6: pins down the EXACT intended
    contract, not just the observable outcome above. `hysteresis_margin`
    is an incumbency bonus that widens `hold_eligible`'s THRESHOLD
    (`effective_delta_ev >= edge_min - hysteresis_margin`) - it does not
    make V_hold numerically competitive with V_cancel. So this scenario
    must hold even though V_hold's own raw value (unweighted, no margin
    baked into the value itself) is strictly LESS than V_cancel's -
    hysteresis intentionally tolerates a "slightly inferior" HOLD within
    the configured band, rather than only ever choosing HOLD when it
    would win a raw value comparison anyway (which would make the margin
    pointless)."""
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=0.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0, hysteresis_margin=1.0, cancel_cost=0.0)
    tracked = _tracked(origin=STATE_A)
    supervisor = OrderSupervisor(cfg)
    supervisor.register(tracked)

    current_delta_ev = -0.5
    v_hold = value_hold(current_delta_ev, STATE_A, STATE_A, tau=200.0, cfg=cfg)
    v_cancel = value_cancel(cfg)
    assert v_hold < v_cancel, "sanity: V_hold's raw value is strictly worse than V_cancel's here"

    decision = supervisor.review_order(tracked, now_ts=1.0, current_regime_state=STATE_A, current_delta_ev=current_delta_ev, current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True)
    assert decision.action is SupervisorActionType.HOLD, "hysteresis must still choose HOLD despite V_hold < V_cancel in raw value terms"


def test_replace_wins_over_hold_only_once_the_new_tick_clears_the_churn_cost():
    """"Hold-vs-replace economic comparison": REPLACE must not win merely
    because a new tick exists - it has to beat V_hold (the order's
    *current* held value) net of the churn cost. A marginal improvement
    stays HOLD; a large one wins REPLACE."""
    cfg = SupervisorConfig(g_min=-1000.0, edge_min=-1000.0, min_tau_for_passive_s=0.0, min_action_interval_s=0.0, churn_threshold=1.0)
    supervisor = OrderSupervisor(cfg)
    tracked_small = _tracked(order_id="o_small", origin=STATE_A)
    supervisor.register(tracked_small)
    decision_small = supervisor.review_order(
        tracked_small, now_ts=1.0, current_regime_state=STATE_A, current_delta_ev=10.0,
        current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True, current_optimal_ev=10.5,
    )
    assert decision_small.action is SupervisorActionType.HOLD, "a small improvement that doesn't clear the churn cost must not trigger a replace"

    supervisor2 = OrderSupervisor(cfg)
    tracked_big = _tracked(order_id="o_big", origin=STATE_A)
    supervisor2.register(tracked_big)
    decision_big = supervisor2.review_order(
        tracked_big, now_ts=1.0, current_regime_state=STATE_A, current_delta_ev=10.0,
        current_g_after_if_fill=-1.0, tau=200.0, is_fresh=True, current_optimal_ev=15.0,
    )
    assert decision_big.action is SupervisorActionType.REPLACE


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
