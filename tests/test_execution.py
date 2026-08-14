"""Phase 7 verification: "Depth walking tests." "Partial FAK tests."
"Pending-delay no-cancel test." Plus maker fill model and order-state
reconciliation coverage.
"""
from __future__ import annotations

import math

import pytest

from xamarinbot.execution.config import ExecutionConfig, MakerFillConfig
from xamarinbot.execution.maker import fill_probability, maker_expected_value, optimal_maker_price, q_fill
from xamarinbot.execution.order_state import OrderLifecycleState, OrderState, replace_order
from xamarinbot.execution.simulator import ExecutionSimulator
from xamarinbot.execution.taker import simulate_taker_order, walk_depth
from xamarinbot.feeds.base import BookLevel
from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, Side

FEE = FeeConfig()
ASKS = (BookLevel(0.50, 100.0), BookLevel(0.52, 150.0), BookLevel(0.54, 200.0))

# --------------------------------------------------------------------------
# Depth walking
# --------------------------------------------------------------------------


def test_depth_walk_fills_within_single_level():
    r = walk_depth(ASKS, 50.0, limit_price=0.99, fee_config=FEE)
    assert r.fully_filled
    assert r.legs == ((0.50, 50.0),)
    assert math.isclose(r.avg_price, 0.50)
    assert math.isclose(r.total_fee, FEE.taker_fee(50.0, 0.50))


def test_depth_walk_crosses_multiple_levels():
    r = walk_depth(ASKS, 180.0, limit_price=0.99, fee_config=FEE)
    assert r.fully_filled
    assert r.legs == ((0.50, 100.0), (0.52, 80.0))
    expected_avg = (100 * 0.50 + 80 * 0.52) / 180.0
    assert math.isclose(r.avg_price, expected_avg)


def test_depth_walk_stops_at_limit_price():
    r = walk_depth(ASKS, 1000.0, limit_price=0.51, fee_config=FEE)  # only the 0.50 level qualifies
    assert not r.fully_filled
    assert r.limited_by_price
    assert not r.limited_by_depth
    assert r.legs == ((0.50, 100.0),)


def test_depth_walk_partial_fak_when_book_exhausted():
    """Roadmap Phase 7 verification: "Partial FAK tests." Requesting more
    than total depth fills what's available and leaves the rest
    unfilled - no resting remainder, no error."""
    total_depth = sum(l.size for l in ASKS)
    r = walk_depth(ASKS, total_depth + 500.0, limit_price=0.99, fee_config=FEE)
    assert not r.fully_filled
    assert r.limited_by_depth
    assert not r.limited_by_price
    assert math.isclose(r.filled_shares, total_depth)


def test_depth_walk_zero_requested_is_a_noop():
    r = walk_depth(ASKS, 0.0, limit_price=0.99, fee_config=FEE)
    assert r.filled_shares == 0.0
    assert r.legs == ()


# --------------------------------------------------------------------------
# 250ms delay + revalidation/repricing
# --------------------------------------------------------------------------


def test_no_delay_fills_immediately_at_submission_book():
    result = simulate_taker_order(ASKS, 50.0, 0.99, FEE, submit_ts=10.0, taker_delay_ms=0.0)
    assert not result.was_delayed
    assert result.matched_ts == 10.0
    assert result.walk.filled_shares == 50.0


def test_delayed_order_uses_revalidation_book_not_submission_book():
    """The core repricing-risk behavior: the fill must reflect the book at
    matched_ts, not at submit_ts, when they differ."""
    asks_later = (BookLevel(0.60, 100.0),)  # price moved up substantially
    result = simulate_taker_order(ASKS, 50.0, limit_price=0.55, fee_config=FEE, submit_ts=10.0, taker_delay_ms=250.0, asks_at_revalidation=asks_later)
    assert result.was_delayed
    assert result.matched_ts == 10.25
    assert result.repriced
    # limit_price 0.55 no longer reaches the revalidation book's 0.60 ask
    assert result.walk.filled_shares == 0.0
    # what would have filled at submission is still reported for slippage analysis
    assert result.walk_at_submission is not None
    assert result.walk_at_submission.filled_shares == 50.0


def test_delayed_order_same_book_is_not_repriced():
    result = simulate_taker_order(ASKS, 50.0, 0.99, FEE, submit_ts=10.0, taker_delay_ms=250.0, asks_at_revalidation=ASKS)
    assert result.was_delayed
    assert not result.repriced
    assert result.walk.filled_shares == 50.0


# --------------------------------------------------------------------------
# Pending-delay no-cancel (order_state.py + simulator.py)
# --------------------------------------------------------------------------


def test_pending_delay_order_cannot_be_canceled_before_match():
    """Roadmap Phase 7 verification: "Pending-delay no-cancel test." /
    Strategy doc SS2.2: "the order is pending and cannot be canceled.\""""
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig())
    order, result = sim.submit_taker_order("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0, taker_delay_ms=250.0)
    assert order.state is OrderLifecycleState.PENDING_DELAY

    cancel_result = order.cancel(now_ts=10.1)  # still inside the 250ms window
    assert not cancel_result.accepted
    assert cancel_result.reason == "pending_taker_delay"
    assert order.state is OrderLifecycleState.PENDING_DELAY  # unchanged by the rejected attempt


def test_order_can_be_canceled_once_delay_window_passes():
    """can_cancel(now_ts) checks the delay window against the given time
    directly - it doesn't require a prior resolve_pending() call to have
    updated .state first, since the caller may want to know cancelability
    at a hypothetical future instant before actually resolving anything."""
    order = OrderState(order_id="o1", side=Side.UP, role=LiquidityRole.TAKER, limit_price=0.5, requested_shares=10.0, submit_ts=0.0, matched_ts=0.25)
    assert not order.can_cancel(0.1)
    assert order.can_cancel(0.3)  # window passed
    assert order.state is OrderLifecycleState.PENDING_DELAY  # .state itself is unchanged until resolve_pending runs


def test_submit_taker_order_does_not_eagerly_resolve_delayed_fills():
    """Regression test: an earlier version of submit_taker_order applied
    the fill immediately at construction regardless of delay, which made
    can_cancel's time parameter meaningless (every delayed order was
    already FILLED at submission)."""
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig())
    order, result = sim.submit_taker_order("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0, taker_delay_ms=250.0)
    assert order.state is OrderLifecycleState.PENDING_DELAY
    assert order.filled_shares == 0.0

    resolved_too_early = sim.resolve_pending(order, result, now_ts=10.1)
    assert not resolved_too_early
    assert order.state is OrderLifecycleState.PENDING_DELAY

    resolved = sim.resolve_pending(order, result, now_ts=10.25)
    assert resolved
    assert order.state is OrderLifecycleState.FILLED
    assert order.filled_shares == 50.0


def test_non_delayed_taker_order_is_immediately_open_or_filled():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig())
    order, _ = sim.submit_taker_order("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0, taker_delay_ms=0.0)
    assert order.state is OrderLifecycleState.FILLED
    assert order.can_cancel(10.0) is False  # already terminal


# --------------------------------------------------------------------------
# Order-state reconciliation: partial fill then cancel remainder
# --------------------------------------------------------------------------


def test_partial_fill_then_cancel_remainder():
    order = OrderState(order_id="o1", side=Side.UP, role=LiquidityRole.MAKER, limit_price=0.4, requested_shares=100.0, submit_ts=0.0)
    order.reconcile_fill(5.0, 40.0)
    assert order.state is OrderLifecycleState.PARTIALLY_FILLED
    assert order.remaining_shares == 60.0

    cancel_result = order.cancel(6.0)
    assert cancel_result.accepted
    assert order.state is OrderLifecycleState.CANCELED
    assert order.filled_shares == 40.0  # the partial fill is preserved, not rolled back

    # further fill events after cancellation are ignored (terminal state)
    order.reconcile_fill(7.0, 10.0)
    assert order.filled_shares == 40.0


def test_cannot_cancel_a_fully_filled_order():
    order = OrderState(order_id="o1", side=Side.UP, role=LiquidityRole.MAKER, limit_price=0.4, requested_shares=10.0, submit_ts=0.0)
    order.reconcile_fill(1.0, 10.0)
    assert order.state is OrderLifecycleState.FILLED
    result = order.cancel(2.0)
    assert not result.accepted
    assert result.reason == "terminal_state:FILLED"


def test_replace_order_creates_new_order_after_successful_cancel():
    order = OrderState(order_id="o1", side=Side.UP, role=LiquidityRole.MAKER, limit_price=0.4, requested_shares=10.0, submit_ts=0.0)
    result = replace_order(order, now_ts=5.0, new_order_id="o2", new_limit_price=0.41, new_requested_shares=15.0)
    assert result.cancel_result.accepted
    assert order.state is OrderLifecycleState.CANCELED
    assert result.new_order is not None
    assert result.new_order.order_id == "o2"
    assert result.new_order.limit_price == 0.41
    assert result.new_order.requested_shares == 15.0
    assert result.new_order.state is OrderLifecycleState.OPEN


def test_replace_fails_atomically_when_original_order_is_pending_delay():
    order = OrderState(order_id="o1", side=Side.UP, role=LiquidityRole.TAKER, limit_price=0.5, requested_shares=10.0, submit_ts=0.0, matched_ts=0.25)
    result = replace_order(order, now_ts=0.1, new_order_id="o2", new_limit_price=0.5, new_requested_shares=10.0)
    assert not result.cancel_result.accepted
    assert result.new_order is None
    assert order.state is OrderLifecycleState.PENDING_DELAY  # unchanged


# --------------------------------------------------------------------------
# Maker fill probability / adverse selection
# --------------------------------------------------------------------------


def test_fill_probability_decreases_with_distance_to_touch():
    cfg = MakerFillConfig()
    near = fill_probability(distance_to_touch_ticks=0.0, queue_ahead_shares=0.0, horizon_s=10.0, cfg=cfg)
    far = fill_probability(distance_to_touch_ticks=5.0, queue_ahead_shares=0.0, horizon_s=10.0, cfg=cfg)
    assert 0.0 <= far < near <= 1.0


def test_fill_probability_increases_with_horizon():
    cfg = MakerFillConfig()
    short = fill_probability(1.0, 0.0, horizon_s=1.0, cfg=cfg)
    long = fill_probability(1.0, 0.0, horizon_s=100.0, cfg=cfg)
    assert short < long


def test_fill_probability_decreases_with_queue_ahead():
    cfg = MakerFillConfig()
    no_queue = fill_probability(1.0, 0.0, 10.0, cfg)
    big_queue = fill_probability(1.0, 5000.0, 10.0, cfg)
    assert big_queue < no_queue


def test_fill_probability_is_bounded_and_handles_edges():
    cfg = MakerFillConfig()
    assert fill_probability(-1.0, 0.0, 10.0, cfg) == 0.0  # invalid distance
    assert fill_probability(1.0, 0.0, 0.0, cfg) == 0.0  # zero horizon
    assert 0.0 <= fill_probability(0.0, 0.0, 1e6, cfg) <= 1.0  # never exceeds 1


def test_q_fill_adverse_selection_direction():
    cfg = MakerFillConfig(adverse_selection_bp=100.0)
    q = 0.6
    assert q_fill(q, Side.UP, cfg) < q  # a filled UP bid skews toward UP losing more often
    assert q_fill(q, Side.DOWN, cfg) > q  # symmetric for DOWN


def test_maker_expected_value_formula():
    ev = maker_expected_value(rho=0.5, quantity=10.0, q_fill_value=0.6, price=0.4, opportunity_cost=1.0, risk_penalty=0.5)
    assert math.isclose(ev, 0.5 * 10.0 * (0.6 - 0.4) - 1.0 - 0.5)


def test_optimal_maker_price_picks_best_ev_candidate():
    cfg = MakerFillConfig()
    # price 0.30 is far cheaper (bigger edge) even though slightly further from touch
    placement = optimal_maker_price(
        candidate_prices=[0.45, 0.30],
        distances_to_touch_ticks=[0.0, 1.0],
        queue_ahead_shares=[0.0, 0.0],
        quantity=100.0,
        horizon_s=30.0,
        q=0.6,
        side=Side.UP,
        cfg=cfg,
    )
    assert placement.price == 0.30


def test_optimal_maker_price_requires_at_least_one_candidate():
    with pytest.raises(ValueError):
        optimal_maker_price([], [], [], 10.0, 10.0, 0.5, Side.UP, MakerFillConfig())


# --------------------------------------------------------------------------
# Reproducible stochastic maker fill draw (reuses Phase 2's seeded_random)
# --------------------------------------------------------------------------


def test_maker_fill_draw_is_reproducible_for_same_round_and_order():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig())
    order = sim.submit_maker_order("m1", Side.UP, 10.0, 0.4, submit_ts=0.0)
    d1 = sim.draw_maker_fill(order, distance_to_touch_ticks=1.0, queue_ahead_shares=10.0, horizon_s=10.0)
    d2 = sim.draw_maker_fill(order, distance_to_touch_ticks=1.0, queue_ahead_shares=10.0, horizon_s=10.0)
    assert d1 == d2


def test_maker_fill_draw_differs_across_order_ids():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig())
    o1 = sim.submit_maker_order("m1", Side.UP, 10.0, 0.4, submit_ts=0.0)
    o2 = sim.submit_maker_order("m2", Side.UP, 10.0, 0.4, submit_ts=0.0)
    d1 = sim.draw_maker_fill(o1, 1.0, 10.0, 10.0)
    d2 = sim.draw_maker_fill(o2, 1.0, 10.0, 10.0)
    assert d1.draw != d2.draw
