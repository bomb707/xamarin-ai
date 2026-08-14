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
from xamarinbot.execution.simulator import ExecutionSimulator, TakerOrderQueue
from xamarinbot.execution.taker import walk_at_submission, walk_depth
from xamarinbot.feeds.base import BookLevel
from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, PortfolioState, Side

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
# walk_at_submission (pure, present-time-only diagnostic) + the
# submit_taker/resolve_taker delay lifecycle (Phase 12B Tranche 1.2 item 2:
# there is no longer any standalone function that computes an order's
# "eventual" fill at submission time - a delayed order's real fill only
# ever comes from resolve_taker(), called with the actual causal book at
# matched_ts, once the caller's own clock has genuinely reached it).
# --------------------------------------------------------------------------


def test_walk_at_submission_reflects_the_book_visible_now():
    result = walk_at_submission(ASKS, 50.0, limit_price=0.99, fee_config=FEE)
    assert result.filled_shares == 50.0
    assert math.isclose(result.avg_price, 0.50)


def test_submit_taker_with_no_delay_resolves_immediately_at_submission_book():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=0.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)
    assert not pending.was_delayed
    assert pending.matched_ts == 10.0
    assert pending.is_resolved
    result = sim.resolve_taker(pending)
    assert not result.was_delayed
    assert result.walk.filled_shares == 50.0


def test_submit_taker_with_delay_does_not_resolve_or_leak_the_future_fill():
    """Phase 12B Tranche 1.2 item 2's core requirement: submit_taker() for
    a delayed order must not compute or expose any future fill information
    at all - not even sitting unused inside the returned object - since a
    caller reading it immediately (as one previously did, when
    execute_taker's PendingTakerOrder-equivalent still carried a fully-
    computed TakerOrderResult) would silently bypass the delay entirely."""
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)
    assert pending.was_delayed
    assert pending.matched_ts == 10.25
    assert not pending.is_resolved
    assert pending.order_state.state is OrderLifecycleState.PENDING_DELAY
    assert pending.order_state.filled_shares == 0.0
    # only present-time diagnostic info exists - never the eventual fill
    assert pending.walk_at_submission.filled_shares == 50.0


def test_resolve_taker_raises_if_called_before_matched_ts():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)
    with pytest.raises(ValueError):
        sim.resolve_taker(pending, ASKS, now_ts=10.1)  # still inside the delay window
    assert pending.order_state.state is OrderLifecycleState.PENDING_DELAY  # unchanged by the rejected attempt


def test_resolve_taker_raises_without_a_book_to_resolve_against():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)
    with pytest.raises(ValueError):
        sim.resolve_taker(pending, now_ts=10.25)  # no asks_at_match given


def test_resolve_taker_uses_the_later_book_not_the_submission_book():
    """The core repricing-risk requirement: the fill must reflect the book
    at matched_ts, not at submit_ts, when they differ - and that book is
    only ever supplied at resolve time, never at submission."""
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    asks_later = (BookLevel(0.60, 100.0),)  # price moved up substantially
    pending = sim.submit_taker("o1", Side.UP, 50.0, limit_price=0.55, asks_at_submission=ASKS, submit_ts=10.0)

    result = sim.resolve_taker(pending, asks_later, now_ts=10.25)

    assert result.was_delayed
    assert result.repriced
    # limit_price 0.55 no longer reaches the later book's 0.60 ask - proves
    # the *later* book determined the outcome, not the submission book
    # (which would have filled fully at 0.50).
    assert result.walk.filled_shares == 0.0
    assert result.walk_at_submission is not None
    assert result.walk_at_submission.filled_shares == 50.0  # diagnostic only - not what actually filled
    assert pending.order_state.state is OrderLifecycleState.CANCELED  # FAK, no resting remainder
    assert pending.order_state.filled_shares == 0.0


def test_resolve_taker_same_book_is_not_repriced():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)

    result = sim.resolve_taker(pending, ASKS, now_ts=10.25)

    assert result.was_delayed
    assert not result.repriced
    assert result.walk.filled_shares == 50.0
    assert pending.order_state.state is OrderLifecycleState.FILLED
    assert pending.order_state.filled_shares == 50.0


def test_resolve_taker_is_idempotent_once_resolved():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)
    first = sim.resolve_taker(pending, ASKS, now_ts=10.25)
    second = sim.resolve_taker(pending)  # no book/now_ts needed - already resolved
    assert first is second


# --------------------------------------------------------------------------
# Phase 12B Tranche 1.2 item 4: a delayed order's real, hard,
# candidate-specific worst-price limit (optimizer.candidates.
# taker_max_execution_price) must prevent an adverse repricing during the
# delay window from breaching g_min/spend_cap - the exact scenario named
# in the reviewer's prompt (p_submit=0.50, p_later=0.80, fee_rate=0.07).
# --------------------------------------------------------------------------


def test_taker_delayed_repricing_cannot_breach_g_min():
    from xamarinbot.optimizer.candidates import taker_max_execution_price
    from xamarinbot.optimizer.config import OneStepConfig

    fee_config = FeeConfig(crypto_fee_rate=0.07)
    p_submit, p_later = 0.50, 0.80
    c_submit = p_submit + fee_config.taker_fee(1.0, p_submit)
    c_later = p_later + fee_config.taker_fee(1.0, p_later)
    assert math.isclose(c_submit, 0.5175, rel_tol=1e-9)
    assert math.isclose(c_later, 0.8112, rel_tol=1e-9)

    portfolio = PortfolioState()  # flat
    cfg = OneStepConfig(g_min=-100.0, min_marginal_edge=0.0)
    q_effective = 0.85  # clears marginal edge at BOTH 0.50 and 0.80 (0.85 - 0.8112 > 0)

    # Size the candidate at exactly its submission-time risk boundary -
    # the largest x still G_min-safe if it fills at c_submit.
    x = (portfolio.G - cfg.g_min) / c_submit  # K(x) = x*c_submit == budget
    g_if_filled_at_submit = min(portfolio.U + x, portfolio.D) - portfolio.C - x * c_submit
    assert g_if_filled_at_submit >= cfg.g_min - 1e-9  # individually safe at the submission price

    # The more expensive level is still positive marginal EV on its own
    # (clears min_marginal_edge) - the old p_max (depth/marginal-edge-only
    # boundary) would have allowed walking into it.
    assert q_effective - c_later > cfg.min_marginal_edge

    # The hard, risk-derived limit for THIS quantity must exclude p_later.
    hard_limit = taker_max_execution_price(portfolio, Side.UP, x, q_effective, fee_config, cfg, tick_size=0.01)
    assert hard_limit is not None
    assert hard_limit < p_later
    assert hard_limit <= p_submit + 0.01  # at (about) the submission price, not materially looser

    # Submit at the cheap book, using the derived hard limit as the
    # candidate's actual limit_price - then the cheap liquidity vanishes
    # and only the expensive level remains by matched_ts.
    sim = ExecutionSimulator("r0", fee_config, ExecutionConfig(taker_delay_ms=250.0))
    book_at_submission = (BookLevel(p_submit, x + 50.0),)
    book_at_matched_ts = (BookLevel(p_later, x + 50.0),)  # cheap liquidity disappeared during the delay

    pending = sim.submit_taker("o1", Side.UP, x, limit_price=hard_limit, asks_at_submission=book_at_submission, submit_ts=10.0)
    result = sim.resolve_taker(pending, book_at_matched_ts, now_ts=10.25)

    # Expected result: zero fill (partial/zero, never a hard-risk
    # violation) - the hard limit protected G_min by refusing the 0.80 level.
    assert result.walk.filled_shares == 0.0
    g_after = min(portfolio.U + pending.order_state.filled_shares, portfolio.D) - portfolio.C - result.walk.total_paid
    assert g_after >= cfg.g_min - 1e-9, "delayed repricing must never be able to breach g_min"


def test_taker_delayed_repricing_partial_fill_still_respects_g_min():
    """Contrast case: if the repriced level offers enough depth that only
    part of the order clears the hard limit, the partial fill itself must
    still be G_min-safe - proving the protection holds for partial fills,
    not only the zero-fill case above."""
    from xamarinbot.optimizer.candidates import taker_max_execution_price
    from xamarinbot.optimizer.config import OneStepConfig

    fee_config = FeeConfig(crypto_fee_rate=0.07)
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-40.0, min_marginal_edge=0.0)
    q_effective = 0.9
    p_submit, p_later = 0.30, 0.35  # a smaller, still-adverse move

    c_submit = p_submit + fee_config.taker_fee(1.0, p_submit)
    x = (portfolio.G - cfg.g_min) / c_submit
    hard_limit = taker_max_execution_price(portfolio, Side.UP, x, q_effective, fee_config, cfg, tick_size=0.01)
    assert hard_limit is not None

    sim = ExecutionSimulator("r0", fee_config, ExecutionConfig(taker_delay_ms=250.0))
    pending = sim.submit_taker("o1", Side.UP, x, limit_price=hard_limit, asks_at_submission=(BookLevel(p_submit, x),), submit_ts=10.0)
    result = sim.resolve_taker(pending, (BookLevel(p_later, x),), now_ts=10.25)

    filled = pending.order_state.filled_shares
    g_after = min(portfolio.U + filled, portfolio.D) - portfolio.C - result.walk.total_paid
    assert g_after >= cfg.g_min - 1e-9


# --------------------------------------------------------------------------
# Pending-delay no-cancel (order_state.py + simulator.py)
# --------------------------------------------------------------------------


def test_pending_delay_order_cannot_be_canceled_before_match():
    """Roadmap Phase 7 verification: "Pending-delay no-cancel test." /
    Strategy doc SS2.2: "the order is pending and cannot be canceled.\""""
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig())
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0, taker_delay_ms=250.0)
    order = pending.order_state
    assert order.state is OrderLifecycleState.PENDING_DELAY

    cancel_result = order.cancel(now_ts=10.1)  # still inside the 250ms window
    assert not cancel_result.accepted
    assert cancel_result.reason == "pending_taker_delay"
    assert order.state is OrderLifecycleState.PENDING_DELAY  # unchanged by the rejected attempt


def test_order_can_be_canceled_once_delay_window_passes():
    """can_cancel(now_ts) checks the delay window against the given time
    directly - it doesn't require a prior resolve_taker() call to have
    updated .state first, since the caller may want to know cancelability
    at a hypothetical future instant before actually resolving anything."""
    order = OrderState(order_id="o1", side=Side.UP, role=LiquidityRole.TAKER, limit_price=0.5, requested_shares=10.0, submit_ts=0.0, matched_ts=0.25)
    assert not order.can_cancel(0.1)
    assert order.can_cancel(0.3)  # window passed
    assert order.state is OrderLifecycleState.PENDING_DELAY  # .state itself is unchanged until resolve_taker runs


def test_non_delayed_taker_order_is_immediately_open_or_filled():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig())
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0, taker_delay_ms=0.0)
    assert pending.order_state.state is OrderLifecycleState.FILLED
    assert pending.order_state.can_cancel(10.0) is False  # already terminal


# --------------------------------------------------------------------------
# TakerOrderQueue - the shared pending-order lifecycle helper every
# execution-oriented backtest/shadow/baseline path now uses (Phase 12B
# Tranche 1.2 items 1 & 5), including its conservative single-pending gate.
# --------------------------------------------------------------------------


def test_taker_order_queue_resolves_only_once_matched_ts_is_reached():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    queue = TakerOrderQueue(sim)
    pending = queue.try_submit("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)
    assert pending is not None and pending.was_delayed
    assert queue.has_pending

    too_early = queue.resolve_ready(now_ts=10.1, book_at_fn=lambda p: ASKS)
    assert too_early == []
    assert queue.has_pending  # still outstanding - matched_ts (10.25) hasn't arrived

    resolved = queue.resolve_ready(now_ts=10.25, book_at_fn=lambda p: ASKS)
    assert len(resolved) == 1
    _, result = resolved[0]
    assert result.walk.filled_shares == 50.0
    assert not queue.has_pending


def test_taker_order_queue_defers_a_second_delayed_submission_while_one_is_pending():
    """Phase 12B Tranche 1.2 item 5's conservative gate: with no aggregate
    exposure/reservation model yet, a second delayed taker order must be
    rejected/deferred while one is already PENDING_DELAY - regardless of
    whether the two would individually or jointly be risk-safe."""
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    queue = TakerOrderQueue(sim)

    first = queue.try_submit("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)
    assert first is not None and first.was_delayed

    second = queue.try_submit("o2", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.05)
    assert second is None  # deferred - queue.has_pending was already True
    assert len(queue._pending) == 1  # no second order was ever actually submitted/queued

    queue.resolve_ready(now_ts=10.25, book_at_fn=lambda p: ASKS)
    assert not queue.has_pending

    third = queue.try_submit("o3", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.30)
    assert third is not None  # admitted now that the queue is clear again


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
