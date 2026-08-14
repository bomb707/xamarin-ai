"""Phase 12B Tranche 1.2 items 5/6: PendingExposure - aggregate risk
exposure from not-yet-confirmed active orders, kept separate from the
confirmed-fills-only PortfolioState kernel.
"""
from __future__ import annotations

import math

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.order_state import OrderState
from xamarinbot.execution.simulator import ExecutionSimulator, TakerOrderQueue
from xamarinbot.feeds.base import BookLevel
from xamarinbot.portfolio.exposure import (
    NO_EXPOSURE,
    PendingExposure,
    exposure_from_open_maker_orders,
    exposure_from_pending_takers,
)
from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, PortfolioState, Side

FEE = FeeConfig()
ASKS = (BookLevel(0.50, 500.0),)


def test_no_exposure_leaves_portfolio_unchanged():
    portfolio = PortfolioState(U=10.0, D=5.0, C=7.0)
    worst = NO_EXPOSURE.worst_case_portfolio(portfolio)
    assert worst == portfolio


def test_worst_case_portfolio_adds_exposure_on_top_of_confirmed():
    portfolio = PortfolioState(U=10.0, D=5.0, C=7.0)
    exposure = PendingExposure(max_committed_spend=20.0, potential_up_shares=15.0, potential_down_shares=0.0)
    worst = exposure.worst_case_portfolio(portfolio)
    assert worst.U == 25.0
    assert worst.D == 5.0
    assert worst.C == 27.0
    assert worst.G < portfolio.G  # strictly more exposed, never less


def test_pending_exposure_addition_sums_all_fields():
    a = PendingExposure(max_committed_spend=10.0, potential_up_shares=5.0, n_pending_delayed_takers=1)
    b = PendingExposure(max_committed_spend=3.0, potential_down_shares=2.0, n_open_maker_orders=1)
    total = a + b
    assert total.max_committed_spend == 13.0
    assert total.potential_up_shares == 5.0
    assert total.potential_down_shares == 2.0
    assert total.n_pending_delayed_takers == 1
    assert total.n_open_maker_orders == 1


def test_exposure_from_pending_takers_uses_worst_case_limit_price():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, limit_price=0.90, asks_at_submission=ASKS, submit_ts=10.0)
    assert pending.was_delayed and not pending.is_resolved

    exposure = exposure_from_pending_takers([pending])
    assert math.isclose(exposure.potential_up_shares, 50.0)
    assert math.isclose(exposure.max_committed_spend, 50.0 * 0.90)  # worst case: fills at the full limit price
    assert exposure.n_pending_delayed_takers == 1


def test_exposure_from_pending_takers_ignores_already_resolved_orders():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=0.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.90, ASKS, submit_ts=10.0)
    assert pending.is_resolved  # no delay - resolved immediately at submission

    exposure = exposure_from_pending_takers([pending])
    assert exposure == NO_EXPOSURE


def test_exposure_from_open_maker_orders_uses_remaining_shares_and_limit_price():
    order = OrderState(order_id="m1", side=Side.DOWN, role=LiquidityRole.MAKER, limit_price=0.40, requested_shares=30.0, submit_ts=0.0)
    order.reconcile_fill(1.0, 10.0)  # 10 of 30 already filled - only the remaining 20 is still exposure

    class _FakeTracked:
        def __init__(self, order_state):
            self.order_state = order_state

    exposure = exposure_from_open_maker_orders([_FakeTracked(order)])
    assert math.isclose(exposure.potential_down_shares, 20.0)
    assert math.isclose(exposure.max_committed_spend, 20.0 * 0.40)
    assert exposure.n_open_maker_orders == 1


# --------------------------------------------------------------------------
# Phase 12B Tranche 1.2 item 5's literal regression: pending order A is
# individually safe, order B is individually safe against the CONFIRMED
# portfolio alone, but A+B jointly would violate g_min - B must be
# rejected/deferred, not admitted merely because confirmed+B looks safe.
# --------------------------------------------------------------------------


def test_second_delayed_taker_is_deferred_even_though_individually_safe_against_confirmed_portfolio():
    confirmed = PortfolioState()  # flat - both A and B look individually safe against this alone
    g_min = -60.0

    # Order A: 60 shares @ ~0.5175 all-in => worst-case cost ~31.05, G' ~ -31.05 - individually safe (>= -60).
    a_shares, a_price = 60.0, 0.50
    a_cost = a_shares * (a_price + FEE.taker_fee(1.0, a_price))
    g_after_a_alone = min(a_shares, 0.0) - a_cost  # D stays 0, min(U,D)=0
    assert g_after_a_alone >= g_min

    # Order B: same size/side, evaluated ALONE against the same confirmed (flat) portfolio - also individually safe.
    b_shares, b_price = 60.0, 0.50
    b_cost = b_shares * (b_price + FEE.taker_fee(1.0, b_price))
    g_after_b_alone = min(b_shares, 0.0) - b_cost
    assert g_after_b_alone >= g_min

    # But A+B together (both UP, both spend, D stays 0) would NOT be safe:
    joint_cost = a_cost + b_cost
    g_after_both = min(a_shares + b_shares, 0.0) - joint_cost
    assert g_after_both < g_min, "test setup must actually construct a jointly-unsafe pair, or this regression is vacuous"

    # The actual admission mechanism: TakerOrderQueue's conservative gate
    # (Phase 12B Tranche 1.2 item 5) must defer B while A is still pending,
    # exactly because this codebase does not yet have the aggregate
    # reservation model that would be required to admit B safely.
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    queue = TakerOrderQueue(sim)

    order_a = queue.try_submit("A", Side.UP, a_shares, a_price, (BookLevel(a_price, 200.0),), submit_ts=10.0)
    assert order_a is not None and order_a.was_delayed

    order_b = queue.try_submit("B", Side.UP, b_shares, b_price, (BookLevel(b_price, 200.0),), submit_ts=10.05)
    assert order_b is None, "B must be rejected/deferred while A is still PENDING_DELAY"

    # Confirm via PendingExposure that the admission gate's caution was
    # actually warranted here - worst-case confirmed+A alone already
    # accounts for A's exposure; had B been (wrongly) admitted too, the
    # combined worst-case portfolio would breach g_min.
    exposure_a_only = queue.exposure
    worst_with_a = exposure_a_only.worst_case_portfolio(confirmed)
    hypothetical_worst_with_both = PortfolioState(
        U=worst_with_a.U + b_shares, D=worst_with_a.D, C=worst_with_a.C + b_cost,
    )
    assert hypothetical_worst_with_both.G < g_min
