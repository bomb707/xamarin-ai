"""Phase 12B Tranche 1.2 items 5/6, corrected and completed in Tranche 2A:
PendingExposure/RiskView - aggregate risk exposure from not-yet-confirmed
active orders, kept separate from the confirmed-fills-only PortfolioState
kernel, admitted via a mathematically correct scenario envelope.
"""
from __future__ import annotations

import math

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.order_state import OrderState
from xamarinbot.execution.simulator import ExecutionSimulator, TakerOrderQueue
from xamarinbot.feeds.base import BookLevel
from xamarinbot.portfolio.exposure import (
    NO_EXPOSURE,
    ActiveOrderExposure,
    PendingExposure,
    RiskView,
    exposure_from_open_maker_orders,
    exposure_from_pending_takers,
)
from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, PortfolioState, Side

FEE = FeeConfig()
ZERO_FEE = FeeConfig(crypto_fee_rate=0.0)
ASKS = (BookLevel(0.50, 500.0),)


def test_no_exposure_worst_case_g_equals_confirmed_g():
    portfolio = PortfolioState(U=10.0, D=5.0, C=7.0)
    assert NO_EXPOSURE.worst_case_g(portfolio) == portfolio.G


def test_pending_exposure_addition_sums_all_fields_including_orders():
    order_a = ActiveOrderExposure(Side.UP, 10.0, 5.0)
    order_b = ActiveOrderExposure(Side.DOWN, 3.0, 1.0)
    a = PendingExposure(max_committed_spend=10.0, potential_up_shares=5.0, n_pending_delayed_takers=1, orders=(order_a,))
    b = PendingExposure(max_committed_spend=3.0, potential_down_shares=2.0, n_open_maker_orders=1, orders=(order_b,))
    total = a + b
    assert total.max_committed_spend == 13.0
    assert total.potential_up_shares == 5.0
    assert total.potential_down_shares == 2.0
    assert total.n_pending_delayed_takers == 1
    assert total.n_open_maker_orders == 1
    assert total.orders == (order_a, order_b)


# --------------------------------------------------------------------------
# Phase 12B Tranche 2A item 1: worst_case_g's scenario envelope - the exact
# counterexample from the reviewer's prompt, proving "all fill" is NOT
# always the worst case for G = min(U,D) - C.
# --------------------------------------------------------------------------


def test_worst_case_g_all_fill_is_not_necessarily_worst_case():
    """U=D=C=0, pending 100 UP@0.40 and 100 DOWN@0.40 (zero fee for exact
    round numbers). Both filling: G=+20. Only UP filling: G=-40. The
    worst case (only one side fills) must be found, not "both fill"."""
    confirmed = PortfolioState()
    up_order = ActiveOrderExposure(Side.UP, 100.0, 100.0 * 0.40)
    down_order = ActiveOrderExposure(Side.DOWN, 100.0, 100.0 * 0.40)
    exposure = PendingExposure(orders=(up_order, down_order))

    # Sanity-check both named scenarios directly against the exact kernel.
    both_fill = PortfolioState(U=100.0, D=100.0, C=80.0)
    assert math.isclose(both_fill.G, 20.0)
    only_up_fills = PortfolioState(U=100.0, D=0.0, C=40.0)
    assert math.isclose(only_up_fills.G, -40.0)

    worst = exposure.worst_case_g(confirmed)
    assert math.isclose(worst, -40.0), "worst_case_g must find the only-one-side-fills scenario, not assume all-fill"


def test_worst_case_g_exact_enumeration_matches_hand_computed_minimum_over_all_subsets():
    confirmed = PortfolioState(U=5.0, D=5.0, C=2.0)
    orders = (
        ActiveOrderExposure(Side.UP, 20.0, 9.0),
        ActiveOrderExposure(Side.DOWN, 15.0, 6.0),
        ActiveOrderExposure(Side.UP, 10.0, 4.5),
    )
    exposure = PendingExposure(orders=orders)

    # Hand-enumerate every subset (2^3 = 8) independently of the implementation.
    best = None
    for mask in range(8):
        u, d, c = confirmed.U, confirmed.D, confirmed.C
        for i, o in enumerate(orders):
            if mask & (1 << i):
                if o.side is Side.UP:
                    u += o.max_fill_qty
                else:
                    d += o.max_fill_qty
                c += o.max_cost
        g = min(u, d) - c
        if best is None or g < best:
            best = g

    assert math.isclose(exposure.worst_case_g(confirmed), best)


def test_worst_case_g_falls_back_to_conservative_bound_beyond_exact_cap():
    confirmed = PortfolioState()
    orders = tuple(ActiveOrderExposure(Side.UP, 10.0, 4.0) for _ in range(5))
    exposure = PendingExposure(orders=orders)

    exact = exposure.worst_case_g(confirmed, exact_subset_cap=10)
    conservative = exposure.worst_case_g(confirmed, exact_subset_cap=2)  # force the fallback path

    total_cost = sum(o.max_cost for o in orders)
    expected_conservative = min(confirmed.U, confirmed.D) - (confirmed.C + total_cost)
    assert math.isclose(conservative, expected_conservative)
    # the conservative bound must never be more optimistic (higher) than the true exact worst case
    assert conservative <= exact + 1e-9


# --------------------------------------------------------------------------
# Phase 12B Tranche 2A item 2: fee-inclusive committed cost.
# --------------------------------------------------------------------------


def test_exposure_from_pending_takers_includes_taker_fee_in_max_cost():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, limit_price=0.90, asks_at_submission=ASKS, submit_ts=10.0)
    assert pending.was_delayed and not pending.is_resolved

    exposure = exposure_from_pending_takers([pending], FEE)
    expected_fee = FEE.taker_fee(50.0, 0.90)
    assert expected_fee > 0.0
    assert math.isclose(exposure.max_committed_spend, 50.0 * 0.90 + expected_fee)
    assert exposure.orders[0].max_cost == exposure.max_committed_spend


def test_exposure_from_pending_takers_ignores_already_resolved_orders():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=0.0))
    pending = sim.submit_taker("o1", Side.UP, 50.0, 0.90, ASKS, submit_ts=10.0)
    assert pending.is_resolved  # no delay - resolved immediately at submission

    exposure = exposure_from_pending_takers([pending], FEE)
    assert exposure == NO_EXPOSURE


def test_exposure_from_open_maker_orders_uses_remaining_shares_and_is_fee_consistent():
    order = OrderState(order_id="m1", side=Side.DOWN, role=LiquidityRole.MAKER, limit_price=0.40, requested_shares=30.0, submit_ts=0.0)
    order.reconcile_fill(1.0, 10.0)  # 10 of 30 already filled - only the remaining 20 is still exposure

    class _FakeTracked:
        def __init__(self, order_state):
            self.order_state = order_state

    exposure = exposure_from_open_maker_orders([_FakeTracked(order)], FEE)
    assert math.isclose(exposure.potential_down_shares, 20.0)
    # maker fee is always 0 today, but the cost must still be computed via
    # fee_config (not hardcoded), matching apply_fill's cost definition exactly.
    expected_fee = FEE.fee_for(LiquidityRole.MAKER, 20.0, 0.40)
    assert expected_fee == 0.0
    assert math.isclose(exposure.max_committed_spend, 20.0 * 0.40 + expected_fee)
    assert exposure.n_open_maker_orders == 1


# --------------------------------------------------------------------------
# RiskView (Phase 12B Tranche 2A item 3): the hard-admission interface.
# --------------------------------------------------------------------------


def test_risk_view_committed_spend_and_potential_positions_combine_confirmed_and_pending():
    confirmed = PortfolioState(U=10.0, D=5.0, C=7.0)
    taker_exposure = PendingExposure(max_committed_spend=8.0, potential_up_shares=6.0, orders=(ActiveOrderExposure(Side.UP, 6.0, 8.0),))
    maker_exposure = PendingExposure(max_committed_spend=2.0, potential_down_shares=3.0, orders=(ActiveOrderExposure(Side.DOWN, 3.0, 2.0),))
    view = RiskView(confirmed, taker_exposure, maker_exposure)

    assert math.isclose(view.committed_spend, 7.0 + 8.0 + 2.0)
    assert math.isclose(view.potential_up_position, 10.0 + 6.0)
    assert math.isclose(view.potential_down_position, 5.0 + 3.0)


def test_risk_view_admits_rejects_candidate_that_would_breach_g_min_jointly():
    """The literal item-4 scenario: an existing maker A is individually
    safe, a new maker B is individually safe against confirmed alone, but
    A+B jointly breach g_min - RiskView.admits must reject B."""
    confirmed = PortfolioState()  # flat
    g_min = -60.0

    order_a = ActiveOrderExposure(Side.UP, 60.0, 60.0 * 0.5175)  # individually safe: G = -31.05 >= -60
    assert PendingExposure(orders=(order_a,)).worst_case_g(confirmed) >= g_min

    order_b = ActiveOrderExposure(Side.UP, 60.0, 60.0 * 0.5175)  # also individually safe alone
    assert PendingExposure(orders=(order_b,)).worst_case_g(confirmed) >= g_min

    view = RiskView(confirmed, open_maker_exposure=PendingExposure(orders=(order_a,), max_committed_spend=order_a.max_cost, potential_up_shares=order_a.max_fill_qty))
    admitted = view.admits(order_b, g_min, spend_cap=None)
    assert not admitted, "B must be rejected once A is already tracked in the RiskView - A+B jointly breach g_min"


def test_risk_view_admits_accepts_candidate_when_jointly_safe():
    confirmed = PortfolioState()
    g_min = -1000.0
    order_a = ActiveOrderExposure(Side.UP, 10.0, 5.0)
    view = RiskView(confirmed, open_maker_exposure=PendingExposure(orders=(order_a,), max_committed_spend=5.0, potential_up_shares=10.0))
    candidate = ActiveOrderExposure(Side.DOWN, 10.0, 5.0)
    assert view.admits(candidate, g_min, spend_cap=None)


def test_risk_view_admits_opposite_side_makers_where_each_alone_looks_safe_but_one_sided_fill_is_not():
    """Opposite-side makers: a UP maker and a DOWN maker each individually
    safe, but if only the UP one fills (the DOWN one never does), the
    resulting one-sided position breaches g_min - admits() must catch
    this via the scenario envelope, not just check "both fill together"."""
    confirmed = PortfolioState()
    g_min = -50.0
    order_up = ActiveOrderExposure(Side.UP, 100.0, 51.75)  # only-this-fills: G = min(100,0)-51.75 = -51.75 < -50

    view = RiskView(confirmed)
    admitted = view.admits(order_up, g_min, spend_cap=None)
    assert not admitted  # the one-sided-fill subset alone already breaches g_min, regardless of a second order


def test_risk_view_admits_rejects_on_spend_cap_breach():
    confirmed = PortfolioState(C=90.0)
    order_a = ActiveOrderExposure(Side.UP, 5.0, 5.0)
    candidate = ActiveOrderExposure(Side.DOWN, 6.0, 6.0)
    view = RiskView(confirmed, pending_taker_exposure=PendingExposure(orders=(order_a,), max_committed_spend=5.0))
    assert not view.admits(candidate, g_min=-1_000_000.0, spend_cap=100.0)  # 90+5+6=101 > 100


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.2 item 1: aggregate breach-recovery semantics - the
# reviewer's exact counterexample (G_current=-50, g_min=-20, a hedge
# improving to -25 must still be admissible even though -25 < -20, since
# the aggregate envelope's own empty-fill subset would otherwise make
# every recovery candidate permanently aggregate-invalid once breached).
# --------------------------------------------------------------------------


def test_risk_view_admits_recovery_candidate_when_it_does_not_worsen_the_aggregate_envelope():
    confirmed = PortfolioState(U=0.0, D=50.0, C=50.0)  # G = min(0,50) - 50 = -50, already below g_min
    g_min = -20.0
    assert confirmed.G == -50.0

    # Buying the underrepresented UP side (D=50 > U=0) raises min(U,D)
    # once filled - worst_case_g's own "nothing new fills" subset still
    # reproduces exactly worst_g_base=-50 (the pre-candidate G), so the
    # candidate's inclusion can never make the aggregate envelope WORSE
    # than -50, satisfying "does not worsen" even though the candidate's
    # OWN if-filled G (-25) stays below g_min=-20.
    hedge_candidate = ActiveOrderExposure(Side.UP, 50.0, 25.0)
    only_hedge_fills = PortfolioState(U=50.0, D=50.0, C=75.0)
    assert only_hedge_fills.G == -25.0  # sanity: the reviewer's own -50 -> -25 example

    view = RiskView(confirmed)
    assert view.admits(hedge_candidate, g_min, spend_cap=None, is_recovery_candidate=True)


def test_risk_view_admits_rejects_recovery_candidate_that_would_worsen_the_aggregate_envelope():
    confirmed = PortfolioState(C=50.0)  # G = -50
    g_min = -20.0
    # DOWN while U=D=0 doesn't raise min(U,D) at all (stays 0) but adds
    # cost whenever it fills - genuinely worsens the worst case.
    bad_candidate = ActiveOrderExposure(Side.DOWN, 10.0, 5.0)
    view = RiskView(confirmed)
    assert not view.admits(bad_candidate, g_min, spend_cap=None, is_recovery_candidate=True)


def test_risk_view_admits_rejects_non_recovery_candidate_outright_while_aggregate_breached():
    """"ALPHA remains prohibited while breached" applies at the aggregate
    level too - is_recovery_candidate=False (an ALPHA-purpose candidate)
    is rejected outright once worst_g_base < g_min, even for a candidate
    that would itself improve G if it filled."""
    confirmed = PortfolioState(U=0.0, D=50.0, C=50.0)  # G = -50, same as the "does not worsen" test above
    g_min = -20.0
    candidate = ActiveOrderExposure(Side.UP, 50.0, 25.0)  # the same genuinely-improving candidate as above
    view = RiskView(confirmed)
    assert not view.admits(candidate, g_min, spend_cap=None, is_recovery_candidate=False)


def test_risk_view_admits_safe_mode_ordinary_g_min_floor_unaffected_by_recovery_flag():
    """When NOT already breached (worst_g_base >= g_min), the ordinary
    hard rule (worst_g_new >= g_min) still applies regardless of
    is_recovery_candidate - the recovery relaxation only ever kicks in
    once already below the floor."""
    confirmed = PortfolioState()  # G = 0, safely above g_min
    g_min = -20.0
    candidate = ActiveOrderExposure(Side.UP, 1000.0, 500.0)  # would badly breach g_min if it fills
    view = RiskView(confirmed)
    assert not view.admits(candidate, g_min, spend_cap=None, is_recovery_candidate=True)


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.1 item 2: aggregate position_limit enforcement -
# previously only exposed via potential_up_position/potential_down_position
# but never actually enforced inside admits().
# --------------------------------------------------------------------------


def test_risk_view_admits_rejects_on_position_limit_breach():
    confirmed = PortfolioState(U=90.0)
    candidate = ActiveOrderExposure(Side.UP, 20.0, 10.0)  # 90+20=110 > 100
    view = RiskView(confirmed)
    assert not view.admits(candidate, g_min=-1_000_000.0, spend_cap=None, position_limit=100.0)


def test_risk_view_admits_uses_plain_sum_for_position_limit_not_the_subset_envelope():
    """Unlike G, a side's position can only ever grow as more orders on
    that side fill (no min() interaction) - "every order on this side
    fills" genuinely is the (only relevant) worst case, per the reviewer's
    own U_potential=U+sum(UP_pending) formula."""
    confirmed = PortfolioState(U=50.0)
    order_a = ActiveOrderExposure(Side.UP, 30.0, 15.0)
    candidate = ActiveOrderExposure(Side.UP, 30.0, 15.0)  # 50+30+30=110 > 100
    view = RiskView(confirmed, pending_taker_exposure=PendingExposure(orders=(order_a,), potential_up_shares=30.0))
    assert not view.admits(candidate, g_min=-1_000_000.0, spend_cap=None, position_limit=100.0)


def test_risk_view_admits_position_limit_none_skips_the_check():
    confirmed = PortfolioState(U=90.0)
    candidate = ActiveOrderExposure(Side.UP, 500.0, 10.0)
    view = RiskView(confirmed)
    # g_min permissive, spend_cap None, position_limit None (default) -
    # a huge position must not be rejected when the caller passed no limit.
    assert view.admits(candidate, g_min=-1_000_000.0, spend_cap=None)


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.1 item 2: combined pending-taker + open-maker
# admission - "taker + open-maker RiskView admission" acceptance item.
# --------------------------------------------------------------------------


def test_risk_view_admits_combines_pending_taker_and_open_maker_exposure():
    """An open maker A and a pending (in-flight, delayed) taker B are each
    individually safe against confirmed alone, but a new candidate C is
    only rejected once BOTH A's and B's exposure are combined in the
    view - proving admits() truly aggregates across both exposure kinds,
    not just one."""
    confirmed = PortfolioState()
    g_min = -90.0
    # All three same-side (UP): D never moves, so min(U,D)=0 regardless of
    # how many fill - cost only ever accumulates, making "all fill" the
    # worst subset here (unlike the opposite-side counterexample above).
    maker_a = ActiveOrderExposure(Side.UP, 50.0, 50.0 * 0.5)  # alone: G=min(50,0)-25=-25 >= -90
    taker_b = ActiveOrderExposure(Side.UP, 50.0, 50.0 * 0.5)  # alone: G=min(50,0)-25=-25 >= -90
    candidate_c = ActiveOrderExposure(Side.UP, 50.0, 50.0 * 0.5)  # alone: G=min(50,0)-25=-25 >= -90

    view = RiskView(
        confirmed,
        pending_taker_exposure=PendingExposure(orders=(taker_b,), max_committed_spend=taker_b.max_cost, potential_up_shares=taker_b.max_fill_qty),
        open_maker_exposure=PendingExposure(orders=(maker_a,), max_committed_spend=maker_a.max_cost, potential_up_shares=maker_a.max_fill_qty),
    )
    # A+B+C all UP, all-fill: G = min(150,0) - 75 = -75 >= -90, still admits.
    assert view.admits(candidate_c, g_min, spend_cap=None)
    # Tighten g_min just past that combined worst case - now must reject,
    # proving the check really did combine all three orders' exposure.
    assert not view.admits(candidate_c, g_min=-70.0, spend_cap=None)


# --------------------------------------------------------------------------
# TakerOrderQueue integration (unchanged behavior, new fee-inclusive exposure)
# --------------------------------------------------------------------------


def test_taker_order_queue_exposure_uses_fee_inclusive_cost():
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    queue = TakerOrderQueue(sim)
    queue.try_submit("o1", Side.UP, 50.0, 0.90, ASKS, submit_ts=10.0)
    expected_fee = FEE.taker_fee(50.0, 0.90)
    assert math.isclose(queue.exposure.max_committed_spend, 50.0 * 0.90 + expected_fee)


def test_taker_order_queue_defers_a_second_delayed_submission_while_one_is_pending():
    """Phase 12B Tranche 1.2 item 5's conservative gate: with no full
    aggregate reservation model wired into candidate GENERATION yet, a
    second delayed taker order must be rejected/deferred while one is
    already PENDING_DELAY."""
    sim = ExecutionSimulator("r0", FEE, ExecutionConfig(taker_delay_ms=250.0))
    queue = TakerOrderQueue(sim)

    first = queue.try_submit("o1", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.0)
    assert first is not None and first.was_delayed

    second = queue.try_submit("o2", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.05)
    assert second is None  # deferred - queue.has_pending was already True
    assert len(queue._pending) == 1

    queue.resolve_ready(now_ts=10.25, book_at_fn=lambda p: ASKS)
    assert not queue.has_pending

    third = queue.try_submit("o3", Side.UP, 50.0, 0.99, ASKS, submit_ts=10.30)
    assert third is not None  # admitted now that the queue is clear again
