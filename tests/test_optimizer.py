"""Phase 8 verification: "Verify optimizer never violates G_min." "Stress
test many book/portfolio states." Plus candidate generation/evaluation
correctness and controller selection logic.
"""
from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.feeds.base import BookLevel, BookSnapshot
from xamarinbot.optimizer.candidates import (
    delta_ev_directional,
    dynamic_maker_candidates,
    evaluate_maker_candidate,
    evaluate_taker_candidate,
    generate_buffer_build_candidates,
    generate_hedge_candidate,
    maker_price_grid,
    purpose_aware_max_execution_price,
    taker_max_execution_price,
    taker_quantities,
    taker_sizing_boundaries,
    wait_candidate,
)
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import OrderMode
from xamarinbot.portfolio.math import OrderPurpose, delta_g_directional, directional_projected_g
from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, PortfolioState, Side
from xamarinbot.regime.types import SeedAction

FEE = FeeConfig()
ASKS = (BookLevel(0.50, 100.0), BookLevel(0.52, 150.0), BookLevel(0.54, 200.0))


# --------------------------------------------------------------------------
# Phase 12B Tranche 1, items 7/8/10: marginal edge, boundary-aware taker
# sizing, and worst-price protection (taker_sizing_boundaries)
# --------------------------------------------------------------------------


def test_taker_sizing_produces_exact_partial_quantity_at_risk_budget_boundary():
    """The audit's own example: a 500-share first ask level, but only a
    fraction of it is actually risk-feasible - the optimizer must be able
    to choose that exact partial quantity, not just 0 or the full 500."""
    asks = (BookLevel(0.50, 500.0),)
    fee_config = FeeConfig()
    c_1 = 0.50 + fee_config.taker_fee(1.0, 0.50)  # fee-per-share at this level's price
    risk_budget = 7.4 * c_1  # engineered so the risk-budget boundary lands at exactly 7.4 shares
    cfg = OneStepConfig(g_min=-risk_budget)  # G_current=0 (empty portfolio) - g_min => max_directional_spend = risk_budget
    portfolio = PortfolioState()

    sizing = taker_sizing_boundaries(asks, q_effective=0.9, fee_config=fee_config, cfg=cfg, portfolio=portfolio, side=Side.UP)

    assert sizing.p_max == 0.50
    assert any(math.isclose(q, 7.4, rel_tol=1e-6) for q in sizing.quantities), sizing.quantities
    assert all(q <= 7.4 + 1e-6 for q in sizing.quantities)  # never offers more than what's risk-feasible


def test_taker_sizing_stops_at_last_level_with_positive_marginal_edge():
    """Worst-price protection: a second, expensive level with negative
    marginal edge must never be walked, regardless of how loose the risk
    budget is - this replaces the old unconditional limit_price=1.0."""
    asks = (BookLevel(0.50, 100.0), BookLevel(0.95, 100.0))
    fee_config = FeeConfig()
    cfg = OneStepConfig(g_min=-1_000_000.0)  # risk budget deliberately not the binding constraint here
    portfolio = PortfolioState()

    sizing = taker_sizing_boundaries(asks, q_effective=0.9, fee_config=fee_config, cfg=cfg, portfolio=portfolio, side=Side.UP)

    assert sizing.p_max == 0.50  # the 0.95 level is never walked
    assert max(sizing.quantities) <= 100.0 + 1e-6  # never offers quantity from the rejected level


def test_taker_sizing_returns_nothing_when_no_level_clears_marginal_edge():
    asks = (BookLevel(0.50, 500.0),)
    fee_config = FeeConfig()
    cfg = OneStepConfig(g_min=-1_000_000.0)
    portfolio = PortfolioState()

    sizing = taker_sizing_boundaries(asks, q_effective=0.1, fee_config=fee_config, cfg=cfg, portfolio=portfolio, side=Side.UP)

    assert sizing.p_max is None
    assert sizing.quantities == ()


def test_taker_sizing_respects_position_and_spend_caps():
    asks = (BookLevel(0.50, 500.0),)
    fee_config = FeeConfig()
    cfg = OneStepConfig(g_min=-1_000_000.0, position_limit=3.0)
    portfolio = PortfolioState()

    sizing = taker_sizing_boundaries(asks, q_effective=0.9, fee_config=fee_config, cfg=cfg, portfolio=portfolio, side=Side.UP)
    assert max(sizing.quantities) <= 3.0 + 1e-6

    cfg_spend = OneStepConfig(g_min=-1_000_000.0, spend_cap=5.0)
    sizing_spend = taker_sizing_boundaries(asks, q_effective=0.9, fee_config=fee_config, cfg=cfg_spend, portfolio=portfolio, side=Side.UP)
    c_1 = 0.50 + fee_config.taker_fee(1.0, 0.50)
    assert max(sizing_spend.quantities) <= (5.0 / c_1) + 1e-6


def test_taker_sizing_uses_exact_side_aware_g_when_buying_the_underrepresented_side():
    """Phase 12B Tranche 1.1 item 5 regression: the flat
    max_directional_spend(g_current, g_min) budget used to cap risk-budget
    sizing at G_current-g_min DOLLARS, which is only exact when the
    purchased side is already the non-minimum side. Buying the currently
    *underrepresented* side also raises min(U,D) itself, so the flat
    budget badly under-sizes what's actually risk-feasible.

    U=0, D=100, C=50, g_min=-100, c_i=0.5175 (=0.50 + taker_fee): the old
    flat budget = (min(0,100)-50)-(-100) = 50 dollars => caps quantity at
    50/0.5175 ~= 96.6 shares - it would reject a 100-share purchase even
    though G'(100) = min(100,100)-(50+51.75) = -1.75 >= -100 (the exact
    example the reviewer's prompt calls out explicitly). The TRUE
    boundary is far higher still, since G keeps rising until x=D-U=100
    (min(U,D) rising alongside the purchase) and only *then* starts
    falling: G(x)=100-50-0.5175x for x>100, crossing g_min=-100 at
    x=150/0.5175 ~= 289.86 shares - almost 3x the old (wrong) cap."""
    asks = (BookLevel(0.50, 500.0),)  # depth well past the true risk boundary, so risk is the binding constraint
    fee_config = FeeConfig()
    cfg = OneStepConfig(g_min=-100.0)
    portfolio = PortfolioState(U=0.0, D=100.0, C=50.0)
    old_wrong_cap = 50.0 / 0.5175  # ~96.6 - what the flat max_directional_spend budget would have allowed
    exact_boundary = 150.0 / 0.5175  # ~289.86 - the true G_U(x)>=g_min crossing point

    sizing = taker_sizing_boundaries(asks, q_effective=0.9, fee_config=fee_config, cfg=cfg, portfolio=portfolio, side=Side.UP)

    assert any(q >= 100.0 for q in sizing.quantities), sizing.quantities  # the prompt's explicit claim: 100 shares must be feasible
    assert any(q > old_wrong_cap for q in sizing.quantities), sizing.quantities  # strictly more than the old (wrong) flat-budget cap
    assert any(math.isclose(q, exact_boundary, rel_tol=1e-6) for q in sizing.quantities), sizing.quantities
    assert max(sizing.quantities) <= exact_boundary + 1e-6  # never offers more than the exact risk-feasible boundary


# --------------------------------------------------------------------------
# Phase 12B Tranche 1.2 item 3: taker_max_execution_price - each
# candidate's own hard, risk-safe worst-price limit, distinct from the
# shared depth/marginal-edge p_max.
# --------------------------------------------------------------------------


def test_taker_max_execution_price_is_tighter_than_g_min_breaching_price():
    """The exact scenario named in the reviewer's prompt: p_submit=0.50,
    p_later=0.80, fee_rate=0.07. A quantity sized to be exactly risk-safe
    at 0.50 must get a derived hard limit below 0.80, even though 0.80
    individually still clears min_marginal_edge."""
    fee_config = FeeConfig(crypto_fee_rate=0.07)
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-100.0, min_marginal_edge=0.0)
    c_submit = 0.50 + fee_config.taker_fee(1.0, 0.50)
    x = (portfolio.G - cfg.g_min) / c_submit

    hard_limit = taker_max_execution_price(portfolio, Side.UP, x, q_effective=0.85, fee_config=fee_config, cfg=cfg, tick_size=0.01)
    assert hard_limit is not None
    assert hard_limit < 0.80
    assert hard_limit <= 0.51  # close to the submission price, not materially looser


def test_taker_max_execution_price_is_none_when_infeasible_at_any_price():
    portfolio = PortfolioState(U=0.0, D=0.0, C=0.0)
    cfg = OneStepConfig(g_min=0.0)  # any spend at all breaches this from a flat start
    result = taker_max_execution_price(portfolio, Side.UP, x=10.0, q_effective=0.9, fee_config=FEE, cfg=cfg, tick_size=0.01)
    assert result is None


def test_taker_max_execution_price_respects_spend_cap_too():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-1_000_000.0, spend_cap=5.0, min_marginal_edge=0.0)  # risk floor loose, spend_cap tight
    x = 100.0
    hard_limit = taker_max_execution_price(portfolio, Side.UP, x, q_effective=0.9, fee_config=FEE, cfg=cfg, tick_size=0.01)
    assert hard_limit is not None
    # spend_cap=5.0 over 100 shares caps the per-share all-in cost at 0.05
    assert hard_limit <= 0.06


def test_taker_max_execution_price_floors_to_the_tick_grid():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-1_000_000.0, min_marginal_edge=0.0)
    hard_limit = taker_max_execution_price(portfolio, Side.UP, x=10.0, q_effective=0.6789, fee_config=FEE, cfg=cfg, tick_size=0.01)
    assert hard_limit is not None
    scaled = hard_limit / 0.01
    assert math.isclose(scaled, round(scaled), abs_tol=1e-6)  # an exact multiple of the tick size


def test_taker_sizing_boundaries_max_execution_price_by_qty_never_exceeds_p_max():
    """End-to-end within taker_sizing_boundaries: every quantity's own
    hard limit must be at most the shared depth/marginal-edge p_max."""
    asks = (BookLevel(0.50, 500.0),)
    cfg = OneStepConfig(g_min=-50.0, min_marginal_edge=0.0)
    portfolio = PortfolioState()

    sizing = taker_sizing_boundaries(asks, q_effective=0.9, fee_config=FEE, cfg=cfg, portfolio=portfolio, side=Side.UP, tick_size=0.01)
    assert sizing.p_max is not None
    for qty, price in sizing.max_execution_price_by_qty.items():
        assert price <= sizing.p_max + 1e-9


def test_taker_sizing_boundaries_uses_tick_size_default_when_not_supplied():
    """Backward-compatible default (tick_size=0.01) so existing callers
    that don't pass it keep working unchanged."""
    asks = (BookLevel(0.50, 500.0),)
    cfg = OneStepConfig(g_min=-1_000_000.0)
    portfolio = PortfolioState()
    sizing = taker_sizing_boundaries(asks, q_effective=0.9, fee_config=FEE, cfg=cfg, portfolio=portfolio, side=Side.UP)
    assert sizing.quantities  # runs without error, produces candidates as before


def test_taker_sizing_boundaries_are_wired_into_controller_candidate_generation():
    """End-to-end: OneStepController's taker candidates must reflect the
    new boundary-aware sizing, not the old raw depth-level-only quantities,
    and must never exceed the derived worst-price boundary."""
    cfg = OneStepConfig(g_min=-1_000_000.0)
    controller = OneStepController(cfg, ExecutionConfig(), FeeConfig())
    portfolio = PortfolioState()
    book_up = BookSnapshot(Side.UP, bids=(BookLevel(0.48, 500.0),), asks=(BookLevel(0.50, 500.0), BookLevel(0.98, 500.0)), ts=0.0, recv_ts=0.0)
    book_down = BookSnapshot(Side.DOWN, bids=(), asks=(), ts=0.0, recv_ts=0.0)
    permitted = frozenset({SeedAction.TAKER_UP, SeedAction.WAIT})

    decision = controller.decide("r0", 10.0, portfolio, q=0.9, permitted_actions=permitted, book_up=book_up, book_down=book_down, tick_size=0.01, is_fresh=True)

    taker_candidates = [c for c in decision.candidates if c.mode is OrderMode.FAK]
    assert taker_candidates  # the good-edge level produced at least one candidate
    assert all(c.price <= 0.50 + 1e-9 for c in taker_candidates)  # the 0.98 level was never used as an execution price


# --------------------------------------------------------------------------
# Phase 12B Tranche 1, item 11: favored-side semantics (no p_min activation)
# --------------------------------------------------------------------------


def test_favored_side_reflects_q_not_payoff_geometry_at_flat_portfolio():
    """At a flat portfolio (Pi_U == Pi_D == 0), the old `_favored_side`
    always returned UP (an arbitrary `>=` tie-break on payoff geometry).
    With q clearly favoring DOWN, the p_min check (once activated) must
    use DOWN as the favored side, not UP."""
    from xamarinbot.optimizer.candidates import _favored_side

    assert _favored_side(q=0.2) is Side.DOWN
    assert _favored_side(q=0.8) is Side.UP


def test_p_min_stays_inactive_by_default_after_favored_side_fix():
    """Fixing _favored_side must not, by itself, activate p_min anywhere -
    OneStepConfig.p_min stays None unless a caller explicitly sets it
    (Phase 12B audit item I: do not introduce an artificial profit-floor
    constraint while fixing the favored-side bug)."""
    cfg = OneStepConfig(g_min=-100.0)
    assert cfg.p_min is None
    portfolio = PortfolioState()
    candidate = evaluate_taker_candidate(
        "taker_up_1", Side.UP, OrderPurpose.ALPHA, 50.0, limit_price=1.0, asks=ASKS,
        portfolio=portfolio, q=0.5, fee_config=FEE, cfg=cfg,
    )
    assert "p_min" not in candidate.violated_constraints

# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------


def test_taker_quantities_are_cumulative_per_level():
    assert taker_quantities(ASKS, max_levels=3) == [100.0, 250.0, 450.0]
    assert taker_quantities(ASKS, max_levels=1) == [100.0]
    assert taker_quantities((), max_levels=3) == []


def test_maker_price_grid_offsets_and_clips_below_best_ask():
    grid = maker_price_grid(best_bid=0.48, best_ask=0.50, tick_size=0.01, offsets_ticks=(0, 1, 2, 3))
    # offsets 0,1,2 -> 0.48, 0.49, 0.50 (but 0.50 == best_ask, excluded); offset 3 would cross too
    prices = [p for p, _ in grid]
    assert 0.50 not in prices
    assert (0.48, 0) in grid
    assert (0.49, 1) in grid


def test_maker_price_grid_empty_when_bid_already_at_ask():
    assert maker_price_grid(best_bid=0.50, best_ask=0.50, tick_size=0.01, offsets_ticks=(0, 1)) == []


# --------------------------------------------------------------------------
# Taker candidate EV matches Strategy doc SS13 DeltaEV formulas
# --------------------------------------------------------------------------


def test_taker_up_ev_matches_ss13_delta_ev_formula():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-1_000_000.0)
    candidate = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.ALPHA, 100.0, 0.99, ASKS, portfolio, q=0.6, fee_config=FEE, cfg=cfg)
    # DeltaEV_U(x) = q*x - K_U(x); K_U(x) = cost + fee
    expected_cost = 100.0 * 0.50
    expected_fee = FEE.taker_fee(100.0, 0.50)
    expected_ev = 0.6 * 100.0 - (expected_cost + expected_fee)
    assert math.isclose(candidate.delta_ev, expected_ev, rel_tol=1e-9)


def test_taker_down_ev_matches_ss13_delta_ev_formula():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-1_000_000.0)
    candidate = evaluate_taker_candidate("t1", Side.DOWN, OrderPurpose.ALPHA, 100.0, 0.99, ASKS, portfolio, q=0.6, fee_config=FEE, cfg=cfg)
    # DeltaEV_D(x) = (1-q)*x - K_D(x)
    expected_cost = 100.0 * 0.50
    expected_fee = FEE.taker_fee(100.0, 0.50)
    expected_ev = 0.4 * 100.0 - (expected_cost + expected_fee)
    assert math.isclose(candidate.delta_ev, expected_ev, rel_tol=1e-9)


def test_taker_candidate_rejected_when_g_min_breached():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=0.0)  # any spend at all breaches this from a zero start
    candidate = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.ALPHA, 100.0, 0.99, ASKS, portfolio, q=0.6, fee_config=FEE, cfg=cfg)
    assert not candidate.is_valid
    assert "g_min" in candidate.violated_constraints


def test_taker_candidate_rejected_below_edge_min():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-1_000_000.0, edge_min=1_000_000.0)  # impossibly high bar
    candidate = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.ALPHA, 100.0, 0.99, ASKS, portfolio, q=0.6, fee_config=FEE, cfg=cfg)
    assert "edge_min" in candidate.violated_constraints


def test_taker_candidate_partial_fak_reflected_in_expected_fill():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-1_000_000.0)
    total_depth = sum(l.size for l in ASKS)
    candidate = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.ALPHA, total_depth + 1000.0, 0.99, ASKS, portfolio, q=0.6, fee_config=FEE, cfg=cfg)
    assert math.isclose(candidate.expected_fill, total_depth)


# --------------------------------------------------------------------------
# Maker candidate EV
# --------------------------------------------------------------------------


def test_maker_candidate_ev_is_probability_weighted():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-1_000_000.0)
    exec_cfg = ExecutionConfig()
    candidate = evaluate_maker_candidate(
        "m1", Side.UP, OrderPurpose.ALPHA, price=0.45, qty=20.0, distance_to_touch_ticks=0.0,
        queue_ahead_shares=0.0, horizon_s=10.0, portfolio=portfolio, q=0.6, exec_cfg=exec_cfg, cfg=cfg,
    )
    assert 0.0 < candidate.expected_fill < 20.0  # rho in (0,1) at the touch with a real horizon
    assert candidate.mode is OrderMode.POST_ONLY


def test_maker_candidate_constraint_uses_if_filled_portfolio():
    """A maker candidate whose full fill would breach G_min must be
    rejected even though the fill itself is only probabilistic - Strategy
    doc SS16's "risk breach: projected fill would push G below G_min ->
    Cancel/shrink" applied pre-emptively (Roadmap Phase 8: "Reject hard-
    constraint violations")."""
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=0.0)
    exec_cfg = ExecutionConfig()
    candidate = evaluate_maker_candidate(
        "m1", Side.UP, OrderPurpose.ALPHA, price=0.45, qty=1000.0, distance_to_touch_ticks=0.0,
        queue_ahead_shares=0.0, horizon_s=10.0, portfolio=portfolio, q=0.6, exec_cfg=exec_cfg, cfg=cfg,
    )
    assert "g_min" in candidate.violated_constraints


# --------------------------------------------------------------------------
# WAIT candidate
# --------------------------------------------------------------------------


def test_wait_candidate_is_always_valid_and_neutral():
    portfolio = PortfolioState(U=5.0, D=3.0, C=4.0)
    candidate = wait_candidate("wait", portfolio)
    assert candidate.is_valid
    assert candidate.delta_ev == 0.0
    assert candidate.g_after == portfolio.G
    assert candidate.mode is OrderMode.WAIT


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


def _up_book() -> BookSnapshot:
    return BookSnapshot(side=Side.UP, bids=(BookLevel(0.48, 500.0),), asks=ASKS, ts=0.0, recv_ts=0.0)


def _down_book() -> BookSnapshot:
    return BookSnapshot(side=Side.DOWN, bids=(BookLevel(0.46, 500.0),), asks=(BookLevel(0.48, 500.0),), ts=0.0, recv_ts=0.0)


def test_controller_candidate_table_includes_every_permitted_family():
    cfg = OneStepConfig(g_min=-1_000_000.0)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    permitted = frozenset({SeedAction.TAKER_UP, SeedAction.MAKER_UP, SeedAction.WAIT})
    decision = controller.decide("r0", 10.0, PortfolioState(), q=0.6, permitted_actions=permitted, book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True)

    modes = {c.mode for c in decision.candidates}
    assert OrderMode.WAIT in modes
    assert OrderMode.FAK in modes
    assert OrderMode.POST_ONLY in modes
    assert len(decision.candidates) > 1  # more than just WAIT


def test_controller_generates_no_extra_candidates_for_wait_only_regime():
    cfg = OneStepConfig(g_min=-1_000_000.0)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    decision = controller.decide("r0", 10.0, PortfolioState(), q=0.6, permitted_actions=frozenset({SeedAction.WAIT}), book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True)
    assert len(decision.candidates) == 1
    assert decision.chosen.mode is OrderMode.WAIT


def test_controller_falls_back_to_wait_when_all_directional_candidates_invalid():
    cfg = OneStepConfig(g_min=1_000_000.0)  # impossible to satisfy
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    permitted = frozenset({SeedAction.TAKER_UP, SeedAction.MAKER_UP})
    decision = controller.decide("r0", 10.0, PortfolioState(), q=0.6, permitted_actions=permitted, book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True)
    assert decision.chosen.mode is OrderMode.WAIT
    assert all(not c.is_valid for c in decision.candidates if c.mode is not OrderMode.WAIT)


def test_controller_selects_highest_ev_valid_candidate():
    cfg = OneStepConfig(g_min=-1_000_000.0)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    permitted = frozenset({SeedAction.TAKER_UP, SeedAction.MAKER_UP, SeedAction.WAIT})
    decision = controller.decide("r0", 10.0, PortfolioState(), q=0.9, permitted_actions=permitted, book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True)
    valid = [c for c in decision.candidates if c.is_valid]
    assert decision.chosen.delta_ev == max(c.delta_ev for c in valid)


def test_controller_stale_data_forces_wait_only():
    cfg = OneStepConfig(g_min=-1_000_000.0)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    permitted = frozenset({SeedAction.TAKER_UP})
    decision = controller.decide("r0", 10.0, PortfolioState(), q=0.9, permitted_actions=permitted, book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=False)
    assert decision.chosen.mode is OrderMode.WAIT
    assert decision.skip_reason == "stale_data"
    assert len(decision.candidates) == 1


# --------------------------------------------------------------------------
# Stress test: G_min is never violated across many random states
# (Roadmap Phase 8 verification, both explicitly named items)
# --------------------------------------------------------------------------


@given(
    u=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    d=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    c=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    q=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    g_min=st.floats(min_value=-500.0, max_value=50.0, allow_nan=False),
    best_bid=st.floats(min_value=0.05, max_value=0.90, allow_nan=False),
    spread=st.floats(min_value=0.01, max_value=0.10, allow_nan=False),
)
@settings(max_examples=200)
def test_optimizer_never_violates_g_min_across_random_states(u, d, c, q, g_min, best_bid, spread):
    portfolio = PortfolioState(U=u, D=d, C=c)
    cfg = OneStepConfig(g_min=g_min)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)

    ask = min(0.99, best_bid + spread)
    book_up = BookSnapshot(Side.UP, bids=(BookLevel(best_bid, 300.0),), asks=(BookLevel(ask, 300.0), BookLevel(min(0.99, ask + 0.02), 400.0)), ts=0.0, recv_ts=0.0)
    down_ask = min(0.99, 1.0 - best_bid + spread)
    book_down = BookSnapshot(Side.DOWN, bids=(BookLevel(max(0.01, 1.0 - ask), 300.0),), asks=(BookLevel(down_ask, 300.0),), ts=0.0, recv_ts=0.0)

    permitted = frozenset({SeedAction.TAKER_UP, SeedAction.TAKER_DOWN, SeedAction.MAKER_UP, SeedAction.MAKER_DOWN, SeedAction.WAIT})
    decision = controller.decide("r0", 10.0, portfolio, q, permitted, book_up, book_down, tick_size=0.01, is_fresh=True)

    # If the starting portfolio is already below g_min (a randomly-drawn
    # portfolio can start there - e.g. g_min > 0 is infeasible from any
    # near-zero starting state, since WAIT itself never changes G), no
    # action can "fix" that; WAIT is then correctly the best available
    # choice and must not be flagged as if the *optimizer* introduced the
    # violation. The real invariant: the optimizer never makes G worse
    # than the better of (the floor) or (what it already started with).
    effective_floor = min(g_min, portfolio.G)
    assert decision.chosen.g_after >= effective_floor - 1e-9, (
        f"chosen candidate {decision.chosen.action_id} has g_after={decision.chosen.g_after} "
        f"< effective_floor={effective_floor} (g_min={g_min}, starting G={portfolio.G})"
    )


# --------------------------------------------------------------------------
# Phase 12B Tranche 2B: BUFFER_BUILD candidates - generated independent of
# a g_min breach, may have negative delta_ev (edge_min exempt), always
# have delta_g > 0 by construction. Contrasted with a plain ALPHA
# candidate, which can have the opposite combination (positive EV,
# negative/breaching G).
# --------------------------------------------------------------------------


def test_delta_ev_directional_matches_the_general_delta_ev_formula_for_one_sided_fills():
    """DeltaEV_U(x)=q*x-K_U(x) must equal the general q*deltaU+(1-q)*deltaD-deltaC
    formula for a pure one-sided fill (delta_D=0) - the identity
    evaluate_taker_candidate silently relies on."""
    assert math.isclose(delta_ev_directional(Side.UP, x=10.0, k_x=4.0, q=0.6), 0.6 * 10.0 - 4.0)
    assert math.isclose(delta_ev_directional(Side.DOWN, x=10.0, k_x=4.0, q=0.6), 0.4 * 10.0 - 4.0)


def test_buffer_build_generates_candidate_with_positive_delta_g_even_when_delta_ev_is_negative():
    """The reviewer's exact case: G comfortably above g_min, a cheap
    opposite-side buffer opportunity available, q low enough that buying
    the underrepresented side has negative standalone EV - BUFFER_BUILD
    must still generate and admit the candidate (delta_g > 0 is its own
    admission bar, not delta_ev > 0)."""
    portfolio = PortfolioState(U=0.0, D=100.0, C=0.0)  # D overrepresented - buying UP raises min(U,D)
    cfg = OneStepConfig(g_min=-1_000_000.0)  # risk floor not the binding constraint here
    low_q = 0.1  # strongly favors DOWN - buying UP is a bad bet standalone
    asks_up = (BookLevel(0.50, 500.0),)

    candidates = generate_buffer_build_candidates("buffer_build", portfolio, asks_up, (), low_q, FEE, cfg)

    assert candidates, "expected at least one BUFFER_BUILD candidate"
    first = candidates[0]
    assert first.purpose is OrderPurpose.BUFFER_BUILD
    assert first.g_after > portfolio.G  # genuine settlement-geometry improvement
    assert first.delta_ev < 0.0  # negative standalone EV, same SS17 allowance as HEDGE
    assert first.is_valid  # not rejected merely for having negative EV - edge_min doesn't apply
    assert "edge_min" not in first.violated_constraints


def test_buffer_build_generates_nothing_when_already_the_overrepresented_side():
    """Buying the side that's already >= the other side can never raise
    min(U,D) (delta_g_directional's own proof) - no candidate should be
    offered in that direction."""
    portfolio = PortfolioState(U=100.0, D=0.0, C=0.0)  # U already overrepresented
    cfg = OneStepConfig(g_min=-1_000_000.0)
    # side selection buys the UNDERrepresented side (D here) - give it an
    # empty down book so there's nothing to walk, proving no candidate
    # materializes rather than incorrectly falling back to buying UP.
    candidates = generate_buffer_build_candidates("buffer_build", portfolio, (), (), 0.9, FEE, cfg)
    assert candidates == []


def test_buffer_build_is_exempt_from_edge_min():
    portfolio = PortfolioState(U=0.0, D=100.0, C=0.0)
    cfg = OneStepConfig(g_min=-1_000_000.0, edge_min=1_000_000.0)  # impossibly high bar
    candidates = generate_buffer_build_candidates("buffer_build", portfolio, (BookLevel(0.50, 500.0),), (), 0.1, FEE, cfg)
    assert candidates
    assert "edge_min" not in candidates[0].violated_constraints


def test_alpha_candidate_can_have_positive_ev_and_negative_g_independently():
    """Contrast case for the BUFFER_BUILD test above: an ordinary ALPHA
    taker candidate with clearly positive EV (q=0.9 against a 0.5 ask)
    but a tight g_min floor it breaches - proving delta_ev and g_after
    are independent axes for ALPHA too, and that positive EV alone does
    not exempt a candidate from the hard g_min gate."""
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=0.0)  # any spend at all breaches this from a flat start
    candidate = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.ALPHA, 100.0, 0.99, ASKS, portfolio, q=0.9, fee_config=FEE, cfg=cfg)
    assert candidate.delta_ev > 0.0  # clearly positive expected value
    assert candidate.g_after < 0.0  # and clearly below the g_min=0.0 floor
    assert not candidate.is_valid
    assert "g_min" in candidate.violated_constraints


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.1 item 7: BUFFER_BUILD generates a bounded MULTI-
# quantity candidate set, not a single "buy the peak amount" candidate.
# --------------------------------------------------------------------------


def test_buffer_build_generates_multiple_distinct_quantity_choices():
    portfolio = PortfolioState(U=0.0, D=200.0, C=0.0)
    cfg = OneStepConfig(g_min=-1_000_000.0, taker_min_size=1.0, taker_qty_step=10.0, taker_qty_grid_points=5)
    asks_up = (BookLevel(0.40, 500.0),)
    candidates = generate_buffer_build_candidates("buffer_build", portfolio, asks_up, (), q=0.5, fee_config=FEE, cfg=cfg)
    quantities = {round(c.qty, 3) for c in candidates}
    assert len(quantities) > 1, "expected multiple distinct BUFFER_BUILD quantity choices, not a single peak candidate"
    assert all(c.expected_delta_g > 0 for c in candidates), "every generated candidate must individually clear ΔG>0"


def test_buffer_build_candidate_quantities_are_all_individually_delta_g_positive():
    """"Keep only ΔG>0" (item 7) must hold per-candidate, not just for the
    single largest one - verified directly via delta_g_directional against
    each candidate's own actual qty/cost."""
    portfolio = PortfolioState(U=10.0, D=150.0, C=0.0)
    cfg = OneStepConfig(g_min=-1_000_000.0, taker_qty_step=5.0)
    asks_up = (BookLevel(0.40, 100.0), BookLevel(0.55, 100.0))
    candidates = generate_buffer_build_candidates("buffer_build", portfolio, asks_up, (), q=0.5, fee_config=FEE, cfg=cfg)
    assert candidates
    for c in candidates:
        k_x = c.expected_fill * c.price  # taker: filled at its own avg price, fee already inside delta_ev's own accounting
        # Recompute ΔG directly from the walk this candidate actually
        # represents (fee-inclusive cost via price*shares undercounts fee
        # slightly, so use a loose tolerance - this is a sign check, not
        # an exact-value check).
        delta_g = delta_g_directional(portfolio.U, portfolio.D, Side.UP, c.expected_fill, k_x)
        assert delta_g > -1e-6


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.1 item 4: purpose-aware execution price ceilings must
# guarantee their repair target survives even at the worst allowed price
# ("delayed repricing cannot destroy ΔG / cannot invalidate the repair
# target").
# --------------------------------------------------------------------------


def test_buffer_build_price_ceiling_guarantees_delta_g_nonnegative_at_the_worst_allowed_price():
    portfolio = PortfolioState(U=0.0, D=100.0, C=0.0)
    cfg = OneStepConfig(g_min=-1_000_000.0)
    x = 40.0
    max_price = purpose_aware_max_execution_price(portfolio, Side.UP, x, OrderPurpose.BUFFER_BUILD, FEE, cfg, tick_size=0.01)
    assert max_price is not None
    fee = FEE.taker_fee(x, max_price)
    k_x = x * max_price + fee
    delta_g = delta_g_directional(portfolio.U, portfolio.D, Side.UP, x, k_x)
    assert delta_g >= -1e-6, "even filling the full quantity at the worst allowed price, ΔG must stay non-negative"


def test_hedge_price_ceiling_guarantees_g_min_survives_at_the_worst_allowed_price():
    portfolio = PortfolioState(U=100.0, D=0.0, C=0.0)  # UP favored, DOWN needs the hedge
    cfg = OneStepConfig(g_min=-20.0)
    x = 30.0
    max_price = purpose_aware_max_execution_price(portfolio, Side.DOWN, x, OrderPurpose.HEDGE, FEE, cfg, tick_size=0.01)
    assert max_price is not None
    fee = FEE.taker_fee(x, max_price)
    k_x = x * max_price + fee
    g_after = directional_projected_g(portfolio.U, portfolio.D, portfolio.C, Side.DOWN, x, k_x)
    assert g_after >= cfg.g_min - 1e-6, "even filling the full hedge quantity at the worst allowed price, G_min must survive"


def test_hedge_candidate_never_uses_unconstrained_limit_price():
    portfolio = PortfolioState(U=190.0, D=0.0, C=100.0)  # Pi_U=90, Pi_D=-100 - a real hedge is needed
    cfg = OneStepConfig(g_min=-10.0)
    hedge = generate_hedge_candidate("hedge", portfolio, (), (BookLevel(0.40, 500.0),), q=0.5, fee_config=FEE, cfg=cfg)
    assert hedge is not None
    assert hedge.max_execution_price is not None
    assert hedge.max_execution_price < 1.0, "HEDGE must never submit at an effectively unconstrained limit_price=1.0"


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.1 item 5: breach-recovery semantics.
# --------------------------------------------------------------------------


def test_breach_recovery_alpha_is_prohibited_when_already_below_g_min():
    portfolio = PortfolioState(U=0.0, D=0.0, C=200.0)  # G = -200, deep in breach
    cfg = OneStepConfig(g_min=-50.0)
    candidate = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.ALPHA, 1.0, 0.50, ASKS, portfolio, q=0.9, fee_config=FEE, cfg=cfg)
    assert not candidate.is_valid
    assert "breach_recovery_alpha_prohibited" in candidate.violated_constraints


def test_breach_recovery_allows_partial_hedge_that_improves_g_without_fully_restoring_it():
    """The bot must not be forced into WAIT merely because g_min cannot be
    restored in a single fill - a partial repair that strictly improves G
    (even while remaining below g_min) must be admitted."""
    portfolio = PortfolioState(U=0.0, D=50.0, C=200.0)  # G = -200; D>U so buying UP raises min(U,D)
    cfg = OneStepConfig(g_min=-50.0)  # full restoration to -50 is not reachable by a small buy
    # a small UP buy improves G (from -200 towards less-negative) without
    # reaching -50 - must NOT be rejected for missing g_min while in breach.
    candidate = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.BUFFER_BUILD, 5.0, 0.50, ASKS, portfolio, q=0.5, fee_config=FEE, cfg=cfg)
    assert candidate.g_after > portfolio.G, "sanity: this candidate does improve G"
    assert candidate.g_after < cfg.g_min, "sanity: it still doesn't reach g_min in one fill"
    assert "g_min" not in candidate.violated_constraints
    assert "breach_recovery_no_improvement" not in candidate.violated_constraints


def test_breach_recovery_rejects_a_repair_candidate_that_makes_g_worse():
    portfolio = PortfolioState(U=0.0, D=0.0, C=200.0)  # G = -200
    cfg = OneStepConfig(g_min=-50.0)
    # DOWN buy while UP=D=0 doesn't move min(U,D) at all (still 0) but adds
    # cost - G strictly worsens.
    candidate = evaluate_taker_candidate("t1", Side.DOWN, OrderPurpose.HEDGE, 5.0, 0.50, ASKS, portfolio, q=0.5, fee_config=FEE, cfg=cfg)
    assert candidate.g_after <= portfolio.G
    assert "breach_recovery_no_improvement" in candidate.violated_constraints


def test_breach_recovery_hard_rule_unaffected_when_not_already_in_breach():
    """If G_before >= g_min, the ordinary hard rule (G_after >= g_min)
    still applies unchanged - breach-recovery relaxation must not leak
    into the normal case."""
    portfolio = PortfolioState()  # G = 0, safely above g_min
    cfg = OneStepConfig(g_min=-10.0)
    candidate = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.ALPHA, 1000.0, 0.99, ASKS, portfolio, q=0.9, fee_config=FEE, cfg=cfg)
    assert candidate.g_after < cfg.g_min  # a large enough buy still breaches it
    assert "g_min" in candidate.violated_constraints
    assert "breach_recovery_no_improvement" not in candidate.violated_constraints  # not the breach-recovery path at all


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.1 item 6: maker expected_delta_g is rho-weighted.
# --------------------------------------------------------------------------


def test_maker_expected_delta_g_is_rho_weighted_not_the_full_if_filled_delta():
    portfolio = PortfolioState(U=0.0, D=50.0, C=0.0)  # D>U so buying UP raises min(U,D)
    cfg = OneStepConfig(g_min=-1_000_000.0)
    c = evaluate_maker_candidate(
        "m1", Side.UP, OrderPurpose.ALPHA, price=0.40, qty=50.0,
        distance_to_touch_ticks=2.0, queue_ahead_shares=10.0, horizon_s=10.0,
        portfolio=portfolio, q=0.6, exec_cfg=ExecutionConfig(), cfg=cfg,
    )
    full_if_filled_delta_g = c.g_after - portfolio.G
    assert full_if_filled_delta_g > 0.0  # sanity: filling raises G here (buying the flat/underrepresented side)
    assert c.expected_delta_g < full_if_filled_delta_g, "expected_delta_g must be rho-weighted, strictly less than the full if-filled delta"
    assert c.expected_delta_g > 0.0


def test_taker_expected_delta_g_equals_the_full_deterministic_delta():
    """A taker fill is deterministic (no fill-probability draw) - its
    expected_delta_g must equal the full delta, unlike a maker's."""
    portfolio = PortfolioState(U=0.0, D=100.0, C=0.0)
    cfg = OneStepConfig(g_min=-1_000_000.0)
    c = evaluate_taker_candidate("t1", Side.UP, OrderPurpose.ALPHA, 20.0, 0.50, ASKS, portfolio, q=0.6, fee_config=FEE, cfg=cfg)
    assert math.isclose(c.expected_delta_g, c.g_after - portfolio.G)


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.1 item 3: a rejected top candidate must rerank to the
# next-best legal one, not silently vanish.
# --------------------------------------------------------------------------


class _RejectAboveQty:
    """Minimal stand-in for a RiskView (duck-typed - decide() only ever
    calls .admits()) that rejects any candidate above a fixed quantity,
    so the test can force a specific top candidate to lose aggregate
    admission deterministically."""

    def __init__(self, reject_above_qty: float):
        self.reject_above_qty = reject_above_qty

    def admits(self, candidate, g_min, spend_cap, position_limit=None, exact_subset_cap=10):
        return candidate.max_fill_qty <= self.reject_above_qty


def test_decide_reranks_to_next_best_candidate_when_top_choice_is_aggregate_unsafe():
    cfg = OneStepConfig(g_min=-1_000_000.0, taker_min_size=10.0, taker_qty_step=10.0, taker_qty_grid_points=5)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    portfolio = PortfolioState()
    book_up = _up_book()

    baseline = controller.decide("r0", 10.0, portfolio, q=0.9, permitted_actions=frozenset({SeedAction.TAKER_UP, SeedAction.WAIT}), book_up=book_up, book_down=_down_book(), tick_size=0.01, is_fresh=True)
    taker_candidates = sorted((c for c in baseline.candidates if c.action_id.startswith("taker_up_")), key=lambda c: c.qty)
    assert len(taker_candidates) >= 2, "need at least two distinct taker quantities to prove reranking"
    assert baseline.chosen.qty == max(c.qty for c in taker_candidates), "sanity: without a risk_view, the largest/highest-EV candidate wins"

    second_best_qty = sorted({c.qty for c in taker_candidates})[-2]
    stub_risk_view = _RejectAboveQty(reject_above_qty=second_best_qty)
    reranked = controller.decide("r0", 10.0, portfolio, q=0.9, permitted_actions=frozenset({SeedAction.TAKER_UP, SeedAction.WAIT}), book_up=book_up, book_down=_down_book(), tick_size=0.01, is_fresh=True, risk_view=stub_risk_view)

    assert reranked.chosen.mode is not OrderMode.WAIT, "must not fall through to WAIT when a smaller legal candidate exists"
    assert reranked.chosen.qty <= second_best_qty
    assert reranked.chosen.qty != baseline.chosen.qty, "the top (now aggregate-unsafe) candidate must not be chosen"


def test_buffer_build_controller_wiring_is_off_by_default_and_on_when_enabled():
    portfolio = PortfolioState(U=0.0, D=100.0, C=0.0)
    book_up = BookSnapshot(Side.UP, bids=(), asks=(BookLevel(0.50, 500.0),), ts=0.0, recv_ts=0.0)
    book_down = BookSnapshot(Side.DOWN, bids=(), asks=(), ts=0.0, recv_ts=0.0)

    cfg_off = OneStepConfig(g_min=-1_000_000.0)  # enable_buffer_build defaults False
    controller_off = OneStepController(cfg_off, ExecutionConfig(), FEE)
    decision_off = controller_off.decide("r0", 10.0, portfolio, q=0.1, permitted_actions=frozenset({SeedAction.WAIT}), book_up=book_up, book_down=book_down, tick_size=0.01, is_fresh=True)
    assert all(c.purpose is not OrderPurpose.BUFFER_BUILD for c in decision_off.candidates)

    cfg_on = OneStepConfig(g_min=-1_000_000.0, enable_buffer_build=True)
    controller_on = OneStepController(cfg_on, ExecutionConfig(), FEE)
    decision_on = controller_on.decide("r0", 10.0, portfolio, q=0.1, permitted_actions=frozenset({SeedAction.WAIT}), book_up=book_up, book_down=book_down, tick_size=0.01, is_fresh=True)
    assert any(c.purpose is OrderPurpose.BUFFER_BUILD for c in decision_on.candidates), (
        "BUFFER_BUILD must be generated even under a WAIT-only regime - a portfolio-level decision, not directional"
    )


# --------------------------------------------------------------------------
# Phase 12B Tranche 2C: soft regime control - candidates from a
# regime-disfavored family are still generated (penalized, not absent),
# while the existing hard-gate ablation mode is unaffected.
# --------------------------------------------------------------------------


def test_hard_regime_gate_still_produces_no_candidate_for_a_disallowed_family():
    """Compatibility check: cfg.soft_regime defaults False, so a family
    absent from permitted_actions must still produce literally zero
    candidates - the exact pre-Tranche-2C behavior, unchanged."""
    cfg = OneStepConfig(g_min=-1_000_000.0)  # soft_regime=False (default)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    decision = controller.decide(
        "r0", 10.0, PortfolioState(), q=0.9, permitted_actions=frozenset({SeedAction.WAIT}),
        book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True,
    )
    assert all(c.mode is OrderMode.WAIT for c in decision.candidates)


def test_soft_regime_generates_disfavored_candidates_with_a_selection_penalty():
    """With soft_regime=True, a family the regime does NOT permit must
    still be generated (not silently absent), tagged with
    regime_prior_penalty as its selection_penalty - proving the candidate
    is now visible to the optimizer's argmax, even if outweighed."""
    cfg = OneStepConfig(g_min=-1_000_000.0, soft_regime=True, regime_prior_penalty=5.0)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    decision = controller.decide(
        "r0", 10.0, PortfolioState(), q=0.9, permitted_actions=frozenset({SeedAction.WAIT}),
        book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True,
    )
    taker_up_candidates = [c for c in decision.candidates if c.action_id.startswith("taker_up_")]
    assert taker_up_candidates, "TAKER_UP must still be generated under soft_regime even though WAIT-only was permitted"
    assert all(math.isclose(c.selection_penalty, 5.0) for c in taker_up_candidates)


def test_soft_regime_can_still_select_a_strong_enough_disfavored_candidate():
    """The whole point of soft mode: a regime-disfavored candidate can
    still win selection if its edge clears the penalty - proving the
    penalty is subtracted from the *selection score*, not an absolute veto."""
    cfg = OneStepConfig(g_min=-1_000_000.0, soft_regime=True, regime_prior_penalty=0.01)  # tiny penalty
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    decision = controller.decide(
        "r0", 10.0, PortfolioState(), q=0.9, permitted_actions=frozenset({SeedAction.WAIT}),
        book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True,
    )
    assert decision.chosen.mode is not OrderMode.WAIT, "a strong-edge candidate must be able to beat WAIT despite a small regime penalty"


def test_permitted_family_never_gets_a_penalty_under_soft_regime():
    cfg = OneStepConfig(g_min=-1_000_000.0, soft_regime=True, regime_prior_penalty=5.0)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    permitted = frozenset({SeedAction.TAKER_UP, SeedAction.WAIT})
    decision = controller.decide("r0", 10.0, PortfolioState(), q=0.9, permitted_actions=permitted, book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True)
    taker_up_candidates = [c for c in decision.candidates if c.action_id.startswith("taker_up_")]
    assert taker_up_candidates
    assert all(c.selection_penalty == 0.0 for c in taker_up_candidates)


# --------------------------------------------------------------------------
# Phase 12B Tranche 2D: dynamic maker price/quantity/TTL - grids derived
# from (q, tau, sigma, G, R) rather than OneStepConfig's fixed constants,
# gated behind cfg.dynamic_maker (default False, preserving Phases 8-12B's
# exact fixed-constant behavior).
# --------------------------------------------------------------------------


def test_dynamic_maker_candidates_shrinks_typical_quantity_and_ttl_as_volatility_rises():
    """The generated set mixes exchange-minimum, "typical size", and
    max-feasible quantities per price - with g_min set so loose that
    max-feasible dwarfs everything, the vol-damped "typical size" point
    (maker_quantity/(1+sigma)) is the one exercising this behavior, so
    check for its expected value directly rather than the set's max
    (which is dominated by the unrelated risk-boundary quantity)."""
    cfg = OneStepConfig(g_min=-1_000_000.0, maker_quantity=20.0, maker_horizon_s=10.0)
    portfolio = PortfolioState()
    calm = dynamic_maker_candidates(q=0.6, side=Side.UP, tau=30.0, sigma=0.0, portfolio=portfolio, best_bid=0.48, best_ask=0.50, tick_size=0.01, fee_config=FEE, cfg=cfg)
    stormy = dynamic_maker_candidates(q=0.6, side=Side.UP, tau=30.0, sigma=3.0, portfolio=portfolio, best_bid=0.48, best_ask=0.50, tick_size=0.01, fee_config=FEE, cfg=cfg)
    assert calm and stormy
    assert any(math.isclose(c.qty, 20.0) for c in calm), "sigma=0 typical size must be the unamped maker_quantity"
    assert any(math.isclose(c.qty, 5.0) for c in stormy), "sigma=3 typical size must be maker_quantity/(1+sigma)=5.0"
    assert all(c.ttl_s < 10.0 for c in stormy), "higher sigma must shorten TTL"


def test_dynamic_maker_candidates_widens_price_offset_grid_with_volatility():
    cfg = OneStepConfig(g_min=-1_000_000.0)
    portfolio = PortfolioState()
    calm = dynamic_maker_candidates(q=0.6, side=Side.UP, tau=30.0, sigma=0.0, portfolio=portfolio, best_bid=0.10, best_ask=0.90, tick_size=0.01, fee_config=FEE, cfg=cfg)
    stormy = dynamic_maker_candidates(q=0.6, side=Side.UP, tau=30.0, sigma=3.0, portfolio=portfolio, best_bid=0.10, best_ask=0.90, tick_size=0.01, fee_config=FEE, cfg=cfg)
    assert len({c.offset_ticks for c in stormy}) > len({c.offset_ticks for c in calm}), "higher sigma must widen (not narrow) the price-offset grid"


def test_dynamic_maker_candidates_quantity_respects_exact_price_aware_risk_budget():
    """Phase 12B Tranche 2.1 item 8 regression: quantity must be derived
    from the exact G_min-feasible ceiling AT EACH PRICE (dollars/price =
    shares), not a raw dollar budget compared directly against a share
    count (the dimensional bug this item fixes) - verified against the
    actual if-filled G, not the implementation's own internals."""
    cfg = OneStepConfig(g_min=-5.0, maker_quantity=1_000.0)
    portfolio = PortfolioState(U=0.0, D=0.0, C=0.0)  # G=0
    specs = dynamic_maker_candidates(q=0.6, side=Side.UP, tau=30.0, sigma=0.0, portfolio=portfolio, best_bid=0.48, best_ask=0.50, tick_size=0.01, fee_config=FEE, cfg=cfg)
    assert specs
    for spec in specs:
        fee = FEE.fee_for(LiquidityRole.MAKER, spec.qty, spec.price)
        k_x = spec.qty * spec.price + fee
        g_after = directional_projected_g(portfolio.U, portfolio.D, portfolio.C, Side.UP, spec.qty, k_x)
        assert g_after >= cfg.g_min - 1e-6, "every generated quantity must be individually G_min-feasible at its own price"


def test_dynamic_maker_candidates_ttl_never_exceeds_tau():
    cfg = OneStepConfig(g_min=-1_000_000.0, maker_horizon_s=1_000.0)
    portfolio = PortfolioState()
    specs = dynamic_maker_candidates(q=0.6, side=Side.UP, tau=2.0, sigma=0.0, portfolio=portfolio, best_bid=0.48, best_ask=0.50, tick_size=0.01, fee_config=FEE, cfg=cfg)
    assert specs
    assert all(c.ttl_s <= 2.0 for c in specs), "TTL must not outlive the round (tau) even when maker_horizon_s is large"


def test_dynamic_maker_candidates_generates_nothing_when_remaining_time_below_min_ttl():
    """Phase 12B Tranche 2.1 item 8: when `tau` is shorter than
    `cfg.min_maker_ttl_s`, no maker candidate is generated at all - the
    prior draft silently clamped TTL back up past `tau` instead."""
    cfg = OneStepConfig(g_min=-1_000_000.0, min_maker_ttl_s=5.0)
    portfolio = PortfolioState()
    specs = dynamic_maker_candidates(q=0.6, side=Side.UP, tau=1.0, sigma=0.0, portfolio=portfolio, best_bid=0.48, best_ask=0.50, tick_size=0.01, fee_config=FEE, cfg=cfg)
    assert specs == []


def test_dynamic_maker_controller_wiring_is_off_by_default_and_matches_fixed_constants():
    """cfg.dynamic_maker defaults False - even when a caller passes
    nonzero tau/sigma, maker candidates must use the exact fixed
    maker_quantity/maker_horizon_s/maker_price_offsets_ticks constants,
    byte-identical to a call that omits tau/sigma entirely."""
    cfg = OneStepConfig(g_min=-1_000_000.0, maker_quantity=20.0, maker_horizon_s=10.0)  # dynamic_maker=False
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    permitted = frozenset({SeedAction.MAKER_UP, SeedAction.WAIT})

    decision_no_vol_args = controller.decide("r0", 10.0, PortfolioState(), q=0.6, permitted_actions=permitted, book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True)
    decision_with_vol_args = controller.decide("r0", 10.0, PortfolioState(), q=0.6, permitted_actions=permitted, book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True, tau=30.0, sigma=5.0)

    maker_up_no_vol = [c for c in decision_no_vol_args.candidates if c.action_id.startswith("maker_up_")]
    maker_up_with_vol = [c for c in decision_with_vol_args.candidates if c.action_id.startswith("maker_up_")]
    assert maker_up_no_vol and maker_up_with_vol
    assert all(math.isclose(c.qty, 20.0) for c in maker_up_no_vol)
    assert all(math.isclose(c.qty, 20.0) for c in maker_up_with_vol), "sigma=5.0 must be ignored entirely while dynamic_maker=False"
    assert all(math.isclose(c.ttl_s, 10.0) for c in maker_up_with_vol)


def test_dynamic_maker_controller_wiring_uses_dynamic_sizing_when_enabled():
    cfg = OneStepConfig(g_min=-1_000_000.0, maker_quantity=20.0, maker_horizon_s=10.0, dynamic_maker=True)
    controller = OneStepController(cfg, ExecutionConfig(), FEE)
    permitted = frozenset({SeedAction.MAKER_UP, SeedAction.WAIT})
    portfolio = PortfolioState()

    expected_specs = dynamic_maker_candidates(q=0.6, side=Side.UP, tau=30.0, sigma=3.0, portfolio=portfolio, best_bid=_up_book().best_bid.price, best_ask=_up_book().best_ask.price, tick_size=0.01, fee_config=FEE, cfg=cfg)
    decision = controller.decide("r0", 10.0, portfolio, q=0.6, permitted_actions=permitted, book_up=_up_book(), book_down=_down_book(), tick_size=0.01, is_fresh=True, tau=30.0, sigma=3.0)

    maker_up = [c for c in decision.candidates if c.action_id.startswith("maker_up_")]
    assert maker_up, "dynamic_maker=True must not suppress maker candidate generation"
    assert len(maker_up) == len(expected_specs)
    assert not any(math.isclose(c.qty, 20.0) for c in maker_up), "fixed maker_quantity must not leak through once dynamic_maker=True"
