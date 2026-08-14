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
    evaluate_maker_candidate,
    evaluate_taker_candidate,
    maker_price_grid,
    taker_max_execution_price,
    taker_quantities,
    taker_sizing_boundaries,
    wait_candidate,
)
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import OrderMode
from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
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
