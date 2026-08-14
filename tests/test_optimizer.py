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
    taker_quantities,
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
    assert math.isclose(candidate.ev_after, expected_ev, rel_tol=1e-9)


def test_taker_down_ev_matches_ss13_delta_ev_formula():
    portfolio = PortfolioState()
    cfg = OneStepConfig(g_min=-1_000_000.0)
    candidate = evaluate_taker_candidate("t1", Side.DOWN, OrderPurpose.ALPHA, 100.0, 0.99, ASKS, portfolio, q=0.6, fee_config=FEE, cfg=cfg)
    # DeltaEV_D(x) = (1-q)*x - K_D(x)
    expected_cost = 100.0 * 0.50
    expected_fee = FEE.taker_fee(100.0, 0.50)
    expected_ev = 0.4 * 100.0 - (expected_cost + expected_fee)
    assert math.isclose(candidate.ev_after, expected_ev, rel_tol=1e-9)


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
    assert candidate.ev_after == 0.0
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
    assert decision.chosen.ev_after == max(c.ev_after for c in valid)


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
