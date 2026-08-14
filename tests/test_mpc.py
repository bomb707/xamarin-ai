"""Phase 10 verification: "MPC action equals one-step action in degenerate
horizon." "Timeout fallback." "Scenario probability sanity checks." Plus
regression coverage for the book-depletion (double-counting) bug and the
churn-penalty consolidation behavior it enables.
"""
from __future__ import annotations

import math

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.feeds.base import BookLevel, BookSnapshot
from xamarinbot.mpc.config import MPCConfig
from xamarinbot.mpc.controller import MPCController, _deplete_book, _portfolio_after_candidate
from xamarinbot.mpc.scenario import TransitionModel, build_transition_model
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import CandidateAction, OrderMode
from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
from xamarinbot.regime.matrix import ActionPermissionMatrix, classify_seed_action
from xamarinbot.regime.types import Direction, GapRegime, RegimeState, RegimeTransition

FEE = FeeConfig()
REGIME_A = RegimeState(GapRegime.UPPER_MIDDLE, Direction.UP, Direction.UP)
REGIME_B = RegimeState(GapRegime.STRONG_POSITIVE, Direction.UP, Direction.UP)

BOOK_UP = BookSnapshot(Side.UP, bids=(BookLevel(0.48, 500.0),), asks=(BookLevel(0.50, 500.0), BookLevel(0.52, 750.0)), ts=0.0, recv_ts=0.0)
BOOK_DOWN = BookSnapshot(Side.DOWN, bids=(BookLevel(0.46, 500.0),), asks=(BookLevel(0.48, 500.0),), ts=0.0, recv_ts=0.0)


def _one_step(g_min=-1000.0, edge_min=0.0, churn_penalty=0.0) -> OneStepController:
    cfg = OneStepConfig(g_min=g_min, edge_min=edge_min, churn_penalty=churn_penalty)
    return OneStepController(cfg, ExecutionConfig(), FEE)


# --------------------------------------------------------------------------
# Transition model + scenario probability sanity checks
# --------------------------------------------------------------------------


def test_build_transition_model_from_hand_crafted_transitions():
    transitions = [
        RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0),
        RegimeTransition("r0", REGIME_A, REGIME_A, None, 2.0, 1.0),
        RegimeTransition("r0", REGIME_A, REGIME_A, None, 3.0, 1.0),
        RegimeTransition("r0", REGIME_A, REGIME_B, None, 4.0, 1.0),
    ]
    model = build_transition_model(transitions)
    row = model.probabilities[GapRegime.UPPER_MIDDLE]
    assert math.isclose(row[GapRegime.UPPER_MIDDLE], 0.75)
    assert math.isclose(row[GapRegime.STRONG_POSITIVE], 0.25)


def test_transition_model_ignores_first_observations():
    transitions = [RegimeTransition("r0", None, REGIME_A, None, 1.0, None)]
    model = build_transition_model(transitions)
    assert model.probabilities == {}


def test_scenario_probabilities_sum_to_one_and_are_bounded():
    """Roadmap Phase 10 verification: "Scenario probability sanity checks.\""""
    transitions = [
        RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0),
        RegimeTransition("r0", REGIME_A, REGIME_B, None, 2.0, 1.0),
        RegimeTransition("r0", REGIME_A, RegimeState(GapRegime.NEAR_CENTER_POSITIVE, Direction.UP, Direction.UP), None, 3.0, 1.0),
    ]
    model = build_transition_model(transitions)
    for gap_regime in GapRegime:
        for k in (1, 2, 3):
            next_states = model.next_states(gap_regime, top_k=k)
            total = sum(p for _, p in next_states)
            assert math.isclose(total, 1.0, rel_tol=1e-9), f"{gap_regime}, k={k} sums to {total}"
            for _, p in next_states:
                assert 0.0 <= p <= 1.0


def test_next_states_falls_back_to_persistence_when_unobserved():
    model = TransitionModel(probabilities={})
    assert model.next_states(GapRegime.LOWER_MIDDLE, top_k=2) == [(GapRegime.LOWER_MIDDLE, 1.0)]


def test_next_states_truncates_to_top_k_and_renormalizes():
    transitions = [
        RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0),  # 0.5
        RegimeTransition("r0", REGIME_A, REGIME_B, None, 1.0, 1.0),  # 0.3
        RegimeTransition("r0", REGIME_A, REGIME_B, None, 1.0, 1.0),
        RegimeTransition("r0", REGIME_A, REGIME_B, None, 1.0, 1.0),
        RegimeTransition("r0", REGIME_A, RegimeState(GapRegime.NEAR_CENTER_POSITIVE, Direction.UP, Direction.UP), None, 1.0, 1.0),  # smallest, dropped at top_k=2
        RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0),
    ]
    model = build_transition_model(transitions)
    top2 = model.next_states(GapRegime.UPPER_MIDDLE, top_k=2)
    states = {s for s, _ in top2}
    assert GapRegime.NEAR_CENTER_POSITIVE not in states  # smallest dropped
    assert math.isclose(sum(p for _, p in top2), 1.0)


# --------------------------------------------------------------------------
# Degenerate horizon (Roadmap Phase 10 verification, named explicitly)
# --------------------------------------------------------------------------


def test_degenerate_horizon_matches_one_step_exactly():
    one_step = _one_step()
    mpc = MPCController(MPCConfig(horizon_steps=1), TransitionModel(probabilities={}), one_step)
    portfolio = PortfolioState()

    mpc_decision = mpc.decide("r0", 10.0, portfolio, 0.65, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True)

    permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(REGIME_A))
    base_decision = one_step.decide("r0", 10.0, portfolio, 0.65, permitted, BOOK_UP, BOOK_DOWN, 0.01, True)

    assert mpc_decision.chosen.action_id == base_decision.chosen.action_id
    assert math.isclose(mpc_decision.chosen.ev_after, base_decision.chosen.ev_after)
    assert not mpc_decision.used_fallback


def test_degenerate_horizon_matches_across_several_states_and_portfolios():
    one_step = _one_step()
    mpc = MPCController(MPCConfig(horizon_steps=1), TransitionModel(probabilities={}), one_step)

    for portfolio in (PortfolioState(), PortfolioState(U=100.0, D=20.0, C=50.0)):
        for regime in (REGIME_A, REGIME_B, RegimeState(GapRegime.LOWER_MIDDLE, Direction.DOWN, Direction.DOWN)):
            mpc_decision = mpc.decide("r0", 10.0, portfolio, 0.55, regime, BOOK_UP, BOOK_DOWN, 0.01, True)
            permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(regime))
            base_decision = one_step.decide("r0", 10.0, portfolio, 0.55, permitted, BOOK_UP, BOOK_DOWN, 0.01, True)
            assert mpc_decision.chosen.action_id == base_decision.chosen.action_id


# --------------------------------------------------------------------------
# Timeout fallback (Roadmap Phase 10 verification, named explicitly)
# --------------------------------------------------------------------------


def test_timeout_fallback_returns_plain_one_step_decision():
    one_step = _one_step()
    transitions = [RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0)]
    model = build_transition_model(transitions)
    mpc = MPCController(MPCConfig(horizon_steps=3, time_budget_ms=0.0), model, one_step)
    portfolio = PortfolioState()

    mpc_decision = mpc.decide("r0", 10.0, portfolio, 0.65, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True)

    permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(REGIME_A))
    base_decision = one_step.decide("r0", 10.0, portfolio, 0.65, permitted, BOOK_UP, BOOK_DOWN, 0.01, True)

    assert mpc_decision.used_fallback
    assert mpc_decision.chosen.action_id == base_decision.chosen.action_id
    assert mpc_decision.sequence_values == {}


def test_generous_time_budget_does_not_fall_back():
    one_step = _one_step()
    model = build_transition_model([RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0)])
    mpc = MPCController(MPCConfig(horizon_steps=2, time_budget_ms=5000.0), model, one_step)
    decision = mpc.decide("r0", 10.0, PortfolioState(), 0.65, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True)
    assert not decision.used_fallback
    assert decision.sequence_values


# --------------------------------------------------------------------------
# Book depletion (regression test for the double-counting bug)
# --------------------------------------------------------------------------


def test_deplete_book_removes_consumed_shares_from_front_levels():
    depleted = _deplete_book(BOOK_UP, shares_consumed=500.0)
    assert depleted.asks == (BookLevel(0.52, 750.0),)  # level 1 fully consumed and dropped


def test_deplete_book_partial_level_consumption():
    depleted = _deplete_book(BOOK_UP, shares_consumed=200.0)
    assert depleted.asks[0] == BookLevel(0.50, 300.0)
    assert depleted.asks[1] == BookLevel(0.52, 750.0)


def test_deplete_book_beyond_total_depth_empties_asks():
    depleted = _deplete_book(BOOK_UP, shares_consumed=10_000.0)
    assert depleted.asks == ()


def test_deplete_book_noop_for_zero_consumption():
    assert _deplete_book(BOOK_UP, 0.0) is BOOK_UP


def test_continuation_does_not_double_count_consumed_liquidity():
    """Regression test for the bug found while building this phase: before
    the fix, splitting a purchase into "some now, rest later" and buying it
    all "now" produced *inflated and unequal* total sequence values because
    the continuation step re-walked the undepleted book. After the fix,
    under zero churn cost, they must tie (provably: same total shares
    bought from the same fixed total depth, no time-discounting in this V1
    model) - not just "close," exactly, since it's the same arithmetic
    either way once depletion is correct."""
    one_step = _one_step(g_min=-1000.0, edge_min=0.0, churn_penalty=0.0)
    model = build_transition_model([
        RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0),
        RegimeTransition("r0", REGIME_A, REGIME_B, None, 2.0, 1.0),
    ])
    mpc = MPCController(MPCConfig(horizon_steps=2, transition_top_k=2, time_budget_ms=1000.0), model, one_step)
    decision = mpc.decide("r0", 10.0, PortfolioState(), 0.7, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True)

    values = list(decision.sequence_values.values())
    assert max(values) - min(values) < 1e-6, f"sequence values should tie under zero churn cost: {decision.sequence_values}"


def test_nonzero_churn_penalty_favors_consolidating_into_fewer_actions():
    """With a churn cost per action, buying everything in one action (this
    step) must beat splitting the same total purchase across this step and
    the next (two penalized actions) - the "measurable timing improvement
    after churn/latency costs" the Phase 10 exit gate names."""
    one_step = _one_step(g_min=-1000.0, edge_min=0.0, churn_penalty=5.0)
    model = build_transition_model([
        RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0),
        RegimeTransition("r0", REGIME_A, REGIME_B, None, 2.0, 1.0),
    ])
    mpc = MPCController(MPCConfig(horizon_steps=2, transition_top_k=2, time_budget_ms=1000.0), model, one_step)
    decision = mpc.decide("r0", 10.0, PortfolioState(), 0.7, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True)

    # the single-action, full-depth candidate must strictly beat the
    # smaller, split-requiring candidate once each action costs something
    small_id = "taker_up_1"  # first depth level only, forces a second action later to match the big candidate's total
    big_id = "taker_up_2"    # both depth levels in one action
    assert small_id in decision.sequence_values and big_id in decision.sequence_values
    assert decision.sequence_values[big_id] > decision.sequence_values[small_id]


# --------------------------------------------------------------------------
# Portfolio reconstruction
# --------------------------------------------------------------------------


def _candidate(side, pi_u_after, pi_d_after) -> CandidateAction:
    return CandidateAction(
        action_id="c", purpose=OrderPurpose.ALPHA, side=side, mode=OrderMode.FAK, price=0.5, qty=10.0,
        ttl_s=0.0, expected_fill=10.0, ev_after=1.0, g_after=min(pi_u_after, pi_d_after),
        pi_u_after=pi_u_after, pi_d_after=pi_d_after,
    )


def test_portfolio_after_wait_is_unchanged():
    portfolio = PortfolioState(U=5.0, D=3.0, C=2.0)
    wait = _candidate(None, portfolio.Pi_U, portfolio.Pi_D)
    assert _portfolio_after_candidate(portfolio, wait) == portfolio


def test_portfolio_after_up_fill_reconstructs_exact_state():
    portfolio = PortfolioState(U=0.0, D=0.0, C=0.0)
    # buying 100 UP @ 0.5 with fee 1.0 -> U=100, D=0, C=51
    expected = PortfolioState(U=100.0, D=0.0, C=51.0)
    candidate = _candidate(Side.UP, expected.Pi_U, expected.Pi_D)
    result = _portfolio_after_candidate(portfolio, candidate)
    assert math.isclose(result.U, expected.U)
    assert math.isclose(result.D, expected.D)
    assert math.isclose(result.C, expected.C)


def test_portfolio_after_down_fill_reconstructs_exact_state():
    portfolio = PortfolioState(U=20.0, D=0.0, C=10.0)
    # buying 50 DOWN @ 0.4 with fee 0.5 -> D += 50, C += 20.5
    expected = PortfolioState(U=20.0, D=50.0, C=30.5)
    candidate = _candidate(Side.DOWN, expected.Pi_U, expected.Pi_D)
    result = _portfolio_after_candidate(portfolio, candidate)
    assert math.isclose(result.U, expected.U)
    assert math.isclose(result.D, expected.D)
    assert math.isclose(result.C, expected.C)
