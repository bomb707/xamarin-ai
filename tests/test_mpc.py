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
from xamarinbot.optimizer.candidates import candidate_selection_score
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import CandidateAction, OrderMode
from xamarinbot.portfolio.exposure import ActiveOrderExposure, PendingExposure, RiskView
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
    assert math.isclose(mpc_decision.chosen.delta_ev, base_decision.chosen.delta_ev)
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
    # smaller, split-requiring candidate once each action costs something.
    # Looked up by quantity, not a hardcoded action_id string - Phase 12B
    # audit items 7/8/10 changed candidate generation from raw depth-level
    # cumulative sums to boundary/marginal-edge-derived sizing, so which
    # index a given quantity lands at is no longer stable, only the set of
    # quantities offered (which still includes both depth-level boundaries
    # as one of several candidate sources).
    small_id = next(c.action_id for c in decision.candidates if c.side is Side.UP and math.isclose(c.qty, 500.0))  # first depth level only
    big_id = next(c.action_id for c in decision.candidates if c.side is Side.UP and math.isclose(c.qty, 1250.0))  # both depth levels in one action
    assert small_id in decision.sequence_values and big_id in decision.sequence_values
    assert decision.sequence_values[big_id] > decision.sequence_values[small_id]


# --------------------------------------------------------------------------
# Portfolio reconstruction
# --------------------------------------------------------------------------


def _candidate(side, pi_u_after, pi_d_after) -> CandidateAction:
    return CandidateAction(
        action_id="c", purpose=OrderPurpose.ALPHA, side=side, mode=OrderMode.FAK, price=0.5, qty=10.0,
        ttl_s=0.0, expected_fill=10.0, delta_ev=1.0, g_after=min(pi_u_after, pi_d_after),
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


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.2 item 2: MPC risk/utility integration
# --------------------------------------------------------------------------


def test_mpc_immediate_decision_respects_risk_view():
    """The action MPC ultimately returns must already be aggregate-risk-
    admissible - risk_view is forwarded into the underlying one-step call
    for the immediate action, not merely consulted at some later dispatch
    step with nothing to fall back to."""
    one_step = _one_step()
    mpc = MPCController(MPCConfig(horizon_steps=1), TransitionModel(probabilities={}), one_step)
    portfolio = PortfolioState()

    unconstrained = mpc.decide("r0", 10.0, portfolio, 0.9, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True)
    assert unconstrained.chosen.mode is not OrderMode.WAIT, "sanity: without a risk_view, MPC picks a real trade"

    # A RiskView that rejects every candidate must force MPC back to WAIT,
    # exactly like OneStepController.decide's own pre-selection check.
    class _RejectEverything:
        def admits(self, candidate, g_min, spend_cap, position_limit=None, is_recovery_candidate=False, exact_subset_cap=10):
            return False

    constrained = mpc.decide("r0", 10.0, portfolio, 0.9, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True, risk_view=_RejectEverything())
    assert constrained.chosen.mode is OrderMode.WAIT, "a RiskView rejecting everything must force MPC's immediate decision to WAIT too"


def test_mpc_sequence_value_uses_the_same_candidate_selection_score_as_one_step():
    """Phase 12B Tranche 2.2 item 2: MPC's immediate sequence value must
    not silently duplicate/diverge from OneStepController's own objective
    - the prior draft used `sequence_value = candidate.delta_ev` alone,
    ignoring lambda_g and selection_penalty entirely."""
    cfg = OneStepConfig(g_min=-1000.0, edge_min=0.0, lambda_g=0.37)  # nonzero lambda_g so the two formulas provably differ if not shared
    one_step = OneStepController(cfg, ExecutionConfig(), FEE)
    mpc = MPCController(MPCConfig(horizon_steps=1), TransitionModel(probabilities={}), one_step)
    portfolio = PortfolioState()

    decision = mpc.decide("r0", 10.0, portfolio, 0.65, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True)
    assert decision.sequence_values
    for c in decision.candidates:
        if c.action_id in decision.sequence_values:
            expected = candidate_selection_score(c, cfg)
            assert math.isclose(decision.sequence_values[c.action_id], expected), f"{c.action_id}: {decision.sequence_values[c.action_id]} != {expected}"
            if not math.isclose(c.expected_delta_g, 0.0):
                assert not math.isclose(decision.sequence_values[c.action_id], c.delta_ev), "sanity: with nonzero lambda_g this must differ from plain delta_ev"


def test_mpc_falls_back_to_risk_aware_one_step_when_real_active_exposure_exists():
    """Phase 12B Tranche 2.2 item 2: rather than inventing a pending-order
    exposure transition model for the hypothetical rollout tree, MPC
    falls back entirely to the risk-aware one-step decision whenever
    real active exposure (pending takers / open makers) is tracked in
    risk_view - the deeper rollout has no model for how that exposure
    evolves, so exploring it without one would silently ignore it."""
    one_step = _one_step()
    model = build_transition_model([RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0)])
    mpc = MPCController(MPCConfig(horizon_steps=3, time_budget_ms=5000.0), model, one_step)
    portfolio = PortfolioState()

    pending_order = ActiveOrderExposure(Side.UP, 50.0, 25.0)
    risk_view = RiskView(portfolio, pending_taker_exposure=PendingExposure(orders=(pending_order,), potential_up_shares=50.0, max_committed_spend=25.0))
    assert risk_view.combined_exposure.orders  # sanity: real active exposure is present

    decision = mpc.decide("r0", 10.0, portfolio, 0.65, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True, risk_view=risk_view)
    assert decision.used_fallback, "MPC must fall back to the plain risk-aware one-step decision when real active exposure exists"

    permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(REGIME_A))
    base_decision = one_step.decide("r0", 10.0, portfolio, 0.65, permitted, BOOK_UP, BOOK_DOWN, 0.01, True, risk_view=risk_view)
    assert decision.chosen.action_id == base_decision.chosen.action_id


def test_mpc_does_not_fall_back_when_risk_view_has_no_active_exposure():
    """Contrast case: a risk_view with zero pending/open orders (only
    confirmed portfolio) must not trigger the active-exposure fallback -
    the deeper rollout still runs normally."""
    one_step = _one_step()
    model = build_transition_model([RegimeTransition("r0", REGIME_A, REGIME_A, None, 1.0, 1.0)])
    mpc = MPCController(MPCConfig(horizon_steps=3, time_budget_ms=5000.0), model, one_step)
    portfolio = PortfolioState()
    risk_view = RiskView(portfolio)  # no pending taker / open maker exposure at all

    decision = mpc.decide("r0", 10.0, portfolio, 0.65, REGIME_A, BOOK_UP, BOOK_DOWN, 0.01, True, risk_view=risk_view)
    assert not decision.used_fallback
    assert decision.sequence_values
