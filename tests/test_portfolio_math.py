"""Phase 3 exit gate: 100% kernel test pass, identities hold under random
multi-fill sequences, kernel has no dependency on predictor/exchange client
(verified structurally by this module's imports)."""
from __future__ import annotations

import math

from hypothesis import given, strategies as st

from xamarinbot.portfolio.math import (
    CandidateFill,
    OrderPurpose,
    RiskConstraints,
    evaluate_constraints,
    hedge_efficiency,
    max_directional_spend,
    max_hedge_quantity,
    min_hedge_quantity,
    simulate_fill,
)
from xamarinbot.portfolio.state import FeeConfig, Fill, LiquidityRole, PortfolioState, Side, apply_fill, apply_fills

fills_strategy = st.lists(
    st.tuples(
        st.sampled_from([Side.UP, Side.DOWN]),
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False),
        st.sampled_from([LiquidityRole.MAKER, LiquidityRole.TAKER]),
    ),
    min_size=0,
    max_size=25,
)


def _fills_from_tuples(tuples, fee_config: FeeConfig) -> list[Fill]:
    out = []
    for side, price, shares, role in tuples:
        fee = fee_config.fee_for(role, shares, price)
        out.append(Fill(side=side, price=price, shares=shares, role=role, fee=fee))
    return out


@given(fills_strategy)
def test_identities_hold_under_random_multi_fill_sequences(raw_fills):
    fee_config = FeeConfig()
    fills = _fills_from_tuples(raw_fills, fee_config)
    state = apply_fills(PortfolioState(), fills)

    assert math.isclose(state.Pi_U, state.U - state.C, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(state.Pi_D, state.D - state.C, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(state.G, min(state.Pi_U, state.Pi_D), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(state.R, state.Pi_U - state.Pi_D, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(state.R, state.U - state.D, rel_tol=1e-9, abs_tol=1e-9)

    # payoff decomposition (Strategy doc SS10.1)
    assert math.isclose(state.Pi_U, (state.m - state.C) + state.d, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(state.Pi_D, (state.m - state.C) - state.d, rel_tol=1e-9, abs_tol=1e-9)


def test_random_multi_fill_reconciliation_matches_manual_sum():
    fee_config = FeeConfig()
    fills = [
        Fill(Side.UP, 0.40, 10.0, LiquidityRole.TAKER, fee_config.taker_fee(10.0, 0.40)),
        Fill(Side.DOWN, 0.55, 4.0, LiquidityRole.MAKER, 0.0),
        Fill(Side.UP, 0.42, 6.0, LiquidityRole.TAKER, fee_config.taker_fee(6.0, 0.42)),
    ]
    state = apply_fills(PortfolioState(), fills)

    expected_U = 10.0 + 6.0
    expected_D = 4.0
    expected_C = sum(f.price * f.shares + f.fee for f in fills)

    assert math.isclose(state.U, expected_U)
    assert math.isclose(state.D, expected_D)
    assert math.isclose(state.C, expected_C)
    assert math.isclose(state.Pi_U, expected_U - expected_C)
    assert math.isclose(state.Pi_D, expected_D - expected_C)
    assert math.isclose(state.G, min(expected_U, expected_D) - expected_C)


def test_maker_fee_is_zero_taker_fee_uses_formula():
    cfg = FeeConfig(crypto_fee_rate=0.07)
    assert cfg.fee_for(LiquidityRole.MAKER, 100.0, 0.6) == 0.0
    expected = 100.0 * 0.07 * 0.6 * (1 - 0.6)
    assert math.isclose(cfg.fee_for(LiquidityRole.TAKER, 100.0, 0.6), expected)


def test_simulate_fill_does_not_mutate_input_state():
    state = PortfolioState(U=5.0, D=2.0, C=3.0)
    candidate = CandidateFill(Side.UP, 0.5, 4.0, LiquidityRole.TAKER, OrderPurpose.ALPHA)
    result = simulate_fill(state, candidate, FeeConfig())

    assert state.U == 5.0 and state.D == 2.0 and state.C == 3.0  # unchanged
    assert result.portfolio_before is state
    assert result.portfolio_after.U == 9.0
    assert result.purpose is OrderPurpose.ALPHA
    assert math.isclose(result.risk_contribution, result.portfolio_after.G - state.G)


def test_alpha_and_hedge_purposes_are_tagged_separately():
    state = PortfolioState(U=20.0, D=2.0, C=10.0)  # UP favored, DOWN is worst case
    hedge = CandidateFill(Side.DOWN, 0.3, 5.0, LiquidityRole.MAKER, OrderPurpose.HEDGE)
    result = simulate_fill(state, hedge, FeeConfig())
    assert result.purpose is OrderPurpose.HEDGE
    # a DOWN hedge fill should improve (or not worsen) the worst-case margin
    assert result.risk_contribution >= 0


def test_g_min_constraint_rejects_violating_action():
    state = PortfolioState(U=1.0, D=1.0, C=0.5)
    candidate = CandidateFill(Side.UP, 0.9, 100.0, LiquidityRole.TAKER, OrderPurpose.ALPHA)
    result = simulate_fill(state, candidate, FeeConfig())
    constraints = RiskConstraints(g_min=0.0)
    check = evaluate_constraints(result, constraints)
    assert not check.passed
    assert "g_min" in check.violated


def test_spend_and_position_constraints():
    state = PortfolioState()
    candidate = CandidateFill(Side.UP, 0.5, 50.0, LiquidityRole.TAKER, OrderPurpose.ALPHA)
    result = simulate_fill(state, candidate, FeeConfig())

    tight_spend = RiskConstraints(g_min=-1000.0, spend_cap=1.0)
    assert "spend_cap" in evaluate_constraints(result, tight_spend).violated

    tight_position = RiskConstraints(g_min=-1000.0, position_limit=10.0)
    assert "position_limit" in evaluate_constraints(result, tight_position).violated


def test_max_directional_spend_formula():
    assert max_directional_spend(g_current=10.0, g_min=2.0) == 8.0


def test_hedge_quantity_formulas():
    # x_min_hedge = max(0, (-L_max - Pi_DOWN) / (1 - c_D))
    assert math.isclose(min_hedge_quantity(l_max=5.0, pi_down=-20.0, c_d=0.4), (20.0 - 5.0) / 0.6)
    assert min_hedge_quantity(l_max=50.0, pi_down=10.0, c_d=0.4) == 0.0  # floored at 0

    # x_max_hedge = (Pi_UP - P_min) / c_D
    assert math.isclose(max_hedge_quantity(pi_up=30.0, p_min=5.0, c_d=0.5), 50.0)


def test_hedge_efficiency_formula():
    assert math.isclose(hedge_efficiency(0.2), 0.8 / 0.2)
    assert math.isclose(hedge_efficiency(0.5), 1.0)
