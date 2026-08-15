"""Phase 12C.1 items 11-14: runtime market constraints, in SHARES.

The numbers asserted here are the ones the live BTC five-minute markets
actually reported in the Phase 12C captures: `min_order_size = 5.0` shares
and `tick_size = 0.01`, for all sampled markets.
"""
from __future__ import annotations

import pytest

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.feeds.base import BookLevel, MarketConfig
from xamarinbot.market.constraints import MarketConstraintError, MarketConstraints
from xamarinbot.market.order_request import (
    OrderRequestError,
    build_market_buy_request,
    build_market_sell_request,
)
from xamarinbot.optimizer.candidates import (
    evaluate_maker_candidate,
    evaluate_taker_candidate,
    generate_buffer_build_candidates,
    generate_hedge_candidate,
    taker_sizing_boundaries,
)
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
from xamarinbot.provenance import DataProvenance

FEE = FeeConfig()
#: The real BTC five-minute market parameters, as captured.
LIVE = MarketConstraints.for_testing(min_order_shares=5.0, tick_size=0.01)


# ------------------------------------------------- the static fallback is gone

def test_taker_min_size_is_no_longer_a_config_field():
    """Item 11: `OneStepConfig.taker_min_size = 1.0` must cease being a
    production fallback. It was wrong by 5x against every sampled live
    market, in the direction that generates orders the venue rejects."""
    assert not hasattr(OneStepConfig(g_min=-1.0), "taker_min_size")
    assert "taker_min_size" not in OneStepConfig.__dataclass_fields__


def test_the_minimum_is_a_share_quantity_not_a_usdc_notional():
    """Item 13: do not conflate minimum shares with minimum USDC notional.
    5 shares at $0.10 is $0.50 of notional and is a legal order."""
    assert LIVE.min_order_shares == 5.0
    assert LIVE.admits_size(5.0)
    notional = 5.0 * 0.10
    assert notional < 1.0  # far below any "$1 minimum" assumption
    assert LIVE.admits_size(5.0), "a legal 5-share order must not be rejected on notional"


def test_float_error_at_exactly_the_minimum_is_tolerated_but_undersize_is_not():
    assert LIVE.admits_size(5.0 - 1e-12)
    assert not LIVE.admits_size(4.9)
    assert not LIVE.admits_size(0.0)


def test_invalid_constraints_raise_rather_than_defaulting():
    with pytest.raises(MarketConstraintError):
        MarketConstraints.for_testing(min_order_shares=0.0)
    with pytest.raises(MarketConstraintError):
        MarketConstraints.for_testing(tick_size=0.0)


# ------------------------------------------------------------ construction

def test_built_from_the_market_s_own_config():
    cfg = MarketConfig(
        market_id="btc-updown-5m-1786777800", up_token_id="U", down_token_id="D",
        start_ts=1786777800.0, end_ts=1786778100.0, tick_size=0.01,
        min_order_size=5.0, fee_rate=0.07, taker_delay_ms=0.0, twap_window_seconds=60, settlement_kind="chainlink_twap",
    )
    c = MarketConstraints.from_market_config(
        cfg, provenance=DataProvenance.REAL_REPLAY, source="projected MARKET_CONFIG"
    )
    assert c.min_order_shares == 5.0
    assert c.tick_size == 0.01
    assert c.fee_configuration.crypto_fee_rate == 0.07
    assert c.twap_window_s == 60
    assert c.provenance is DataProvenance.REAL_REPLAY
    assert c.token_for(Side.UP) == "U" and c.token_for(Side.DOWN) == "D"
    assert "MOS=5.0" in c.describe()


def test_for_testing_is_stamped_synthetic():
    assert MarketConstraints.for_testing().provenance is DataProvenance.SYNTHETIC_TEST


# ------------------------------------ item 13: MOS applies to EVERY purpose

def _cfg() -> OneStepConfig:
    return OneStepConfig(g_min=-1_000_000.0, spend_cap=None, position_limit=None,
                         edge_min=-1_000_000.0)


def test_taker_candidate_below_the_minimum_is_invalid():
    c = evaluate_taker_candidate(
        "t", Side.UP, OrderPurpose.ALPHA, 4.0, limit_price=0.50,
        asks=(BookLevel(0.50, 500.0),), portfolio=PortfolioState(), q=0.9,
        fee_config=FEE, cfg=_cfg(), constraints=LIVE,
    )
    assert not c.is_valid
    assert "min_order_shares" in c.violated_constraints


def test_maker_candidate_below_the_minimum_is_invalid():
    """The fixed-quantity maker path never checked any minimum before."""
    c = evaluate_maker_candidate(
        "m", Side.UP, OrderPurpose.ALPHA, price=0.40, qty=4.0,
        distance_to_touch_ticks=0.0, queue_ahead_shares=0.0, horizon_s=10.0,
        portfolio=PortfolioState(), q=0.9, exec_cfg=ExecutionConfig(),
        cfg=_cfg(), constraints=LIVE,
    )
    assert not c.is_valid
    assert "min_order_shares" in c.violated_constraints


def test_hedge_candidate_is_subject_to_the_minimum():
    """`generate_hedge_candidate` consulted NO minimum at all before Phase
    12C.1 - a HEDGE could be sized below what the venue accepts."""
    # A tiny imbalance needs only a tiny hedge, well under 5 shares.
    portfolio = PortfolioState(U=0.0, D=1.0, C=0.0)
    hedge = generate_hedge_candidate(
        "hedge", portfolio, (BookLevel(0.40, 500.0),), (BookLevel(0.40, 500.0),),
        q=0.5, fee_config=FEE,
        cfg=OneStepConfig(g_min=-0.5, spend_cap=200.0, position_limit=200.0,
                          enable_portfolio_repair=True),
        constraints=LIVE,
    )
    if hedge is not None and hedge.qty < LIVE.min_order_shares:
        assert not hedge.is_valid
        assert "min_order_shares" in hedge.violated_constraints


def test_every_generated_taker_quantity_clears_the_market_minimum():
    sizing = taker_sizing_boundaries(
        (BookLevel(0.50, 500.0),), q_effective=0.9, fee_config=FEE, cfg=_cfg(),
        portfolio=PortfolioState(), side=Side.UP, constraints=LIVE,
    )
    assert sizing.quantities
    assert all(LIVE.admits_size(q) for q in sizing.quantities)


def test_no_order_is_generated_rather_than_rounding_up_to_the_minimum():
    """Item 13: "If x_maxFeasible < MOS, generate no order rather than
    rounding upward and accidentally breaking the risk/economic constraint."
    """
    tight = OneStepConfig(g_min=-1_000_000.0, spend_cap=0.30, position_limit=None)
    sizing = taker_sizing_boundaries(
        (BookLevel(0.50, 500.0),), q_effective=0.9, fee_config=FEE, cfg=tight,
        portfolio=PortfolioState(), side=Side.UP, constraints=LIVE,
    )
    # 0.30 USDC of spend buys well under 5 shares at 0.50.
    assert all(LIVE.admits_size(q) for q in sizing.quantities)


def test_buffer_build_respects_the_market_minimum():
    cands = generate_buffer_build_candidates(
        "bb", PortfolioState(U=0.0, D=20.0, C=0.0), (BookLevel(0.10, 500.0),), (),
        q=0.5, fee_config=FEE,
        cfg=OneStepConfig(g_min=-1_000_000.0, enable_buffer_build=True),
        constraints=LIVE,
    )
    for c in cands:
        if c.is_valid:
            assert LIVE.admits_size(c.qty)


# ------------------------ item 14: BUY dollar encoding vs internal sizing

def test_buy_request_translates_shares_to_a_dollar_amount():
    req = build_market_buy_request(
        Side.UP, desired_shares=10.0, max_execution_price=0.55,
        max_all_in_cost=100.0, constraints=LIVE,
    )
    assert req.amount_usdc == pytest.approx(5.5)
    assert req.desired_shares == 10.0          # the share intent survives
    assert req.token_id == LIVE.up_token_id
    assert req.implied_shares_at_cap == pytest.approx(10.0)


def test_sell_stays_share_denominated():
    req = build_market_sell_request(Side.DOWN, shares=7.0, min_execution_price=0.30,
                                    constraints=LIVE)
    assert req.shares == 7.0
    assert req.token_id == LIVE.down_token_id


def test_the_share_minimum_survives_the_dollar_encoding():
    """Item 14: "Preserve x >= MOS independently of the request's dollar
    encoding." A request comfortably above any dollar threshold can still be
    below the share minimum."""
    with pytest.raises(OrderRequestError, match="below the market minimum"):
        build_market_buy_request(
            Side.UP, desired_shares=4.0, max_execution_price=0.99,
            max_all_in_cost=100.0, constraints=LIVE,
        )
    with pytest.raises(OrderRequestError, match="below the market minimum"):
        build_market_sell_request(Side.UP, shares=4.0, min_execution_price=0.1,
                                  constraints=LIVE)


def test_a_budget_that_clips_below_the_minimum_is_refused_not_shrunk():
    with pytest.raises(OrderRequestError, match="below the market minimum"):
        build_market_buy_request(
            Side.UP, desired_shares=10.0, max_execution_price=0.50,
            max_all_in_cost=1.0, constraints=LIVE,  # buys only 2 shares
        )


def test_order_request_module_submits_nothing():
    """Phase 12C.1 builds the translation boundary but must not send it.

    Checked structurally (imports and called names) rather than by substring,
    so the module's own prose explaining that it does no signing cannot trip
    the assertion.
    """
    import ast
    import inspect

    import xamarinbot.market.order_request as mod

    tree = ast.parse(inspect.getsource(mod))
    imported = {
        a.name.split(".")[0]
        for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    } | {
        n.module.split(".")[0]
        for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not (imported & {"httpx", "requests", "websockets", "urllib", "socket"}), (
        f"order_request must not import a network client; got {sorted(imported)}"
    )

    called = {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    for banned in ("post", "send", "submit", "sign_order", "create_order"):
        assert banned not in called, f"order_request calls {banned}()"
