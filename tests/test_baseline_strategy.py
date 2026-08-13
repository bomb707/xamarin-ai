"""Roadmap Phase 0 verification: "Verify all baseline skip reasons are
observable." Every SkipReason is exercised directly here with a crafted
BaselineInputs, plus the average-price guard and risk caps."""
from __future__ import annotations

from xamarinbot.baseline.config import BaselineConfig
from xamarinbot.baseline.strategy import BaselineInputs, SkipReason, decide
from xamarinbot.portfolio.state import Side

CFG = BaselineConfig()


def _inputs(**overrides) -> BaselineInputs:
    """Defaults are internally consistent with the ask_range guard: best_ask
    sits just above (for UP) / just above the complement of (for DOWN) the
    *current* clob_mid, well inside the default ask_range_bp=800 tolerance
    (a fixed ~1c spread is already several hundred bp at these price
    levels, so ask_range_bp is a wide-quote cap, not a tight range)."""
    clob_mid = 0.42
    base = dict(
        t=100.0,
        p0=100_000.0,
        twap=100_050.0,  # gap_bp = 5 > minimum_gap_bp -> UP
        clob_mid=clob_mid,
        clob_mid_prev=0.40,  # UP direction (0.42 > 0.40)
        spot=100_100.0,
        spot_prev=100_000.0,  # UP direction
        best_ask_up=clob_mid + 0.001,
        best_ask_down=(1.0 - clob_mid) + 0.001,
        is_fresh=True,
    )
    base.update(overrides)
    return BaselineInputs(**base)


def test_unanimous_up_signal_produces_taker_up_order():
    decision = decide(_inputs(), 0.0, 0.0, 0.0, CFG)
    assert decision.skip_reason is None
    assert decision.order is not None
    assert decision.order.side is Side.UP


def test_outside_decision_window_skip():
    decision = decide(_inputs(t=5.0), 0.0, 0.0, 0.0, CFG)
    assert decision.skip_reason is SkipReason.OUTSIDE_DECISION_WINDOW

    decision2 = decide(_inputs(t=290.0), 0.0, 0.0, 0.0, CFG)
    assert decision2.skip_reason is SkipReason.OUTSIDE_DECISION_WINDOW


def test_stale_data_skip():
    decision = decide(_inputs(is_fresh=False), 0.0, 0.0, 0.0, CFG)
    assert decision.skip_reason is SkipReason.STALE_DATA


def test_gap_below_minimum_skip():
    decision = decide(_inputs(twap=100_000.05), 0.0, 0.0, 0.0, CFG)  # ~0.005bp gap
    assert decision.skip_reason is SkipReason.GAP_BELOW_MINIMUM


def test_conflicting_signals_skip():
    # TWAP says UP, spot says DOWN -> not unanimous
    decision = decide(_inputs(spot=99_900.0, spot_prev=100_000.0), 0.0, 0.0, 0.0, CFG)
    assert decision.skip_reason is SkipReason.CONFLICTING_SIGNALS


def test_no_liquidity_skip_when_ask_missing():
    decision = decide(_inputs(best_ask_up=None), 0.0, 0.0, 0.0, CFG)
    assert decision.skip_reason is SkipReason.NO_LIQUIDITY


def test_no_liquidity_skip_when_ask_range_exceeded():
    # best_ask_up is far from the current UP reference (clob_mid) - moderate
    # enough not to also trip the average-price guard first.
    decision = decide(_inputs(best_ask_up=0.05, clob_mid=0.42), 0.0, 0.0, 0.0, CFG)
    assert decision.skip_reason is SkipReason.NO_LIQUIDITY


def test_price_guard_breach_skip():
    cfg = BaselineConfig(avg_price_guard=0.5)
    decision = decide(_inputs(best_ask_up=0.48), 0.0, 0.0, 0.0, cfg)
    assert decision.skip_reason is SkipReason.PRICE_GUARD_BREACH


def test_residual_cap_breach_skip():
    cfg = BaselineConfig(residual_cap=1.0)
    decision = decide(_inputs(), state_U=0.0, state_D=0.0, state_C=0.0, cfg=cfg)
    assert decision.skip_reason is SkipReason.RESIDUAL_CAP_BREACH


def test_spend_cap_breach_skip():
    cfg = BaselineConfig(spend_cap=0.01)
    decision = decide(_inputs(), state_U=0.0, state_D=0.0, state_C=0.0, cfg=cfg)
    assert decision.skip_reason is SkipReason.SPEND_CAP_BREACH


def test_directional_lead_sizing_adds_bonus_when_lead_exceeds_threshold():
    cfg = BaselineConfig(lead_bonus_threshold_bp=1.0, lead_size_bonus=3.0, clip=5.0)
    # spot far above twap -> large positive lead, same direction as twap_direction (UP)
    decision = decide(_inputs(spot=101_000.0, spot_prev=100_000.0, twap=100_050.0), 0.0, 0.0, 0.0, cfg)
    assert decision.order is not None
    assert decision.order.quantity == cfg.clip + cfg.lead_size_bonus


def test_down_side_unanimous_signal():
    clob_mid = 0.45
    decision = decide(
        _inputs(
            twap=99_950.0,  # gap_bp negative -> DOWN
            clob_mid=clob_mid,
            clob_mid_prev=0.50,  # DOWN direction (0.45 < 0.50)
            spot=99_900.0,
            spot_prev=100_000.0,  # DOWN direction
            best_ask_down=(1.0 - clob_mid) + 0.001,  # stay within ask_range of the current DOWN reference
        ),
        0.0,
        0.0,
        0.0,
        CFG,
    )
    assert decision.skip_reason is None
    assert decision.order.side is Side.DOWN
