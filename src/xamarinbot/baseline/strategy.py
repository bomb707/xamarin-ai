"""Reconstructed Xamarinbot V1 baseline strategy (Roadmap Phase 0).

Source basis (Strategy doc SS1, "Source Basis and Design Boundary"): "a
buy-only BTC five-minute strategy that evaluates from 15 to 270 seconds,
uses CLOB midpoint momentum, BTC spot momentum, and a Chainlink TWAP gap,
requires unanimous direction, executes bounded FAK taker orders, and holds
fills to settlement. The baseline also uses a fixed minimum clip,
directional-lead sizing, an average-price guard, and risk caps."

No V1 source code exists in this repo (see PHASE_STATUS.md), so this is a
reconstruction from that description plus the fee/tick-size mechanics in
Strategy doc SS2. The source material describes sizing/guard behavior
narratively, not as exact formulas; every place this module had to choose a
concrete formula that isn't literally given, it's called out below and in
BaselineConfig so it's auditable rather than silently invented.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xamarinbot.baseline.config import BaselineConfig
from xamarinbot.portfolio.state import LiquidityRole, Side


class SkipReason(str, Enum):
    OUTSIDE_DECISION_WINDOW = "OUTSIDE_DECISION_WINDOW"
    STALE_DATA = "STALE_DATA"
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
    GAP_BELOW_MINIMUM = "GAP_BELOW_MINIMUM"
    PRICE_GUARD_BREACH = "PRICE_GUARD_BREACH"
    RESIDUAL_CAP_BREACH = "RESIDUAL_CAP_BREACH"
    SPEND_CAP_BREACH = "SPEND_CAP_BREACH"
    NO_LIQUIDITY = "NO_LIQUIDITY"


@dataclass(frozen=True)
class BaselineInputs:
    """Causal market state at one decision point."""

    t: float  # elapsed seconds in round
    p0: float  # price-to-beat / opening reference
    twap: float  # current Chainlink TWAP
    clob_mid: float
    clob_mid_prev: float  # midpoint `clob_lookback_s` seconds earlier
    spot: float
    spot_prev: float  # spot `spot_lookback_s` seconds earlier
    best_ask_up: float | None
    best_ask_down: float | None
    is_fresh: bool  # all required feeds within `freshness_s`


@dataclass(frozen=True)
class BaselineOrder:
    side: Side
    role: LiquidityRole
    price: float
    quantity: float


@dataclass(frozen=True)
class BaselineDecision:
    order: BaselineOrder | None
    skip_reason: SkipReason | None
    # diagnostics, always populated so every skip reason is observable
    # (Roadmap Phase 0 verification: "Verify all baseline skip reasons are
    # observable.")
    clob_direction: int  # -1, 0, +1
    spot_direction: int
    twap_direction: int
    gap_bp: float


def _sign(delta: float) -> int:
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def decide(inputs: BaselineInputs, state_U: float, state_D: float, state_C: float, cfg: BaselineConfig) -> BaselineDecision:
    """Pure decision function: causal inputs + current portfolio state ->
    one candidate order or a skip reason. No mutation, no I/O."""

    clob_direction = _sign(inputs.clob_mid - inputs.clob_mid_prev)
    spot_direction = _sign(inputs.spot - inputs.spot_prev)
    gap_bp = 10_000.0 * (inputs.twap - inputs.p0) / inputs.p0
    twap_direction = _sign(gap_bp) if abs(gap_bp) >= cfg.minimum_gap_bp else 0

    diagnostics = dict(
        clob_direction=clob_direction,
        spot_direction=spot_direction,
        twap_direction=twap_direction,
        gap_bp=gap_bp,
    )

    if not (cfg.decision_window_start_s <= inputs.t <= cfg.decision_window_end_s):
        return BaselineDecision(order=None, skip_reason=SkipReason.OUTSIDE_DECISION_WINDOW, **diagnostics)

    if not inputs.is_fresh:
        return BaselineDecision(order=None, skip_reason=SkipReason.STALE_DATA, **diagnostics)

    if twap_direction == 0:
        return BaselineDecision(order=None, skip_reason=SkipReason.GAP_BELOW_MINIMUM, **diagnostics)

    # requires unanimous direction (Strategy doc SS1)
    if not (clob_direction == spot_direction == twap_direction):
        return BaselineDecision(order=None, skip_reason=SkipReason.CONFLICTING_SIGNALS, **diagnostics)

    side = Side.UP if twap_direction > 0 else Side.DOWN
    best_ask = inputs.best_ask_up if side is Side.UP else inputs.best_ask_down
    if best_ask is None:
        return BaselineDecision(order=None, skip_reason=SkipReason.NO_LIQUIDITY, **diagnostics)

    limit_price = min(1.0, best_ask + cfg.limit_delta)

    # average-price guard: don't chase an already-expensive outcome
    if limit_price >= cfg.avg_price_guard:
        return BaselineDecision(order=None, skip_reason=SkipReason.PRICE_GUARD_BREACH, **diagnostics)

    # ask range: reject if the executable ask sits too far (in bp, same
    # gap-formula shape as SS5) from the *current* fair-value reference -
    # effectively an executable bid-ask-spread cap, guarding against a wide/
    # illiquid quote. clob_mid is the UP midpoint (Strategy doc SS4 M_t);
    # the DOWN-side reference is its complement. Deliberately uses the
    # current mid, not clob_mid_prev, since best_ask has no "previous ask"
    # counterpart in BaselineInputs to compare against directly.
    side_ref_now = inputs.clob_mid if side is Side.UP else (1.0 - inputs.clob_mid)
    ask_range_bp = 10_000.0 * abs(best_ask - side_ref_now) / max(side_ref_now, 1e-6)
    if ask_range_bp > cfg.ask_range_bp:
        return BaselineDecision(order=None, skip_reason=SkipReason.NO_LIQUIDITY, **diagnostics)

    # fixed minimum clip + directional-lead sizing (see module docstring:
    # the source docs name this behavior but not its formula)
    lead_bp = 10_000.0 * (inputs.spot - inputs.twap) / inputs.twap
    quantity = cfg.clip
    if abs(lead_bp) >= cfg.lead_bonus_threshold_bp and _sign(lead_bp) == twap_direction:
        quantity += cfg.lead_size_bonus

    # risk caps
    current_R = state_U - state_D
    projected_R = current_R + quantity if side is Side.UP else current_R - quantity
    if abs(projected_R) > cfg.residual_cap:
        return BaselineDecision(order=None, skip_reason=SkipReason.RESIDUAL_CAP_BREACH, **diagnostics)

    projected_spend = state_C + quantity * limit_price
    if projected_spend > cfg.spend_cap:
        return BaselineDecision(order=None, skip_reason=SkipReason.SPEND_CAP_BREACH, **diagnostics)

    order = BaselineOrder(side=side, role=LiquidityRole.TAKER, price=limit_price, quantity=quantity)
    return BaselineDecision(order=order, skip_reason=None, **diagnostics)
