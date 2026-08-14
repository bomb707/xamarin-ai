"""Execution simulator parameters (Roadmap Phase 7)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MakerFillConfig:
    """Parametric maker fill-probability / adverse-selection model
    (Strategy doc SS14). No real historical fill data exists yet to
    estimate rho(p,h) and adverse selection from (Roadmap Phase 7 step:
    "Estimate q_fill / maker adverse selection from historical fills") -
    these are clearly-labeled, uncalibrated placeholder parameters, in the
    same spirit as Phase 4's V_T,model. Calibrate against real fills before
    trusting any EV_maker output (Phase 11's job).

    Model: rho(p, h) = 1 - exp(-lambda * h), a standard "constant hazard
    rate" fill-probability shape, where the per-second fill rate lambda
    decays exponentially with distance-to-touch (in ticks) and is damped by
    queue position ahead of the order.
    """

    base_fill_rate_per_s: float = 0.15  # lambda at the touch (distance=0, no queue ahead)
    distance_decay_per_tick: float = 0.6  # lambda multiplied by exp(-k * distance_ticks)
    queue_normalization_shares: float = 200.0  # lambda halved roughly every this many shares of queue ahead

    # adverse selection: q_fill = q - adverse_selection_bp/10000 for the
    # side being bought (a filled maker bid is disproportionately likely to
    # have filled because the market moved against it - SS14).
    adverse_selection_bp: float = 50.0


@dataclass(frozen=True)
class ExecutionConfig:
    taker_delay_ms: float = 0.0  # read from MarketConfig.taker_delay_ms per round in practice
    maker: MakerFillConfig = MakerFillConfig()
