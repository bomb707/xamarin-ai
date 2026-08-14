"""Maker fill probability, adverse selection, and EV (Roadmap Phase 7,
Strategy doc SS14).

"Implement maker queue/fill model using distance to touch, queue depth,
trade flow, state and time." "Estimate q_fill / maker adverse selection
from historical fills." See execution/config.py's MakerFillConfig
docstring for why this is a clearly-labeled, uncalibrated placeholder model
rather than something fit to real fills (none exist yet).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from xamarinbot.execution.config import MakerFillConfig
from xamarinbot.portfolio.state import Side


def fill_probability(distance_to_touch_ticks: float, queue_ahead_shares: float, horizon_s: float, cfg: MakerFillConfig) -> float:
    """rho(p, h) = P(fill before horizon h | state, price, queue)
    (Strategy doc SS14). Modeled as a constant-hazard-rate process: the
    per-second fill rate lambda decays exponentially with distance to
    touch and is damped by queue position ahead."""
    if distance_to_touch_ticks < 0 or horizon_s <= 0:
        return 0.0
    queue_damping = cfg.queue_normalization_shares / (cfg.queue_normalization_shares + max(0.0, queue_ahead_shares))
    lam = cfg.base_fill_rate_per_s * math.exp(-cfg.distance_decay_per_tick * distance_to_touch_ticks) * queue_damping
    return 1.0 - math.exp(-lam * horizon_s)


def q_fill(q: float, side: Side, cfg: MakerFillConfig) -> float:
    """q_fill can differ from q because a resting order is disproportion-
    ately likely to fill when the market moves against it (SS14). A
    resting UP bid's conditional fill probability skews toward states
    where UP is *less* likely to win than the unconditional q, and
    symmetrically for DOWN."""
    adjustment = cfg.adverse_selection_bp / 10_000.0
    adjusted = q - adjustment if side is Side.UP else q + adjustment
    return min(1.0, max(0.0, adjusted))


def maker_expected_value(
    rho: float, quantity: float, q_fill_value: float, price: float, opportunity_cost: float = 0.0, risk_penalty: float = 0.0
) -> float:
    """EV_maker_UP(p,x,h) ~= rho(p,h) * x * (q_fill - p) - opportunity_cost
    - risk_penalty (SS14). Same formula for DOWN with q_fill computed for
    Side.DOWN and price interpreted as the DOWN price."""
    return rho * quantity * (q_fill_value - price) - opportunity_cost - risk_penalty


@dataclass(frozen=True)
class MakerPlacement:
    price: float
    expected_value: float
    fill_probability: float


def optimal_maker_price(
    candidate_prices: list[float],
    distances_to_touch_ticks: list[float],
    queue_ahead_shares: list[float],
    quantity: float,
    horizon_s: float,
    q: float,
    side: Side,
    cfg: MakerFillConfig,
    opportunity_cost: float = 0.0,
    risk_penalty: float = 0.0,
) -> MakerPlacement:
    """p*_maker = argmax_p EV_maker(p,x,h) (SS14), searched over a caller-
    supplied candidate price grid (the current valid tick grid near the
    touch - "The maker price should be optimized on the live tick grid. Do
    not use a fixed number of ticks from BESTASK as a universal rule.")."""
    if not candidate_prices:
        raise ValueError("optimal_maker_price requires at least one candidate price")

    qf = q_fill(q, side, cfg)
    best: MakerPlacement | None = None
    for price, distance, queue in zip(candidate_prices, distances_to_touch_ticks, queue_ahead_shares):
        rho = fill_probability(distance, queue, horizon_s, cfg)
        ev = maker_expected_value(rho, quantity, qf, price, opportunity_cost, risk_penalty)
        if best is None or ev > best.expected_value:
            best = MakerPlacement(price=price, expected_value=ev, fill_probability=rho)
    return best
