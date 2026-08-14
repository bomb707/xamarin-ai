"""OneStepController parameters (Roadmap Phase 8; field names match the
Strategy doc SS21 Parameter Registry where it names them: G_min, P_min,
edge_min, max spend, max order churn)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OneStepConfig:
    g_min: float  # Portfolio risk floor - hard constraint, never violated (Phase 8 exit gate)
    p_min: float | None = None  # favored-side profit floor, if used
    spend_cap: float | None = None  # B_max
    position_limit: float | None = None

    # SS21 "edge_min: minimum predicted net edge | Alpha". Not formularized
    # beyond the name in the source docs; this build applies it as a flat
    # floor on ev_after for directional (non-WAIT) candidates.
    edge_min: float = 0.0

    # taker candidate generation: how many cumulative depth levels to
    # generate as separate quantity candidates (Roadmap Phase 8: "Generate
    # taker quantities from depth levels and portfolio risk budget").
    max_taker_depth_levels: int = 3

    # maker candidate generation: price offsets (in ticks, toward the
    # touch) from the best bid/ask to generate as candidates (Roadmap Phase
    # 8: "Generate maker prices on the valid current tick grid").
    maker_price_offsets_ticks: tuple[int, ...] = (0, 1, 2)
    maker_quantity: float = 20.0
    maker_horizon_s: float = 10.0

    # SS18 operational penalties (lambda_churn*OrderChurn style terms) -
    # flat, uncalibrated placeholders subtracted from ev_after for any
    # non-WAIT candidate, representing the operational cost of acting.
    churn_penalty: float = 0.0
    opportunity_cost: float = 0.0
