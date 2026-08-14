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

    # Roadmap Phase 11 / SS20.1 ablation toggles. All three default to the
    # pre-Phase-11 behavior (repair off, all execution modes on, lambda_g
    # zero), so existing Phase 8/9/10 callers that never set these are
    # unaffected; ablations turn them on/off explicitly to isolate each
    # piece's contribution.
    enable_portfolio_repair: bool = False  # SS17 hedge candidates (ablation #5 vs #6+)
    taker_only: bool = False  # skip maker candidate generation entirely (ablation #6 vs #7)

    # SS18's actual objective is J = E[PnL_T] + lambda_G*G_T - ... , not
    # plain EV - selection under pure ev_after can *never* pick a hedge
    # candidate (SS17 hedges have negative standalone EV by construction,
    # so they always lose to WAIT at lambda_g=0), which would make
    # enable_portfolio_repair wired in but functionally inert. lambda_g=0.0
    # preserves Phases 8-10's exact prior ev_after-only ranking; set it
    # above 0 to let a worse-EV, better-G candidate actually win selection.
    # Applied only to the selection score, not to the reported ev_after
    # field itself, so ev_after keeps meaning "expected PnL" everywhere
    # else (journaling, reports, tests).
    lambda_g: float = 0.0
