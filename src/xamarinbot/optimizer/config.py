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
    # floor on total candidate delta_ev for directional (non-WAIT)
    # candidates - this is "min_total_delta_ev" in Phase 12B audit item
    # 6/7's terminology (kept as `edge_min` rather than renamed, to limit
    # blast radius across the ~14 files that already reference it under
    # this name; see docs/PHASE_12B_AUDIT.md for the scope note).
    edge_min: float = 0.0

    # Phase 12B audit item 6/7: min_marginal_edge is a genuinely separate
    # concept from edge_min above - a *per-share* floor
    # (e_U,i = q - c_i, e_D,i = (1-q) - c_i at ask level i, c_i = price +
    # fee-per-share) so a large trade with poor per-share edge can't pass
    # merely because its total dollar delta_ev clears edge_min, and a
    # small trade with excellent per-share edge isn't rejected merely
    # because its total dollar delta_ev is small. Also doubles as the
    # taker worst-price boundary (item 10): a depth level is only walked
    # while its own marginal edge still clears this floor - "q > c_marginal
    # + safetyMargin" from the audit's formula is exactly
    # "e_i > min_marginal_edge" once whatever safety margin is wanted is
    # folded into this single threshold, rather than a second field.
    min_marginal_edge: float = 0.0

    # Phase 12B audit item 8: boundary-quantity generation inputs for
    # taker candidates. `taker_qty_step` is the configured small-quantity
    # grid spacing.
    #
    # `taker_min_size` USED TO LIVE HERE, defaulting to 1.0, described in
    # its own comment as a fallback for the real exchange minimum "once
    # wired in". Phase 12C.1 item 11 removed it: every BTC five-minute
    # market sampled in the Phase 12C captures reports a minimum of 5.0
    # SHARES, so the static 1.0 was not merely un-wired but wrong by 5x, in
    # the direction that generates orders the venue rejects. The executable
    # minimum now comes from `MarketConstraints.min_order_shares`, read per
    # round from that market's own metadata and passed down the call chain -
    # it is deliberately NOT copied back into this static config, which
    # would recreate a second, staleable source of truth (item 12).
    taker_qty_step: float = 1.0
    # Caps the small-quantity grid to this many steps near the market
    # minimum,
    # regardless of how large the feasible range turns out to be - an
    # unbounded dense grid exploded candidate counts (and MPC latency)
    # whenever the risk/spend budget was loose. The boundary quantities
    # (marginal-edge, risk-budget, spend-cap, position-limit, depth-level)
    # already cover the rest of the range.
    taker_qty_grid_points: int = 5

    # taker candidate generation: how many cumulative depth levels to
    # generate as separate quantity candidates (Roadmap Phase 8: "Generate
    # taker quantities from depth levels and portfolio risk budget") -
    # kept alongside the item 8 boundary-derived quantities below as an
    # additional, not replaced, source of candidates.
    max_taker_depth_levels: int = 3

    # maker candidate generation: price offsets (in ticks, toward the
    # touch) from the best bid/ask to generate as candidates (Roadmap Phase
    # 8: "Generate maker prices on the valid current tick grid").
    maker_price_offsets_ticks: tuple[int, ...] = (0, 1, 2)
    maker_quantity: float = 20.0
    maker_horizon_s: float = 10.0

    # SS18 operational penalties (lambda_churn*OrderChurn style terms) -
    # flat, uncalibrated placeholders subtracted from delta_ev for any
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

    # Phase 12B Tranche 2B: SS17 BUFFER_BUILD candidates - proactively
    # accumulate cheap opposite-side inventory to improve settlement
    # geometry, generated independent of any G<g_min breach (unlike
    # HEDGE, which only reacts once already in trouble). Defaults False
    # so every existing ablation/demo/test is unaffected until explicitly
    # enabled - this is new candidate-generation behavior, not a bugfix.
    enable_buffer_build: bool = False

    # Phase 12B Tranche 2C: soft regime control - when True, a regime that
    # would otherwise hard-gate out a candidate family instead applies a
    # continuous prior/penalty (regime_prior_penalty, subtracted from the
    # selection score) rather than never generating the candidate at all.
    # Defaults False so the existing hard-gate ablations are unaffected;
    # kept as an explicit, separately-selectable ablation mode per the
    # prompt's own instruction ("keep current hard regime matrix as an
    # ablation").
    soft_regime: bool = False
    # STRUCTURAL / UNCALIBRATED (Phase 12B Tranche 2.1 item 11): a flat
    # dollar penalty subtracted from the selection score, NOT a calibrated
    # prior - a fixed dollar amount has quantity-scale dependence (the
    # same $5 penalty is decisive against a 1-share candidate and
    # meaningless against a 1000-share one), which a genuine prior
    # shouldn't. Left as a flat placeholder deliberately, pending real
    # data to determine whether the correct representation is a per-share
    # penalty, a probability/logit adjustment, or a state-dependent EV
    # penalty - none of which should be guessed and tuned against this
    # repository's synthetic PnL.
    regime_prior_penalty: float = 0.0

    # Phase 12B Tranche 2D: dynamic maker price/quantity/TTL - when True,
    # `dynamic_maker_candidates` derives (price, quantity, TTL) from
    # (q, tau, sigma, G, R) instead of the fixed maker_quantity/
    # maker_horizon_s constants below, and price offsets widen with
    # volatility instead of using a fixed tick grid. Defaults False so
    # every existing ablation/demo/test is unaffected until explicitly
    # enabled - only after aggregate hard-risk admission (Tranche 2A) is
    # wired, per the prompt's own sequencing. Explicitly NOT calibrated
    # against synthetic PnL (item 37/Addendum J) - a structural placeholder
    # for real-data calibration.
    dynamic_maker: bool = False
    # Phase 12B Tranche 2.1 item 8: the minimum passive-order lifetime
    # considered sensible to propose at all - when `cfg.dynamic_maker=True`
    # and the round's remaining time (`tau`) is positive but shorter than
    # this, no maker candidate is generated for that decision (rather than
    # clamping the TTL back up past the round's own remaining time, the
    # bug this item fixes). Only consulted in dynamic-maker mode; the
    # fixed-constant path is unaffected regardless of this value.
    min_maker_ttl_s: float = 1.0

    # SS18's actual objective is J = E[PnL_T] + lambda_G*G_T - ... , not
    # plain EV - selection under pure delta_ev can *never* pick a hedge
    # candidate (SS17 hedges have negative standalone EV by construction,
    # so they always lose to WAIT at lambda_g=0), which would make
    # enable_portfolio_repair wired in but functionally inert. lambda_g=0.0
    # preserves Phases 8-10's exact prior delta_ev-only ranking; set it
    # above 0 to let a worse-EV, better-G candidate actually win selection.
    # Applied only to the selection score, not to the reported delta_ev
    # field itself, so delta_ev keeps meaning "expected PnL" everywhere
    # else (journaling, reports, tests).
    lambda_g: float = 0.0
