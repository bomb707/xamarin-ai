"""Candidate action generation and evaluation (Roadmap Phase 8).

"Generate taker quantities from depth levels and portfolio risk budget."
"Generate maker prices on the valid current tick grid and candidate
quantities." "For each action, simulate EV_after, G_after, Pi_U/Pi_D,
cost and operational penalties." "Reject hard-constraint violations."

EV_after uses one formula throughout, derived directly from Strategy doc
SS13's DeltaEV_U(x) = q*x - K_U(x): for any candidate fill with portfolio
deltas (delta_U, delta_D, delta_C),

    EV_after = q*delta_U + (1-q)*delta_D - delta_C

which reduces exactly to SS13's formula for a pure UP taker fill
(delta_D=0) and to SS14's EV_maker_UP core term (rho*x*(q_fill-p)) for a
maker fill once weighted by fill probability and evaluated with q_fill
instead of raw q (adverse selection, Phase 7). This keeps alpha and hedge
candidates, taker and maker, on one consistent economic footing rather than
several different ad hoc formulas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.maker import fill_probability, q_fill
from xamarinbot.execution.taker import walk_depth
from xamarinbot.feeds.base import BookLevel, BookSnapshot
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.types import CandidateAction, OrderMode
from xamarinbot.portfolio.exposure import ActiveOrderExposure
from xamarinbot.portfolio.math import (
    FillSimulationResult,
    OrderPurpose,
    RiskConstraints,
    delta_g_directional,
    directional_projected_g,
    evaluate_constraints,
    min_hedge_quantity,
)
from xamarinbot.portfolio.state import FeeConfig, Fill, LiquidityRole, PortfolioState, Side, apply_fill


def taker_quantities(levels: tuple[BookLevel, ...], max_levels: int) -> list[float]:
    """Cumulative depth at each of the first `max_levels` book levels - one
    quantity candidate per level, per the roadmap step. Kept as a small,
    independently-tested building block; `taker_sizing_boundaries` below
    is what candidate generation actually uses now (Phase 12B audit item
    8) - raw depth-level sums alone gave the optimizer nothing to choose
    from whenever even the first level's full size didn't fit the current
    risk/position/spend budget."""
    out = []
    cumulative = 0.0
    for level in levels[:max_levels]:
        cumulative += level.size
        out.append(cumulative)
    return out


@dataclass(frozen=True)
class TakerSizing:
    """Phase 12B audit items 8 (boundary-aware sizing) and 10 (worst-price
    protection), computed together in one pass since both fall out of the
    same level-by-level walk. `quantities` are the candidate order sizes
    to evaluate (deduped, ascending); `p_max` is the depth/marginal-edge-
    derived worst acceptable execution price - the price of the last book
    level whose own marginal edge still clears `cfg.min_marginal_edge` -
    meant to replace a hardcoded `limit_price=1.0` (which is not real
    worst-price protection, since prices are probabilities in [0,1] and
    1.0 is effectively unconstraining).

    `max_execution_price_by_qty` (Phase 12B Tranche 1.2 item 3) maps each
    of `quantities` to its OWN hard, risk-safe execution price limit -
    `min(p_max, taker_max_execution_price(...))` - since `p_max` alone
    does not protect `g_min`/`spend_cap` against an adverse repricing
    during a taker delay window (a deeper, pricier level can individually
    clear `min_marginal_edge` while collectively breaching the hard risk
    floor once revalidated against a moved book). Callers evaluating a
    specific quantity should use this map, not the shared `p_max`, as the
    candidate's actual submitted/evaluated limit price."""

    quantities: tuple[float, ...]
    p_max: float | None
    max_execution_price_by_qty: dict[float, float] = field(default_factory=dict)


def _risk_boundary_step(
    u: float, d: float, c: float, g_min: float, side: Side, x0: float,
    cum_qty: float, cum_cost: float, level_size: float, c_i: float,
) -> tuple[float, bool]:
    """Advances the exact risk-budget-feasible quantity through one book
    level, using `directional_projected_g` (G_U(x)=min(U+x,D)-[C+K_U(x)],
    G_D symmetric) instead of a flat G_current-g_min budget (Phase 12B
    Tranche 1.1 item 5). `x0` is the breakpoint where `side`'s shares
    catch up to the other side (`max(0, D-U)` for UP, `max(0, U-D)` for
    DOWN) - G_side(x) is non-decreasing for x<=x0 (buying still-scarcer
    inventory raises min(U,D) itself) and non-increasing for x>x0 (once
    `side` is no longer the minimum, buying more only adds cost), so it is
    unimodal with its peak at x0 and, once it drops below g_min past that
    peak, every deeper level only makes it worse - safe to stop walking.

    `cum_qty`/`cum_cost` are this walk's state at the START of this level
    (before adding it); `c_i` is this level's fee-inclusive per-share
    cost. Returns `(feasible_qty_through_this_level, keep_walking)`.
    """

    def g_at(x: float) -> float:
        k_x = cum_cost + (x - cum_qty) * c_i
        return directional_projected_g(u, d, c, side, x, k_x)

    level_end = cum_qty + level_size
    kink = min(max(x0, cum_qty), level_end)  # clamp the peak into this level's own range

    g_start = g_at(cum_qty)
    if g_start < g_min - 1e-12:
        return cum_qty, False  # already infeasible entering this level

    g_kink = g_at(kink) if kink > cum_qty else g_start
    if kink > cum_qty and g_kink < g_min:
        # crosses within the non-decreasing sub-range (only possible if
        # c_i > 1, a pathological book state) - interpolate linearly.
        # Guard against a near-zero denominator: at an astronomically
        # small x0 (e.g. u and d differing only in the last few bits of
        # float precision), `kink` can end up numerically identical to
        # `cum_qty` after rounding, making g_kink == g_start exactly even
        # though the < g_min check above (with its own tolerance) still
        # fired - in that degenerate case there is no real interpolation
        # to do, so just report no further feasible movement.
        denom = g_start - g_kink
        if denom <= 1e-15:
            return cum_qty, False
        frac = (g_start - g_min) / denom
        return cum_qty + frac * (kink - cum_qty), False

    g_end = g_at(level_end)
    if g_end >= g_min:
        return level_end, True  # whole level feasible (including any kink) - keep walking

    # crosses within the non-increasing sub-range [kink, level_end]
    ref_x, ref_g = (kink, g_kink) if kink > cum_qty else (cum_qty, g_start)
    denom = ref_g - g_end
    if denom <= 1e-15:
        return ref_x, False
    frac = (ref_g - g_min) / denom
    return ref_x + frac * (level_end - ref_x), False


def _invert_all_in_cost(c_max: float, fee_config: FeeConfig) -> float | None:
    """Largest `p` in `[0,1]` with `p + fee_config.taker_fee(1, p) <= c_max`,
    found by monotonic binary search rather than a hardcoded analytic
    inverse (Phase 12B Tranche 1.2 item 3) - stays correct under any
    `FeeConfig.taker_fee` formula, not coupled to the current
    `shares*rate*p*(1-p)` fallback specifically. Assumes the all-in-cost
    function `p -> p + feePerShare(p)` is non-decreasing in `p`, true of
    the standard fee formula at any realistic fee rate. Returns `None` if
    even `p=0` exceeds `c_max` (nothing is affordable)."""

    def all_in(p: float) -> float:
        return p + fee_config.taker_fee(1.0, p)

    if c_max < 0.0 or all_in(0.0) > c_max:
        return None
    lo, hi = 0.0, 1.0
    if all_in(hi) <= c_max:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if all_in(mid) <= c_max:
            lo = mid
        else:
            hi = mid
    return lo


def taker_max_execution_price(
    portfolio: PortfolioState,
    side: Side,
    x: float,
    q_effective: float,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
    tick_size: float,
) -> float | None:
    """Derives the hard, candidate-specific worst-price limit for a taker
    candidate of quantity `x` on `side` (Phase 12B Tranche 1.2 item 3):
    the largest execution price such that filling all `x` shares at (up
    to) that price stays both `g_min`-safe and `spend_cap`-safe, while
    still respecting the `min_marginal_edge` floor:

        K_s^G(x) = min(U+x,D) - C - g_min          (UP; DOWN symmetric via directional_projected_g)
        K^B      = spend_cap - C                    (only when spend_cap is set)
        K_max,s(x) = min(K_s^G(x), K^B)
        q_s = q_effective                            (q for UP, 1-q for DOWN, by this module's convention)
        c_max,s(x) = min(q_s - min_marginal_edge, K_max,s(x) / x)

    `c_max,s(x)` (the maximum fee-inclusive per-share cost) is then
    inverted into a raw exchange price via `_invert_all_in_cost` and
    floored to the valid tick grid. This replaces `TakerSizing.p_max`
    (the last book level whose own marginal edge still clears
    `min_marginal_edge`) as the actual submitted/evaluated limit price -
    `p_max` alone does not protect `g_min`/`spend_cap` against an adverse
    repricing during a taker delay window, since a deeper, more expensive
    level can individually clear `min_marginal_edge` while collectively
    breaching the hard risk floor once revalidated against a moved book.

    Returns `None` if `x` is not feasible at any positive price."""
    if x <= 0:
        return None
    k_g = directional_projected_g(portfolio.U, portfolio.D, portfolio.C, side, x, 0.0) - cfg.g_min
    k_max = k_g
    if cfg.spend_cap is not None:
        k_max = min(k_max, cfg.spend_cap - portfolio.C)
    if k_max <= 0:
        return None

    c_max = min(q_effective - cfg.min_marginal_edge, k_max / x)
    if c_max <= 0:
        return None

    p = _invert_all_in_cost(c_max, fee_config)
    if p is None or p <= 0:
        return None

    ticks = math.floor(p / tick_size + 1e-9)
    return max(0.0, round(ticks * tick_size, 10))


def purpose_aware_max_execution_price(
    portfolio: PortfolioState,
    side: Side,
    x: float,
    purpose: OrderPurpose,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
    tick_size: float,
    q_effective: float | None = None,
) -> float | None:
    """Purpose-aware worst-price limit (Phase 12B Tranche 2.1 item 4) -
    HEDGE and BUFFER_BUILD must never be evaluated/submitted with an
    effectively unconstrained `limit_price=1.0` (a bug: neither purpose's
    quantity is re-derived from the actual execution price, so a deep,
    expensive walk could silently blow through the repair target the
    quantity was sized for). This is the single execution-price function
    every purpose routes through:

    ALPHA: delegates to `taker_max_execution_price` unchanged - `K_max` is
    still bounded by marginal EV (`min_marginal_edge`), `g_min`, and
    `spend_cap`, exactly as before.

    HEDGE / BUFFER_BUILD: no marginal-edge term (both are explicitly
    allowed negative standalone EV - `_EDGE_MIN_EXEMPT_PURPOSES` - so
    gating their price on per-share EV would be wrong). BUFFER_BUILD adds
    the parity-headroom bound so the walk can never pay more than the
    settlement-geometry benefit it is buying:

        K_max^buffer(x) = min[ min(U+x,D)-min(U,D), min(U+x,D)-C-G_min, B-C ]  (UP; DOWN symmetric)

    HEDGE uses only the G_min/spend bound (its quantity is already sized
    to land at exactly `G_min` against the best-ask assumption at sizing
    time - this ceiling is what guarantees that repair target survives
    even if the walk executes worse than that assumption, instead of
    silently allowing execution up to 1.00):

        K_max^hedge(x) = min[ min(U+x,D)-C-G_min, B-C ]  (UP; DOWN symmetric)

    Returns `None` if `x` is not feasible at any positive price."""
    if x <= 0:
        return None
    if purpose is OrderPurpose.ALPHA:
        if q_effective is None:
            raise ValueError("q_effective is required for ALPHA")
        return taker_max_execution_price(portfolio, side, x, q_effective, fee_config, cfg, tick_size)

    u, d, c = portfolio.U, portfolio.D, portfolio.C
    g_before = portfolio.G
    # Phase 12B Tranche 2.2 item 1: once already in aggregate breach
    # (g_before < g_min), the ordinary g_min floor is unreachable by
    # construction - K_max would be negative for every x that doesn't
    # instantly restore g_min in one fill, the same empty-fill-subset-
    # style contradiction item 1 fixes for RiskView.admits(). Matching
    # _finalize's own breach-recovery relaxation, the effective floor
    # becomes g_before itself (must not make G worse) rather than the
    # unreachable g_min, so a genuine partial repair still gets a real,
    # usable price ceiling instead of being priced out of existence.
    g_floor = cfg.g_min if g_before >= cfg.g_min else g_before
    k_max = directional_projected_g(u, d, c, side, x, 0.0) - g_floor
    if purpose is OrderPurpose.BUFFER_BUILD:
        k_max = min(k_max, delta_g_directional(u, d, side, x, 0.0))
    if cfg.spend_cap is not None:
        k_max = min(k_max, cfg.spend_cap - c)
    if k_max <= 0:
        return None

    c_max = k_max / x
    if c_max <= 0:
        return None

    p = _invert_all_in_cost(c_max, fee_config)
    if p is None or p <= 0:
        return None

    ticks = math.floor(p / tick_size + 1e-9)
    return max(0.0, round(ticks * tick_size, 10))


def taker_sizing_boundaries(
    asks: tuple[BookLevel, ...],
    q_effective: float,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
    portfolio: PortfolioState,
    side: Side,
    tick_size: float = 0.01,
) -> TakerSizing:
    """Generates economically meaningful candidate quantities from the
    union of boundaries named in Phase 12B audit item 7: exchange minimum
    size, small quantity steps, CLOB depth boundaries, the marginal-edge
    boundary (item 10's worst-price protection), the risk-budget boundary
    (the exact side-aware `directional_projected_g` walk, Phase 12B
    Tranche 1.1 item 5 - `max_directional_spend` is exact only in the
    special case where `side` is already the non-minimum side and is not
    used here as a universal boundary, see its own docstring), the
    position-limit boundary, and the spend-cap boundary - including the
    *exact partial quantity* inside whichever boundary is tightest, not
    just whole-level sums (so e.g. a 500-share first level with only 7.4
    shares of feasible budget yields a 7.4-share candidate, not a 0- or
    500-share one).

    `q_effective` is `q` for a UP-side walk, `1-q` for a DOWN-side walk -
    the "success probability" for whichever side `asks` belongs to.
    `side` is which side of the portfolio (`U` or `D`) is being sized -
    used for both the position-capacity boundary and the exact risk-budget
    walk's own side-aware kernel evaluation.
    """
    if not asks:
        return TakerSizing((), None)

    side_position = portfolio.U if side is Side.UP else portfolio.D
    x0 = max(0.0, (portfolio.D - portfolio.U) if side is Side.UP else (portfolio.U - portfolio.D))

    cum_qty = 0.0
    cum_cost = 0.0
    marginal_edge_qty = 0.0
    p_max: float | None = None

    spend_budget = (cfg.spend_cap - portfolio.C) if cfg.spend_cap is not None else None
    position_capacity = (cfg.position_limit - side_position) if cfg.position_limit is not None else None

    spend_capacity_qty: float | None = None
    risk_budget_qty = 0.0
    risk_walk_done = False

    for level in asks:
        c_i = level.price + fee_config.taker_fee(1.0, level.price)  # fee-per-share at this level's price
        e_i = q_effective - c_i
        if e_i <= cfg.min_marginal_edge:
            break  # worst-price boundary: this and every worse (later) level isn't worth walking

        level_cost = level.size * c_i
        if spend_budget is not None and spend_capacity_qty is None and cum_cost + level_cost > spend_budget:
            remaining_budget = max(0.0, spend_budget - cum_cost)
            spend_capacity_qty = cum_qty + remaining_budget / c_i
        if not risk_walk_done:
            risk_budget_qty, keep_going = _risk_boundary_step(
                portfolio.U, portfolio.D, portfolio.C, cfg.g_min, side, x0, cum_qty, cum_cost, level.size, c_i
            )
            risk_walk_done = not keep_going

        p_max = level.price
        cum_qty += level.size
        cum_cost += level_cost
        marginal_edge_qty = cum_qty

    if spend_budget is not None and spend_capacity_qty is None:
        spend_capacity_qty = cum_qty  # budget never exhausted within the walked (edge-acceptable) levels

    boundaries = [marginal_edge_qty, risk_budget_qty]
    if position_capacity is not None:
        boundaries.append(max(0.0, position_capacity))
    if spend_capacity_qty is not None:
        boundaries.append(spend_capacity_qty)
    max_feasible = max(0.0, min(boundaries))

    quantities: set[float] = set()
    step = max(cfg.taker_qty_step, 1e-9)
    x = cfg.taker_min_size
    # Bounded to `taker_qty_grid_points` steps near the minimum, not a
    # dense grid across the whole feasible range - when the risk/spend
    # budget is loose, max_feasible can be in the hundreds or thousands
    # of shares, and a 1-unit-step grid across that whole range exploded
    # candidate counts into the thousands (caught by the MPC latency
    # tests timing out under exactly this condition). The grid's purpose
    # is fine-grained *small*-trade sizing; larger sizes are already
    # covered by the boundary quantities and depth-level candidates below.
    for _ in range(cfg.taker_qty_grid_points):
        if x >= max_feasible - 1e-9:
            break
        quantities.add(round(x, 6))
        x += step
    if max_feasible >= cfg.taker_min_size - 1e-9 and max_feasible > 1e-9:
        quantities.add(round(max_feasible, 6))  # the exact tightest-boundary partial quantity

    cum = 0.0
    for level in asks[: cfg.max_taker_depth_levels]:
        cum += level.size
        capped = min(cum, max_feasible)
        if capped >= cfg.taker_min_size - 1e-9 and capped > 1e-9:
            quantities.add(round(capped, 6))

    final_quantities = tuple(sorted(quantities))
    # Phase 12B Tranche 1.2 item 3: each quantity gets its OWN hard,
    # risk-safe execution price limit, not the shared depth/marginal-edge
    # p_max alone - see taker_max_execution_price's docstring.
    max_execution_price_by_qty: dict[float, float] = {}
    for qty in final_quantities:
        hard_price = taker_max_execution_price(portfolio, side, qty, q_effective, fee_config, cfg, tick_size)
        if hard_price is None:
            continue
        max_execution_price_by_qty[qty] = min(p_max, hard_price) if p_max is not None else hard_price

    return TakerSizing(final_quantities, p_max, max_execution_price_by_qty)


# Purposes exempt from the edge_min alpha-edge floor (Phase 12B Tranche
# 2B): both are explicitly allowed negative standalone EV in exchange for
# improving worst-case risk/settlement geometry - see _finalize's docstring.
# The same set is exactly the "recovery-capable" purposes _finalize's
# breach-recovery logic (Tranche 2.1 item 5) admits while G_before <
# g_min - reused as RiskView.admits()'s is_recovery_candidate flag
# (Tranche 2.2 item 1) via is_recovery_purpose() below.
_EDGE_MIN_EXEMPT_PURPOSES = (OrderPurpose.HEDGE, OrderPurpose.BUFFER_BUILD)


def is_recovery_purpose(purpose: OrderPurpose) -> bool:
    """Whether `purpose` is a portfolio-repair type allowed to admit
    partial breach-recovery (HEDGE/BUFFER_BUILD) rather than being
    prohibited outright while already below g_min (ALPHA) - the single
    source of truth every RiskView.admits() call site uses for its
    `is_recovery_candidate` argument, so aggregate admission and
    `_finalize`'s single-candidate breach-recovery check can never
    silently diverge on which purposes qualify."""
    return purpose in _EDGE_MIN_EXEMPT_PURPOSES


def maker_price_grid(best_bid: float, best_ask: float, tick_size: float, offsets_ticks: tuple[int, ...]) -> list[tuple[float, int]]:
    """(price, offset_ticks) pairs moving from the current best bid toward
    (but never at or past) the best ask, on the live tick grid - "Do not
    use a fixed number of ticks from BESTASK as a universal rule" (SS14) is
    respected by generating from the bid side, not a fixed BESTASK offset."""
    out = []
    for offset in offsets_ticks:
        price = round(best_bid + offset * tick_size, 10)
        if price < best_ask - 1e-12:
            out.append((price, offset))
    return out


@dataclass(frozen=True)
class MakerCandidateSpec:
    """One (price, quantity, TTL) maker candidate to evaluate (Phase 12B
    Tranche 2.1 item 8) - the common unit both the fixed-constant and
    dynamic maker paths in `OneStepController.decide()` produce, so the
    controller's own candidate-generation loop treats them identically
    regardless of source."""

    price: float
    offset_ticks: int
    qty: float
    ttl_s: float


# Effectively-unbounded upper search range for _maker_feasible_quantity's
# single-flat-price walk (Phase 12B Tranche 2.1 item 8) - a resting maker
# has no book-depth limit of its own the way a taker's walk does, so the
# real ceiling always comes from the g_min/spend_cap/position_limit
# constraints computed inside that walk, never from this cap itself.
_MAKER_QTY_SEARCH_CAP = 1e9


def _maker_feasible_quantity(
    u: float, d: float, c: float, g_min: float, side: Side, price: float,
    fee_config: FeeConfig, spend_cap: float | None, position_limit: float | None,
) -> float:
    """Phase 12B Tranche 2.1 item 8: the exact maximum resting-maker
    quantity at a FIXED price `p` such that, if it fully fills:

        G_after(x, p) >= G_min
        C + p*x + fee <= spend_cap
        position_after <= position_limit

    Corrects the prior draft's `qty = min(vol_damped_qty, portfolio.G -
    g_min)`, which compared a SHARE quantity directly against a DOLLAR
    budget (`G - g_min` is dollars; `qty` is shares) - dimensionally
    invalid whenever `price != 1.0`. A maker resting at one fixed price is
    equivalent to a single "infinite-depth level" at per-share cost
    `c_i = p + fee_per_share(p)` - reuses `_risk_boundary_step`'s own
    tested unimodal `G(x)` walk (the same shape a taker's multi-level
    depth walk has) rather than re-deriving the algebra."""
    c_i = price + fee_config.fee_for(LiquidityRole.MAKER, 1.0, price)
    if c_i <= 0:
        return 0.0
    x0 = max(0.0, (d - u) if side is Side.UP else (u - d))
    max_feasible, _ = _risk_boundary_step(u, d, c, g_min, side, x0, 0.0, 0.0, _MAKER_QTY_SEARCH_CAP, c_i)
    if spend_cap is not None:
        max_feasible = min(max_feasible, max(0.0, spend_cap - c) / c_i)
    if position_limit is not None:
        side_position = u if side is Side.UP else d
        max_feasible = min(max_feasible, max(0.0, position_limit - side_position))
    return max(0.0, max_feasible)


def dynamic_maker_candidates(
    q: float, side: Side, tau: float | None, sigma: float, portfolio: PortfolioState,
    best_bid: float, best_ask: float, tick_size: float, fee_config: FeeConfig, cfg: OneStepConfig,
) -> list[MakerCandidateSpec]:
    """Derives maker (price, quantity, TTL) candidates from `(q, tau,
    sigma, G, R, purpose)` instead of `OneStepConfig`'s fixed
    `maker_quantity`/`maker_horizon_s`/`maker_price_offsets_ticks`
    constants (Phase 12B Tranche 2D; quantity formula corrected, TTL
    floor-clamp bug fixed, in Tranche 2.1 item 8). STRUCTURAL, not a
    calibrated economic model - per item 37/Addendum J, no coefficient
    here has been tuned against synthetic PnL; gated behind
    `cfg.dynamic_maker` so the default (fixed) behavior is exactly Phases
    8-12B's.

    `tau` (Phase 12B Tranche 2.2 item 5): `None` means "not supplied /
    unknown" (no time constraint applied); `0` means "no remaining time at
    all" - a real, meaningful value in a 5-minute market, previously
    conflated with "not provided" by defaulting the parameter to `0.0`.
    `tau <= 0` (a REAL, supplied zero-or-negative remaining time) always
    suppresses maker generation entirely; `tau is None` never does.

    TTL: damped by volatility (a stale quote is riskier in a fast market),
    floored at 1.0, then capped by remaining round time (`tau`) LAST, so
    `TTL <= tau` is guaranteed whenever `tau` is known even when the
    volatility-damped value would otherwise floor above it. Item 8: if
    `tau` is known and positive but shorter than `cfg.min_maker_ttl_s`
    (the minimum sensible passive-order lifetime), NO maker candidate is
    generated at all this decision - the prior draft's `ttl = max(1.0,
    ttl)` alone could silently clamp the TTL back UP past the round's own
    remaining time instead, proposing a resting order that could never
    legitimately live out its horizon.

    Price offsets: widen with volatility - three offsets in calm
    conditions, growing with `sigma`, capped to avoid an unbounded grid.

    Quantity: for EACH price in the offset grid, `_maker_feasible_quantity`
    derives the exact G_min/spend/position-feasible maximum AT THAT PRICE
    (a maker's cost-per-share varies with its own quoted price, unlike a
    taker's fixed book levels) - not one dollar-budget-derived number
    reused across every price. A small quantity set per price (exchange
    minimum, a volatility-damped "typical size" point - preserving the
    original "larger resting orders carry more adverse-selection risk in
    a faster-moving market" intent, now correctly clamped against the
    price-aware feasible ceiling instead of a raw dollar budget - and the
    max feasible itself) lets the caller's own EV/fill-probability model
    (`evaluate_maker_candidate`) rank which quantity is actually worth it,
    rather than this function picking a single one."""
    if tau is not None and tau <= 0:
        return []
    if tau is not None and cfg.min_maker_ttl_s > 0 and tau < cfg.min_maker_ttl_s:
        return []

    vol = max(0.0, sigma)
    ttl = max(1.0, cfg.maker_horizon_s / (1.0 + vol))
    if tau is not None:
        ttl = min(ttl, tau)

    n_offsets = min(8, 3 + round(vol * 4))
    offsets = tuple(range(n_offsets))
    vol_damped_typical = cfg.maker_quantity / (1.0 + vol)

    u, d, c = portfolio.U, portfolio.D, portfolio.C
    specs: list[MakerCandidateSpec] = []
    for price, offset in maker_price_grid(best_bid, best_ask, tick_size, offsets):
        max_feasible = _maker_feasible_quantity(u, d, c, cfg.g_min, side, price, fee_config, cfg.spend_cap, cfg.position_limit)
        if max_feasible < cfg.taker_min_size - 1e-9:
            continue
        candidate_qtys = sorted({round(v, 6) for v in (cfg.taker_min_size, min(vol_damped_typical, max_feasible), max_feasible)})
        for qty in candidate_qtys:
            if qty < cfg.taker_min_size - 1e-9:
                continue
            specs.append(MakerCandidateSpec(price=price, offset_ticks=offset, qty=qty, ttl_s=ttl))
    return specs


def _risk_constraints(cfg: OneStepConfig) -> RiskConstraints:
    return RiskConstraints(g_min=cfg.g_min, p_min=cfg.p_min, spend_cap=cfg.spend_cap, position_limit=cfg.position_limit)


def _favored_side(q: float) -> Side:
    """Phase 12B audit item 10/11: previously inferred from payoff
    geometry (`Pi_U >= Pi_D`), which is arbitrary at a flat portfolio
    (`Pi_U == Pi_D == 0` defaults to UP) and has nothing to do with
    prediction. The favored side must instead reflect current predictive/
    executable opportunity - `q` (calibrated `P(UP)`) is the direct driver
    of both `DeltaEV_U ~ q*x` and `DeltaEV_D ~ (1-q)*x`, so `q >= 0.5`
    means UP currently has the larger raw per-share edge. This is a
    simplified first-correct step, not the fuller
    `argmax(BestDeltaJ_U, BestDeltaJ_D)` (which would additionally weigh
    each side's current best executable price/cost, not just q) - that
    requires visibility into the whole candidate table, which this
    per-candidate evaluation function doesn't have; deferred to when
    `p_min` (still dormant everywhere in this codebase) is ever actually
    activated for a real risk-policy reason."""
    return Side.UP if q >= 0.5 else Side.DOWN


def _finalize(
    action_id: str,
    purpose: OrderPurpose,
    side: Side | None,
    mode: OrderMode,
    price: float | None,
    qty: float,
    ttl_s: float | None,
    expected_fill: float,
    delta_U: float,
    delta_D: float,
    delta_C: float,
    delta_ev_raw: float,
    portfolio: PortfolioState,
    portfolio_after: PortfolioState,
    q: float,
    cfg: OneStepConfig,
    apply_edge_min: bool,
    expected_delta_g: float,
    max_execution_price: float | None = None,
) -> CandidateAction:
    result = FillSimulationResult(
        purpose=purpose,
        delta_U=delta_U,
        delta_D=delta_D,
        delta_C=delta_C,
        portfolio_before=portfolio,
        portfolio_after=portfolio_after,
        risk_contribution=portfolio_after.G - portfolio.G,
    )
    favored = _favored_side(q) if cfg.p_min is not None else None
    check = evaluate_constraints(result, _risk_constraints(cfg), favored_side=favored)
    violated = list(check.violated)

    delta_ev = delta_ev_raw - cfg.churn_penalty
    # Hedge and BUFFER_BUILD orders are explicitly allowed negative
    # standalone EV in exchange for improving worst-case risk (SS17:
    # "Hedge orders may have negative standalone EV if they efficiently
    # improve the whole portfolio risk. They must be labeled separately in
    # the journal and optimizer." - Phase 12B Tranche 2B extends the same
    # allowance to BUFFER_BUILD, which trades EV for settlement geometry
    # by the same design) - edge_min is an alpha-edge floor and does not
    # apply to either, regardless of what the caller passed.
    if apply_edge_min and purpose not in _EDGE_MIN_EXEMPT_PURPOSES and delta_ev < cfg.edge_min:
        violated.append("edge_min")

    # Phase 12B Tranche 2.1 item 5: breach-recovery semantics. The
    # ordinary hard rule (G_after >= g_min, enforced above via
    # evaluate_constraints) only makes sense when the portfolio was
    # already safe before this candidate - once already breached
    # (G_before < g_min), demanding full restoration in one fill would
    # reject every partial repair, including the best one available. In
    # that state: ALPHA (new speculative one-sided risk) is prohibited
    # outright regardless of its own numbers; every other purpose
    # (HEDGE/BUFFER_BUILD/REPAIR-type) is admitted precisely when it
    # improves G relative to where it started (G_after > G_before), even
    # if it can't reach g_min in a single fill, and rejected if it would
    # make things worse (G_after <= G_before) - "never allow G_after <
    # G_before" while already in recovery.
    g_before = portfolio.G
    g_after = portfolio_after.G
    if g_before < cfg.g_min:
        if purpose is OrderPurpose.ALPHA:
            if "breach_recovery_alpha_prohibited" not in violated:
                violated.append("breach_recovery_alpha_prohibited")
        else:
            if "g_min" in violated and g_after > g_before:
                violated.remove("g_min")
            if g_after <= g_before and "breach_recovery_no_improvement" not in violated:
                violated.append("breach_recovery_no_improvement")

    return CandidateAction(
        action_id=action_id,
        purpose=purpose,
        side=side,
        mode=mode,
        price=price,
        qty=qty,
        ttl_s=ttl_s,
        expected_fill=expected_fill,
        delta_ev=delta_ev,
        g_after=g_after,
        pi_u_after=portfolio_after.Pi_U,
        pi_d_after=portfolio_after.Pi_D,
        violated_constraints=tuple(violated),
        max_execution_price=max_execution_price,
        expected_delta_g=expected_delta_g,
    )


def evaluate_taker_candidate(
    action_id: str,
    side: Side,
    purpose: OrderPurpose,
    requested_qty: float,
    limit_price: float,
    asks: tuple[BookLevel, ...],
    portfolio: PortfolioState,
    q: float,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
) -> CandidateAction:
    walk = walk_depth(asks, requested_qty, limit_price, fee_config)
    fill = Fill(side=side, price=walk.avg_price if walk.filled_shares > 0 else limit_price, shares=walk.filled_shares, role=LiquidityRole.TAKER, fee=walk.total_fee)
    portfolio_after = apply_fill(portfolio, fill)

    delta_U = portfolio_after.U - portfolio.U
    delta_D = portfolio_after.D - portfolio.D
    delta_C = portfolio_after.C - portfolio.C
    delta_ev_raw = q * delta_U + (1.0 - q) * delta_D - delta_C
    # Phase 12B Tranche 2.1 item 6: a taker fill is deterministic (FAK
    # against the book, no probability draw involved) - its expected risk
    # contribution IS its full delta, not a fill-probability-weighted
    # fraction of it.
    expected_delta_g = portfolio_after.G - portfolio.G

    return _finalize(
        action_id, purpose, side, OrderMode.FAK, walk.avg_price if walk.filled_shares > 0 else limit_price,
        requested_qty, ttl_s=0.0, expected_fill=walk.filled_shares,
        delta_U=delta_U, delta_D=delta_D, delta_C=delta_C, delta_ev_raw=delta_ev_raw,
        portfolio=portfolio, portfolio_after=portfolio_after, q=q, cfg=cfg, apply_edge_min=True,
        expected_delta_g=expected_delta_g, max_execution_price=limit_price,
    )


def evaluate_maker_candidate(
    action_id: str,
    side: Side,
    purpose: OrderPurpose,
    price: float,
    qty: float,
    distance_to_touch_ticks: float,
    queue_ahead_shares: float,
    horizon_s: float,
    portfolio: PortfolioState,
    q: float,
    exec_cfg: ExecutionConfig,
    cfg: OneStepConfig,
) -> CandidateAction:
    """Constraint checks use the *if-filled* portfolio (conservative: a
    maker order that would breach G_min if it actually fills is rejected
    now, not left to be caught after the fact - Strategy doc SS16's "risk
    breach: projected fill would push G below G_min -> Cancel/shrink" is
    exactly this check, applied pre-emptively)."""
    fill = Fill(side=side, price=price, shares=qty, role=LiquidityRole.MAKER, fee=0.0)
    portfolio_after_if_filled = apply_fill(portfolio, fill)

    delta_U = portfolio_after_if_filled.U - portfolio.U
    delta_D = portfolio_after_if_filled.D - portfolio.D
    delta_C = portfolio_after_if_filled.C - portfolio.C

    rho = fill_probability(distance_to_touch_ticks, queue_ahead_shares, horizon_s, exec_cfg.maker)
    qf = q_fill(q, side, exec_cfg.maker)
    ev_if_filled = qf * delta_U + (1.0 - qf) * delta_D - delta_C
    delta_ev_raw = rho * ev_if_filled - cfg.opportunity_cost
    # Phase 12B Tranche 2.1 item 6: a maker only realizes its G contribution
    # with probability rho - using the unweighted if-filled delta here
    # (the bug this item fixes) would let the selection score treat an
    # unlikely-to-fill maker as if its risk contribution were certain,
    # exactly the "maker soft-risk weighting" the reviewer's fix targets.
    expected_delta_g = rho * (portfolio_after_if_filled.G - portfolio.G)

    return _finalize(
        action_id, purpose, side, OrderMode.POST_ONLY, price, qty, ttl_s=horizon_s, expected_fill=rho * qty,
        delta_U=delta_U, delta_D=delta_D, delta_C=delta_C, delta_ev_raw=delta_ev_raw,
        portfolio=portfolio, portfolio_after=portfolio_after_if_filled, q=q, cfg=cfg, apply_edge_min=True,
        expected_delta_g=expected_delta_g,
    )


@dataclass(frozen=True)
class ReplacementPlan:
    """A concrete, re-evaluated REPLACE proposal for an open maker order
    (Phase 12B Tranche 2.1 item 9) - "re-evaluate the current book and
    create a real ReplacementPlan" instead of leaving
    `current_optimal_ev=None` (which silently made REPLACE unreachable
    through `OrderSupervisor.review_order` in every integrated harness).
    `exposure` is the plan's own `ActiveOrderExposure`, ready for a
    `RiskView.admits()` hard-admission check before it is actually
    dispatched - a REPLACE must clear the same aggregate risk bar any
    other new order would (item 2)."""

    side: Side
    price: float
    qty: float
    ttl_s: float
    delta_ev: float
    expected_delta_g: float
    exposure: ActiveOrderExposure


def evaluate_replacement_plan(
    side: Side,
    remaining_shares: float,
    best_bid: float,
    best_ask: float,
    tick_size: float,
    offsets_ticks: tuple[int, ...],
    horizon_s: float,
    portfolio: PortfolioState,
    q: float,
    exec_cfg: ExecutionConfig,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
) -> ReplacementPlan | None:
    """Re-evaluates every price on the current maker grid for `side` (via
    `evaluate_maker_candidate`, the same EV/fill-probability model any
    other maker candidate uses) and returns the highest-`delta_ev` valid
    tick as a `ReplacementPlan` - the real "best alternative tick" a
    REPLACE decision needs, rather than blindly taking the first (tightest)
    grid price. Returns `None` if the book has no valid grid at all, or
    every grid price is itself hard-constraint-invalid (e.g. would breach
    g_min if filled) - REPLACE is then simply not offered this review."""
    best: CandidateAction | None = None
    for price, offset in maker_price_grid(best_bid, best_ask, tick_size, offsets_ticks):
        c = evaluate_maker_candidate(
            "replacement_eval", side, OrderPurpose.ALPHA, price, remaining_shares,
            distance_to_touch_ticks=float(offset), queue_ahead_shares=0.0, horizon_s=horizon_s,
            portfolio=portfolio, q=q, exec_cfg=exec_cfg, cfg=cfg,
        )
        if not c.is_valid:
            continue
        if best is None or c.delta_ev > best.delta_ev:
            best = c
    if best is None:
        return None
    fee = fee_config.fee_for(LiquidityRole.MAKER, best.qty, best.price)
    exposure = ActiveOrderExposure(side, best.qty, best.qty * best.price + fee)
    return ReplacementPlan(
        side=side, price=best.price, qty=best.qty, ttl_s=horizon_s,
        delta_ev=best.delta_ev, expected_delta_g=best.expected_delta_g, exposure=exposure,
    )


def generate_hedge_candidate(
    action_id: str,
    portfolio: PortfolioState,
    asks_up: tuple[BookLevel, ...],
    asks_down: tuple[BookLevel, ...],
    q: float,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
    tick_size: float = 0.01,
) -> CandidateAction | None:
    """Portfolio-repair / hedge candidate (Strategy doc SS17, Roadmap
    Phase 11 / SS20.1 ablation #5 "Lead-lag + CLOB without portfolio
    repair" vs #6+ "with portfolio control"). None of Phases 8-10 ever
    generated one of these - `portfolio/math.py`'s hedge formulas
    (`min_hedge_quantity`, `max_hedge_quantity`, `hedge_efficiency`) have
    existed since Phase 3 but were only ever unit-tested in isolation
    until this function actually calls them from the candidate pipeline.

    SS17 states the UP-favored case explicitly
    (`x_min_hedge = max(0, (-L_max - Pi_DOWN) / (1 - c_D))`); the
    DOWN-favored case here is a symmetric, undocumented-in-the-source-docs
    mirror (buy UP to restore the floor), the same kind of extension used
    for Phase 6's regime matrix. `L_max` (max acceptable loss) is taken as
    `-cfg.g_min` - the same worst-case floor concept G_min already
    represents elsewhere. `c_D`/`c_U` (all-in per-share hedge cost) uses
    the opposite side's best ask *plus* its taker fee at that price
    (`FeeConfig.taker_fee`, per-share) rather than the raw price alone -
    using raw price here first, then discovering the resulting candidate's
    actual (fee-inclusive) G_after landed just short of g_min because the
    formula never budgeted for the fee, is what motivated computing an
    effective cost instead.

    Returns None if the portfolio is already balanced or already meets
    the floor (`x_min_hedge <= 0` - no repair needed) or the opposite
    side's book has no ask to hedge against.
    """
    if math.isclose(portfolio.Pi_U, portfolio.Pi_D):
        return None

    l_max = -cfg.g_min
    if portfolio.Pi_U > portfolio.Pi_D:
        # UP favored, DOWN is the worst case - hedge by buying DOWN (SS17, as stated)
        side, asks = Side.DOWN, asks_down
        price = asks[0].price if asks else None
        if price is None or price >= 1.0:
            return None
        c_effective = price + fee_config.taker_fee(1.0, price)
        x_min = min_hedge_quantity(l_max, portfolio.Pi_D, c_effective)
    else:
        # DOWN favored, UP is the worst case - symmetric mirror
        side, asks = Side.UP, asks_up
        price = asks[0].price if asks else None
        if price is None or price >= 1.0:
            return None
        c_effective = price + fee_config.taker_fee(1.0, price)
        x_min = min_hedge_quantity(l_max, portfolio.Pi_U, c_effective)

    if x_min <= 0:
        return None  # already at/above the floor, no repair needed

    # x_min sizes the hedge to land *exactly* on the g_min boundary; chained
    # floating-point arithmetic (division in min_hedge_quantity, then the
    # fee formula again inside evaluate_taker_candidate) can put the actual
    # result a few ULPs on the wrong side of that boundary, which
    # evaluate_constraints' strict `<` then rejects for essentially no
    # reason. A tiny relative safety margin avoids that without touching
    # Phase 3's general (and already covered by property tests) constraint
    # comparison.
    x_min *= 1.0001

    # Phase 12B Tranche 2.1 item 4: HEDGE must never execute at an
    # effectively unconstrained limit_price=1.0 - x_min was sized against
    # the best ask's own price, but nothing previously stopped the actual
    # FAK walk from filling worse than that (a thinner top level, or a
    # book that moved), silently blowing through the g_min repair target
    # x_min was computed to hit. This ceiling guarantees the repair target
    # survives even under adverse execution.
    max_price = purpose_aware_max_execution_price(portfolio, side, x_min, OrderPurpose.HEDGE, fee_config, cfg, tick_size)
    if max_price is None or max_price <= 0:
        return None

    return evaluate_taker_candidate(
        action_id, side, OrderPurpose.HEDGE, x_min, limit_price=max_price, asks=asks,
        portfolio=portfolio, q=q, fee_config=fee_config, cfg=cfg,
    )


def delta_ev_directional(side: Side, x: float, k_x: float, q: float) -> float:
    """ΔEV_U(x) = q*x - K_U(x); ΔEV_D(x) = (1-q)*x - K_D(x) (Phase 12B
    Tranche 2B) - computed independently of `delta_g_directional`
    (`portfolio/math.py`), which stays prediction-free and never sees
    `q`. For a one-sided fill this is provably identical to the module's
    general `q*delta_U + (1-q)*delta_D - delta_C` formula (delta_D=0 for
    a pure UP buy reduces it to exactly this), so `evaluate_taker_candidate`
    doesn't need a separate code path to get the same number - this
    function exists so the identity is explicit and independently
    testable, matching the reviewer's "independently calculate" ask.
    ΔG and ΔEV must never be conflated: `ΔG>0, ΔEV<0` is a real,
    non-contradictory possibility (Addendum G) - a BUFFER_BUILD
    candidate is allowed exactly that combination."""
    if side is Side.UP:
        return q * x - k_x
    return (1.0 - q) * x - k_x


def generate_buffer_build_candidates(
    action_id_prefix: str,
    portfolio: PortfolioState,
    asks_up: tuple[BookLevel, ...],
    asks_down: tuple[BookLevel, ...],
    q: float,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
    tick_size: float = 0.01,
) -> list[CandidateAction]:
    """BUFFER_BUILD candidates (Phase 12B Tranche 2B, Strategy doc SS17;
    redesigned into a bounded multi-quantity set in Tranche 2.1 item 7):
    proactively accumulate cheap opposite-side inventory to improve
    settlement geometry, generated independent of any `G < g_min` breach -
    unlike `generate_hedge_candidate`, which only ever fires once already
    at/below the risk floor. Buys the currently underrepresented side
    (`U <= D` -> buy UP, else buy DOWN) - buying that side raises
    `min(U,D)`; buying the already-overrepresented side never can (see
    `delta_g_directional`'s proof).

    Item 7 correction: the prior draft generated exactly ONE candidate, at
    the single quantity where `ΔG_side(x)` peaks. That silently assumed
    "buy the maximum parity-improving amount" is always the best choice,
    when a smaller quantity might have better EV per share, or a tighter
    risk/spend budget might make the peak infeasible for other reasons.
    This generates a whole BOUNDED candidate SET instead - exchange
    minimum, a small-quantity grid, book-depth boundaries, the spend
    boundary, the `g_min` risk boundary, and the parity boundary (the
    `ΔG=0` crossing, `x0 = |U-D|`'s own peak) - evaluates `ΔEV(x)` and
    `ΔG(x)` independently at each, and keeps only `ΔG(x) > 0` (SS17: a
    BUFFER_BUILD trade whose cost exceeds its parity benefit is never
    generated at all). The controller's own `argmax` then picks whichever
    quantity is actually worth it - 5, 20, 50, or the full parity gap -
    rather than this function assuming more is always better.

    Every returned candidate's own `delta_ev` may be negative (SS17:
    BUFFER_BUILD trades expected value for improved worst-case
    settlement, same allowance as HEDGE) - `edge_min` does not apply (see
    `_EDGE_MIN_EXEMPT_PURPOSES`)."""
    side = Side.UP if portfolio.U <= portfolio.D else Side.DOWN
    asks = asks_up if side is Side.UP else asks_down
    if not asks:
        return []

    u, d, c = portfolio.U, portfolio.D, portfolio.C
    x0 = max(0.0, (d - u) if side is Side.UP else (u - d))
    if x0 <= 0:
        return []  # already at parity or overrepresented - no buffer opportunity in this direction

    g_current = min(u, d) - c
    spend_budget = (cfg.spend_cap - c) if cfg.spend_cap is not None else None

    cum_qty = 0.0
    cum_cost = 0.0
    risk_boundary_qty = 0.0
    risk_walk_done = False
    parity_boundary_qty = 0.0
    parity_walk_done = False
    spend_capacity_qty: float | None = None
    depth_points: list[float] = []

    for level in asks[: cfg.max_taker_depth_levels]:
        c_i = level.price + fee_config.taker_fee(1.0, level.price)
        level_cost = level.size * c_i
        if spend_budget is not None and spend_capacity_qty is None and cum_cost + level_cost > spend_budget:
            remaining_budget = max(0.0, spend_budget - cum_cost)
            spend_capacity_qty = cum_qty + remaining_budget / c_i
        if not risk_walk_done:
            risk_boundary_qty, keep_going = _risk_boundary_step(u, d, c, cfg.g_min, side, x0, cum_qty, cum_cost, level.size, c_i)
            risk_walk_done = not keep_going
        if not parity_walk_done:
            parity_boundary_qty, keep_going = _risk_boundary_step(u, d, c, g_current, side, x0, cum_qty, cum_cost, level.size, c_i)
            parity_walk_done = not keep_going
        cum_qty += level.size
        cum_cost += level_cost
        depth_points.append(cum_qty)

    if spend_budget is not None and spend_capacity_qty is None:
        spend_capacity_qty = cum_qty

    # ΔG(x) > 0 only inside (0, min(x0, parity_boundary_qty)] - beyond the
    # parity boundary, cost exceeds the parity benefit even at the peak
    # (delta_g_directional's proof), so this is a hard prefilter, not the
    # g_min risk boundary (which stays a candidate QUANTITY in the set
    # below, not a truncation - a partial buffer build below the g_min
    # boundary is still a legitimate, independently-evaluated candidate;
    # _finalize's own hard-constraint/breach-recovery check is what
    # ultimately decides admissibility, not this generator).
    max_feasible = min(x0, parity_boundary_qty)
    if spend_capacity_qty is not None:
        max_feasible = min(max_feasible, spend_capacity_qty)
    max_feasible = max(0.0, max_feasible)
    if max_feasible <= 0:
        return []

    # Phase 12B Tranche 2.2 item 4: the 0.999 float-safety margin must
    # apply ONLY to quantities anchored to a genuine upper boundary
    # (parity/risk/spend), never to the exchange minimum, an interior
    # small-quantity-grid point, or a depth point strictly below the
    # feasible ceiling - margining every quantity uniformly (the prior
    # draft's bug) turned `raw_qty=taker_min_size=1.0` into `qty=0.999`,
    # which the very next check then rejected as below the exchange
    # minimum, silently discarding the exact-minimum candidate even when
    # it was individually delta_g>0-feasible.
    exact_quantities: set[float] = set()
    exact_quantities.add(min(cfg.taker_min_size, max_feasible))  # exchange minimum - never margined
    step = max(cfg.taker_qty_step, 1e-9)
    x = cfg.taker_min_size
    for _ in range(cfg.taker_qty_grid_points):  # small-quantity grid - interior points, exact
        if x >= max_feasible - 1e-9:
            break
        exact_quantities.add(x)
        x += step
    for dp in depth_points:  # book-depth boundaries - exact where strictly interior
        capped = min(dp, max_feasible)
        if capped < max_feasible - 1e-9:
            exact_quantities.add(capped)

    boundary_quantities: set[float] = {max_feasible}  # the parity (or spend-capped) boundary itself
    if spend_capacity_qty is not None:
        boundary_quantities.add(min(spend_capacity_qty, max_feasible))
    if risk_boundary_qty > 0:
        boundary_quantities.add(min(risk_boundary_qty, max_feasible))

    quantities: set[float] = set()
    for raw_qty in exact_quantities:
        if raw_qty >= cfg.taker_min_size - 1e-9:
            quantities.add(raw_qty)
    for raw_qty in boundary_quantities:
        # Safety margin (matching generate_hedge_candidate's own
        # x_min*1.0001 pattern) - a quantity landing exactly on the
        # parity/risk/spend boundary can end up a few ULPs on the wrong
        # side of it after chained floating-point arithmetic. Applied
        # only here, never to an already-exact quantity above.
        margined = raw_qty * 0.999
        if margined >= cfg.taker_min_size - 1e-9:
            quantities.add(margined)

    candidates: list[CandidateAction] = []
    idx = 0
    for qty in sorted(quantities):
        if qty <= 0:
            continue
        max_price = purpose_aware_max_execution_price(portfolio, side, qty, OrderPurpose.BUFFER_BUILD, fee_config, cfg, tick_size)
        if max_price is None or max_price <= 0:
            continue
        walk = walk_depth(asks, qty, limit_price=max_price, fee_config=fee_config)
        if walk.filled_shares <= 0:
            continue
        k_x = walk.total_paid
        delta_g = delta_g_directional(u, d, side, walk.filled_shares, k_x)
        if delta_g <= 0:
            continue  # keep only ΔG>0 (SS17) - cost exceeds the parity benefit at this quantity
        idx += 1
        candidates.append(evaluate_taker_candidate(
            f"{action_id_prefix}_{idx}", side, OrderPurpose.BUFFER_BUILD, walk.filled_shares, limit_price=max_price,
            asks=asks, portfolio=portfolio, q=q, fee_config=fee_config, cfg=cfg,
        ))
    return candidates


def candidate_exposure(candidate: CandidateAction, fee_config: FeeConfig) -> ActiveOrderExposure | None:
    """Converts a not-yet-submitted `CandidateAction` into the same
    `ActiveOrderExposure` shape `portfolio/exposure.py` uses for already-
    pending taker/maker orders (Phase 12B Tranche 2.1 items 2/3) - the
    single conversion every dispatch site and `OneStepController.decide`
    itself route through before checking `RiskView.admits()`, so a
    not-yet-submitted candidate is judged by the exact same worst-case
    convention an already-active order would be.

    FAK (taker): worst case is the full REQUESTED quantity filling at its
    own hard price ceiling (`max_execution_price`, falling back to
    `price`) - mirrors `exposure_from_pending_takers`'s convention for an
    order that has already been submitted and is awaiting delayed
    resolution, since a delayed fill can still consume the full requested
    size at that ceiling.

    POST_ONLY (maker): worst case is the full quoted quantity filling at
    its quoted price - mirrors `exposure_from_open_maker_orders`.

    Returns `None` for WAIT (no exposure) or a candidate missing the
    price/side information needed to build one."""
    if candidate.side is None or candidate.qty <= 0:
        return None
    if candidate.mode is OrderMode.FAK:
        price = candidate.max_execution_price if candidate.max_execution_price is not None else candidate.price
        if price is None:
            return None
        fee = fee_config.taker_fee(candidate.qty, price)
        return ActiveOrderExposure(candidate.side, candidate.qty, candidate.qty * price + fee)
    if candidate.mode is OrderMode.POST_ONLY:
        if candidate.price is None:
            return None
        fee = fee_config.fee_for(LiquidityRole.MAKER, candidate.qty, candidate.price)
        return ActiveOrderExposure(candidate.side, candidate.qty, candidate.qty * candidate.price + fee)
    return None


def candidate_selection_score(candidate: CandidateAction, cfg: OneStepConfig) -> float:
    """SS18: `J(a) = ΔEV(a) + lambda_G*E[ΔG(a)] - selectionPenalty(a)`
    (Phase 12B Tranche 2.1 item 6, made the single reusable function every
    ranking caller must use in Tranche 2.2 item 2) - `OneStepController`'s
    own `argmax` and `MPCController`'s immediate sequence value must never
    silently diverge on what "best" means. The prior MPC draft used
    `sequence_value = candidate.delta_ev` alone, ignoring `lambda_g` and
    `selection_penalty` entirely - a duplicated, drifted-from-one-step
    objective this function closes."""
    return candidate.delta_ev + cfg.lambda_g * candidate.expected_delta_g - candidate.selection_penalty


def wait_candidate(action_id: str, portfolio: PortfolioState) -> CandidateAction:
    """Roadmap Phase 8: "include WAIT as a candidate with value 0 /
    continuation estimate." Always valid - the optimizer must always have a
    safe fallback."""
    return CandidateAction(
        action_id=action_id,
        purpose=OrderPurpose.ALPHA,
        side=None,
        mode=OrderMode.WAIT,
        price=None,
        qty=0.0,
        ttl_s=None,
        expected_fill=0.0,
        delta_ev=0.0,
        g_after=portfolio.G,
        pi_u_after=portfolio.Pi_U,
        pi_d_after=portfolio.Pi_D,
        violated_constraints=(),
    )
