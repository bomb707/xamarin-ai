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
from dataclasses import dataclass

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.maker import fill_probability, q_fill
from xamarinbot.execution.taker import walk_depth
from xamarinbot.feeds.base import BookLevel, BookSnapshot
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.types import CandidateAction, OrderMode
from xamarinbot.portfolio.math import (
    FillSimulationResult,
    OrderPurpose,
    RiskConstraints,
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
    to evaluate (deduped, ascending); `p_max` is the derived worst
    acceptable execution price - the price of the last book level whose
    own marginal edge still clears `cfg.min_marginal_edge` - meant to
    replace a hardcoded `limit_price=1.0` (which is not real worst-price
    protection, since prices are probabilities in [0,1] and 1.0 is
    effectively unconstraining)."""

    quantities: tuple[float, ...]
    p_max: float | None


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
        frac = (g_start - g_min) / (g_start - g_kink)
        return cum_qty + frac * (kink - cum_qty), False

    g_end = g_at(level_end)
    if g_end >= g_min:
        return level_end, True  # whole level feasible (including any kink) - keep walking

    # crosses within the non-increasing sub-range [kink, level_end]
    ref_x, ref_g = (kink, g_kink) if kink > cum_qty else (cum_qty, g_start)
    frac = (ref_g - g_min) / (ref_g - g_end)
    return ref_x + frac * (level_end - ref_x), False


def taker_sizing_boundaries(
    asks: tuple[BookLevel, ...],
    q_effective: float,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
    portfolio: PortfolioState,
    side: Side,
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

    return TakerSizing(tuple(sorted(quantities)), p_max)


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
    # Hedge orders are explicitly allowed negative standalone EV in
    # exchange for improving worst-case risk (SS17: "Hedge orders may have
    # negative standalone EV if they efficiently improve the whole
    # portfolio risk. They must be labeled separately in the journal and
    # optimizer.") - edge_min is an alpha-edge floor and does not apply to
    # them, regardless of what the caller passed.
    if apply_edge_min and purpose is not OrderPurpose.HEDGE and delta_ev < cfg.edge_min:
        violated.append("edge_min")

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
        g_after=portfolio_after.G,
        pi_u_after=portfolio_after.Pi_U,
        pi_d_after=portfolio_after.Pi_D,
        violated_constraints=tuple(violated),
        max_execution_price=max_execution_price,
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

    return _finalize(
        action_id, purpose, side, OrderMode.FAK, walk.avg_price if walk.filled_shares > 0 else limit_price,
        requested_qty, ttl_s=0.0, expected_fill=walk.filled_shares,
        delta_U=delta_U, delta_D=delta_D, delta_C=delta_C, delta_ev_raw=delta_ev_raw,
        portfolio=portfolio, portfolio_after=portfolio_after, q=q, cfg=cfg, apply_edge_min=True,
        max_execution_price=limit_price,
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

    return _finalize(
        action_id, purpose, side, OrderMode.POST_ONLY, price, qty, ttl_s=horizon_s, expected_fill=rho * qty,
        delta_U=delta_U, delta_D=delta_D, delta_C=delta_C, delta_ev_raw=delta_ev_raw,
        portfolio=portfolio, portfolio_after=portfolio_after_if_filled, q=q, cfg=cfg, apply_edge_min=True,
    )


def generate_hedge_candidate(
    action_id: str,
    portfolio: PortfolioState,
    asks_up: tuple[BookLevel, ...],
    asks_down: tuple[BookLevel, ...],
    q: float,
    fee_config: FeeConfig,
    cfg: OneStepConfig,
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

    return evaluate_taker_candidate(
        action_id, side, OrderPurpose.HEDGE, x_min, limit_price=1.0, asks=asks,
        portfolio=portfolio, q=q, fee_config=fee_config, cfg=cfg,
    )


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
