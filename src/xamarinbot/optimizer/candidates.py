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

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.maker import fill_probability, q_fill
from xamarinbot.execution.taker import walk_depth
from xamarinbot.feeds.base import BookLevel, BookSnapshot
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.types import CandidateAction, OrderMode
from xamarinbot.portfolio.math import FillSimulationResult, OrderPurpose, RiskConstraints, evaluate_constraints
from xamarinbot.portfolio.state import FeeConfig, Fill, LiquidityRole, PortfolioState, Side, apply_fill


def taker_quantities(levels: tuple[BookLevel, ...], max_levels: int) -> list[float]:
    """Cumulative depth at each of the first `max_levels` book levels - one
    quantity candidate per level, per the roadmap step."""
    out = []
    cumulative = 0.0
    for level in levels[:max_levels]:
        cumulative += level.size
        out.append(cumulative)
    return out


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


def _favored_side(portfolio: PortfolioState) -> Side:
    return Side.UP if portfolio.Pi_U >= portfolio.Pi_D else Side.DOWN


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
    ev_after_raw: float,
    portfolio: PortfolioState,
    portfolio_after: PortfolioState,
    cfg: OneStepConfig,
    apply_edge_min: bool,
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
    favored = _favored_side(portfolio) if cfg.p_min is not None else None
    check = evaluate_constraints(result, _risk_constraints(cfg), favored_side=favored)
    violated = list(check.violated)

    ev_after = ev_after_raw - cfg.churn_penalty
    if apply_edge_min and ev_after < cfg.edge_min:
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
        ev_after=ev_after,
        g_after=portfolio_after.G,
        pi_u_after=portfolio_after.Pi_U,
        pi_d_after=portfolio_after.Pi_D,
        violated_constraints=tuple(violated),
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
    ev_after_raw = q * delta_U + (1.0 - q) * delta_D - delta_C

    return _finalize(
        action_id, purpose, side, OrderMode.FAK, walk.avg_price if walk.filled_shares > 0 else limit_price,
        requested_qty, ttl_s=0.0, expected_fill=walk.filled_shares,
        delta_U=delta_U, delta_D=delta_D, delta_C=delta_C, ev_after_raw=ev_after_raw,
        portfolio=portfolio, portfolio_after=portfolio_after, cfg=cfg, apply_edge_min=True,
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
    ev_after_raw = rho * ev_if_filled - cfg.opportunity_cost

    return _finalize(
        action_id, purpose, side, OrderMode.POST_ONLY, price, qty, ttl_s=horizon_s, expected_fill=rho * qty,
        delta_U=delta_U, delta_D=delta_D, delta_C=delta_C, ev_after_raw=ev_after_raw,
        portfolio=portfolio, portfolio_after=portfolio_after_if_filled, cfg=cfg, apply_edge_min=True,
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
        ev_after=0.0,
        g_after=portfolio.G,
        pi_u_after=portfolio.Pi_U,
        pi_d_after=portfolio.Pi_D,
        violated_constraints=(),
    )
