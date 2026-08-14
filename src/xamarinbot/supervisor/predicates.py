"""Cancel/replace trigger predicates (Roadmap Phase 9, Strategy doc SS16
trigger table). Each predicate answers one yes/no question about whether an
open order's originating thesis still holds; `supervisor.py` combines them
in priority order. Kept as small pure functions so every cell is testable
in isolation - "Unit test every matrix cell" was Phase 6's phrasing, but
the same principle applies here.

"Condition-based cancellation: Cancel order o when V_o(t) < V_min or its
originating state predicate is false" (SS16) - `edge_failure` is the V_o(t)
< V_min check; `regime_flip` is the "originating state predicate is false"
check.
"""
from __future__ import annotations

from xamarinbot.regime.types import RegimeState
from xamarinbot.supervisor.config import SupervisorConfig


def edge_failure(current_delta_ev: float, cfg: SupervisorConfig) -> bool:
    """"Edge failure: Expected filled edge falls below threshold -> Cancel
    or reprice.\""""
    return current_delta_ev < cfg.edge_min


def regime_flip(origin_state: RegimeState, current_state: RegimeState) -> bool:
    """"Regime flip: TWAP/spot/CLOB state leaves the regime that created
    the order -> Cancel immediately.\""""
    return origin_state != current_state


def risk_breach(current_g_after_if_fill: float, cfg: SupervisorConfig) -> bool:
    """"Risk breach: Projected fill would push G below G_min -> Cancel /
    shrink.\""""
    return current_g_after_if_fill < cfg.g_min


def time_compression(tau: float, cfg: SupervisorConfig) -> bool:
    """"Time compression: Remaining time makes passive fill too slow ->
    Cancel, convert to taker only if taker EV passes.\" (the
    convert-to-taker half is the supervisor's/caller's job once it sees
    this trigger fire - this predicate only detects the condition)."""
    return tau < cfg.min_tau_for_passive_s


def feed_stale(is_fresh: bool) -> bool:
    """"Feed freshness failure: TWAP/spot/book input stale or timestamp
    invalid -> Cancel risky open orders / freeze new orders.\""""
    return not is_fresh


def book_displacement(current_optimal_ev: float, ev_at_submit: float, cfg: SupervisorConfig) -> bool:
    """"Book displacement: Queue/price becomes unattractive relative to new
    fair value -> Replace on new optimal tick." Only true once the new
    opportunity clears the churn threshold - "Replace only when the new
    expected value exceeds the old order by a churn threshold.\""""
    return current_optimal_ev > ev_at_submit + cfg.churn_threshold


def value_hold(
    current_delta_ev: float,
    origin_regime_state: RegimeState,
    current_regime_state: RegimeState,
    tau: float,
    cfg: SupervisorConfig,
) -> float:
    """Phase 12B Tranche 2E: SS18's "V_hold" for the argmax(V_hold,
    V_cancel, V_replace) cancel/replace policy - replaces the old
    unconditional "regime flip -> cancel immediately" rule with a *soft*
    economic degradation: a flipped regime or compressed remaining time
    lowers the order's held value by a configurable penalty rather than
    vetoing it outright, so a still strongly profitable order can survive
    a flip it can economically absorb, while a marginal one gets canceled
    precisely because the flip pushed it under the edge floor - not merely
    because a flip occurred. At `cfg`'s default zero penalties this
    reduces to plain `current_delta_ev`."""
    v = current_delta_ev
    if regime_flip(origin_regime_state, current_regime_state):
        v -= cfg.regime_flip_penalty
    if time_compression(tau, cfg):
        v -= cfg.time_compression_penalty
    return v


def value_cancel(cfg: SupervisorConfig) -> float:
    """Phase 12B Tranche 2E: SS18's "V_cancel" - the baseline value of
    walking away: no further forward EV from this order, floored at
    `edge_min` (the same "expected filled edge" floor `edge_failure` used,
    now framed as an indifference point rather than a hard veto) minus the
    fixed cost of actually executing a cancel."""
    return cfg.edge_min - cfg.cancel_cost


def value_replace(current_optimal_ev: float | None, cfg: SupervisorConfig) -> float | None:
    """Phase 12B Tranche 2E: SS18's "V_replace" - only defined once the
    caller has evaluated a concrete new tick's EV (`current_optimal_ev`);
    `None` otherwise, since there is nothing to replace *to* this review.
    Compared against the order's *current* held value (not its stale
    submit-time EV, unlike the old `book_displacement` predicate) minus
    the churn cost of tearing up and re-posting - "replace only when the
    new expected value clears a churn threshold" (SS16), now measured
    against what continuing to hold is actually worth right now."""
    if current_optimal_ev is None:
        return None
    return current_optimal_ev - cfg.churn_threshold
