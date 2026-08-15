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
    """Phase 12B Tranche 2E: SS18's "V_hold" for the cancel/replace policy
    - a flipped regime or compressed remaining time lowers the order's
    held value by a configurable penalty rather than vetoing it outright,
    so a still strongly profitable order can survive a flip it can
    economically absorb. This is a pure economic value (what the order is
    actually worth if left resting) - `edge_min` is deliberately NOT
    subtracted here (Tranche 2.1 item 10): it is a decision THRESHOLD on
    whether holding is even eligible (see `hold_eligible` below), not a
    value term, so it must not distort this number's comparison against
    `value_replace`, which has no symmetric edge_min term of its own. At
    `cfg`'s default zero penalties this reduces to plain `current_delta_ev`."""
    v = current_delta_ev
    if regime_flip(origin_regime_state, current_regime_state):
        v -= cfg.regime_flip_penalty
    if time_compression(tau, cfg):
        v -= cfg.time_compression_penalty
    return v


def hold_eligible(effective_delta_ev: float, cfg: SupervisorConfig) -> bool:
    """Phase 12B Tranche 2.1 item 10: `edge_min` is a decision THRESHOLD -
    "is this order's (penalty-adjusted) edge still above the floor worth
    continuing to hold for" - not an economic value folded into any V()
    term. `cfg.hysteresis_margin` widens the threshold itself (classic
    control-theory hysteresis: don't flip state until crossing threshold
    minus/plus a margin), rather than being added as a flat bonus to
    `value_hold`'s magnitude - a flat bonus would also (wrongly) tilt the
    HOLD-vs-REPLACE comparison, which has nothing to do with edge_min or
    hysteresis at all. `effective_delta_ev` should be `value_hold(...)`'s
    own return value, so regime-flip/time-compression penalties are
    already netted in before this threshold is applied."""
    return effective_delta_ev >= cfg.edge_min - cfg.hysteresis_margin


def value_cancel(cfg: SupervisorConfig) -> float:
    """Phase 12B Tranche 2.1 item 10: SS18's "V_cancel" is the economic
    value actually received from canceling - none, beyond avoiding the
    fixed cost of executing the cancel itself. `edge_min` is a decision
    threshold on whether an order is worth *holding* (see
    `hold_eligible`), not a value received from walking away, so it must
    not appear here (a bug in the prior Tranche 2E draft, which folded it
    in as `edge_min - cancel_cost`)."""
    return -cfg.cancel_cost


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
