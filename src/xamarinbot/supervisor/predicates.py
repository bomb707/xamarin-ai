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


def edge_failure(current_ev_after: float, cfg: SupervisorConfig) -> bool:
    """"Edge failure: Expected filled edge falls below threshold -> Cancel
    or reprice.\""""
    return current_ev_after < cfg.edge_min


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
