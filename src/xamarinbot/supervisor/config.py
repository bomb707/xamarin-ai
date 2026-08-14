"""OrderSupervisor parameters (Roadmap Phase 9)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupervisorConfig:
    edge_min: float = 0.0  # same concept as OneStepConfig.edge_min (Phase 8), reused as the "expected filled edge" floor
    g_min: float = -1_000_000.0
    min_tau_for_passive_s: float = 15.0  # "time compression: remaining time makes passive fill too slow"

    # "Replace only when the new expected value exceeds the old order by a
    # churn threshold" (SS16) - not a formula in the source docs beyond the
    # name; a flat EV-improvement floor here. Phase 12B Tranche 2E: also
    # doubles as V_replace's fixed action cost (see `predicates.value_replace`).
    churn_threshold: float = 0.5

    # "Rate-limit cancel/replace to protect latency, API limits and queue
    # priority" (Roadmap Phase 9 step) - no formula given; a simple minimum
    # dwell time between actions on the same order.
    min_action_interval_s: float = 1.0

    # Phase 12B Tranche 2E: V_hold/V_cancel/V_replace argmax cancel/replace
    # policy - replaces the old unconditional "regime flip -> cancel
    # immediately" rule with a soft economic degradation. All four default
    # to 0.0 (inert placeholders, like every other Tranche 2 economic
    # coefficient - not tuned against synthetic PnL per item 37/Addendum
    # J); at these defaults the policy reduces exactly to the pre-2E
    # `edge_failure` comparison for CANCEL and the pre-2E `book_displacement`
    # comparison (against current value rather than submit-time value) for
    # REPLACE. Non-zero values are exercised explicitly in tests to prove
    # the mechanism, not asserted as calibrated production settings.
    regime_flip_penalty: float = 0.0  # V_hold degradation once the origin regime no longer holds
    time_compression_penalty: float = 0.0  # V_hold degradation once remaining time makes a passive fill unlikely
    cancel_cost: float = 0.0  # V_cancel's fixed action cost (queue priority / API cost of cancelling)
    hysteresis_margin: float = 0.0  # incumbency bonus added to V_hold so tiny economic wobbles don't cause thrashing
