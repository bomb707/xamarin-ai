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
    # name; a flat EV-improvement floor here.
    churn_threshold: float = 0.5

    # "Rate-limit cancel/replace to protect latency, API limits and queue
    # priority" (Roadmap Phase 9 step) - no formula given; a simple minimum
    # dwell time between actions on the same order.
    min_action_interval_s: float = 1.0
