"""ShadowRunner parameters (Roadmap Phase 12)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowConfig:
    heartbeat_s: float = 10.0

    # SS21 "Latency gate: Controller misses deadline -> Fallback to
    # simpler one-step/WAIT policy." This build's controller under shadow
    # is already the one-step controller (the simplest tier), so a missed
    # deadline falls all the way back to WAIT.
    decision_deadline_ms: float = 50.0

    # "24/7 reconnect stability" (Roadmap Phase 12 verification) - bounded
    # retry count before a feed outage is treated as a missed decision
    # rather than retried forever.
    reconnect_max_retries: int = 3
