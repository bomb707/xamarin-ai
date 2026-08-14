"""Shadow-run record types (Roadmap Phase 12 deliverable: "Record desired
order, hypothetical submit time, expected fill and real subsequent book
evolution.")."""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.portfolio.state import PortfolioState


@dataclass(frozen=True)
class ShadowDecisionRecord:
    round_id: str
    decision_ts: float
    action_id: str
    mode: str
    side: str | None
    price: float | None
    qty: float
    expected_fill: float
    ev_after: float
    g_after: float
    decide_elapsed_ms: float
    missed_deadline: bool  # True if decide_elapsed_ms exceeded the configured deadline (SS21 Latency gate)
    reconnected: bool  # True if a simulated feed outage was recovered from just before this decision


@dataclass(frozen=True)
class ShadowRoundResult:
    round_id: str
    records: tuple[ShadowDecisionRecord, ...]
    final_portfolio: PortfolioState  # caller derives hypothetical PnL against the real settlement outcome
    n_reconnects: int
    n_missed_deadlines: int
