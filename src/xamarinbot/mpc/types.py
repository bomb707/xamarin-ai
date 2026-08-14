"""MPCController result type (Roadmap Phase 10)."""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.optimizer.types import CandidateAction


@dataclass(frozen=True)
class MPCDecision:
    round_id: str
    decision_ts: float
    chosen: CandidateAction  # only the first action is ever executed (receding horizon)
    candidates: tuple[CandidateAction, ...]  # first-level candidate table (Phase 8 diagnostics, reused)
    sequence_values: dict[str, float]  # action_id -> candidate.delta_ev + expected continuation value
    used_fallback: bool  # True if the time budget was exceeded and this is the plain one-step decision
    elapsed_ms: float
