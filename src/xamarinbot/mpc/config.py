"""MPCController parameters (Roadmap Phase 10).

"Start with a horizon of 1-3 decision steps before expanding." "Define
small discrete scenario tree for spot/TWAP/CLOB/book evolution over
0.25-5 seconds." "Bound computation time; fall back to one-step controller
if deadline exceeded."
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MPCConfig:
    # horizon_steps=1 means "evaluate first-level candidates only, no
    # recursive continuation" - by construction this must reduce exactly
    # to OneStepController's decision (Phase 10 verification: "MPC action
    # equals one-step action in degenerate horizon").
    horizon_steps: int = 2

    step_dt_s: float = 1.0  # within SS15's named 0.25-5s scenario-tree range

    # "small discrete scenario tree": only the top-k most probable next
    # GapRegime states are expanded at each level, renormalized to sum to
    # 1 - not the full transition distribution's tail.
    transition_top_k: int = 2

    time_budget_ms: float = 50.0
