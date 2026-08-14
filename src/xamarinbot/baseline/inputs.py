"""Shared helper for constructing `BaselineInputs` correctly (Phase 12B
audit items 3/D): every prior baseline-runner reimplementation of
"elapsed time into the round" and "value N seconds ago" was a chance to
get it wrong, and one of them (`walkforward/ablations.py`) did, twice,
independently. This module exists so there is exactly one place that
knows what "elapsed round time" means, used by every runner.
"""
from __future__ import annotations


def elapsed_t(decision_ts: float, round_start_ts: float) -> float:
    """Elapsed seconds into the round. `BaselineConfig.decision_window_*`
    and `BaselineInputs.t` are both defined relative to round start, never
    to an absolute replay/wall-clock timestamp - passing the absolute
    timestamp directly (as `walkforward/ablations.py` did before this fix)
    makes every round whose `start_ts != 0` immediately fail the decision
    window gate, since the window bounds ([15, 270] by default) are only
    ever meaningful relative to round start."""
    return decision_ts - round_start_ts
