"""Scenario/transition model (Roadmap Phase 10 deliverable).

"Estimate transition probabilities from historical state transitions." No
real historical data exists yet, so this is estimated from Phase 6's own
`RegimeTransition` records collected over a causal replay - the same
"historical state transitions" concept, just synthetic for now, exactly
like Phase 5's `q` model being trained on synthetic replay data rather than
real fills.

The scenario tree deliberately evolves only `GapRegime` (6 states), not the
full 54-state `RegimeState` (which also carries CLOB/spot direction) -
"small discrete scenario tree" (Roadmap) and a 54-state chain would need far
more transition data than a synthetic dataset this size can estimate
reliably. CLOB/spot direction and the order book are held at their current
values in continuation projections (see controller.py) - a documented
simplification, not a full state-evolution model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.regime.types import GapRegime, RegimeTransition


@dataclass(frozen=True)
class TransitionModel:
    """probabilities[from_regime][to_regime] = P(to | from), each row
    summing to 1.0. A GapRegime with no observed outgoing transitions is
    simply absent - `next_states` falls back to "stays put" for it."""

    probabilities: dict[GapRegime, dict[GapRegime, float]] = field(default_factory=dict)

    def next_states(self, current: GapRegime, top_k: int) -> list[tuple[GapRegime, float]]:
        """Top-k most probable next states, renormalized to sum to 1.0 -
        the "small discrete" truncation. Falls back to [(current, 1.0)]
        (assume persistence) if `current` was never observed transitioning
        anywhere in the training data."""
        row = self.probabilities.get(current)
        if not row:
            return [(current, 1.0)]
        ranked = sorted(row.items(), key=lambda kv: -kv[1])[:top_k]
        total = sum(p for _, p in ranked)
        if total <= 0:
            return [(current, 1.0)]
        return [(state, p / total) for state, p in ranked]


def build_transition_model(transitions: list[RegimeTransition]) -> TransitionModel:
    counts: dict[GapRegime, dict[GapRegime, int]] = {}
    for t in transitions:
        if t.from_state is None:
            continue
        from_g = t.from_state.gap_regime
        to_g = t.to_state.gap_regime
        row = counts.setdefault(from_g, {})
        row[to_g] = row.get(to_g, 0) + 1

    probabilities: dict[GapRegime, dict[GapRegime, float]] = {}
    for from_g, next_counts in counts.items():
        total = sum(next_counts.values())
        probabilities[from_g] = {to_g: c / total for to_g, c in next_counts.items()}

    return TransitionModel(probabilities=probabilities)
