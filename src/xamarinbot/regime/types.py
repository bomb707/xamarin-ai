"""Seed regime types (Roadmap Phase 6): "Represent raw/normalized gap
states and CLOB/spot directions as explicit enums."
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


class GapRegime(str, Enum):
    """Normalized-gap (Z_gap) buckets at the -1/-0.5/0/0.5/1 breakpoints
    named in Roadmap Phase 6 verification ("Replay transitions around
    ±1/±0.5/0 seeds"). The Strategy doc SS8 table only names three regions
    (Positive/upper middle, Near center/weakening positive, Negative/lower
    middle) - this build splits each side at ±1 into a "middle" and
    "strong" sub-bucket, and mirrors the doc's "near center / weakening
    positive" case to the negative side, so every breakpoint the roadmap
    names is an actual state boundary (see matrix.py for how the extra
    buckets map back onto the doc's action families)."""

    STRONG_NEGATIVE = "STRONG_NEGATIVE"  # z_gap < -1.0
    LOWER_MIDDLE = "LOWER_MIDDLE"  # -1.0 <= z_gap < -0.5
    NEAR_CENTER_NEGATIVE = "NEAR_CENTER_NEGATIVE"  # -0.5 <= z_gap < 0.0
    NEAR_CENTER_POSITIVE = "NEAR_CENTER_POSITIVE"  # 0.0 <= z_gap < 0.5
    UPPER_MIDDLE = "UPPER_MIDDLE"  # 0.5 <= z_gap < 1.0
    STRONG_POSITIVE = "STRONG_POSITIVE"  # z_gap >= 1.0


class SeedAction(str, Enum):
    """Candidate action *families* (Strategy doc SS8: "Every seed action is
    only an allowed action family. Final choice among TAKER, MAKER, WAIT,
    CANCEL, or REPLACE is determined by expected value, fill/timing value,
    and the post-fill portfolio state" - i.e. Phase 8's optimizer, not this
    module, makes the final call)."""

    TAKER_UP = "TAKER_UP"
    TAKER_DOWN = "TAKER_DOWN"
    MAKER_UP = "MAKER_UP"
    MAKER_DOWN = "MAKER_DOWN"
    WAIT = "WAIT"
    CANCEL = "CANCEL"


DIRECTIONAL_ACTIONS = frozenset({SeedAction.TAKER_UP, SeedAction.TAKER_DOWN, SeedAction.MAKER_UP, SeedAction.MAKER_DOWN})


@dataclass(frozen=True)
class RegimeState:
    """The atomic classification key - Roadmap Phase 6: "All seed states
    are deterministic and auditable.\""""

    gap_regime: GapRegime
    clob_direction: Direction
    spot_direction: Direction


@dataclass(frozen=True)
class RegimeSnapshot:
    round_id: str
    decision_ts: float
    state: RegimeState
    seed_action: SeedAction
    permitted_actions: frozenset[SeedAction]


@dataclass(frozen=True)
class RegimeTransition:
    """Roadmap Phase 6 deliverable: "Log regime transitions and dwell
    times." from_state is None for a round's first observation (nothing to
    transition from yet). seed_action is the effective action for
    to_state (post CANCEL-substitution - see classifier.py)."""

    round_id: str
    from_state: RegimeState | None
    to_state: RegimeState
    seed_action: SeedAction
    transition_ts: float
    dwell_time_s: float | None
