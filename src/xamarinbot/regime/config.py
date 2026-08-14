"""Seed regime classifier parameters (Roadmap Phase 6)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeConfig:
    # which per-horizon spot/CLOB signal to classify on - reuses the same
    # canonical horizon convention as features/model (see
    # FeatureConfig.canonical_horizon_s).
    canonical_horizon_s: float = 1.0

    # Z_gap breakpoints (Roadmap Phase 6: "Replay transitions around
    # ±1/±0.5/0 seeds").
    gap_strong_threshold: float = 1.0
    gap_middle_threshold: float = 0.5

    # spot momentum (bp) below which direction is FLAT rather than UP/DOWN.
    # CLOB direction doesn't need an equivalent threshold - it's already a
    # discrete sign(M_t - M_{t-h}) in FeatureVector.clob_direction.
    spot_flat_threshold_bp: float = 0.5
