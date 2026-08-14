"""Seed regime classification (Roadmap Phase 6, Strategy doc SS8).

The source table gives 5 example rows plus a catch-all ("Any region,
Conflict/stale -> WAIT/CANCEL"), not an exhaustive 54-cell matrix (6 gap
regimes x 3 CLOB directions x 3 spot directions). `classify_seed_action`
below is a complete, deterministic extension of the table's stated logic to
every cell - documented here since the docs don't specify the un-named
cells:

  1. Either leg FLAT -> WAIT (no clear direction to act on).
  2. CLOB and spot synchronized (same direction) and the gap regime agrees
     with that direction (a "positive" family regime + UP, or "negative"
     family + DOWN) -> TAKER in that direction (SS8 rows 1 and 4).
  3. CLOB and spot synchronized against a *near-center* (weak) gap regime
     of the opposite sign -> TAKER in the synchronized direction ("fast
     reversal can lead the slower TWAP state", SS8 row 3, mirrored here for
     the symmetric near-center-negative/UP case since the doc only gives
     the DOWN example).
  4. CLOB and spot synchronized but opposing a *strong/middle* gap regime
     -> WAIT (too large a conflict with the TWAP anchor to act on fast
     signals alone; the doc doesn't cover this case, and this is the more
     conservative reading).
  5. CLOB pulls back against spot while the gap regime agrees with spot's
     direction -> MAKER in spot's direction (SS8 rows 2 and 5).
  6. CLOB pulls back against spot while the gap regime disagrees with
     spot's direction too -> WAIT (three-way conflict; no doc guidance,
     conservative default).

Controller override still applies (SS8): every action returned here is a
candidate action family only, not a placed order. Nothing in this module
touches PortfolioState, simulates a fill, or checks G_min/spend/position
constraints (Phase 3) - the Phase 8 optimizer (not yet built) is where a
seed candidate gets evaluated for EV and risk before ever becoming an
order, per Roadmap Phase 6 verification "no matrix entry directly bypasses
EV/risk gates."
"""
from __future__ import annotations

from xamarinbot.regime.config import RegimeConfig
from xamarinbot.regime.types import DIRECTIONAL_ACTIONS, Direction, GapRegime, RegimeState, SeedAction

_POSITIVE_FAMILY = frozenset({GapRegime.UPPER_MIDDLE, GapRegime.STRONG_POSITIVE})
_NEGATIVE_FAMILY = frozenset({GapRegime.LOWER_MIDDLE, GapRegime.STRONG_NEGATIVE})


def gap_regime_for(z_gap: float, cfg: RegimeConfig) -> GapRegime:
    if z_gap >= cfg.gap_strong_threshold:
        return GapRegime.STRONG_POSITIVE
    if z_gap >= cfg.gap_middle_threshold:
        return GapRegime.UPPER_MIDDLE
    if z_gap >= 0.0:
        return GapRegime.NEAR_CENTER_POSITIVE
    if z_gap >= -cfg.gap_middle_threshold:
        return GapRegime.NEAR_CENTER_NEGATIVE
    if z_gap >= -cfg.gap_strong_threshold:
        return GapRegime.LOWER_MIDDLE
    return GapRegime.STRONG_NEGATIVE


def clob_direction_from_sign(sign: int) -> Direction:
    if sign > 0:
        return Direction.UP
    if sign < 0:
        return Direction.DOWN
    return Direction.FLAT


def spot_direction_from_bp(return_bp: float | None, cfg: RegimeConfig) -> Direction:
    if return_bp is None or abs(return_bp) < cfg.spot_flat_threshold_bp:
        return Direction.FLAT
    return Direction.UP if return_bp > 0 else Direction.DOWN


def classify_seed_action(state: RegimeState) -> SeedAction:
    if state.clob_direction is Direction.FLAT or state.spot_direction is Direction.FLAT:
        return SeedAction.WAIT

    if state.clob_direction == state.spot_direction:
        sync_dir = state.clob_direction
        if state.gap_regime in _POSITIVE_FAMILY and sync_dir is Direction.UP:
            return SeedAction.TAKER_UP
        if state.gap_regime in _NEGATIVE_FAMILY and sync_dir is Direction.DOWN:
            return SeedAction.TAKER_DOWN
        if state.gap_regime is GapRegime.NEAR_CENTER_POSITIVE and sync_dir is Direction.DOWN:
            return SeedAction.TAKER_DOWN  # fast reversal leads the slow TWAP state (SS8 row 3)
        if state.gap_regime is GapRegime.NEAR_CENTER_NEGATIVE and sync_dir is Direction.UP:
            return SeedAction.TAKER_UP  # symmetric fast reversal
        return SeedAction.WAIT  # synchronized fast signals but strongly opposing the TWAP anchor

    # CLOB pulls back against spot's direction
    if state.gap_regime in _POSITIVE_FAMILY and state.spot_direction is Direction.UP and state.clob_direction is Direction.DOWN:
        return SeedAction.MAKER_UP
    if state.gap_regime in _NEGATIVE_FAMILY and state.spot_direction is Direction.DOWN and state.clob_direction is Direction.UP:
        return SeedAction.MAKER_DOWN
    return SeedAction.WAIT


class ActionPermissionMatrix:
    """Roadmap Phase 6 deliverable. `permitted_actions` always includes
    WAIT (the optimizer must always be free to do nothing) alongside
    whatever `effective_action` the caller computed (typically
    classify_seed_action's result, or CANCEL once RegimeClassifier
    determines a prior thesis needs invalidating)."""

    @staticmethod
    def permitted_actions(effective_action: SeedAction) -> frozenset[SeedAction]:
        if effective_action in DIRECTIONAL_ACTIONS:
            return frozenset({effective_action, SeedAction.WAIT})
        return frozenset({effective_action}) | {SeedAction.WAIT}
