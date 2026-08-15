"""Stateful per-round regime classifier (Roadmap Phase 6 deliverable:
RegimeClassifier). "Keep the original baseline rule available as a
separate policy" - this is an additional candidate-action source
(gated by `xamarinbot.config.FeatureFlags.use_regime_seed_policy`), not a
replacement for baseline/strategy.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.features.types import FeatureVector
from xamarinbot.journal.schema import RegimeTransitionRecord
from xamarinbot.regime.config import RegimeConfig
from xamarinbot.regime.matrix import (
    ActionPermissionMatrix,
    classify_seed_action,
    clob_direction_from_sign,
    gap_regime_for,
    spot_direction_from_bp,
)
from xamarinbot.regime.types import DIRECTIONAL_ACTIONS, RegimeSnapshot, RegimeState, RegimeTransition, SeedAction


def _state_dict(state: RegimeState | None) -> dict | None:
    if state is None:
        return None
    return {"gap_regime": state.gap_regime.value, "clob_direction": state.clob_direction.value, "spot_direction": state.spot_direction.value}


def to_journal_record(transition: RegimeTransition) -> RegimeTransitionRecord:
    return RegimeTransitionRecord(
        round_id=transition.round_id,
        transition_ts=transition.transition_ts,
        from_state=_state_dict(transition.from_state),
        to_state=_state_dict(transition.to_state),
        seed_action=transition.seed_action.value,
        dwell_time_s=transition.dwell_time_s,
    )


def state_for(fv: FeatureVector, cfg: RegimeConfig) -> RegimeState:
    clob_sign = fv.clob_direction.get(cfg.canonical_horizon_s, 0)
    spot_bp = fv.spot_returns_bp.get(cfg.canonical_horizon_s)
    return RegimeState(
        gap_regime=gap_regime_for(fv.z_gap, cfg),
        clob_direction=clob_direction_from_sign(clob_sign),
        spot_direction=spot_direction_from_bp(spot_bp, cfg),
    )


@dataclass
class RegimeClassifier:
    """One instance per round - regime transitions/dwell times are only
    meaningful within a single round's 0-300s timeline."""

    round_id: str
    cfg: RegimeConfig = field(default_factory=RegimeConfig)
    transitions: list[RegimeTransition] = field(default_factory=list, init=False)
    _state: RegimeState | None = field(default=None, init=False)
    _last_directional_action: SeedAction | None = field(default=None, init=False)
    _regime_start_ts: float | None = field(default=None, init=False)

    @property
    def current_state(self) -> RegimeState | None:
        """The last classified state, or None before the first observation.

        Added for Phase 12C item 10: when a decision point is reached with
        stale or missing inputs, the runner still has to review its resting
        orders, and it must do so against the last state it genuinely
        classified rather than fabricating one."""
        return self._state

    def observe(self, fv: FeatureVector) -> RegimeSnapshot:
        state = state_for(fv, self.cfg)
        seed_action = classify_seed_action(state)

        # Encode WAIT/CANCEL for conflict, flat, stale or invalid states
        # (Roadmap Phase 6): if the regime just lapsed out of a directional
        # thesis into WAIT, any resting order built on that thesis needs
        # canceling - a plain WAIT (never had a thesis) doesn't.
        effective_action = seed_action
        if seed_action is SeedAction.WAIT and self._last_directional_action is not None:
            effective_action = SeedAction.CANCEL

        is_first_observation = self._state is None
        if is_first_observation or state != self._state:
            dwell = None if is_first_observation else fv.decision_ts - self._regime_start_ts
            self.transitions.append(
                RegimeTransition(
                    round_id=self.round_id,
                    from_state=self._state,
                    to_state=state,
                    seed_action=effective_action,
                    transition_ts=fv.decision_ts,
                    dwell_time_s=dwell,
                )
            )
            self._regime_start_ts = fv.decision_ts

        self._last_directional_action = seed_action if seed_action in DIRECTIONAL_ACTIONS else None
        self._state = state

        return RegimeSnapshot(
            round_id=self.round_id,
            decision_ts=fv.decision_ts,
            state=state,
            seed_action=effective_action,
            permitted_actions=ActionPermissionMatrix.permitted_actions(effective_action),
        )
