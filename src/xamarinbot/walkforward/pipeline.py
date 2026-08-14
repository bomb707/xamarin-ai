"""Genuine per-window walk-forward pipeline (Phase 12B audit item 4/B).

Before this module existed, `scripts/run_walk_forward_ablation_demo.py`
fit one q-model and one transition model *once*, globally, before
`rolling_windows()` was ever constructed, and reused that same fixed
model across every window's test segment - Phase 11 was measuring how
strategy/execution behavior varies across time-ordered data slices given
one fixed model, not genuine walk-forward model validation (where each
window's own model must be fit only from that window's own past).

Every window here runs:
    TRAIN_i  -> fit feature standardization + probability model (+ transition model)
    VALIDATE_i -> calibrate (Platt) the probability model
    FREEZE_i -> model weights, standardization, calibrator, transition model are now fixed
    TEST_i   -> exactly one evaluation pass, using only the frozen artifacts above

TEST data is never touched by any earlier stage - `LeakageTrace` records
every round_id actually consumed by each stage so this is asserted, not
assumed (see `tests/test_walkforward_pipeline.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.model.calibrated import CalibratedModel, fit_calibrated_model
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import FeatureSet
from xamarinbot.model.walkforward import time_ordered_split
from xamarinbot.mpc.scenario import TransitionModel, build_transition_model
from xamarinbot.portfolio.state import FeeConfig
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.synthetic.rounds import SyntheticRoundResult
from xamarinbot.walkforward.ablations import AblationSpec, RoundResult, run_ablation_round
from xamarinbot.walkforward.windows import WalkForwardWindow

HEARTBEAT_S = 10.0

# Held out of each window's own TRAIN pool as that window's internal
# calibration-validation slice, per Phase 5's own established discipline
# (fit on train, calibrate on a disjoint validation slice) - applied here
# *within* a window's train_round_ids, distinct from the window's own
# separate `validate_round_ids` (which Phase 11's ablations/sensitivity
# code already uses for parameter selection, per
# `parameter_stability_across_windows`). Splitting the window's train
# examples again avoids calibrating on the exact same rows the raw model
# was fit on.
_CALIBRATION_HOLDOUT_FRAC = 0.2


@dataclass
class LeakageTrace:
    """Records every round_id actually consumed by each pipeline stage,
    across every window - the basis for the required end-to-end
    no-leakage test (Phase 12B audit item 4's explicit ask: "add an
    end-to-end leakage test that records every round_id consumed by:
    model fit, standardizer, calibrator, transition-model fit, parameter
    sweep, final test evaluation")."""

    model_fit_round_ids: set[str] = field(default_factory=set)
    calibrator_fit_round_ids: set[str] = field(default_factory=set)
    transition_fit_round_ids: set[str] = field(default_factory=set)
    test_eval_round_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class WindowArtifacts:
    window_index: int
    models_by_feature_set: dict[str, CalibratedModel | None]
    transition_model: TransitionModel


def _rounds_for_ids(all_rounds: list[SyntheticRoundResult], round_ids: tuple[str, ...]) -> list[SyntheticRoundResult]:
    wanted = set(round_ids)
    return [r for r in all_rounds if r.round_id in wanted]


def _train_transition_model(
    store: EventStore, train_rounds: list[SyntheticRoundResult], feature_cfg: FeatureConfig
) -> TransitionModel:
    all_transitions = []
    for result in train_rounds:
        events = store.all_events(result.round_id)
        clock = ReplayClock(store, result.round_id)
        classifier = RegimeClassifier(round_id=result.round_id)
        for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
            fv = compute(events, result.round_id, decision_ts, result.p0, feature_cfg)
            if isinstance(fv, FeatureVector):
                classifier.observe(fv)
        all_transitions.extend(classifier.transitions)
    return build_transition_model(all_transitions)


def fit_window_artifacts(
    window: WalkForwardWindow,
    all_rounds: list[SyntheticRoundResult],
    store: EventStore,
    feature_cfg: FeatureConfig,
    feature_sets: list[FeatureSet],
    trace: LeakageTrace,
) -> WindowArtifacts:
    """TRAIN_i -> FIT_i -> VALIDATE_i -> CALIBRATE_i -> FREEZE_i, for
    every feature_set the ablation matrix actually needs (each ablation
    names its own FeatureSet; a single shared model would silently make
    e.g. "TWAP-only" and "spot-only" identical). The transition model
    (MPC ablation #8 only) doesn't depend on feature_set choice, so one
    per window suffices."""
    train_rounds = _rounds_for_ids(all_rounds, window.train_round_ids)

    models: dict[str, CalibratedModel | None] = {}
    for fs in feature_sets:
        by_fs = build_examples_multi(store, train_rounds, feature_cfg, [fs], heartbeat_s=HEARTBEAT_S)
        examples = by_fs[fs.name]
        # TRAIN_i's own internal fit/calibrate split - never window.validate_round_ids
        # or window.test_round_ids, only a chronological slice of this
        # window's own train_round_ids.
        split = time_ordered_split(examples, train_frac=1.0 - _CALIBRATION_HOLDOUT_FRAC - 1e-9, val_frac=_CALIBRATION_HOLDOUT_FRAC)
        trace.model_fit_round_ids.update(e.round_id for e in split.train)
        trace.calibrator_fit_round_ids.update(e.round_id for e in split.validation)
        models[fs.name] = fit_calibrated_model(split.train, split.validation, fs)

    transition_model = _train_transition_model(store, train_rounds, feature_cfg)
    trace.transition_fit_round_ids.update(r.round_id for r in train_rounds)

    return WindowArtifacts(window_index=window.window_index, models_by_feature_set=models, transition_model=transition_model)


@dataclass(frozen=True)
class WindowRoundResult:
    window_index: int
    ablation_name: str
    round_id: str
    result: RoundResult


def run_walk_forward_ablations(
    store: EventStore,
    all_rounds: list[SyntheticRoundResult],
    windows: list[WalkForwardWindow],
    feature_cfg: FeatureConfig,
    fee_config: FeeConfig,
    exec_cfg: ExecutionConfig,
    ablation_specs: tuple[AblationSpec, ...],
) -> tuple[list[WindowRoundResult], LeakageTrace]:
    """Runs every ablation's TEST_i segment against that window's own
    frozen artifacts. `ablation_specs` with `controller == "baseline"`
    need no model (the baseline strategy is deterministic sign logic, not
    a fitted q model) and are evaluated directly."""
    feature_sets = sorted(
        {spec.feature_set for spec in ablation_specs if spec.feature_set is not None}, key=lambda fs: fs.name
    )
    trace = LeakageTrace()
    all_round_results: list[WindowRoundResult] = []

    for window in windows:
        artifacts = fit_window_artifacts(window, all_rounds, store, feature_cfg, feature_sets, trace)
        test_rounds = _rounds_for_ids(all_rounds, window.test_round_ids)

        for r in test_rounds:
            trace.test_eval_round_ids.add(r.round_id)
            for spec in ablation_specs:
                model = artifacts.models_by_feature_set.get(spec.feature_set.name) if spec.feature_set is not None else None
                result = run_ablation_round(
                    spec, store, r.round_id, r.p0, r.outcome, feature_cfg, fee_config, exec_cfg, model, artifacts.transition_model
                )
                all_round_results.append(WindowRoundResult(window.window_index, spec.name, r.round_id, result))

    return all_round_results, trace
