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
from xamarinbot.mpc.scenario import TransitionModel, build_transition_model
from xamarinbot.portfolio.state import FeeConfig, Side
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.rounds import RoundLabel
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
# ROUNDS (never individual examples - see _split_rounds_for_fit_and_calibration)
# again avoids calibrating on the exact same rows the raw model was fit on.
_CALIBRATION_HOLDOUT_FRAC = 0.2


def _split_rounds_for_fit_and_calibration(
    train_round_ids: tuple[str, ...],
    calibration_holdout_frac: float,
    round_outcomes: dict[str, Side] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Splits a window's own `train_round_ids` into a chronologically
    earlier fit sub-range and a later calibration sub-range, at ROUND
    granularity (Phase 12B Tranche 1.1 item 1).

    Before this function existed, `fit_window_artifacts` built examples
    from every train round and then ran `time_ordered_split()` (an
    example-count-fraction split) over the resulting flat example list -
    since a single round contributes many decision-point examples that
    all share one eventual settlement outcome and are strongly serially
    correlated, that split could and did place some of a round's own
    examples in the raw-model fit set and the rest of that *same round*
    in the calibration set. The required invariant is
    `R_modelFit ∩ R_calibration = ∅` at the round level, not just at the
    example level.

    `train_round_ids` is already chronologically ordered (guaranteed by
    `rolling_windows()` slicing `round_ids_chronological`), so fit is
    always the earlier prefix and calibration the later suffix - that
    chronological ordering (TRAIN_i -> VALIDATE_i) is preserved by every
    split point this function ever returns.

    Phase 12B Tranche 1.2 item 7 / Tranche 2A item 6: with
    `round_outcomes` given (maps round_id -> settlement `Side`), this
    searches for a split point where BOTH the fit prefix AND the
    calibration suffix independently contain both UP and DOWN outcomes
    (`|{y_fit}| >= 2` and `|{y_calibration}| >= 2`) - a small default
    holdout (e.g. `n_train=6, calibration_holdout_frac=0.2` usually
    yields exactly one calibration round) is otherwise likely to land on
    a single-class window purely by chance, since every decision example
    from one round shares that round's one settlement outcome. Growing
    the calibration suffix can only ever shrink (never help) the fit
    prefix's own diversity, so this searches outward from the default
    split point in both directions (more AND less calibration) for the
    closest point satisfying both sides at once, rather than only ever
    expanding calibration monotonically.

    Returns `None` if no split point achieves diversity on both sides -
    the caller must not fit a model at all in that case (an unstable
    one-class raw model is not an acceptable fallback merely because
    `fit_calibrated_model`'s own guard would still catch the calibration
    half; the FIT half needs the same guarantee)."""
    n = len(train_round_ids)
    if n <= 1:
        return train_round_ids, ()
    n_calib = max(1, round(n * calibration_holdout_frac))
    n_calib = min(n_calib, n - 1)  # always leave at least one round to fit on

    if round_outcomes is None:
        n_fit = n - n_calib
        return train_round_ids[:n_fit], train_round_ids[n_fit:]

    def outcomes_for(rids: tuple[str, ...]) -> set[Side]:
        return {round_outcomes[rid] for rid in rids if rid in round_outcomes}

    for offset in range(0, n):
        for candidate in (n_calib + offset, n_calib - offset):
            if candidate < 1 or candidate > n - 1:
                continue
            n_fit = n - candidate
            fit_ids = train_round_ids[:n_fit]
            calib_ids = train_round_ids[n_fit:]
            if len(outcomes_for(fit_ids)) >= 2 and len(outcomes_for(calib_ids)) >= 2:
                return fit_ids, calib_ids

    return None  # no split achieves class diversity on both sides


@dataclass
class LeakageTrace:
    """Records every round_id actually consumed by one window's own
    pipeline stages - the basis for the required end-to-end no-leakage
    test (Phase 12B audit item 4's explicit ask: "add an end-to-end
    leakage test that records every round_id consumed by: model fit,
    standardizer, calibrator, transition-model fit, parameter sweep,
    final test evaluation").

    One `LeakageTrace` covers exactly one walk-forward window
    (Phase 12B Tranche 1.1 item 4) - pooling traces across every window
    into one shared set is wrong for a rolling walk-forward run, since a
    round that was one window's TEST round is explicitly permitted to
    become a *later* window's TRAIN/VALIDATE round (windows.py's own
    "future data only ever appears in a later window's train, never a
    given window's own test" docstring) - a pooled check would flag that
    legitimate reuse as if it were leakage. See
    `run_walk_forward_ablations`, which now returns one trace per window
    keyed by `window_index` rather than a single shared trace."""

    model_fit_round_ids: set[str] = field(default_factory=set)
    calibrator_fit_round_ids: set[str] = field(default_factory=set)
    transition_fit_round_ids: set[str] = field(default_factory=set)
    test_eval_round_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class WindowArtifacts:
    window_index: int
    models_by_feature_set: dict[str, CalibratedModel | None]
    transition_model: TransitionModel


def _rounds_for_ids(all_rounds: list[RoundLabel], round_ids: tuple[str, ...]) -> list[RoundLabel]:
    wanted = set(round_ids)
    return [r for r in all_rounds if r.round_id in wanted]


def _train_transition_model(
    store: EventStore, train_rounds: list[RoundLabel], feature_cfg: FeatureConfig
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
    all_rounds: list[RoundLabel],
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

    # TRAIN_i's own internal fit/calibrate split, at ROUND granularity
    # (Phase 12B Tranche 1.1 item 1) - never window.validate_round_ids or
    # window.test_round_ids, only a chronologically-earlier/later
    # partition of this window's own train_round_ids, with every example
    # from a given round landing entirely in one side of the split. The
    # split searches for class diversity on BOTH sides (Phase 12B Tranche
    # 1.2 item 7 / Tranche 2A item 6); if no split anywhere achieves that,
    # no model is fit at all for this window (an unstable one-class raw
    # model is not an acceptable silent fallback) - every feature_set maps
    # to None, and callers already treat model=None as "fall back to
    # q=0.5" (see ablations.py/run_one_step_controller_demo.py/
    # shadow/runner.py's `q = model.predict_proba(...) if model is not
    # None ... else 0.5` pattern).
    round_outcomes = {r.round_id: r.outcome for r in train_rounds}
    split_result = _split_rounds_for_fit_and_calibration(window.train_round_ids, _CALIBRATION_HOLDOUT_FRAC, round_outcomes)

    models: dict[str, CalibratedModel | None] = {}
    if split_result is None:
        models = {fs.name: None for fs in feature_sets}
    else:
        fit_round_ids, calib_round_ids = split_result
        fit_round_id_set, calib_round_id_set = set(fit_round_ids), set(calib_round_ids)
        for fs in feature_sets:
            by_fs = build_examples_multi(store, train_rounds, feature_cfg, [fs], heartbeat_s=HEARTBEAT_S)
            examples = by_fs[fs.name]
            fit_examples = [e for e in examples if e.round_id in fit_round_id_set]
            calib_examples = [e for e in examples if e.round_id in calib_round_id_set]
            trace.model_fit_round_ids.update(e.round_id for e in fit_examples)
            trace.calibrator_fit_round_ids.update(e.round_id for e in calib_examples)
            models[fs.name] = fit_calibrated_model(fit_examples, calib_examples, fs)

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
    all_rounds: list[RoundLabel],
    windows: list[WalkForwardWindow],
    feature_cfg: FeatureConfig,
    fee_config: FeeConfig,
    exec_cfg: ExecutionConfig,
    ablation_specs: tuple[AblationSpec, ...],
) -> tuple[list[WindowRoundResult], dict[int, LeakageTrace]]:
    """Runs every ablation's TEST_i segment against that window's own
    frozen artifacts. `ablation_specs` with `controller == "baseline"`
    need no model (the baseline strategy is deterministic sign logic, not
    a fitted q model) and are evaluated directly.

    Returns one `LeakageTrace` per window, keyed by `window_index`
    (Phase 12B Tranche 1.1 item 4) - never one trace shared/pooled across
    every window, since a round that was one window's TEST round is
    explicitly allowed to become a later window's TRAIN/VALIDATE round
    (see `LeakageTrace`'s own docstring); leakage must be asserted
    per-window, against that window's own fit/calibrator/transition
    round_ids and that window's own test_round_ids only."""
    feature_sets = sorted(
        {spec.feature_set for spec in ablation_specs if spec.feature_set is not None}, key=lambda fs: fs.name
    )
    traces: dict[int, LeakageTrace] = {}
    all_round_results: list[WindowRoundResult] = []

    for window in windows:
        trace = LeakageTrace()
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

        traces[window.window_index] = trace

    return all_round_results, traces
