"""Phase 12B Tranche 1 item 4/B: genuine per-window walk-forward pipeline.

Covers the reviewer's explicit requirements:
- each window performs its own TRAIN -> FIT -> VALIDATE -> CALIBRATE ->
  FREEZE -> TEST cycle, not one globally-fit model reused everywhere;
- an end-to-end leakage test recording every round_id consumed by model
  fit, the calibrator, the transition-model fit, and the final test
  evaluation, proving TrainIDs/ValidateIDs/TestIDs stay disjoint across
  every stage, not only `sweep_parameter()` (already covered separately
  in `tests/test_walkforward.py`).
"""
from __future__ import annotations

from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.features import COMBINED_LEAD_LAG, SPOT_ONLY, TWAP_ONLY
from xamarinbot.portfolio.state import FeeConfig, Side
from xamarinbot.synthetic.rounds import generate_synthetic_dataset
from xamarinbot.walkforward.ablations import MANDATORY_ABLATIONS
from xamarinbot.walkforward.pipeline import (
    LeakageTrace,
    _split_rounds_for_fit_and_calibration,
    fit_window_artifacts,
    run_walk_forward_ablations,
)
from xamarinbot.walkforward.windows import rolling_windows


def _dataset(n_rounds=18):
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)
    return store, results


def test_each_window_fits_a_distinct_model_not_one_global_model():
    """Direct evidence the per-window TRAIN->FIT->VALIDATE->CALIBRATE
    cycle actually runs independently per window, not once globally -
    two windows with disjoint train_round_ids must produce two distinct
    fitted model objects (different weights), not the same one reused."""
    # n_train=8 (not 4): Phase 12B Tranche 2A item 6 requires class
    # diversity on BOTH the fit and calibration sides, which a very small
    # window can genuinely fail to achieve (correctly - see
    # test_split_returns_none_when_diversity_is_unreachable_on_either_side);
    # this test's own purpose is "two windows produce two distinct
    # models," which needs a window large enough to realistically succeed.
    store, results = _dataset(n_rounds=24)
    round_ids = [r.round_id for r in results]
    windows = rolling_windows(round_ids, n_train=8, n_validate=2, n_test=2)
    assert len(windows) >= 2

    feature_cfg = FeatureConfig()
    trace = LeakageTrace()
    artifacts_0 = fit_window_artifacts(windows[0], results, store, feature_cfg, [COMBINED_LEAD_LAG], trace)
    artifacts_1 = fit_window_artifacts(windows[1], results, store, feature_cfg, [COMBINED_LEAD_LAG], trace)

    model_0 = artifacts_0.models_by_feature_set[COMBINED_LEAD_LAG.name]
    model_1 = artifacts_1.models_by_feature_set[COMBINED_LEAD_LAG.name]
    assert model_0 is not None and model_1 is not None
    assert model_0.raw_model.weights != model_1.raw_model.weights, "different windows' train data produced identical model weights - looks like one shared model, not per-window fitting"


def test_window_artifacts_are_calibrated_not_raw():
    store, results = _dataset(n_rounds=16)  # n_train=8: see the item 6 comment above
    round_ids = [r.round_id for r in results]
    windows = rolling_windows(round_ids, n_train=8, n_validate=2, n_test=2)
    feature_cfg = FeatureConfig()
    trace = LeakageTrace()

    artifacts = fit_window_artifacts(windows[0], results, store, feature_cfg, [COMBINED_LEAD_LAG], trace)
    model = artifacts.models_by_feature_set[COMBINED_LEAD_LAG.name]
    assert model is not None
    assert model.calibration_version == "platt-v1"
    assert hasattr(model, "predict_proba_raw")  # q_raw stays available for diagnostics


def test_end_to_end_leakage_trace_keeps_every_stage_disjoint_from_test():
    """The core requirement: for *each* window, that window's own TEST_i
    round_ids must never appear in what that window's own model fit,
    calibrator, or transition model were built from - checked per-window,
    not by pooling every window's test round_ids together first.

    Per-window, not pooled, deliberately: `rolling_windows`'s default
    `step=n_test` still lets a smaller step reuse a round that was one
    window's TEST as a *later* window's TRAIN (documented and tested
    already in `windows.py`/`test_walkforward.py`'s own "no test-period
    tuning" check) - that is legitimate reuse across windows, not a leak
    within a window. A global pooled-disjointness check would flag that
    legitimate reuse as a false positive, exactly as this test did before
    being fixed to check per-window instead."""
    store, results = _dataset(n_rounds=12)
    round_ids = [r.round_id for r in results]
    windows = rolling_windows(round_ids, n_train=4, n_validate=2, n_test=2)
    assert windows

    feature_cfg = FeatureConfig()
    fee_config = FeeConfig()
    exec_cfg = ExecutionConfig()
    specs = tuple(s for s in MANDATORY_ABLATIONS if s.name in ("1_baseline_unanimous", "2_twap_only"))
    feature_sets = [s.feature_set for s in specs if s.feature_set is not None]

    any_test_eval = False
    for window in windows:
        trace = LeakageTrace()
        fit_window_artifacts(window, results, store, feature_cfg, feature_sets, trace)
        window_test_ids = set(window.test_round_ids)

        assert trace.model_fit_round_ids.isdisjoint(window_test_ids), f"window {window.window_index}: model fit touched its own test round_id"
        assert trace.calibrator_fit_round_ids.isdisjoint(window_test_ids), f"window {window.window_index}: calibrator fit touched its own test round_id"
        assert trace.transition_fit_round_ids.isdisjoint(window_test_ids), f"window {window.window_index}: transition model fit touched its own test round_id"

        window_train_validate_ids = set(window.train_round_ids) | set(window.validate_round_ids)
        assert trace.model_fit_round_ids <= window_train_validate_ids
        assert trace.calibrator_fit_round_ids <= window_train_validate_ids
        assert trace.transition_fit_round_ids <= window_train_validate_ids

        # Phase 12B Tranche 1.1 item 4: TRAIN_i/VALIDATE_i/TEST_i must be
        # pairwise disjoint within this window, checked explicitly rather
        # than only implied by rolling_windows()'s own slicing.
        train_ids = set(window.train_round_ids)
        validate_ids = set(window.validate_round_ids)
        assert train_ids.isdisjoint(validate_ids)
        assert train_ids.isdisjoint(window_test_ids)
        assert validate_ids.isdisjoint(window_test_ids)

        any_test_eval = any_test_eval or bool(window_test_ids)

    assert any_test_eval, "expected at least one window with a non-empty test segment"

    # separately, confirm the real end-to-end entry point actually
    # evaluates every window's test rounds (aggregate check - this one
    # can be pooled safely since it's not a leakage assertion). Phase 12B
    # Tranche 1.1 item 4: run_walk_forward_ablations now returns one
    # LeakageTrace per window (keyed by window_index), not one shared
    # trace - pool only test_eval_round_ids across windows here, never
    # the fit/calibrator/transition sets (those must stay per-window).
    _, traces = run_walk_forward_ablations(store, results, windows, feature_cfg, fee_config, exec_cfg, specs)
    all_test_ids = {rid for w in windows for rid in w.test_round_ids}
    pooled_test_eval_ids = {rid for t in traces.values() for rid in t.test_eval_round_ids}
    assert pooled_test_eval_ids == all_test_ids


def test_calibrator_fit_round_ids_are_disjoint_from_model_fit_round_ids_within_a_window():
    """Within one window's own TRAIN segment, the raw model and its
    calibrator must be fit on genuinely different ROUNDS, not merely
    different individual examples (Phase 12B Tranche 1.1 item 1): every
    observation from a given 5-minute market shares one eventual
    settlement outcome and is strongly serially correlated with every
    other observation from that same round, so a round split partly into
    the fit set and partly into the calibration set would leak that
    round's label across the split. R_modelFit ∩ R_calibration = ∅ is the
    required invariant, at round granularity - example-level overlap
    within a shared round is exactly what must NOT happen (this test used
    to assert the opposite before the round-disjoint split fix)."""
    store, results = _dataset(n_rounds=12)
    round_ids = [r.round_id for r in results]
    windows = rolling_windows(round_ids, n_train=6, n_validate=2, n_test=2)
    feature_cfg = FeatureConfig()
    trace = LeakageTrace()

    fit_window_artifacts(windows[0], results, store, feature_cfg, [TWAP_ONLY], trace)

    assert trace.model_fit_round_ids
    assert trace.calibrator_fit_round_ids
    assert trace.model_fit_round_ids.isdisjoint(trace.calibrator_fit_round_ids), (
        "a round_id was consumed by both the raw-model fit and the calibrator fit - "
        "R_modelFit ∩ R_calibration must be empty"
    )


# --------------------------------------------------------------------------
# Phase 12B Tranche 1.2 item 7 / Tranche 2A item 6: the split searches for
# a point where BOTH the fit prefix and the calibration suffix
# independently have class diversity, and never invents a model (fit OR
# calibration) when that genuinely can't be achieved.
# --------------------------------------------------------------------------


def test_split_expands_calibration_window_backward_until_both_classes_present():
    round_ids = tuple(f"r{i}" for i in range(10))
    # Naive holdout (frac=0.2 of 10 -> 2 rounds) would land on r8,r9, both UP.
    outcomes = {rid: Side.UP for rid in round_ids}
    outcomes["r0"] = Side.DOWN  # gives the fit side its own diversity
    outcomes["r7"] = Side.DOWN  # only reachable by expanding calibration one round further back

    fit_ids, calib_ids = _split_rounds_for_fit_and_calibration(round_ids, 0.2, outcomes)

    assert "r7" in calib_ids
    assert {outcomes[rid] for rid in calib_ids} == {Side.UP, Side.DOWN}
    assert {outcomes[rid] for rid in fit_ids} == {Side.UP, Side.DOWN}
    assert fit_ids and set(fit_ids).isdisjoint(calib_ids)


def test_split_can_shrink_calibration_below_the_default_to_fix_fit_side_diversity():
    """Phase 12B Tranche 2A item 6: the default calibration window can
    already be diverse while the fit prefix is NOT - fixable only by
    SHRINKING calibration (pulling a round back into fit), which the
    old, monotonic-only-growing search (Tranche 1.2 item 7 alone) could
    never find. Default (frac=0.3 of 10 -> n_calib=3) calib=r7,r8,r9;
    fit=r0..r6 (all UP - not diverse). Shrinking to n_calib=2 pulls r7
    (DOWN) into fit, fixing it, while calib=r8,r9 stays diverse."""
    round_ids = tuple(f"r{i}" for i in range(10))
    outcomes = {rid: Side.UP for rid in round_ids}
    outcomes["r7"] = Side.DOWN
    outcomes["r9"] = Side.DOWN

    fit_ids, calib_ids = _split_rounds_for_fit_and_calibration(round_ids, 0.3, outcomes)

    assert fit_ids is not None
    assert {outcomes[rid] for rid in fit_ids} == {Side.UP, Side.DOWN}
    assert {outcomes[rid] for rid in calib_ids} == {Side.UP, Side.DOWN}
    assert set(fit_ids).isdisjoint(calib_ids)
    assert "r7" in fit_ids  # pulled back into fit by the shrink search


def test_split_returns_none_when_diversity_is_unreachable_on_either_side():
    round_ids = tuple(f"r{i}" for i in range(6))
    outcomes = {rid: Side.UP for rid in round_ids}  # every round UP - diversity is impossible anywhere
    result = _split_rounds_for_fit_and_calibration(round_ids, 0.2, outcomes)
    assert result is None


def test_split_returns_none_when_only_one_round_could_ever_supply_the_second_class():
    """A single DOWN round among many UP rounds can go to fit XOR
    calibration, never both - genuinely impossible, must return None,
    not a partially-diverse split."""
    round_ids = tuple(f"r{i}" for i in range(8))
    outcomes = {rid: Side.UP for rid in round_ids}
    outcomes["r4"] = Side.DOWN
    result = _split_rounds_for_fit_and_calibration(round_ids, 0.2, outcomes)
    assert result is None


def test_split_without_round_outcomes_keeps_original_fixed_fraction_behavior():
    """Backward-compatible default: omitting round_outcomes (or passing
    None) must reproduce the exact pre-item-7 fixed-fraction split."""
    round_ids = tuple(f"r{i}" for i in range(10))
    fit_ids, calib_ids = _split_rounds_for_fit_and_calibration(round_ids, 0.2)
    assert len(calib_ids) == 2
    assert fit_ids == round_ids[:8]
    assert calib_ids == round_ids[8:]


def test_fit_window_artifacts_produces_a_genuinely_calibrated_model_not_identity_fallback():
    """End-to-end: on a real synthetic dataset with enough rounds, the
    per-window calibration split (with backward expansion) should
    ordinarily reach class diversity and produce a real Platt calibrator,
    not silently fall back to IdentityCalibrator."""
    store, results = _dataset(n_rounds=18)
    round_ids = [r.round_id for r in results]
    windows = rolling_windows(round_ids, n_train=10, n_validate=3, n_test=3)
    assert windows
    feature_cfg = FeatureConfig()
    trace = LeakageTrace()

    artifacts = fit_window_artifacts(windows[0], results, store, feature_cfg, [TWAP_ONLY], trace)
    model = artifacts.models_by_feature_set[TWAP_ONLY.name]
    assert model is not None
    assert model.is_calibrated, f"expected genuine calibration, got fallback version {model.calibration_version!r}"
