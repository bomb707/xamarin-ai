"""Phase 5 verification: Brier/log-loss correctness, walk-forward split
ordering (no leakage), calibrator monotonicity/validity, and the
ModelRegistry promotion gate (Roadmap Phase 5 exit gate: "No production use
until calibration is acceptable.")
"""
from __future__ import annotations

import math
import random

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.calibration import fit_isotonic, fit_platt
from xamarinbot.model.dataset import Example, build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG, SPOT_ONLY, TWAP_ONLY, base_value_map, design_vector
from xamarinbot.model.logistic import fit_logistic_regression
from xamarinbot.model.metrics import (
    brier_score,
    calibration_by_group,
    calibration_table,
    log_loss,
    settlement_accuracy,
)
from xamarinbot.model.registry import ModelRegistry, PromotionGateError, make_artifact
from xamarinbot.model.walkforward import round_ordered_split, time_ordered_split
from devtools.synthetic.rounds import generate_synthetic_dataset

# --------------------------------------------------------------------------
# Logistic regression
# --------------------------------------------------------------------------


def _separable_dataset(n: int = 200, seed: int = 0) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        label = rng.randint(0, 1)
        center = 3.0 if label == 1 else -3.0
        X.append([center + rng.gauss(0, 0.5)])
        y.append(label)
    return X, y


def test_logistic_regression_discriminates_separable_data():
    X, y = _separable_dataset()
    # weak regularization: a single, cleanly separated feature should let
    # the model become confidently correct.
    model = fit_logistic_regression(X, y, "test", ("x",), l2=0.01, n_iters=300)
    assert model.predict_proba([3.0]) > 0.9
    assert model.predict_proba([-3.0]) < 0.1


def test_l2_regularization_shrinks_weights_monotonically_and_stably():
    """Also guards against the proximal-update regression: an explicit
    gradient step on the L2 term diverges to +-inf once lr*l2 >= 2 (was a
    real bug here), so this checks a large l2 stays finite and keeps
    shrinking weights rather than blowing up."""
    X, y = _separable_dataset()
    weights_by_l2 = [
        abs(fit_logistic_regression(X, y, "test", ("x",), l2=l2, n_iters=300).weights[0])
        for l2 in (0.01, 1.0, 10.0, 100.0)
    ]
    assert all(math.isfinite(w) for w in weights_by_l2)
    assert weights_by_l2 == sorted(weights_by_l2, reverse=True)


def test_logistic_regression_rejects_empty_input():
    with pytest.raises(ValueError):
        fit_logistic_regression([], [], "test", ())


# --------------------------------------------------------------------------
# Walk-forward split
# --------------------------------------------------------------------------


def _fake_examples(n: int) -> list[Example]:
    # decision_ts intentionally unsorted on input - split must sort by time.
    order = list(range(n))
    random.Random(1).shuffle(order)
    return [Example(round_id=f"r{i}", decision_ts=float(i), features=None, x=[float(i)], y=i % 2) for i in order]


def test_walkforward_split_is_chronological_with_no_overlap():
    examples = _fake_examples(100)
    split = time_ordered_split(examples, train_frac=0.6, val_frac=0.2)

    assert len(split.train) + len(split.validation) + len(split.test) == 100
    assert max(e.decision_ts for e in split.train) < min(e.decision_ts for e in split.validation)
    assert max(e.decision_ts for e in split.validation) < min(e.decision_ts for e in split.test)


def test_walkforward_split_rejects_invalid_fractions():
    examples = _fake_examples(10)
    with pytest.raises(ValueError):
        time_ordered_split(examples, train_frac=0.7, val_frac=0.4)  # sums >= 1.0


def test_walkforward_training_window_matches_train_slice_bounds():
    examples = _fake_examples(100)
    split = time_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    start, end = split.training_window
    assert start == min(e.decision_ts for e in split.train)
    assert end == max(e.decision_ts for e in split.train)


# --------------------------------------------------------------------------
# round_ordered_split (Phase 12B Tranche 1.2 item 8): the round-aware
# drop-in replacement for time_ordered_split, for callers outside the main
# walk-forward pipeline that must still not split a round across the
# fit/calibration/test boundary.
# --------------------------------------------------------------------------


def _fake_multi_example_rounds(n_rounds: int, examples_per_round: int = 5) -> list[Example]:
    """Several examples per round, all sharing that round's own y label -
    the shape a real round actually has (every decision-point example
    within one round shares that round's single eventual outcome)."""
    examples = []
    order = list(range(n_rounds))
    random.Random(2).shuffle(order)
    for i in order:
        round_id = f"round-{i}"
        y = i % 2
        for j in range(examples_per_round):
            examples.append(Example(round_id=round_id, decision_ts=i * 100.0 + j, features=None, x=[float(i)], y=y))
    return examples


def test_round_ordered_split_never_splits_a_single_round_across_slices():
    examples = _fake_multi_example_rounds(n_rounds=20, examples_per_round=5)
    split = round_ordered_split(examples, train_frac=0.6, val_frac=0.2)

    train_rids = {e.round_id for e in split.train}
    val_rids = {e.round_id for e in split.validation}
    test_rids = {e.round_id for e in split.test}
    assert train_rids.isdisjoint(val_rids)
    assert train_rids.isdisjoint(test_rids)
    assert val_rids.isdisjoint(test_rids)
    assert len(split.train) + len(split.validation) + len(split.test) == len(examples)


def test_round_ordered_split_is_chronological_by_round():
    examples = _fake_multi_example_rounds(n_rounds=20, examples_per_round=5)
    split = round_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    assert max(e.decision_ts for e in split.train) < min(e.decision_ts for e in split.validation)
    assert max(e.decision_ts for e in split.validation) < min(e.decision_ts for e in split.test)


def test_round_ordered_split_matches_time_ordered_split_when_one_example_per_round():
    """Sanity/equivalence check: when every round contributes exactly one
    example (as time_ordered_split implicitly assumes), the two split
    functions must produce identical results."""
    examples = _fake_examples(100)
    by_round = round_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    by_time = time_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    assert [e.round_id for e in by_round.train] == [e.round_id for e in by_time.train]
    assert [e.round_id for e in by_round.validation] == [e.round_id for e in by_time.validation]
    assert [e.round_id for e in by_round.test] == [e.round_id for e in by_time.test]


def test_round_ordered_split_rejects_invalid_fractions():
    examples = _fake_multi_example_rounds(n_rounds=10)
    with pytest.raises(ValueError):
        round_ordered_split(examples, train_frac=0.7, val_frac=0.4)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_isotonic_calibrator_output_is_monotonic_non_decreasing():
    rng = random.Random(2)
    q_raw = [rng.random() for _ in range(200)]
    y = [1 if rng.random() < q else 0 for q in q_raw]
    cal = fit_isotonic(q_raw, y)

    probes = sorted(rng.random() for _ in range(50))
    outputs = [cal.transform(p) for p in probes]
    assert all(outputs[i] <= outputs[i + 1] for i in range(len(outputs) - 1))


def test_isotonic_calibrator_improves_or_matches_in_sample_brier():
    rng = random.Random(3)
    q_raw = [rng.random() for _ in range(300)]
    y = [1 if rng.random() < q else 0 for q in q_raw]
    cal = fit_isotonic(q_raw, y)
    q_cal = [cal.transform(q) for q in q_raw]

    assert brier_score(q_cal, y) <= brier_score(q_raw, y) + 1e-9  # in-sample: PAVA minimizes this by construction


def test_platt_calibrator_outputs_valid_probabilities():
    rng = random.Random(4)
    q_raw = [rng.random() for _ in range(200)]
    y = [1 if rng.random() < q else 0 for q in q_raw]
    cal = fit_platt(q_raw, y)
    for q in [0.0, 0.01, 0.5, 0.99, 1.0]:
        p = cal.transform(q)
        assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_brier_score_known_value():
    # perfectly wrong predictions -> brier = 1.0; perfect -> 0.0
    assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_log_loss_known_value():
    ll = log_loss([0.5, 0.5], [1, 0])
    assert ll == pytest.approx(-math.log(0.5))


def test_settlement_accuracy():
    q = [0.9, 0.4, 0.6, 0.1]
    y = [1, 0, 1, 1]
    assert settlement_accuracy(q, y) == pytest.approx(0.75)


def test_calibration_table_buckets_by_predicted_probability():
    q = [0.05, 0.15, 0.95]
    y = [0, 1, 1]
    table = calibration_table(q, y, n_bins=10)
    assert table[0].n == 1 and table[0].mean_predicted == pytest.approx(0.05)
    assert table[1].n == 1
    assert table[9].n == 1 and table[9].empirical_rate == pytest.approx(1.0)


def test_calibration_by_group_groups_correctly():
    examples = [Example(round_id="r", decision_ts=float(i), features=None, x=[], y=i % 2) for i in range(4)]
    q = [0.1, 0.2, 0.8, 0.9]
    groups = calibration_by_group(examples, q, key_fn=lambda e: "A" if e.decision_ts < 2 else "B")
    assert groups["A"].n == 2
    assert groups["B"].n == 2
    assert groups["A"].mean_predicted == pytest.approx(0.15)
    assert groups["B"].mean_predicted == pytest.approx(0.85)


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------


def test_registry_promotes_model_below_brier_threshold():
    X, y = _separable_dataset()
    model = fit_logistic_regression(X, y, "twap_only", ("x",))
    artifact = make_artifact(model, "features-v1", (0.0, 100.0), metrics={"brier": 0.1})
    registry = ModelRegistry()
    registry.register(artifact)
    promoted = registry.promote(artifact.model_id, max_brier=0.25)
    assert promoted.promoted
    assert registry.champion().model_id == artifact.model_id


def test_registry_rejects_promotion_above_brier_threshold():
    X, y = _separable_dataset()
    model = fit_logistic_regression(X, y, "twap_only", ("x",))
    artifact = make_artifact(model, "features-v1", (0.0, 100.0), metrics={"brier": 0.4})
    registry = ModelRegistry()
    registry.register(artifact)
    with pytest.raises(PromotionGateError):
        registry.promote(artifact.model_id, max_brier=0.25)
    assert registry.champion() is None


def test_registry_rollback_switches_champion_without_retraining():
    X, y = _separable_dataset()
    model_a = fit_logistic_regression(X, y, "a", ("x",))
    model_b = fit_logistic_regression(X, y, "b", ("x",))
    artifact_a = make_artifact(model_a, "v1", (0.0, 1.0), metrics={"brier": 0.1})
    artifact_b = make_artifact(model_b, "v1", (0.0, 1.0), metrics={"brier": 0.05})
    registry = ModelRegistry()
    registry.register(artifact_a)
    registry.register(artifact_b)
    registry.promote(artifact_b.model_id)
    assert registry.champion().model_id == artifact_b.model_id

    registry.rollback(artifact_a.model_id)
    assert registry.champion().model_id == artifact_a.model_id


# --------------------------------------------------------------------------
# Feature-set design vectors + dataset construction (integration)
# --------------------------------------------------------------------------


def test_feature_set_column_names_match_design_vector_length():
    for fs in (TWAP_ONLY, SPOT_ONLY, COMBINED_LEAD_LAG):
        assert len(fs.column_names) == len(fs.base) + len(fs.interactions)


def test_build_examples_multi_shares_one_feature_computation_across_feature_sets():
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=2)
    cfg = FeatureConfig()
    by_fs = build_examples_multi(store, results, cfg, [TWAP_ONLY, SPOT_ONLY, COMBINED_LEAD_LAG], heartbeat_s=15.0)

    for fs in (TWAP_ONLY, SPOT_ONLY, COMBINED_LEAD_LAG):
        examples = by_fs[fs.name]
        assert examples, f"no examples produced for {fs.name}"
        for ex in examples:
            assert len(ex.x) == len(fs.column_names)
            assert ex.y in (0, 1)
            assert ex.round_id in {r.round_id for r in results}

    # every feature set should see the same set of decision points, since
    # they're all derived from the same underlying FeatureVector computation
    twap_points = {(e.round_id, e.decision_ts) for e in by_fs[TWAP_ONLY.name]}
    combined_points = {(e.round_id, e.decision_ts) for e in by_fs[COMBINED_LEAD_LAG.name]}
    assert twap_points == combined_points


def _make_feature_vector(**overrides):
    from xamarinbot.features.types import FeatureVector, TimeRegime

    defaults = dict(
        feature_version="v1",
        round_id="r0",
        decision_ts=100.0,
        t=100.0,
        tau=200.0,
        time_regime=TimeRegime.CORE_TRADING,
        p0=100_000.0,
        twap=100_010.0,
        spot=100_020.0,
        clob_mid=0.55,
        gap_twap_bp=1.0,
        gap_spot_bp=2.0,
        lead_gap_bp=1.0,
        twap_pressure_model=0.0,
        realized_vol=0.01,
        z_gap=0.5,
        z_spot=0.3,
        clob_log_odds=0.2,
        z_clob=0.1,
        ofi=0.0,
        spread=0.02,
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


def test_design_vector_returns_vector_when_z_spot_present():
    fv = _make_feature_vector(z_spot=0.3)
    vec = design_vector(fv, COMBINED_LEAD_LAG)
    assert vec is not None
    assert len(vec) == len(COMBINED_LEAD_LAG.column_names)


def test_design_vector_none_when_z_spot_unavailable():
    fv = _make_feature_vector(z_spot=None)
    assert base_value_map(fv) is None
    assert design_vector(fv, COMBINED_LEAD_LAG) is None
    assert design_vector(fv, TWAP_ONLY) is None  # even feature sets that don't use z_spot are excluded
