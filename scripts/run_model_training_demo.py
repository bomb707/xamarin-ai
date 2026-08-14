#!/usr/bin/env python3
"""Phase 5 end-to-end demo: build a training dataset from causal replay,
fit TWAP-only / current-BTC-only (spot-only) / combined lead-lag logistic
models on a chronological (walk-forward) train split, calibrate each on the
validation split, evaluate Brier/log-loss/accuracy out-of-sample on the
test split, print a calibration-by-group report for the combined model, and
register all three in a ModelRegistry (attempting to promote the combined
model against the Phase 5 exit gate).

Calibrator choice: Platt scaling, not isotonic, despite both being
implemented in model/calibration.py ("isotonic or Platt-style calibration",
Roadmap Phase 5). This synthetic data is close to deterministically
separable (see docs/PHASE_STATUS.md), so raw q lands near 0/1 with almost
no repeated values - isotonic's PAVA then pools almost nothing (one block
per near-unique point) and just memorizes the validation set instead of
learning a smooth curve, which generalizes badly to test. Platt's 2-
parameter smooth fit doesn't have that failure mode at this data scale.

Usage: python scripts/run_model_training_demo.py [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.events.store import EventStore
from xamarinbot.features.config import FeatureConfig
from xamarinbot.journal.schema import ModelOutputRecord
from xamarinbot.journal.writer import JournalWriter
from xamarinbot.model.calibration import fit_platt
from xamarinbot.model.dataset import Example, build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG, SPOT_ONLY, TWAP_ONLY
from xamarinbot.model.logistic import fit_logistic_regression
from xamarinbot.model.metrics import brier_score, calibration_by_group, calibration_table, log_loss, settlement_accuracy
from xamarinbot.model.registry import ModelRegistry, PromotionGateError, make_artifact
from xamarinbot.model.walkforward import round_ordered_split
from xamarinbot.synthetic.rounds import generate_synthetic_dataset

HEARTBEAT_S = 10.0
FEATURE_SETS = [TWAP_ONLY, SPOT_ONLY, COMBINED_LEAD_LAG]


def _bucket(value: float, size: float) -> float:
    return round(value / size) * size


def _volatility_regime(ex: Example, terciles: tuple[float, float]) -> str:
    v = ex.features.realized_vol
    if v <= terciles[0]:
        return "LOW"
    if v <= terciles[1]:
        return "MED"
    return "HIGH"


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)
    feature_cfg = FeatureConfig()

    print(f"Generating training data from {n_rounds} synthetic rounds (heartbeat={HEARTBEAT_S}s)...")
    by_feature_set = build_examples_multi(store, results, feature_cfg, FEATURE_SETS, heartbeat_s=HEARTBEAT_S)

    registry = ModelRegistry()
    journal = JournalWriter(":memory:")

    print(f"\n{'model':<20}{'n_train':>9}{'n_val':>7}{'n_test':>8}{'brier':>9}{'brier_cal':>11}{'logloss':>10}{'acc':>7}")
    combined_test_examples: list[Example] | None = None
    combined_q_test_calibrated: list[float] | None = None

    for fs in FEATURE_SETS:
        examples = by_feature_set[fs.name]
        split = round_ordered_split(examples, train_frac=0.6, val_frac=0.2)
        if not split.train or not split.validation or not split.test:
            print(f"{fs.name:<20} insufficient examples to split (n={len(examples)}) - skipping")
            continue

        X_train = [e.x for e in split.train]
        y_train = [e.y for e in split.train]
        model = fit_logistic_regression(X_train, y_train, fs.name, fs.column_names)

        q_val_raw = [model.predict_proba(e.x) for e in split.validation]
        y_val = [e.y for e in split.validation]
        calibrator = fit_platt(q_val_raw, y_val)

        q_test_raw = [model.predict_proba(e.x) for e in split.test]
        q_test_cal = [calibrator.transform(q) for q in q_test_raw]
        y_test = [e.y for e in split.test]

        b_raw = brier_score(q_test_raw, y_test)
        b_cal = brier_score(q_test_cal, y_test)
        ll = log_loss(q_test_cal, y_test)
        acc = settlement_accuracy(q_test_cal, y_test)

        artifact = make_artifact(
            model,
            feature_version=feature_cfg.feature_version,
            training_window=split.training_window,
            metrics={"brier": b_cal, "log_loss": ll, "accuracy": acc, "n_test": len(y_test)},
            calibrator=calibrator,
        )
        registry.register(artifact)

        print(
            f"{fs.name:<20}{len(split.train):>9}{len(split.validation):>7}{len(split.test):>8}"
            f"{b_raw:>9.4f}{b_cal:>11.4f}{ll:>10.4f}{acc:>7.1%}   id={artifact.model_id}"
        )

        if fs.name == COMBINED_LEAD_LAG.name:
            combined_test_examples = split.test
            combined_q_test_calibrated = q_test_cal
            combined_artifact = artifact

    if combined_test_examples is None:
        print("\nCombined model was not trained (insufficient data) - stopping.")
        return

    print("\n=== Calibration table (combined model, test split, calibrated q) ===")
    for b, stats in calibration_table(combined_q_test_calibrated, [e.y for e in combined_test_examples]).items():
        print(f"  bin {b}: n={stats.n:<5} mean_q={stats.mean_predicted:.3f} empirical={stats.empirical_rate:.3f} gap={stats.gap:+.3f}")

    print("\n=== Calibration by time_regime ===")
    by_time = calibration_by_group(combined_test_examples, combined_q_test_calibrated, lambda e: e.features.time_regime.value)
    for key, stats in sorted(by_time.items()):
        print(f"  {key:<24} n={stats.n:<5} mean_q={stats.mean_predicted:.3f} empirical={stats.empirical_rate:.3f} gap={stats.gap:+.3f}")

    print("\n=== Calibration by gap_twap_bp bucket (200bp) ===")
    by_gap = calibration_by_group(combined_test_examples, combined_q_test_calibrated, lambda e: _bucket(e.features.gap_twap_bp, 200.0))
    for key, stats in sorted(by_gap.items()):
        print(f"  {key:>8.0f}bp  n={stats.n:<5} mean_q={stats.mean_predicted:.3f} empirical={stats.empirical_rate:.3f} gap={stats.gap:+.3f}")

    print("\n=== Calibration by CLOB mid (price) bucket (0.1) ===")
    by_price = calibration_by_group(combined_test_examples, combined_q_test_calibrated, lambda e: _bucket(e.features.clob_mid, 0.1))
    for key, stats in sorted(by_price.items()):
        print(f"  {key:>5.2f}  n={stats.n:<5} mean_q={stats.mean_predicted:.3f} empirical={stats.empirical_rate:.3f} gap={stats.gap:+.3f}")

    train_vols = sorted(e.features.realized_vol for e in by_feature_set[COMBINED_LEAD_LAG.name])
    terciles = (train_vols[len(train_vols) // 3], train_vols[2 * len(train_vols) // 3]) if train_vols else (0.0, 0.0)
    print("\n=== Calibration by volatility regime (train-set terciles) ===")
    by_vol = calibration_by_group(combined_test_examples, combined_q_test_calibrated, lambda e: _volatility_regime(e, terciles))
    for key, stats in sorted(by_vol.items()):
        print(f"  {key:<6} n={stats.n:<5} mean_q={stats.mean_predicted:.3f} empirical={stats.empirical_rate:.3f} gap={stats.gap:+.3f}")

    for ex, q in zip(combined_test_examples, combined_q_test_calibrated):
        journal.write(
            ModelOutputRecord(
                round_id=ex.round_id,
                decision_ts=ex.decision_ts,
                model_version=combined_artifact.model_id,
                q=q,
                uncertainty={},
            )
        )
    print(f"\nJournaled {len(combined_test_examples)} ModelOutputRecord rows for {combined_artifact.model_id}.")

    print("\n=== Promotion gate ===")
    try:
        registry.promote(combined_artifact.model_id, max_brier=0.25)
        print(f"Promoted {combined_artifact.model_id} as champion (brier={combined_artifact.metrics['brier']:.4f} <= 0.25).")
    except PromotionGateError as exc:
        print(f"Promotion rejected: {exc}")


if __name__ == "__main__":
    main()
