"""Calibrated q model wrapper (Phase 12B audit item 5/C).

Every Phase 8-12 demo/harness `train_q_model()` previously returned the
*raw* `fit_logistic_regression` output directly - skipping the
calibration step Phase 5's own reference demo
(`scripts/run_model_training_demo.py`) already implements correctly, and
that Phase 5's own exit gate ("No production use until calibration is
acceptable") exists specifically to require. `DeltaEV_U(x) = q*x - K_U(x)`
uses `q` quantitatively, so a poorly-calibrated q can make a trade look
positive-EV when it isn't.

`CalibratedModel.predict_proba` is duck-type-compatible with
`LogisticModel.predict_proba` (same `(x: list[float]) -> float`
signature), so every existing caller that calls `model.predict_proba(vec)`
keeps working unchanged once a `CalibratedModel` is passed in place of a
raw `LogisticModel` - only the functions that *construct* the model need
to change.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.model.calibration import IsotonicCalibrator, PlattCalibrator, fit_platt
from xamarinbot.model.dataset import Example
from xamarinbot.model.features import FeatureSet
from xamarinbot.model.logistic import LogisticModel, fit_logistic_regression


@dataclass(frozen=True)
class CalibratedModel:
    """Wraps a raw `LogisticModel` with its fitted calibrator (fit on a
    validation split disjoint from the raw model's own training data -
    Phase 5's own discipline, reused here). `predict_proba` returns
    q_calibrated (what every caller should use); `predict_proba_raw`
    returns q_raw for diagnostics only - "Keep raw probability available
    for diagnostics if useful, but record separately: q_raw, q_calibrated,
    calibration_version." Does not implement q_safe/uncertainty logic -
    deferred to when real data exists to estimate it from (Phase 12B audit
    item 26/Tranche 4)."""

    raw_model: LogisticModel
    calibrator: PlattCalibrator | IsotonicCalibrator
    calibration_version: str

    def predict_proba(self, x: list[float]) -> float:
        return self.calibrator.transform(self.raw_model.predict_proba(x))

    def predict_proba_raw(self, x: list[float]) -> float:
        return self.raw_model.predict_proba(x)


def fit_calibrated_model(
    train_examples: list[Example],
    validate_examples: list[Example],
    feature_set: FeatureSet,
    calibration_version: str = "platt-v1",
) -> CalibratedModel | None:
    """Fits the raw logistic model on `train_examples` only, then Platt-
    calibrates it on `validate_examples` only - never the same examples
    for both (Phase 5's leakage discipline, reused verbatim here). Platt
    is used rather than isotonic for the same reason
    `run_model_training_demo.py` documents: this synthetic data is close
    to deterministically separable, so isotonic just memorizes validation
    instead of learning a smooth curve. Returns None if either split is
    empty - there's nothing to fit/calibrate."""
    if not train_examples or not validate_examples:
        return None

    X_train = [e.x for e in train_examples]
    y_train = [e.y for e in train_examples]
    raw_model = fit_logistic_regression(X_train, y_train, feature_set.name, feature_set.column_names)

    q_val_raw = [raw_model.predict_proba(e.x) for e in validate_examples]
    y_val = [e.y for e in validate_examples]
    calibrator = fit_platt(q_val_raw, y_val)

    return CalibratedModel(raw_model=raw_model, calibrator=calibrator, calibration_version=calibration_version)
