"""Phase 12B Tranche 1.2 item 7: probability calibration must be robust
to one-class calibration windows - Platt/isotonic must never be silently
fit on a constant y (all-UP or all-DOWN), which produces a degenerate,
actively wrong calibrator rather than an honestly-uncalibrated one.
"""
from __future__ import annotations

from xamarinbot.model.calibrated import UNCALIBRATED_INSUFFICIENT_CLASS_DIVERSITY, fit_calibrated_model
from xamarinbot.model.calibration import IdentityCalibrator
from xamarinbot.model.dataset import Example
from xamarinbot.model.features import TWAP_ONLY
from xamarinbot.model.logistic import fit_logistic_regression


def _example(round_id: str, decision_ts: float, x: list[float], y: int) -> Example:
    return Example(round_id=round_id, decision_ts=decision_ts, features=None, x=x, y=y)


def _train_examples(n: int = 20) -> list[Example]:
    # alternating labels so the raw model itself has something to learn
    # (TWAP_ONLY has 3 design-vector columns: z_gap, tau, z_gap*tau)
    return [_example(f"train-{i}", float(i), [float(i % 5), 0.5, 0.1], i % 2) for i in range(n)]


def test_all_up_calibration_rounds_fall_back_to_identity():
    train = _train_examples()
    calib = [_example("calib-0", 100.0 + i, [1.0, 0.5, 0.1], 1) for i in range(5)]  # every label UP (y=1)

    model = fit_calibrated_model(train, calib, TWAP_ONLY)

    assert model is not None
    assert isinstance(model.calibrator, IdentityCalibrator)
    assert model.calibration_version == UNCALIBRATED_INSUFFICIENT_CLASS_DIVERSITY
    assert not model.is_calibrated
    # pass-through: predict_proba must equal the raw model's own output
    for e in calib:
        assert model.predict_proba(e.x) == model.predict_proba_raw(e.x)


def test_all_down_calibration_rounds_fall_back_to_identity():
    train = _train_examples()
    calib = [_example("calib-0", 100.0 + i, [1.0, 0.5, 0.1], 0) for i in range(5)]  # every label DOWN (y=0)

    model = fit_calibrated_model(train, calib, TWAP_ONLY)

    assert model is not None
    assert isinstance(model.calibrator, IdentityCalibrator)
    assert model.calibration_version == UNCALIBRATED_INSUFFICIENT_CLASS_DIVERSITY
    assert not model.is_calibrated


def test_mixed_calibration_rounds_calibrate_normally():
    train = _train_examples()
    calib = [_example("calib-0", 100.0 + i, [float(i % 3), 0.5, 0.1], i % 2) for i in range(10)]  # both classes present

    model = fit_calibrated_model(train, calib, TWAP_ONLY)

    assert model is not None
    assert not isinstance(model.calibrator, IdentityCalibrator)
    assert model.calibration_version == "platt-v1"
    assert model.is_calibrated


def test_insufficient_number_of_calibration_examples_returns_none():
    """Distinct from the class-diversity fallback: zero calibration
    examples at all is still the pre-existing "nothing to fit" case, not
    a one-class case - fit_calibrated_model must still return None, not
    silently produce an IdentityCalibrator for a genuinely empty split."""
    train = _train_examples()
    model = fit_calibrated_model(train, [], TWAP_ONLY)
    assert model is None
