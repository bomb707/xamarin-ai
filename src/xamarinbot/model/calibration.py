"""Probability calibration (Roadmap Phase 5: "Calibrate probability with
isotonic or Platt-style calibration only on validation data if needed.")

Both calibrators are fit on the validation split only, never train or test
- fitting on train would just re-learn the training distribution, and
fitting on test would leak the evaluation set into the model.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from xamarinbot.model.logistic import LogisticModel, fit_logistic_regression

_LOGIT_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(1.0 - _LOGIT_EPS, max(_LOGIT_EPS, p))
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Pool-Adjacent-Violators isotonic regression: a monotonic non-
    decreasing step function mapping raw q -> calibrated q. Flat
    extrapolation outside the fitted range (standard isotonic-regression
    behavior)."""

    x_starts: tuple[float, ...]
    ys: tuple[float, ...]

    def transform(self, q_raw: float) -> float:
        idx = bisect.bisect_right(self.x_starts, q_raw) - 1
        idx = max(0, min(idx, len(self.ys) - 1))
        return self.ys[idx]


def fit_isotonic(q_raw: list[float], y: list[int]) -> IsotonicCalibrator:
    if not q_raw:
        raise ValueError("cannot calibrate on zero examples")
    order = sorted(range(len(q_raw)), key=lambda i: q_raw[i])
    # PAVA: each block is [x_start, mean_y, weight]; merge back-to-front
    # whenever adjacent block means violate monotonicity.
    blocks: list[list[float]] = []
    for i in order:
        blocks.append([q_raw[i], float(y[i]), 1.0])
        while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
            x2, y2, w2 = blocks.pop()
            x1, y1, w1 = blocks.pop()
            merged_w = w1 + w2
            merged_y = (y1 * w1 + y2 * w2) / merged_w
            blocks.append([x1, merged_y, merged_w])
    return IsotonicCalibrator(x_starts=tuple(b[0] for b in blocks), ys=tuple(b[1] for b in blocks))


@dataclass(frozen=True)
class PlattCalibrator:
    model: LogisticModel

    def transform(self, q_raw: float) -> float:
        return self.model.predict_proba([_logit(q_raw)])


def fit_platt(q_raw: list[float], y: list[int]) -> PlattCalibrator:
    if not q_raw:
        raise ValueError("cannot calibrate on zero examples")
    X = [[_logit(q)] for q in q_raw]
    model = fit_logistic_regression(X, y, feature_set_name="platt", column_names=("logit_q",), l2=0.0, lr=0.3, n_iters=300)
    return PlattCalibrator(model=model)


@dataclass(frozen=True)
class IdentityCalibrator:
    """Pass-through calibrator: returns `q_raw` unchanged. Used only when
    a genuine calibration fit is not possible - Phase 12B Tranche 1.2 item
    7: fitting Platt/isotonic on a one-class calibration set (e.g. every
    example labeled UP) would silently produce a degenerate, actively
    wrong calibrator (a constant output near 0 or 1, since the fit has no
    contrast to learn from), not a genuinely-uncalibrated-but-honest one.
    `CalibratedModel.calibration_version` records this state explicitly
    (`model.calibrated.UNCALIBRATED_INSUFFICIENT_CLASS_DIVERSITY`) so it
    is visible/auditable, never silently indistinguishable from "properly
    calibrated."."""

    def transform(self, q_raw: float) -> float:
        return q_raw
