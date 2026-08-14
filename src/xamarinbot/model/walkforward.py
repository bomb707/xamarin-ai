"""Time-ordered train/validation/test split (Roadmap Phase 5: "Create
walk-forward train/validation/test splits by time, never random row splits
across adjacent observations.")

This is a single chronological split (train = earliest slice, validation =
middle slice, test = latest slice). The full rolling multi-window
walk-forward procedure across many train/validate/test segments is Roadmap
Phase 11's job ("Walk-Forward Calibration and Ablations") - conflating the
two would overstate what Phase 5 delivers.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.model.dataset import Example


@dataclass(frozen=True)
class WalkForwardSplit:
    train: list[Example]
    validation: list[Example]
    test: list[Example]

    @property
    def training_window(self) -> tuple[float, float]:
        if not self.train:
            return (0.0, 0.0)
        return (self.train[0].decision_ts, self.train[-1].decision_ts)


def time_ordered_split(examples: list[Example], train_frac: float = 0.6, val_frac: float = 0.2) -> WalkForwardSplit:
    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0) or train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0, leaving a non-empty test slice")

    ordered = sorted(examples, key=lambda e: e.decision_ts)
    n = len(ordered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return WalkForwardSplit(
        train=ordered[:n_train],
        validation=ordered[n_train : n_train + n_val],
        test=ordered[n_train + n_val :],
    )
