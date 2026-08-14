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


def round_ordered_split(examples: list[Example], train_frac: float = 0.6, val_frac: float = 0.2) -> WalkForwardSplit:
    """Same chronological train/validation/test split as `time_ordered_split`,
    but splits at ROUND granularity, never mid-round (Phase 12B Tranche
    1.2 item 8).

    `time_ordered_split` splits by example-count fraction - since a
    single round contributes many decision-point examples that all share
    that round's one eventual settlement outcome and are strongly
    serially correlated, an example-count split can and does place some
    of a round's own examples in one split and the rest of that *same
    round* in another (the same round-leakage problem the walk-forward
    pipeline's own `_split_rounds_for_fit_and_calibration` was fixed for
    - Phase 12B Tranche 1.1 item 1). This is the drop-in, round-aware
    replacement for every caller that fits a model outside the main
    walk-forward pipeline (a single global split, not itself Phase 11's
    rolling-window procedure) but must still not leak a round across the
    fit/calibration boundary merely because it isn't participating in
    that pipeline.

    Rounds are ordered by their own earliest `decision_ts` (each round's
    examples are already internally time-ordered, and rounds do not
    interleave in this codebase's replay model - `synthetic/rounds.py`
    lays out one round strictly after the previous one's end)."""
    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0) or train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0, leaving a non-empty test slice")

    by_round: dict[str, list[Example]] = {}
    for e in examples:
        by_round.setdefault(e.round_id, []).append(e)
    round_ids_ordered = sorted(by_round.keys(), key=lambda rid: min(e.decision_ts for e in by_round[rid]))

    n = len(round_ids_ordered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_rids = round_ids_ordered[:n_train]
    val_rids = round_ids_ordered[n_train : n_train + n_val]
    test_rids = round_ids_ordered[n_train + n_val :]

    def _gather(rids: list[str]) -> list[Example]:
        out: list[Example] = []
        for rid in rids:
            out.extend(by_round[rid])
        out.sort(key=lambda e: e.decision_ts)
        return out

    return WalkForwardSplit(train=_gather(train_rids), validation=_gather(val_rids), test=_gather(test_rids))
