"""Per-round training eligibility (Gate A.0 item 1).

    training_eligible = label_valid AND data_valid AND projection_valid

The bug this replaces
---------------------
`run_continuous_capture.py --status` equated `LabelStatus.CONFIRMED` with
"trainable". That conflates two independent questions:

  * **Is the label right?**  Did our independent reconstruction of the
    settlement outcome agree with the venue's?
  * **Is the market data right?**  Did the recorder capture this round's book
    and reference streams without dropping, mangling, or mis-reconstructing
    anything?

A round can have a perfectly CONFIRMED label sitting on top of a book that
went out of sync with the venue mid-round. Training on it teaches the model
features that never existed. In the first 32 captured rounds, 31 labels are
CONFIRMED and several of those rounds carry explicit book-integrity
mismatches - so "CONFIRMED" over-counted the usable dataset.

Conservative by construction
----------------------------
Every check answers "is there positive evidence this round is clean?", not
"is there evidence it is dirty?". Missing evidence disqualifies. A dataset
that is too small is a delay; a dataset that is quietly wrong is a false
result that survives every downstream check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Disqualifier(str, Enum):
    """Why a round is not training-eligible. One per distinct cause so the
    preflight can report reasons BY CATEGORY rather than a single count."""

    # --- label ---
    LABEL_NOT_CONFIRMED = "label_not_confirmed"
    LABEL_AMBIGUOUS = "label_ambiguous"
    LABEL_UNRESOLVED = "label_unresolved"
    NO_RECONSTRUCTED_OUTCOME = "no_reconstructed_outcome"
    NO_REPORTED_OUTCOME = "no_reported_outcome"
    RECONSTRUCTION_DISAGREES = "reconstruction_disagrees"
    RULE_TEXT_DISAGREES = "rule_text_disagrees"

    # --- recorder / market-data quality ---
    DROPPED_EVENTS = "dropped_events"
    PARSE_FAILURES = "parse_failures"
    BOOK_INTEGRITY_MISMATCH = "book_integrity_mismatch"
    NO_RECORDER_METRICS = "no_recorder_metrics"

    # --- projection ---
    PROJECTION_FAILED = "projection_failed"
    MISSING_SETTLEMENT_RULE = "missing_settlement_rule"
    NO_REFERENCE_AT_BOUNDARY = "no_reference_at_boundary"
    ROUND_NOT_FINALIZED = "round_not_finalized"


#: Which disqualifiers belong to which of the three independent gates.
LABEL_DISQUALIFIERS = frozenset({
    Disqualifier.LABEL_NOT_CONFIRMED, Disqualifier.LABEL_AMBIGUOUS,
    Disqualifier.LABEL_UNRESOLVED, Disqualifier.NO_RECONSTRUCTED_OUTCOME,
    Disqualifier.NO_REPORTED_OUTCOME, Disqualifier.RECONSTRUCTION_DISAGREES,
    Disqualifier.RULE_TEXT_DISAGREES,
})
DATA_DISQUALIFIERS = frozenset({
    Disqualifier.DROPPED_EVENTS, Disqualifier.PARSE_FAILURES,
    Disqualifier.BOOK_INTEGRITY_MISMATCH, Disqualifier.NO_RECORDER_METRICS,
})
PROJECTION_DISQUALIFIERS = frozenset({
    Disqualifier.PROJECTION_FAILED, Disqualifier.MISSING_SETTLEMENT_RULE,
    Disqualifier.NO_REFERENCE_AT_BOUNDARY, Disqualifier.ROUND_NOT_FINALIZED,
})


@dataclass(frozen=True)
class RoundEligibility:
    """One round's answer to "may this be used to fit a model?"."""

    round_id: str
    label_valid: bool
    data_valid: bool
    projection_valid: bool
    disqualifiers: tuple[Disqualifier, ...] = ()
    #: Free-text detail per disqualifier, for the preflight report.
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def training_eligible(self) -> bool:
        return self.label_valid and self.data_valid and self.projection_valid

    @property
    def label_disqualifiers(self) -> tuple[Disqualifier, ...]:
        return tuple(d for d in self.disqualifiers if d in LABEL_DISQUALIFIERS)

    @property
    def data_disqualifiers(self) -> tuple[Disqualifier, ...]:
        return tuple(d for d in self.disqualifiers if d in DATA_DISQUALIFIERS)

    @property
    def projection_disqualifiers(self) -> tuple[Disqualifier, ...]:
        return tuple(d for d in self.disqualifiers if d in PROJECTION_DISQUALIFIERS)

    def as_index_fields(self) -> dict:
        """The fields Gate A.0 item 1 requires on every INDEX.jsonl record."""
        return {
            "label_valid": self.label_valid,
            "data_training_grade": self.data_valid,
            "projection_valid": self.projection_valid,
            "training_eligible": self.training_eligible,
            "data_disqualifiers": [d.value for d in self.data_disqualifiers],
            "label_disqualifiers": [d.value for d in self.label_disqualifiers],
            "projection_disqualifiers": [d.value for d in self.projection_disqualifiers],
            "disqualifiers": [d.value for d in self.disqualifiers],
        }


def evaluate_label(
    label_status: str | None,
    reconstructed_outcome: str | None,
    reported_outcome: str | None,
    declared_agrees: bool | None,
) -> list[Disqualifier]:
    """Item 1's `label_valid` predicate, in full.

    Every one of these is a genuinely different failure. `LabelStatus`
    already folds several together, so they are re-derived here rather than
    trusted, and each is reported separately.
    """
    out: list[Disqualifier] = []
    if label_status == "LABEL_AMBIGUOUS":
        out.append(Disqualifier.LABEL_AMBIGUOUS)
    elif label_status == "UNRESOLVED":
        out.append(Disqualifier.LABEL_UNRESOLVED)
    elif label_status != "CONFIRMED":
        out.append(Disqualifier.LABEL_NOT_CONFIRMED)
    if reconstructed_outcome is None:
        out.append(Disqualifier.NO_RECONSTRUCTED_OUTCOME)
    if reported_outcome is None:
        out.append(Disqualifier.NO_REPORTED_OUTCOME)
    if declared_agrees is not True:
        out.append(Disqualifier.RECONSTRUCTION_DISAGREES)
    return out


def evaluate_data_quality(
    metrics: dict | None,
    round_integrity_mismatches: int,
) -> list[Disqualifier]:
    """Item 1's `data_valid` predicate.

    `metrics` are SESSION-level (one recorder run captures several rounds),
    so a drop or parse failure anywhere in the batch disqualifies every round
    in it. That is deliberate: the recorder's bounded queue drops without
    recording WHICH round lost an event, so no round in that session can be
    proven clean. Book-integrity mismatches, by contrast, are attributable to
    a specific round and are checked per round.
    """
    out: list[Disqualifier] = []
    if metrics is None:
        out.append(Disqualifier.NO_RECORDER_METRICS)
        return out
    if metrics.get("dropped_events"):
        out.append(Disqualifier.DROPPED_EVENTS)
    if metrics.get("parse_failures"):
        out.append(Disqualifier.PARSE_FAILURES)
    if round_integrity_mismatches:
        out.append(Disqualifier.BOOK_INTEGRITY_MISMATCH)
    return out


def build(
    round_id: str,
    *,
    label_status: str | None,
    reconstructed_outcome: str | None,
    reported_outcome: str | None,
    declared_agrees: bool | None,
    metrics: dict | None,
    round_integrity_mismatches: int,
    projection_problems: list[Disqualifier] | None = None,
    detail: dict[str, str] | None = None,
) -> RoundEligibility:
    label_bad = evaluate_label(
        label_status, reconstructed_outcome, reported_outcome, declared_agrees
    )
    data_bad = evaluate_data_quality(metrics, round_integrity_mismatches)
    proj_bad = list(projection_problems or ())
    return RoundEligibility(
        round_id=round_id,
        label_valid=not label_bad,
        data_valid=not data_bad,
        projection_valid=not proj_bad,
        disqualifiers=tuple(label_bad + data_bad + proj_bad),
        detail=dict(detail or {}),
    )


def summarize(records: list[RoundEligibility]) -> dict:
    """The Gate A.0 item 10 preflight counts, plus reasons by category."""
    import collections

    by_reason: collections.Counter = collections.Counter()
    for r in records:
        for d in r.disqualifiers:
            by_reason[d.value] += 1
    return {
        "captured": len(records),
        "label_valid": sum(1 for r in records if r.label_valid),
        "data_training_grade": sum(1 for r in records if r.data_valid),
        "projection_valid": sum(1 for r in records if r.projection_valid),
        "training_eligible": sum(1 for r in records if r.training_eligible),
        "disqualifiers_by_reason": dict(by_reason.most_common()),
        "eligible_round_ids": [r.round_id for r in records if r.training_eligible],
    }
