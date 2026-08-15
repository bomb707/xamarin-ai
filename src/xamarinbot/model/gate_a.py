"""The one canonical REAL training-dataset path (Gate A.0.1 item 4).

    RawCapture
        -> RoundEligibility        (label, market data, rule text)
        -> actual projection       (not a precondition screen)
        -> verified RoundLabel     (CONFIRMED reconstruction only)
        -> REAL causal features    (receive-time gated, fixed decision grid)
        -> training examples       (round-balanced, total weight 1 per round)

Why this module exists
----------------------
Every stage above already existed and was already correct in isolation. The
gap was that they were connected BY THE CALLER. A script that projected a
capture, collected the `RoundLabel`s the projection returned, and passed
them to `build_real_examples` would produce a dataset containing rounds with
book-integrity mismatches, unattributed parse failures, or a rule text that
contradicts the settlement basis - not through any bug, but because nothing
in the type system said eligibility had to be consulted.

Correctness that depends on the caller remembering is not correctness. This
module is the only supported way to turn a real capture into training data,
and it consults eligibility structurally: an ineligible round is never
projected into the training store in the first place, and
`build_real_examples` independently refuses it even if one gets through.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.eligibility import RoundEligibility
from xamarinbot.events.store import EventStore
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.features import FeatureSet
from xamarinbot.model.real_dataset import (
    RealDatasetResult,
    TrainingEligibilityError,
    build_real_examples,
)
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.preflight import evaluate_capture
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.replay.projection import project_round
from xamarinbot.rounds import RoundLabel


@dataclass
class GateADataset:
    """The dataset plus the full audit trail of what was excluded and why."""

    dataset: RealDatasetResult
    eligibility: dict[str, RoundEligibility]
    included_rounds: list[str] = field(default_factory=list)
    excluded_rounds: dict[str, list[str]] = field(default_factory=dict)
    #: Rounds that passed eligibility but produced no usable target or no
    #: valid decision point - reported, never silently dropped.
    projected_without_label: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict:
        return {
            "rounds_in_capture": len(self.eligibility),
            "rounds_eligible": len(self.included_rounds) + len(self.projected_without_label),
            "rounds_in_dataset": len(self.included_rounds),
            "rounds_excluded": len(self.excluded_rounds),
            "examples": len(self.dataset.examples),
        }


def build_gate_a_dataset(
    raw: RawEventStore,
    feature_cfg: FeatureConfig,
    feature_set: FeatureSet,
    *,
    out_store: EventStore | None = None,
    grid: tuple[float, ...] | None = None,
) -> GateADataset:
    """Raw capture in, training examples out, eligibility enforced throughout.

    Gate A.0.2 item 4: there is deliberately NO `eligibility=` parameter.

    A.0.1 accepted a caller-supplied map "as a cache", enforcing it rather
    than trusting it as a filter - but that distinction does not survive
    contact with a caller who simply constructs
    `RoundEligibility(training_eligible=True)`. A plain dict is an
    assertion, not evidence, and an invariant that any caller can assert
    their way past is not an invariant. Eligibility is therefore always
    derived HERE, from the raw capture, by the same code the preflight runs.

    The cost is real - `evaluate_capture` projects every round for item 3's
    verification, about nine seconds each - and it is the price of the
    guarantee. Callers who want the verdicts without the dataset should call
    `evaluate_capture(raw)` directly and read them; what they cannot do is
    hand a verdict back in and have it believed.
    """
    by_round = {r.round_id: r for r in evaluate_capture(raw)}

    store = out_store or EventStore(":memory:", provenance=DataProvenance.REAL_REPLAY)
    included: list[str] = []
    excluded: dict[str, list[str]] = {}
    no_label: list[str] = []
    labels: list[RoundLabel] = []

    for round_id, record in by_round.items():
        if not record.training_eligible:
            excluded[round_id] = [d.value for d in record.disqualifiers] or ["not_eligible"]
            continue
        try:
            result = project_round(raw, round_id, store)
        except Exception as exc:  # noqa: BLE001
            # Eligibility said this round projects. If it does not, that is a
            # disagreement worth surfacing, not an exception to swallow.
            excluded[round_id] = [f"projection_failed:{type(exc).__name__}: {exc}"[:200]]
            continue
        if result.label is None:
            no_label.append(round_id)
            continue
        labels.append(result.label)
        included.append(round_id)

    dataset = (
        build_real_examples(
            store, labels, feature_cfg, feature_set,
            eligibility=by_round, grid=grid,
        )
        if labels
        else RealDatasetResult(feature_set=feature_set.name)
    )
    return GateADataset(
        dataset=dataset,
        eligibility=by_round,
        included_rounds=included,
        excluded_rounds=excluded,
        projected_without_label=no_label,
    )


__all__ = [
    "GateADataset",
    "TrainingEligibilityError",
    "build_gate_a_dataset",
]
