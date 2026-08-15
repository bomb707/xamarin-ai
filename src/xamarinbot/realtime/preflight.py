"""Capture -> per-round training eligibility (Gate A.0 items 1, 10).

Reads a real capture and answers, per round, "may this be used to fit a
model?" - separating label validity, market-data quality and projection
validity rather than collapsing them into `LabelStatus`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from xamarinbot.eligibility import Disqualifier, RoundEligibility, build, summarize
from xamarinbot.realtime.raw_events import Topic
from xamarinbot.realtime.raw_store import RawEventStore


def round_integrity_mismatches(raw: RawEventStore, round_id: str) -> list[str]:
    """Book-integrity checks that FAILED for this round.

    The recorder resnapshots after a mismatch, so the book is repaired going
    forward - but the interval between the last passing check and the
    mismatch is of unknown validity, and that interval is inside the round.
    Those decision points would train the model on a book that had drifted
    from the venue's, so the round is disqualified rather than partially
    salvaged. (Salvaging would mean trusting our own guess about when the
    drift started, which is precisely what the check just proved we cannot do.)
    """
    out = []
    for e in raw.events(round_id=round_id, topics=[Topic.RECORDER_CONTROL]):
        if e.event_type != "book_integrity_check":
            continue
        payload = e.payload
        if not payload.get("matched"):
            out.append(payload.get("detail") or "book integrity mismatch")
    return out


def capture_records_parse_failure_events(raw: RawEventStore) -> bool:
    """Whether this capture is new enough to timestamp its parse failures.

    Captures recorded before Gate A.0 only have a session counter, so a
    failure cannot be attributed to a round and the conservative
    session-wide rule applies to all of them.
    """
    return any(
        e.event_type == "parse_failure"
        for e in raw.events(topics=[Topic.RECORDER_CONTROL])
    )


def round_parse_failures(raw: RawEventStore, round_id: str) -> int:
    """Parse failures recorded while THIS round was the active one."""
    return sum(
        1 for e in raw.events(round_id=round_id, topics=[Topic.RECORDER_CONTROL])
        if e.event_type == "parse_failure"
    )


def session_metrics(raw: RawEventStore, round_id: str) -> dict | None:
    """The recorder's health counters for the session that captured this
    round, taken from its own persisted `round_results` row."""
    for res in raw.round_results():
        if res.get("round_id") != round_id:
            continue
        blob = res.get("metrics_json")
        if not blob:
            return None
        try:
            metrics = json.loads(blob)
        except json.JSONDecodeError:
            return None
        return metrics.get("session_metrics") or metrics
    return None


def label_fields(raw: RawEventStore, round_id: str) -> dict:
    """The round's label outcome as the recorder persisted it, including the
    `LabelStatus` (which now incorporates the rule-text cross-check)."""
    fields = {
        "label_status": None, "reconstructed_outcome": None,
        "reported_outcome": None, "declared_agrees": None,
        "rule_text_agrees": None,
    }
    for e in raw.events(round_id=round_id, topics=[Topic.RECORDER_CONTROL]):
        if e.event_type not in ("label_reconstruction", "label_reconstruction_resolved"):
            continue
        p = e.payload
        fields["label_status"] = p.get("status") or fields["label_status"]
        fields["declared_agrees"] = p.get("declared_agrees")
        fields["rule_text_agrees"] = p.get("rule_text_agrees")
        fields["reported_outcome"] = p.get("reported_outcome")
        fields["reconstructed_outcome"] = (p.get("declared") or {}).get("outcome")
    for res in raw.round_results():
        if res.get("round_id") != round_id:
            continue
        fields["reported_outcome"] = res.get("reported_outcome") or fields["reported_outcome"]
        fields["reconstructed_outcome"] = (
            res.get("reconstructed_outcome") or fields["reconstructed_outcome"]
        )
        if fields["declared_agrees"] is None and res.get("label_agreement") is not None:
            fields["declared_agrees"] = bool(res["label_agreement"])

    # Captures written before Gate A.0 item 3 have no `status` in their
    # `label_reconstruction` payload, because the field did not exist and the
    # rule-text cross-check was never wired in. Item 1 says to REVALIDATE
    # such captures rather than discard them, so the status is re-derived
    # here from the outcomes that WERE recorded, using the same rule
    # `LabelReconstruction.status` applies.
    if fields["label_status"] is None:
        fields["label_status"] = _derive_label_status(fields)
    return fields


def _derive_label_status(fields: dict) -> str:
    """`LabelStatus` re-derived for a pre-Gate-A.0 capture."""
    reported = fields["reported_outcome"]
    reconstructed = fields["reconstructed_outcome"]
    if reported is None or reconstructed is None:
        return "UNRESOLVED"
    if reported != reconstructed:
        return "LABEL_AMBIGUOUS"
    if fields["rule_text_agrees"] is False:
        return "LABEL_AMBIGUOUS"
    return "CONFIRMED"


def projection_problems(raw: RawEventStore, round_id: str) -> list[Disqualifier]:
    """Whether this round can be projected at all, checked WITHOUT running the
    (expensive) projection: the same preconditions `project_round` enforces."""
    out: list[Disqualifier] = []
    row = raw.get_round(round_id)
    if row is None:
        return [Disqualifier.PROJECTION_FAILED]
    if not row.get("settlement_kind"):
        out.append(Disqualifier.MISSING_SETTLEMENT_RULE)
    if row.get("state") != "FINALIZED":
        out.append(Disqualifier.ROUND_NOT_FINALIZED)

    from xamarinbot.replay.projection import ProjectionError, settlement_topic_for

    if row.get("settlement_kind"):
        try:
            topic = settlement_topic_for(row["settlement_kind"], row.get("twap_window_s"))
        except ProjectionError:
            out.append(Disqualifier.MISSING_SETTLEMENT_RULE)
            return out
        from xamarinbot.replay.projection import reference_events

        start_ts = (row.get("start_ts_ns") or 0) / 1e9
        end_ts = (row.get("end_ts_ns") or 0) / 1e9
        # Reference feeds are GLOBAL - see `projection.reference_events`.
        # Selecting by `round_id` here would report almost every round in a
        # multi-round batch as missing boundary data that is in fact present,
        # merely filed under a neighbouring round.
        obs = [
            e.source_timestamp_ns / 1e9
            for e in reference_events(raw, topic)
            if e.source_timestamp_ns is not None
        ]
        # The label needs a reference observation on BOTH sides of the round.
        if not obs or min(obs) > start_ts or max(obs) < end_ts:
            out.append(Disqualifier.NO_REFERENCE_AT_BOUNDARY)
    return out


def evaluate_round(raw: RawEventStore, round_id: str) -> RoundEligibility:
    labels = label_fields(raw, round_id)
    mismatches = round_integrity_mismatches(raw, round_id)
    detail = {}
    if mismatches:
        detail[Disqualifier.BOOK_INTEGRITY_MISMATCH.value] = "; ".join(mismatches[:2])
    if labels["rule_text_agrees"] is False:
        detail[Disqualifier.RULE_TEXT_DISAGREES.value] = (
            "the market's structured settlement config contradicts its own rule text"
        )
    metrics = session_metrics(raw, round_id)
    # If the capture timestamps its parse failures, attribute them to the
    # round they actually happened in; otherwise fall back to the
    # session-wide counter, which condemns the whole batch.
    # A capture with zero session parse failures needs no attribution at all.
    if metrics is not None and not metrics.get("parse_failures"):
        pass
    elif metrics is not None and capture_records_parse_failure_events(raw):
        metrics = dict(metrics, parse_failures=round_parse_failures(raw, round_id))
        detail["parse_failure_attribution"] = "per-round (timestamped events present)"
    elif metrics is not None and metrics.get("parse_failures"):
        detail["parse_failure_attribution"] = (
            "session-wide: this capture predates timestamped parse-failure events, so "
            "the failure cannot be attributed to a round and every round in the batch "
            "is disqualified"
        )
    record = build(
        round_id,
        label_status=labels["label_status"],
        reconstructed_outcome=labels["reconstructed_outcome"],
        reported_outcome=labels["reported_outcome"],
        declared_agrees=labels["declared_agrees"],
        metrics=metrics,
        round_integrity_mismatches=len(mismatches),
        projection_problems=projection_problems(raw, round_id),
        detail=detail,
    )
    if labels["rule_text_agrees"] is False:
        record = RoundEligibility(
            round_id=record.round_id,
            label_valid=False,
            data_valid=record.data_valid,
            projection_valid=record.projection_valid,
            disqualifiers=record.disqualifiers + (Disqualifier.RULE_TEXT_DISAGREES,),
            detail=record.detail,
        )
    return record


def evaluate_capture(raw: RawEventStore) -> list[RoundEligibility]:
    return [evaluate_round(raw, rid) for rid in raw.round_ids()]


@dataclass(frozen=True)
class PreflightReport:
    counts: dict
    records: list[RoundEligibility]

    def format(self) -> str:
        c = self.counts
        lines = [
            "=" * 78,
            "GATE A.0 - REAL TRAINING-DATASET INTEGRITY PREFLIGHT",
            "=" * 78,
            "",
            "  Label validity, market-data quality and projection validity are",
            "  INDEPENDENT. A CONFIRMED label on a round whose book drifted from",
            "  the venue is not a trainable round.",
            "",
            f"  captured rounds              {c['captured']}",
            f"  label CONFIRMED (valid)      {c['label_valid']}",
            f"  data-quality clean           {c['data_training_grade']}",
            f"  projection valid             {c['projection_valid']}",
            f"  FINAL training eligible      {c['training_eligible']}",
            "",
            "  DISQUALIFICATION REASONS BY CATEGORY",
            "  " + "-" * 74,
        ]
        if c["disqualifiers_by_reason"]:
            for reason, n in c["disqualifiers_by_reason"].items():
                lines.append(f"    {reason:<40} {n:>5}")
        else:
            lines.append("    (none)")
        lines.append("")
        lines.append("=" * 78)
        return "\n".join(lines)


def preflight(raw: RawEventStore) -> PreflightReport:
    records = evaluate_capture(raw)
    return PreflightReport(counts=summarize(records), records=records)
