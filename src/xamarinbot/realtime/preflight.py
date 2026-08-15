"""Capture -> per-round training eligibility (Gate A.0 items 1, 10).

Reads a real capture and answers, per round, "may this be used to fit a
model?" - separating label validity, market-data quality and projection
validity rather than collapsing them into `LabelStatus`.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from xamarinbot.eligibility import Disqualifier, RoundEligibility, build, summarize
from xamarinbot.realtime.attribution import (
    FAILURE_EVENT_TYPES,
    MATERIAL_GAP_NS,
    AttributionSummary,
    FailureAttribution,
    RoundWindow,
    StreamGap,
    attribute_gap,
)
from xamarinbot.realtime.label import RuleTextStatus, verify_rule_text
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
    """Whether this capture recorded any structured failure at all.

    Kept as a diagnostic only. Gate A.0.1 item 1: this must NOT be the
    switch that turns off the session-wide fallback - see
    `attribution_summary`. One recorded failure says nothing about the
    others.
    """
    return any(
        e.event_type == "parse_failure"
        for e in raw.events(topics=[Topic.RECORDER_CONTROL])
    )


def round_windows(raw: RawEventStore) -> list[RoundWindow]:
    """Every round's required recording interval, from the capture itself."""
    out = []
    for rid in raw.round_ids():
        row = raw.get_round(rid) or {}
        if row.get("start_ts_ns") is None or row.get("end_ts_ns") is None:
            continue
        out.append(RoundWindow(
            round_id=rid,
            condition_id=row.get("condition_id"),
            up_token_id=row.get("up_token_id"),
            down_token_id=row.get("down_token_id"),
            start_ts_ns=int(row["start_ts_ns"]),
            end_ts_ns=int(row["end_ts_ns"]),
        ))
    return out


#: Which raw topics carry the DATA of each stream, for reconstructing a
#: legacy outage's true interval from the observations around it.
_STREAM_DATA_TOPICS = {
    "rtds": [Topic.RTDS_BINANCE, Topic.RTDS_CHAINLINK,
             Topic.RTDS_TWAP_30, Topic.RTDS_TWAP_60],
    "clob": [Topic.CLOB_MARKET],
    "clob_market": [Topic.CLOB_MARKET],
}


def reconstruct_gap_interval(
    raw: RawEventStore, stream: str, at_ns: int
) -> tuple[int, int]:
    """The TRUE outage interval around a control event, from the data itself.

    Gate A.0.2 item 1 requires the interval to be the actual missing-
    observation period. Captures written before A.0.2 recorded only the
    instant the watchdog fired, which is neither end of the outage: the
    watchdog waits 30 seconds of silence before firing, and the stream stays
    dead through the reconnect afterwards.

    Both ends are recoverable from the raw log, because the log records
    exactly what did and did not arrive: the last data event on that stream
    before the control event, and the first one after it. That is a
    measurement, not an estimate - which is what makes legacy captures
    revalidatable rather than merely re-judged.
    """
    topics = _STREAM_DATA_TOPICS.get(stream, list(_STREAM_DATA_TOPICS["rtds"]))
    before = raw.last_recv_before(topics, at_ns)
    after = raw.first_recv_after(topics, at_ns)
    return (before if before is not None else at_ns,
            after if after is not None else at_ns)


def structured_failure_events(raw: RawEventStore) -> list[FailureAttribution]:
    """Every data-quality failure in a capture, as structured attributions.

    Gate A.0.2 item 3: ONE reader for all of them. Preflight previously
    looked only for `parse_failure`, so `stream_stalled` - the event the
    watchdog exists to produce - was outside the eligibility gate entirely.
    A 37-second RTDS blackout could be fully recorded and still leave every
    overlapping round `data_valid=True`.

    Three shapes are handled:

    * `data_gap` (A.0.2): already carries a measured interval and a verdict.
    * `parse_failure` (A.0.1): a point event about one frame.
    * `stream_stalled` / `reconnect` (legacy): no attribution at all, so the
      interval is reconstructed from the surrounding data and the affected
      rounds are recomputed. Outages shorter than one publication interval
      are dropped - a reconnect that resubscribes in milliseconds lost
      nothing, and treating it as damage would make the signal noise.
    """
    windows = round_windows(raw)
    out: list[FailureAttribution] = []
    for e in raw.events(topics=[Topic.RECORDER_CONTROL]):
        if e.event_type not in FAILURE_EVENT_TYPES:
            continue
        payload = e.payload
        if e.event_type in ("data_gap", "parse_failure"):
            out.append(FailureAttribution.from_payload(payload, e.event_type))
            continue

        # Legacy stall/reconnect: reconstruct rather than discard.
        stream = payload.get("stream") or ("clob" if e.event_type == "reconnect" else "rtds")
        at_ns = e.recv_wall_timestamp_ns
        start_ns, end_ns = reconstruct_gap_interval(raw, str(stream), at_ns)
        if end_ns - start_ns <= MATERIAL_GAP_NS:
            continue
        gap = StreamGap(
            stream="rtds" if str(stream).startswith("rtds") else "clob",
            failure_kind=e.event_type,
            last_data_ns=start_ns,
            detected_ns=at_ns,
            recovered_ns=end_ns,
        )
        out.append(dataclasses.replace(
            attribute_gap(gap, windows), source_event_type=e.event_type,
        ))
    return out


def attribution_summary(raw: RawEventStore) -> AttributionSummary:
    """How this capture's parse failures map onto its rounds.

    The session counter is the ground truth for HOW MANY failures happened;
    the control events are the record of WHICH markets each one damaged. The
    two are compared rather than one being trusted: a capture whose counter
    exceeds its records has failures nobody can place, and per-round
    attribution is then not available at all.
    """
    attributions = structured_failure_events(raw)

    # `RecorderMetrics.parse_failures` is a MONOTONIC session-wide counter,
    # snapshotted into each round's result as that round finalizes. Every
    # round in the batch therefore carries a running total, not its own
    # count. Summing them would multiply one failure by the round count
    # (measured: the eight rounds of a real batch report `events_received`
    # 1640670, 1640672, 1640674, ... - the same counter at eight instants).
    # The session total is the LAST snapshot, i.e. the maximum.
    session_parse, session_gaps = 0, 0
    for res in raw.round_results():
        blob = res.get("metrics_json")
        if not blob:
            continue
        try:
            metrics = json.loads(blob)
        except json.JSONDecodeError:
            continue
        m = metrics.get("session_metrics") or metrics
        session_parse = max(session_parse, int(m.get("parse_failures") or 0))
        session_gaps = max(session_gaps, int(m.get("data_gaps") or 0))

    # Compare like with like. A capture with three uncounted parse failures
    # and five well-attributed gaps is NOT fully attributed, however many
    # records it has in total.
    recorded_parse = sum(1 for a in attributions if a.source_event_type == "parse_failure")
    recorded_gaps = sum(1 for a in attributions if a.source_event_type == "data_gap")
    unrecorded = (max(0, session_parse - recorded_parse)
                  + max(0, session_gaps - recorded_gaps))

    return AttributionSummary(
        attributions=tuple(attributions),
        session_failure_count=session_parse + session_gaps,
        all_round_ids=tuple(raw.round_ids()),
        unrecorded_count=unrecorded,
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


def rule_text_status(raw: RawEventStore, round_id: str) -> RuleTextStatus:
    """Item 2: RE-DERIVE the rule-text verdict from the market metadata the
    capture persisted, rather than reading whatever the recorder wrote.

    Legacy captures recorded `rule_text_agrees=None` for every round -
    because nothing passed the text into `reconstruct_label`, not because
    the text was absent. It IS present: `rounds.resolution_source` and
    `rounds.description` were persisted at DISCOVERED all along. Reading the
    persisted verdict would therefore inherit the bug; recomputing it from
    the stored text is what "revalidate" means.
    """
    row = raw.get_round(round_id) or {}
    kind = row.get("settlement_kind")
    if not kind:
        return RuleTextStatus.SOURCE_TEXT_UNAVAILABLE
    window = row.get("twap_window_s")
    return verify_rule_text(
        kind,
        int(window) if window is not None else None,
        row.get("resolution_source"),
        row.get("description"),
    )


def verify_projection(raw: RawEventStore, round_id: str) -> tuple[bool, str | None]:
    """Item 3: actually PROJECT the round and report whether it succeeded.

    `projection_problems` below is a cheap screen over the round row. It
    cannot see a malformed book frame, a settlement topic with no usable
    observation at the boundary after time filtering, a payload the feature
    engine's contract rejects, or any of the other ways a real capture
    breaks halfway through. `projection_valid=True` therefore did not mean
    "this round can be projected" - it meant "nothing obvious forbids it".

    For Gate-A eligibility the projection is run for real, into a throwaway
    in-memory store. The exception text is returned verbatim; nothing is
    suppressed.
    """
    from xamarinbot.events.store import EventStore
    from xamarinbot.provenance import DataProvenance
    from xamarinbot.replay.projection import project_round

    out = EventStore(":memory:", provenance=DataProvenance.REAL_REPLAY)
    try:
        project_round(raw, round_id, out)
        return True, None
    except Exception as exc:  # noqa: BLE001 - the reason is the product here
        return False, f"{type(exc).__name__}: {exc}"[:400]
    finally:
        out.close()


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


def evaluate_round(
    raw: RawEventStore,
    round_id: str,
    *,
    verify_projection_run: bool = True,
    attribution: AttributionSummary | None = None,
) -> RoundEligibility:
    """One round's full Gate-A verdict.

    `verify_projection_run=False` skips item 3's real projection, which is
    the expensive part (~180k events per round). It is a diagnostic mode
    only: the resulting record has `projection_verified=False` and is
    therefore NOT training-eligible whatever else it passes, so a fast run
    can never be mistaken for a gate.
    """
    labels = label_fields(raw, round_id)
    mismatches = round_integrity_mismatches(raw, round_id)
    detail = {}
    if mismatches:
        detail[Disqualifier.BOOK_INTEGRITY_MISMATCH.value] = "; ".join(mismatches[:2])

    rule_status = rule_text_status(raw, round_id)
    if rule_status is RuleTextStatus.VERIFIED_FALSE:
        detail[Disqualifier.RULE_TEXT_DISAGREES.value] = (
            "the market's own text does not corroborate the settlement basis its "
            "structured configuration declares"
        )

    metrics = session_metrics(raw, round_id)
    summary = attribution if attribution is not None else attribution_summary(raw)
    affected = summary.affected_rounds()
    parse_failure_count = affected.get(round_id, 0)
    if metrics is not None and (summary.session_failure_count or summary.recorded_count):
        detail["parse_failure_attribution"] = summary.reason()
        if parse_failure_count:
            detail["parse_failures_affecting_this_round"] = str(parse_failure_count)

    problems = projection_problems(raw, round_id)
    projected_ok, projection_error = True, None
    if verify_projection_run:
        # Only run the real projection when the screen has not already ruled
        # the round out - projecting a round with no settlement rule would
        # just re-raise what the screen already reported, at 180k events of
        # cost per round.
        if problems:
            projected_ok = False
            projection_error = None
        else:
            projected_ok, projection_error = verify_projection(raw, round_id)
            if projection_error:
                detail[Disqualifier.PROJECTION_FAILED.value] = projection_error

    return build(
        round_id,
        label_status=labels["label_status"],
        reconstructed_outcome=labels["reconstructed_outcome"],
        reported_outcome=labels["reported_outcome"],
        declared_agrees=labels["declared_agrees"],
        metrics=metrics,
        round_integrity_mismatches=len(mismatches),
        projection_problems=problems,
        detail=detail,
        rule_text_status=rule_status.value,
        projection_error=projection_error,
        projection_verified=verify_projection_run and projected_ok,
        recorder_generation=raw.recorder_identity().recorder_generation,
        parse_failure_count=parse_failure_count,
    )


def evaluate_capture(
    raw: RawEventStore, *, verify_projection_run: bool = True
) -> list[RoundEligibility]:
    # One attribution pass per capture, not per round: it reads every
    # control event and every round result, and the answer is a property of
    # the capture as a whole.
    summary = attribution_summary(raw)
    return [
        evaluate_round(
            raw, rid,
            verify_projection_run=verify_projection_run,
            attribution=summary,
        )
        for rid in raw.round_ids()
    ]


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
            f"  rule-text VERIFIED_TRUE      {c.get('rule_text_verified', 0)}",
            f"  data-quality clean           {c['data_training_grade']}",
            f"  projection preconditions ok  {c.get('projection_preconditions_valid', 0)}",
            f"  ACTUAL projection valid      {c['projection_valid']}",
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


def preflight(raw: RawEventStore, *, verify_projection_run: bool = True) -> PreflightReport:
    records = evaluate_capture(raw, verify_projection_run=verify_projection_run)
    return PreflightReport(counts=summarize(records), records=records)
