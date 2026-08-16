"""Structured parse-failure attribution (Gate A.0.1 item 1).

What was wrong
--------------
Gate A.0 replaced a session-wide parse-failure counter with a timestamped
control event, and tagged that event with

    the round that happened to be ACTIVE

which is not attribution at all. It is a guess that is right only when the
failure concerns the currently-trading market. Three cases where it is
positively wrong:

* a `/book` bootstrap or integrity resnapshot for a **future** round's token
  fails while an earlier round is trading - the ACTIVE round is blamed for a
  failure that did not touch it, and the round that actually lost data is
  recorded as clean;
* a CLOB **connection** drop affects every subscribed market whose recording
  window overlaps the outage, not one of them;
* an **RTDS** failure concerns the global BTC reference series, which feeds
  the PRE_ROUND and ACTIVE windows of several overlapping rounds at once.

Mis-attribution is worse than no attribution, because it produces a clean
verdict for a round that lost data. This module therefore refuses to guess:
a failure it cannot place becomes `UNATTRIBUTED`, and an UNATTRIBUTED
failure conservatively affects every round it could possibly have touched.

The completeness invariant
--------------------------
The presence of *one* well-attributed failure must never license dropping
the session-wide fallback for the rest. Per-round attribution may replace
the session-wide rule only when EVERY failure in the capture is accounted
for and placed - see `attribution_is_complete`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum


class AttributionStatus(str, Enum):
    """How well a failure could be placed. Ordered worst-last for reporting."""

    #: A token or condition id in the failure itself names the market.
    EXACT = "EXACT"
    #: No identifier, but the failure has a known interval and the affected
    #: markets are those whose required recording window overlaps it.
    WINDOW_INFERRED = "WINDOW_INFERRED"
    #: A global stream (RTDS) failed; every round whose required window
    #: overlaps the outage lost the same observations.
    GLOBAL_WINDOW = "GLOBAL_WINDOW"
    #: Could not be placed. Fails conservatively - see the module docstring.
    UNATTRIBUTED = "UNATTRIBUTED"

    @property
    def is_trustworthy(self) -> bool:
        return self is not AttributionStatus.UNATTRIBUTED


class Stream(str, Enum):
    CLOB = "clob"
    RTDS = "rtds"
    UNKNOWN = "unknown"


#: Bumped when the persisted attribution payload changes shape.
#: 1 = Gate A.0.1 (point failures); 2 = Gate A.0.2 (measured outage
#: intervals, `data_gap` events); 3 = Gate A.0.2.1 (per-topic:
#: `wire_topic` and `expected_symbol` name the series that went dark).
ATTRIBUTION_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class RoundWindow:
    """A round's REQUIRED recording interval, in wall nanoseconds.

    "Required" is wider than the trading window: PRE_ROUND lookback feeds
    the volatility and lead-lag features that exist at t=15s, and the
    post-round tail carries the settling book and the closing reference
    observation the label is reconstructed from. A failure anywhere in this
    interval damages the round.
    """

    round_id: str
    condition_id: str | None
    up_token_id: str | None
    down_token_id: str | None
    start_ts_ns: int
    end_ts_ns: int
    pre_round_lead_ns: int = 420_000_000_000
    post_round_tail_ns: int = 90_000_000_000

    @property
    def required_start_ns(self) -> int:
        return self.start_ts_ns - self.pre_round_lead_ns

    @property
    def required_end_ns(self) -> int:
        return self.end_ts_ns + self.post_round_tail_ns

    def covers(self, ts_ns: int) -> bool:
        return self.required_start_ns <= ts_ns <= self.required_end_ns

    def overlaps(self, start_ns: int, end_ns: int) -> bool:
        return start_ns <= self.required_end_ns and end_ns >= self.required_start_ns

    def owns_token(self, token_id: str | None) -> bool:
        return token_id is not None and token_id in (self.up_token_id, self.down_token_id)


@dataclass(frozen=True)
class FailureAttribution:
    """One parse/stream failure and the rounds it damaged."""

    stream: str
    failure_kind: str
    recv_timestamp_ns: int
    token_id: str | None
    condition_id: str | None
    affected_round_ids: tuple[str, ...]
    attribution_status: AttributionStatus
    raw_excerpt: str
    #: Outage interval when the failure spans time rather than one message.
    interval_start_ns: int | None = None
    interval_end_ns: int | None = None
    detail: str = ""
    #: Gate A.0.2.1 item 4: the specific BTC series, for RTDS gaps.
    wire_topic: str | None = None
    expected_symbol: str | None = None
    #: Which control event this came from (`parse_failure`, `data_gap`, or a
    #: legacy `stream_stalled`/`reconnect` reconstructed after the fact).
    #: Needed so completeness compares like with like - unrecorded PARSE
    #: failures against parse records, unrecorded GAPS against gap records.
    #: Without it, five reconstructed gaps would "cover" three unrecorded
    #: parse failures and wrongly declare the capture fully attributed.
    source_event_type: str = ""

    @property
    def is_trustworthy(self) -> bool:
        return self.attribution_status.is_trustworthy

    @property
    def is_gap(self) -> bool:
        """A missing-observation OUTAGE rather than an unreadable frame."""
        return (
            self.source_event_type in ("data_gap", "stream_stalled", "reconnect",
                                       "topic_stalled")
            or self.failure_kind in ("stream_stalled", "connection_gap",
                                     "resync_gap", "reconnect", "topic_stalled")
        )

    def as_payload(self) -> dict:
        return {
            "schema_version": ATTRIBUTION_SCHEMA_VERSION,
            "stream": self.stream,
            "failure_kind": self.failure_kind,
            "recv_timestamp_ns": self.recv_timestamp_ns,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "affected_round_ids": list(self.affected_round_ids),
            "attribution_status": self.attribution_status.value,
            "raw_excerpt": self.raw_excerpt,
            "interval_start_ns": self.interval_start_ns,
            "interval_end_ns": self.interval_end_ns,
            "detail": self.detail,
            "wire_topic": self.wire_topic,
            "expected_symbol": self.expected_symbol,
        }

    @classmethod
    def from_payload(cls, payload: dict, source_event_type: str = "") -> "FailureAttribution":
        """Rebuild from a persisted control event.

        A payload written before this module existed has no
        `attribution_status`; it is read back as UNATTRIBUTED rather than
        being trusted, because its `active_round_id` was exactly the guess
        this module replaces.
        """
        raw_status = payload.get("attribution_status")
        try:
            status = AttributionStatus(raw_status)
        except ValueError:
            status = AttributionStatus.UNATTRIBUTED
        rounds = payload.get("affected_round_ids")
        if rounds is None:
            legacy = payload.get("active_round_id")
            rounds = [legacy] if legacy else []
            status = AttributionStatus.UNATTRIBUTED
        return cls(
            stream=payload.get("stream") or Stream.UNKNOWN.value,
            failure_kind=payload.get("failure_kind") or payload.get("error_type") or "unknown",
            recv_timestamp_ns=int(payload.get("recv_timestamp_ns") or 0),
            token_id=payload.get("token_id"),
            condition_id=payload.get("condition_id"),
            affected_round_ids=tuple(r for r in rounds if r),
            attribution_status=status,
            raw_excerpt=payload.get("raw_excerpt") or "",
            interval_start_ns=payload.get("interval_start_ns"),
            interval_end_ns=payload.get("interval_end_ns"),
            detail=payload.get("detail") or "",
            wire_topic=payload.get("wire_topic"),
            expected_symbol=payload.get("expected_symbol"),
            source_event_type=source_event_type,
        )


# --------------------------------------------------------------- extraction

#: `bootstrap:<token>` / `integrity:<token>` - the CLOB adapter's own
#: failure contexts, which already name the token that failed.
_CONTEXT_TOKEN = re.compile(r"^(bootstrap|integrity|resnapshot):(?P<token>\S+)")

#: Connection-scoped failure contexts, which name no market at all.
CONNECTION_CONTEXTS = frozenset({"ws_connection", "rtds_connection"})

#: Failure kinds that are scoped to a CONNECTION rather than a message, and
#: therefore damage every market whose required window overlaps the outage.
#: Gate A.0.2 item 2: these are intervals, not points.
CONNECTION_SCOPED_KINDS = CONNECTION_CONTEXTS | frozenset({
    "stream_stalled", "connection_gap", "resync_gap", "reconnect",
    "topic_stalled",
})

#: Control-event types that carry a structured data-quality failure.
#: Gate A.0.2 item 3: preflight must read ALL of these, not just
#: `parse_failure` - an RTDS outage is a data loss whether or not any frame
#: failed to parse.
FAILURE_EVENT_TYPES = frozenset({
    "parse_failure", "data_gap", "stream_stalled", "reconnect", "topic_stalled",
})

#: How long a feed must be silent before observations were CERTAINLY lost.
#:
#: Measured publication rates in the real captures: RTDS reference streams
#: (Chainlink, TWAP-30/60, Binance) publish at ~1 Hz; the CLOB market socket
#: at ~130 messages/s. A gap longer than one second therefore means at least
#: one reference observation is missing, and around 130 book updates.
#:
#: This exists so that item 3's "a harmless control event that produced no
#: observation gap must not automatically disqualify a round" is decided by
#: the DURATION of the outage rather than by the presence of a control
#: event. A clean reconnect that resubscribes in 200ms loses nothing; the
#: 30-second watchdog stalls this phase is really about are two orders of
#: magnitude past this line and are never in doubt.
MATERIAL_GAP_S = 1.0
MATERIAL_GAP_NS = int(MATERIAL_GAP_S * 1e9)


def extract_token_ids(raw: str | None) -> list[str]:
    """Every token id identifiable in a failed frame.

    Handles the three shapes the CLOB market socket actually produces:
    a top-level `asset_id`, a `price_changes[]` array whose ELEMENTS each
    carry their own `asset_id` (the Phase 12C headline bug - the top level
    has none), and the adapter's `bootstrap:`/`integrity:` contexts.
    """
    if not raw:
        return []
    m = _CONTEXT_TOKEN.match(raw)
    if m:
        return [m.group("token")]

    out: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key in ("asset_id", "token_id", "assetId", "tokenId"):
                v = node.get(key)
                if isinstance(v, str) and v:
                    out.append(v)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    try:
        walk(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        # Not JSON (a truncated frame, or a repr). Fall back to a literal
        # scan for a long hex/decimal token, which is what Polymarket ids
        # look like. Better to find nothing than to invent an id.
        for token in re.findall(r"\b\d{60,}\b|\b0x[0-9a-fA-F]{40,}\b", raw):
            out.append(token)

    seen, unique = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def extract_condition_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        blob = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    out: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key in ("market", "condition_id", "conditionId"):
                v = node.get(key)
                if isinstance(v, str) and v.startswith("0x"):
                    out.append(v)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(blob)
    return list(dict.fromkeys(out))


# -------------------------------------------------------------- attribution

def attribute_failure(
    *,
    stream: Stream | str,
    failure_kind: str,
    recv_timestamp_ns: int,
    raw: str | None,
    windows: list[RoundWindow],
    interval_start_ns: int | None = None,
    interval_end_ns: int | None = None,
) -> FailureAttribution:
    """Place one failure against the rounds being recorded.

    The order of the branches is the order of decreasing evidence, and each
    branch that fails to place the failure falls through to a WIDER blast
    radius rather than to silence.
    """
    stream_value = stream.value if isinstance(stream, Stream) else str(stream)
    excerpt = (raw or "")[:400]
    tokens = extract_token_ids(raw)
    conditions = extract_condition_ids(raw)

    # 1. EXACT - the failure names a market we are recording.
    for token in tokens:
        owners = [w for w in windows if w.owns_token(token)]
        if owners:
            return FailureAttribution(
                stream=stream_value, failure_kind=failure_kind,
                recv_timestamp_ns=recv_timestamp_ns, token_id=token,
                condition_id=owners[0].condition_id,
                affected_round_ids=tuple(w.round_id for w in owners),
                attribution_status=AttributionStatus.EXACT,
                raw_excerpt=excerpt,
                detail=f"token {token} belongs to this round",
            )
    for cond in conditions:
        owners = [w for w in windows if w.condition_id == cond]
        if owners:
            return FailureAttribution(
                stream=stream_value, failure_kind=failure_kind,
                recv_timestamp_ns=recv_timestamp_ns,
                token_id=tokens[0] if tokens else None, condition_id=cond,
                affected_round_ids=tuple(w.round_id for w in owners),
                attribution_status=AttributionStatus.EXACT,
                raw_excerpt=excerpt,
                detail=f"condition {cond} belongs to this round",
            )

    # A token we are NOT recording is still exact information: the failure
    # provably concerns a market outside this capture, so it damages no round
    # here. This is the case the ACTIVE-round heuristic got backwards.
    if tokens and not any(w.owns_token(t) for t in tokens for w in windows):
        return FailureAttribution(
            stream=stream_value, failure_kind=failure_kind,
            recv_timestamp_ns=recv_timestamp_ns, token_id=tokens[0],
            condition_id=conditions[0] if conditions else None,
            affected_round_ids=(),
            attribution_status=AttributionStatus.EXACT,
            raw_excerpt=excerpt,
            detail=f"token {tokens[0]} belongs to no round in this capture",
        )

    # 2. An interval + a global stream: every overlapping required window.
    start = interval_start_ns if interval_start_ns is not None else recv_timestamp_ns
    end = interval_end_ns if interval_end_ns is not None else recv_timestamp_ns
    overlapping = tuple(w.round_id for w in windows if w.overlaps(start, end))

    if stream_value == Stream.RTDS.value:
        # RTDS is ONE global BTC series. A gap in it is a gap in every
        # round's reference history that spans the gap - not in "the" round.
        return FailureAttribution(
            stream=stream_value, failure_kind=failure_kind,
            recv_timestamp_ns=recv_timestamp_ns, token_id=None, condition_id=None,
            affected_round_ids=overlapping,
            attribution_status=AttributionStatus.GLOBAL_WINDOW,
            raw_excerpt=excerpt,
            interval_start_ns=start, interval_end_ns=end,
            detail="global BTC reference stream; every overlapping required window",
        )

    if failure_kind in CONNECTION_SCOPED_KINDS or (raw or "") in CONNECTION_SCOPED_KINDS:
        return FailureAttribution(
            stream=stream_value, failure_kind=failure_kind,
            recv_timestamp_ns=recv_timestamp_ns, token_id=None, condition_id=None,
            affected_round_ids=overlapping,
            attribution_status=AttributionStatus.WINDOW_INFERRED,
            raw_excerpt=excerpt,
            interval_start_ns=start, interval_end_ns=end,
            detail="connection-wide outage; every subscribed market overlapping it",
        )

    # 3. UNATTRIBUTED. A message we could not identify at all, on a
    #    per-market stream. It could concern any market being recorded, so
    #    every round whose window covers the moment is treated as damaged.
    return FailureAttribution(
        stream=stream_value, failure_kind=failure_kind,
        recv_timestamp_ns=recv_timestamp_ns, token_id=None, condition_id=None,
        affected_round_ids=overlapping,
        attribution_status=AttributionStatus.UNATTRIBUTED,
        raw_excerpt=excerpt,
        interval_start_ns=start, interval_end_ns=end,
        detail="no market identifier in the failed frame",
    )


# -------------------------------------------------------------- gap tracking

@dataclass
class StreamGap:
    """One feed outage, as an INTERVAL of missing observations."""

    stream: str
    failure_kind: str
    #: Wall time of the last observation actually received before the gap.
    last_data_ns: int
    #: When the watchdog or the socket noticed. Strictly after
    #: `last_data_ns`, and NOT the start of the outage - data stopped
    #: arriving up to `stall_timeout_s` earlier.
    detected_ns: int
    #: When usable data resumed. `None` while the gap is still open.
    recovered_ns: int | None = None
    #: Gate A.0.2.1 item 4: WHICH series went dark. RTDS carries four
    #: independent BTC feeds and the strategy depends on each separately, so
    #: "the RTDS socket had an outage" is not a usable statement - a TWAP-60
    #: blackout with the other three healthy is invisible in the aggregate
    #: and fatal to the label.
    wire_topic: str | None = None
    expected_symbol: str | None = None

    @property
    def duration_ns(self) -> int:
        end = self.recovered_ns if self.recovered_ns is not None else self.detected_ns
        return max(0, end - self.last_data_ns)

    @property
    def is_material(self) -> bool:
        """STRICTLY longer than one publication interval.

        The boundary case is not academic: an outage bracketed by two
        CONSECUTIVE 1 Hz observations spans exactly one second, and by
        definition lost nothing - the two observations either side of it are
        the two the feed was due to publish. `>=` would call that damage and
        disqualify a round for a reconnect that cost no data at all.
        """
        return self.duration_ns > MATERIAL_GAP_NS


class StreamGapTracker:
    """Turns a watchdog firing into an interval with a real start and end.

    Gate A.0.2 items 1 and 2. The recorder previously wrote `stream_stalled`
    at the moment the watchdog fired and moved on, which is wrong in both
    directions:

    * the outage did NOT start when the watchdog noticed - the watchdog
      waits `stall_timeout_s` (30s) of silence before firing, so the last
      real observation is up to 30 seconds EARLIER;
    * the outage did not END there either - the stream is still dead through
      the reconnect, the resubscribe, and for the CLOB through the REST
      resnapshot that makes the books usable again.

    Recording it as the single point where the watchdog happened to fire
    understates a 37-second data loss as a zero-duration event, and a
    zero-duration event intersects almost no round window. The gap is
    therefore opened at the last observation actually received and closed
    only when usable data has genuinely resumed.
    """

    def __init__(self, stream: Stream | str, on_gap=None, clock=None,
                 wire_topic: str | None = None, expected_symbol: str | None = None):
        self.stream = stream.value if isinstance(stream, Stream) else str(stream)
        #: Gate A.0.2.1 items 2-3: one tracker per REQUIRED SERIES, not one
        #: per socket. `note_data` on this tracker may close only THIS
        #: series' gap - a Binance tick is not evidence that TWAP-60 is
        #: alive, and an ETH tick is not evidence that anything BTC is.
        self.wire_topic = wire_topic
        self.expected_symbol = expected_symbol
        self._on_gap = on_gap or (lambda gap: None)
        self._clock = clock or (lambda: __import__("time").time_ns())
        self.last_data_ns: int | None = None
        self.open_gap: StreamGap | None = None

    def note_data(self, at_ns: int | None = None) -> None:
        """A valid DATA observation arrived (not a PONG, not a control frame).

        This both advances the liveness mark and closes any open gap: the
        first genuinely usable observation is what ends an outage.
        """
        now = at_ns if at_ns is not None else self._clock()
        if self.open_gap is not None:
            self.close(now)
        self.last_data_ns = now

    def begin(self, failure_kind: str, detected_ns: int | None = None) -> StreamGap | None:
        """The watchdog fired, or the socket dropped. Opens a gap whose start
        is the last observation we actually received."""
        if self.open_gap is not None:
            return self.open_gap
        now = detected_ns if detected_ns is not None else self._clock()
        self.open_gap = StreamGap(
            stream=self.stream, failure_kind=failure_kind,
            last_data_ns=self.last_data_ns if self.last_data_ns is not None else now,
            detected_ns=now,
            wire_topic=self.wire_topic, expected_symbol=self.expected_symbol,
        )
        return self.open_gap

    def close(self, recovered_ns: int | None = None) -> StreamGap | None:
        """Usable data has resumed; publish the completed interval."""
        gap = self.open_gap
        if gap is None:
            return None
        gap.recovered_ns = recovered_ns if recovered_ns is not None else self._clock()
        self.open_gap = None
        self.last_data_ns = gap.recovered_ns
        self._on_gap(gap)
        return gap

    def abandon(self) -> StreamGap | None:
        """Shutdown with a gap still open. It is still a real outage - the
        recorder simply stopped before data resumed - so it is published
        with the end left at the moment we stopped looking."""
        return self.close() if self.open_gap is not None else None


def attribute_gap(gap: StreamGap, windows: list[RoundWindow]) -> FailureAttribution:
    """Place a completed outage against the rounds it damaged."""
    import dataclasses

    end = gap.recovered_ns if gap.recovered_ns is not None else gap.detected_ns
    return dataclasses.replace(
        attribute_failure(
            stream=gap.stream,
            failure_kind=gap.failure_kind,
            recv_timestamp_ns=gap.detected_ns,
            raw=gap.failure_kind,
            windows=windows,
            interval_start_ns=gap.last_data_ns,
            interval_end_ns=end,
        ),
        wire_topic=gap.wire_topic,
        expected_symbol=gap.expected_symbol,
    )


# ------------------------------------------------------------ completeness

@dataclass(frozen=True)
class AttributionSummary:
    """Whether a capture's failures can be trusted round by round."""

    attributions: tuple[FailureAttribution, ...]
    #: `RecorderMetrics.parse_failures` for the session, i.e. how many
    #: failures actually happened.
    session_failure_count: int
    #: Rounds recorded in the capture, used for the session-wide fallback.
    all_round_ids: tuple[str, ...]
    #: Failures the recorder COUNTED but never wrote a record for. Non-zero
    #: for every capture written before Gate A.0, and for any capture where
    #: writing the record itself failed. Each one is a failure we know
    #: happened and cannot place. Computed by the caller per failure KIND,
    #: so gap records cannot silently account for parse failures.
    unrecorded_count: int = 0

    @property
    def recorded_count(self) -> int:
        return len(self.attributions)

    @property
    def untrustworthy(self) -> tuple[FailureAttribution, ...]:
        return tuple(a for a in self.attributions if not a.is_trustworthy)

    @property
    def is_complete(self) -> bool:
        """The invariant: per-round attribution replaces the session-wide
        rule only when EVERY failure is both recorded and placed.

        One well-attributed failure does not license trusting the rest.
        """
        return self.unrecorded_count == 0 and not self.untrustworthy

    def affected_rounds(self) -> dict[str, int]:
        """Damaged-failure count per round.

        When attribution is incomplete, every round in the capture is
        treated as affected: an unplaced failure could have damaged any of
        them, and a round cannot be proven clean by the absence of evidence
        that was never written.
        """
        counts: dict[str, int] = {}
        if not self.is_complete:
            n = self.unrecorded_count + len(self.untrustworthy)
            # An UNATTRIBUTED record still names the rounds whose windows it
            # covers; unrecorded failures have no interval at all, so they
            # fall on everything.
            for rid in self.all_round_ids:
                counts[rid] = n
            for a in self.attributions:
                if a.is_trustworthy:
                    for rid in a.affected_round_ids:
                        counts[rid] = counts.get(rid, 0) + 1
            return counts
        for a in self.attributions:
            for rid in a.affected_round_ids:
                counts[rid] = counts.get(rid, 0) + 1
        return counts

    def affected_rounds_by_kind(self) -> tuple[dict[str, int], dict[str, int]]:
        """`(parse_failures_per_round, data_gaps_per_round)`.

        Gate A.0.2.1 item 7. A.0.2 folded both into one count injected
        through `parse_failure_count`, so a round excluded for a 32-second
        TWAP-60 blackout reported `parse_failures` - when nothing had failed
        to parse. The boolean verdict was right and the reason was
        unreadable, which is the difference between knowing the dataset is
        small and knowing why.
        """
        parse: dict[str, int] = {}
        gaps: dict[str, int] = {}
        if not self.is_complete:
            # Unplaceable failures fall on everything; they are counted as
            # parse failures because that is what the session counter that
            # revealed them was counting.
            n = self.unrecorded_count + len(self.untrustworthy)
            for rid in self.all_round_ids:
                parse[rid] = n
            for a in self.attributions:
                if a.is_trustworthy:
                    target = gaps if a.is_gap else parse
                    for rid in a.affected_round_ids:
                        target[rid] = target.get(rid, 0) + 1
            return parse, gaps
        for a in self.attributions:
            target = gaps if a.is_gap else parse
            for rid in a.affected_round_ids:
                target[rid] = target.get(rid, 0) + 1
        return parse, gaps

    def gap_topics_for_round(self, round_id: str) -> list[str]:
        """`["rtds:crypto_prices_twap_sixty", "clob", ...]` - which feeds
        actually went dark for this round (item 7's diagnostic detail)."""
        out = []
        for a in self.attributions:
            if not a.is_gap or round_id not in a.affected_round_ids:
                continue
            label = f"{a.stream}:{a.wire_topic}" if a.wire_topic else a.stream
            if label not in out:
                out.append(label)
        return sorted(out)

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {s.value: 0 for s in AttributionStatus}
        for a in self.attributions:
            out[a.attribution_status.value] += 1
        if self.unrecorded_count:
            out["UNRECORDED"] = self.unrecorded_count
        return out

    def reason(self) -> str:
        if self.is_complete:
            return "per-round (every failure recorded and attributed)"
        parts = []
        if self.unrecorded_count:
            parts.append(f"{self.unrecorded_count} failure(s) counted but never recorded")
        if self.untrustworthy:
            parts.append(f"{len(self.untrustworthy)} failure(s) could not be attributed")
        return ("session-wide fallback: " + "; ".join(parts)
                + " - every round in the capture is treated as affected")
