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
ATTRIBUTION_SCHEMA_VERSION = 1


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

    @property
    def is_trustworthy(self) -> bool:
        return self.attribution_status.is_trustworthy

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
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "FailureAttribution":
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
        )


# --------------------------------------------------------------- extraction

#: `bootstrap:<token>` / `integrity:<token>` - the CLOB adapter's own
#: failure contexts, which already name the token that failed.
_CONTEXT_TOKEN = re.compile(r"^(bootstrap|integrity|resnapshot):(?P<token>\S+)")

#: Connection-scoped failure contexts, which name no market at all.
CONNECTION_CONTEXTS = frozenset({"ws_connection", "rtds_connection"})


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

    if failure_kind in CONNECTION_CONTEXTS or (raw or "") in CONNECTION_CONTEXTS:
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

    @property
    def recorded_count(self) -> int:
        return len(self.attributions)

    @property
    def unrecorded_count(self) -> int:
        """Failures the recorder counted but never wrote a record for.

        Non-zero for every capture written before Gate A.0, and for any
        future capture where writing the record itself failed. Each one is a
        failure we know happened and cannot place.
        """
        return max(0, self.session_failure_count - self.recorded_count)

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
