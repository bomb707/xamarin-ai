"""Append-only causal event types (Roadmap Phase 2).

"Define append-only event types: MARKET_CONFIG, TWAP, SPOT, BOOK_SNAPSHOT,
BOOK_DELTA, ORDER_SUBMIT, ORDER_STATUS, FILL, CANCEL, SETTLEMENT."
"Persist nanosecond/millisecond timestamps as received and source timestamps
separately."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    MARKET_CONFIG = "MARKET_CONFIG"
    TWAP = "TWAP"
    SPOT = "SPOT"
    BOOK_SNAPSHOT = "BOOK_SNAPSHOT"
    BOOK_DELTA = "BOOK_DELTA"
    ORDER_SUBMIT = "ORDER_SUBMIT"
    ORDER_STATUS = "ORDER_STATUS"
    FILL = "FILL"
    CANCEL = "CANCEL"
    SETTLEMENT = "SETTLEMENT"


# Deterministic secondary tie-break ordering when two events share the same
# effective event_time (Roadmap Phase 2: "Create deterministic event
# ordering rules for ties.").
#
# Phase 12C item 13 REWROTE this table. The previous version assigned every
# event type its own distinct rank, which had two separate defects:
#
#   1. It ranked ORDER_STATUS(5)/FILL(6)/CANCEL(7) BEFORE ORDER_SUBMIT(8).
#      On a timestamp tie that literally sorted an order's own fill ahead
#      of the submit that created it - a causally impossible ordering
#      manufactured purely by the priority table. `causal_sort` below now
#      also repairs this structurally, per order_id, even when the
#      timestamps themselves are skewed.
#   2. It imposed a total order on genuinely EXTERNAL observations
#      (TWAP < SPOT < BOOK_SNAPSHOT < BOOK_DELTA). Two external events
#      carrying the same source timestamp are simply not causally ordered
#      with respect to each other, and inventing "TWAP happened before the
#      book update" is fabricating a causal fact from an arbitrary constant.
#      All external observations therefore now share ONE rank, so ties fall
#      through to the honest keys: local receive time, then insertion
#      sequence (i.e. arrival order). `ordering_ambiguities()` reports
#      exactly where that fallback was load-bearing rather than hiding it.
#
# What remains ranked is only what this system genuinely knows the causal
# direction of: MARKET_CONFIG is a round precondition rather than an
# observation of the market; our OWN submits are caused by the data visible
# at that instant; an order's status/fill/cancel is caused by its submit;
# SETTLEMENT closes the round.
_RANK_MARKET_CONFIG = 0
_RANK_EXTERNAL_OBSERVATION = 1
_RANK_OWN_ORDER_SUBMIT = 2
_RANK_OWN_ORDER_REACTION = 3
_RANK_SETTLEMENT = 4

_TYPE_PRIORITY: dict[EventType, int] = {
    EventType.MARKET_CONFIG: _RANK_MARKET_CONFIG,
    # --- external observations: deliberately all equal, see above ---
    EventType.TWAP: _RANK_EXTERNAL_OBSERVATION,
    EventType.SPOT: _RANK_EXTERNAL_OBSERVATION,
    EventType.BOOK_SNAPSHOT: _RANK_EXTERNAL_OBSERVATION,
    EventType.BOOK_DELTA: _RANK_EXTERNAL_OBSERVATION,
    # --- our own order lifecycle: submit strictly before its reactions ---
    EventType.ORDER_SUBMIT: _RANK_OWN_ORDER_SUBMIT,
    EventType.ORDER_STATUS: _RANK_OWN_ORDER_REACTION,
    EventType.FILL: _RANK_OWN_ORDER_REACTION,
    EventType.CANCEL: _RANK_OWN_ORDER_REACTION,
    EventType.SETTLEMENT: _RANK_SETTLEMENT,
}

#: Own-order events that cannot precede their own ORDER_SUBMIT, whatever
#: the timestamps say.
ORDER_REACTION_TYPES = frozenset(
    {EventType.ORDER_STATUS, EventType.FILL, EventType.CANCEL}
)


@dataclass(frozen=True)
class Event:
    """One append-only event.

    sequence: monotonic insertion order assigned by the EventStore; the
      final, always-unique tie-break key.
    source_ts: timestamp assigned by the origin (exchange/oracle), seconds
      since epoch as a float (sub-millisecond precision preserved as
      fractional seconds). None if the event has no external source
      timestamp (e.g. locally synthesized).
    recv_ts: local receive timestamp, seconds since epoch.
    round_id: market/round this event belongs to.
    payload: type-specific structured data.
    """

    sequence: int
    event_type: EventType
    round_id: str
    recv_ts: float
    source_ts: float | None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def event_time(self) -> float:
        """Effective causal timestamp: prefer the source timestamp, fall
        back to local receive time when the source provides none."""
        return self.source_ts if self.source_ts is not None else self.recv_ts

    @property
    def sort_key(self) -> tuple[float, int, float, int]:
        """(event_time, type rank, recv_ts, sequence) - the full
        deterministic ordering rule for ties.

        Phase 12C item 13 added `recv_ts` ahead of `sequence`: when two
        external observations tie on source timestamp AND on rank (which is
        now the normal case, since all external observations share a rank),
        the next most defensible discriminator is which one this system
        actually received first, not merely which one happened to be
        inserted first. `sequence` remains the final, always-unique key.
        """
        return (
            self.event_time,
            _TYPE_PRIORITY[self.event_type],
            self.recv_ts,
            self.sequence,
        )


def _order_id_of(event: Event) -> str | None:
    oid = event.payload.get("order_id")
    return str(oid) if oid is not None else None


def causal_sort(events: list[Event]) -> list[Event]:
    """Deterministic ordering that additionally guarantees no own-order
    reaction (ORDER_STATUS / FILL / CANCEL) can precede the ORDER_SUBMIT
    for the same `order_id` (Phase 12C item 13).

    `sort_key` alone gets this right whenever the timestamps agree with
    reality, but a real venue can stamp a fill with a source timestamp at
    or even fractionally before its own acknowledged submit (clock skew
    between the matching engine and the order gateway, or a submit whose
    only available timestamp is our local one). Ranking cannot fix that -
    ordering by timestamp would still emit the fill first. This pass
    repairs it structurally: for each `order_id`, its ORDER_SUBMIT is
    moved to immediately before the earliest reaction event that would
    otherwise precede it. Everything else keeps its `sort_key` order.

    Events with no `order_id` in the payload are untouched, and an
    order whose submit was never recorded (e.g. a reconciliation-only
    stream) is left exactly as sorted rather than being invented.
    """
    ordered = sorted(events, key=lambda e: e.sort_key)

    submit_index: dict[str, int] = {}
    earliest_reaction: dict[str, int] = {}
    for i, e in enumerate(ordered):
        oid = _order_id_of(e)
        if oid is None:
            continue
        if e.event_type is EventType.ORDER_SUBMIT:
            submit_index.setdefault(oid, i)
        elif e.event_type in ORDER_REACTION_TYPES:
            earliest_reaction.setdefault(oid, i)

    # Only orders whose submit is genuinely out of place need moving.
    misplaced = {
        oid: (submit_index[oid], earliest_reaction[oid])
        for oid in submit_index
        if oid in earliest_reaction and earliest_reaction[oid] < submit_index[oid]
    }
    if not misplaced:
        return ordered

    moving = {si for si, _ in misplaced.values()}
    # Insert each displaced submit immediately before the reaction that
    # jumped ahead of it, preserving relative order everywhere else.
    inserts: dict[int, list[Event]] = {}
    for _oid, (si, ri) in sorted(misplaced.items(), key=lambda kv: kv[1][1]):
        inserts.setdefault(ri, []).append(ordered[si])

    repaired: list[Event] = []
    for i, e in enumerate(ordered):
        for pending in inserts.get(i, ()):
            repaired.append(pending)
        if i in moving:
            continue
        repaired.append(e)
    return repaired


def ordering_ambiguities(events: list[Event]) -> list[tuple[Event, Event]]:
    """Every adjacent pair of EXTERNAL observations whose relative order is
    decided by nothing stronger than local arrival (identical `event_time`
    and identical rank).

    This is the honest counterpart to no longer fabricating a type
    priority between external events: rather than silently picking one, the
    recorder can count and report how often the source data left the order
    genuinely undetermined. A high count over a capture interval is a
    data-quality finding about that interval, not a bug to be tuned away.
    """
    ordered = causal_sort(events)
    out: list[tuple[Event, Event]] = []
    for a, b in zip(ordered, ordered[1:]):
        if (
            _TYPE_PRIORITY[a.event_type] == _RANK_EXTERNAL_OBSERVATION
            and _TYPE_PRIORITY[b.event_type] == _RANK_EXTERNAL_OBSERVATION
            and a.event_time == b.event_time
        ):
            out.append((a, b))
    return out
