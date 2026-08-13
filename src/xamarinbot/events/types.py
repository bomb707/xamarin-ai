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
# ordering rules for ties."). Market-truth events are ordered before the
# order-lifecycle events they may causally trigger.
_TYPE_PRIORITY: dict[EventType, int] = {
    EventType.MARKET_CONFIG: 0,
    EventType.TWAP: 1,
    EventType.SPOT: 2,
    EventType.BOOK_SNAPSHOT: 3,
    EventType.BOOK_DELTA: 4,
    EventType.ORDER_STATUS: 5,
    EventType.FILL: 6,
    EventType.CANCEL: 7,
    EventType.ORDER_SUBMIT: 8,
    EventType.SETTLEMENT: 9,
}


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
    def sort_key(self) -> tuple[float, int, int]:
        """(event_time, type priority, sequence) - the full deterministic
        ordering rule for ties."""
        return (self.event_time, _TYPE_PRIORITY[self.event_type], self.sequence)
