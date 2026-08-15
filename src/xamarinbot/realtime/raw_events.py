"""Immutable raw-event record (Phase 12C item 6).

The existing `events/store.py` `EventStore` is a good deterministic-replay
substrate, but it is the wrong shape for a real recorder, for two concrete
reasons the brief names:

  1. It commits SQLite on EVERY appended event. The live BTC 5-minute
     market channel was measured at roughly 130 `price_change` messages per
     second across a single round's two tokens (see
     docs/REAL_RECORDER_ARCHITECTURE.md's "Measured event rates"), and the
     RTDS reference feeds add four more streams at ~1 Hz each. One
     transaction per event puts an fsync in the WebSocket reader's path.
  2. It stores timestamps as floating-point seconds. A float64 holds a
     nanosecond-resolution epoch timestamp only to ~microsecond precision,
     and, worse, `event_time` COLLAPSES source and receive time into one
     number by falling back from one to the other. For a normalized feature
     store that is acceptable; for the immutable record of what arrived on
     the wire it is lossy in exactly the dimension latency analysis needs.

`RawEvent` therefore keeps every clock separately, as integers, and keeps
the original wire payload verbatim. Normalization is a *downstream* view
built from these rows - never a replacement for them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Topic(str, Enum):
    """Stream a raw event arrived on. Deliberately named after the real
    transport topic rather than a strategy-level concept, so a row can
    always be traced back to the exact subscription that produced it."""

    # Polymarket CLOB market channel
    CLOB_MARKET = "clob_market"
    # Polymarket CLOB REST (bootstrap snapshot / resync / integrity check)
    CLOB_REST = "clob_rest"
    # Polymarket RTDS reference streams
    RTDS_BINANCE = "rtds_binance"
    RTDS_CHAINLINK = "rtds_chainlink"
    RTDS_TWAP_30 = "rtds_twap_30"
    RTDS_TWAP_60 = "rtds_twap_60"
    # Recorder's own control plane (lifecycle transitions, reconnects,
    # reconciliation results). Recorded in the same immutable log so the
    # capture's own history is replayable alongside the market data.
    RECORDER_CONTROL = "recorder_control"
    # Market metadata as fetched (Gamma + CLOB market-info)
    MARKET_METADATA = "market_metadata"
    # Hypothetical (never-submitted) paper maker quotes and their
    # counterfactual observations - item 11.
    PAPER_QUOTE = "paper_quote"


#: Topics carrying externally-sourced market data. Only these count toward
#: data-integrity statistics; the recorder's own control events do not.
EXTERNAL_TOPICS = frozenset(
    {
        Topic.CLOB_MARKET,
        Topic.CLOB_REST,
        Topic.RTDS_BINANCE,
        Topic.RTDS_CHAINLINK,
        Topic.RTDS_TWAP_30,
        Topic.RTDS_TWAP_60,
    }
)

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


def ms_to_ns(ms: int | float | str | None) -> int | None:
    """Millisecond epoch -> nanosecond epoch, preserving `None`.

    Polymarket sends millisecond epochs as JSON *strings* on the CLOB
    market channel (`"timestamp": "1786771382832"`) and as JSON *numbers*
    on RTDS (`"timestamp": 1786771253000`); both are accepted. Returns
    `None` rather than 0 for a missing value - "no source timestamp" and
    "the epoch" must never be confused, since freshness is computed from
    this.

    The conversion is done in INTEGER arithmetic wherever the input is an
    integral millisecond count, which every observed payload is. Going via
    `float` corrupts the value: `float(1786771254918) * 1e6` evaluates to
    1786771254918000128, i.e. 128 nanoseconds of pure fabrication, because
    a float64 has 53 bits of mantissa and the product needs 61. That is
    small, but it is precisely the "timestamps stored as floating point"
    defect this module exists to remove, and it would make two events with
    identical wire timestamps compare unequal.
    """
    if ms is None or ms == "":
        return None
    if isinstance(ms, int):
        return ms * NS_PER_MS
    if isinstance(ms, str):
        text = ms.strip()
        if text.lstrip("-").isdigit():
            return int(text) * NS_PER_MS
    # Genuinely fractional milliseconds: no exact integer path exists, so
    # accept the float rounding rather than reject the observation.
    return int(round(float(ms) * NS_PER_MS))


@dataclass(frozen=True)
class RawEvent:
    """One immutable observation, exactly as received.

    Timestamps, all nanosecond epochs except `recv_monotonic_ns`:

      source_timestamp_ns    Stamped by the ORIGIN of the observation -
                             the CLOB matching engine for book/price_change,
                             the Chainlink observation time for reference and
                             TWAP values, the Binance trade time. This is the
                             only timestamp freshness may be judged against
                             (item 4: "Do not replace Chainlink observation
                             time with local receive time").
      publisher_timestamp_ns Stamped by the RELAY that forwarded it - RTDS's
                             own outer `timestamp`. Distinct from the source:
                             the gap between them is the oracle-to-Polymarket
                             hop, which we do not control and must be able to
                             measure separately from our own network hop.
      recv_wall_timestamp_ns Local wall clock at the moment the frame was
                             read. Comparable to the two above (same epoch),
                             but subject to NTP steps.
      recv_monotonic_ns      Local monotonic clock at the same moment. NOT an
                             epoch - only differences between two of these are
                             meaningful. This is what inter-event deltas and
                             the local sequence must be computed from, because
                             a wall-clock step backwards during a capture
                             would otherwise silently reorder events.

    `payload_json` is the verbatim wire payload. Item 6: "Do not discard
    the original wire payload after normalization." `normalized_side` is a
    convenience projection only, and is `None` whenever the mapping is not
    known with certainty rather than being guessed.
    """

    recorder_sequence: int
    session_id: str
    topic: Topic
    event_type: str  # the ORIGINAL event type string from the wire
    round_id: str | None
    condition_id: str | None
    token_id: str | None
    source_timestamp_ns: int | None
    publisher_timestamp_ns: int | None
    recv_wall_timestamp_ns: int
    recv_monotonic_ns: int
    payload_json: str
    normalized_side: str | None
    reconnect_generation: int

    @property
    def payload(self) -> Any:
        return json.loads(self.payload_json)

    @property
    def source_to_recv_latency_ns(self) -> int | None:
        """How long between the origin stamping this observation and this
        process reading it. `None` when there is no source timestamp -
        never 0, which would read as a perfect zero-latency observation."""
        if self.source_timestamp_ns is None:
            return None
        return self.recv_wall_timestamp_ns - self.source_timestamp_ns

    @property
    def publisher_to_recv_latency_ns(self) -> int | None:
        """How long between the relay publishing and this process reading -
        i.e. only the hop we could plausibly influence."""
        if self.publisher_timestamp_ns is None:
            return None
        return self.recv_wall_timestamp_ns - self.publisher_timestamp_ns

    @property
    def dedupe_key(self) -> tuple:
        """Identity used to detect a genuine duplicate (the same wire
        observation delivered twice, e.g. across a reconnect resubscribe).

        Deliberately EXCLUDES both receive timestamps and the reconnect
        generation: a redelivered event is the same observation even though
        we read it at a different local time on a different connection.
        Deliberately INCLUDES the full payload, because two distinct
        observations of the same token can share a source timestamp
        (millisecond resolution against a high-rate feed) and are not
        duplicates.
        """
        return (
            self.topic,
            self.event_type,
            self.token_id,
            self.source_timestamp_ns,
            self.payload_json,
        )


@dataclass
class RawEventBuilder:
    """Assigns `recorder_sequence` and both local clocks at the single
    moment of ingestion, so every event in one session shares one
    monotonic ordering that no later processing can perturb.

    The counter is guarded by the caller (the recorder's reader threads each
    own one builder, or share one under its lock); `next_sequence` is the
    only mutable state.
    """

    session_id: str
    next_sequence: int = 0
    reconnect_generation: int = 0
    _clock_wall_ns: Any = field(default=None, repr=False)
    _clock_mono_ns: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import time

        if self._clock_wall_ns is None:
            self._clock_wall_ns = time.time_ns
        if self._clock_mono_ns is None:
            self._clock_mono_ns = time.monotonic_ns

    def build(
        self,
        topic: Topic,
        event_type: str,
        payload: Any,
        *,
        round_id: str | None = None,
        condition_id: str | None = None,
        token_id: str | None = None,
        source_timestamp_ns: int | None = None,
        publisher_timestamp_ns: int | None = None,
        normalized_side: str | None = None,
        raw_json: str | None = None,
    ) -> RawEvent:
        """`raw_json`, when given, is the untouched wire text - preferred
        over re-serializing `payload`, so the stored bytes are exactly what
        arrived (key order, number formatting and all)."""
        self.next_sequence += 1
        return RawEvent(
            recorder_sequence=self.next_sequence,
            session_id=self.session_id,
            topic=topic,
            event_type=event_type,
            round_id=round_id,
            condition_id=condition_id,
            token_id=token_id,
            source_timestamp_ns=source_timestamp_ns,
            publisher_timestamp_ns=publisher_timestamp_ns,
            recv_wall_timestamp_ns=self._clock_wall_ns(),
            recv_monotonic_ns=self._clock_mono_ns(),
            payload_json=raw_json if raw_json is not None else json.dumps(payload, separators=(",", ":")),
            normalized_side=normalized_side,
            reconnect_generation=self.reconnect_generation,
        )
