"""REAL raw capture -> normalized event projection (Phase 12C.1 item 8).

    RawEventStore_real  ->  NormalizedEventStore_real  ->  FeatureEngine

Phase 12C built a recorder that captures real BTC five-minute rounds
faithfully, and Phase 12C.1's audit found the resulting data was unreachable:
`features/engine.py` and `ShadowRunner` both read the Phase-2 normalized
`EventStore`, and nothing translated a raw capture into that shape. Every
number the system could produce therefore still came from fabricated data.
This module closes that gap.

What it will not do
-------------------
**It never synthesizes a missing observation.** Not a TWAP, not a spot price,
not a book observation, not `p0`, not a timestamp. Where the capture has a
gap, the projection leaves a gap, and the feature engine's existing
`InvalidFeatureState` machinery reports it downstream. `ProjectionResult`
counts and names every skip so a gap is visible rather than inferred from a
suspiciously smooth feature series.

Which real stream becomes which normalized event
------------------------------------------------
==================  ==========================================================
`MARKET_CONFIG`     the `rounds` row the recorder persisted for this round
`TWAP`              the round's **declared settlement basis** - chosen from
                    its own `settlement_kind`/`twap_window_s`, never a global
                    constant (item 15)
`SPOT`              Binance BTCUSDT, the leading signal
`BOOK_SNAPSHOT`     `clob_market:book` plus the REST bootstrap/resync
                    snapshots
`BOOK_DELTA`        each `clob_market:price_change` element
`SETTLEMENT`        the venue's resolved outcome, when the capture has one
==================  ==========================================================

Plain Chainlink and TWAP-30 are deliberately **not** projected into
`EventType.TWAP`. `features/engine.py` builds one TWAP series with
`_series(causal, EventType.TWAP)`, so emitting two different quantities into
it would silently interleave them and corrupt `gap_twap_bp` and `z_gap`. Both
remain fully available in the raw log for diagnostics, which is what item 15
asks for.

Timestamps
----------
`source_ts` and `recv_ts` are carried across separately, because the
normalized layer genuinely uses both: `Event.sort_key` orders on
`event_time` (source-preferring) while `ShadowRunner` gates visibility on
`recv_ts`. Collapsing them would destroy the live-vs-replay distinction the
whole shadow design rests on. The two timestamps the normalized `Event` has
no field for - publisher time and the local monotonic clock - are preserved
inside the payload's `_provenance` block, along with the raw event's identity
and a hash of the original wire bytes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.raw_events import RawEvent, Topic
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.rounds import RoundLabel

#: How much reference history each round carries into its projection.
#: 600s comfortably exceeds the largest feature window (the 300s round plus
#: the 60s TWAP warm-up) with margin, matching the recorder's own 420s
#: PRE_ROUND lead.
REFERENCE_LOOKBACK_S = 600.0
#: Reference observations after the close, needed for the end boundary.
REFERENCE_TAIL_S = 120.0

#: Reserved payload key holding the raw-event provenance block. Chosen with a
#: leading underscore so it cannot collide with a market field, and filtered
#: out by `replay/feeds.py::market_config_from_payload`.
PROVENANCE_KEY = "_provenance"

#: RTDS wire topic supplying the leading spot signal.
SPOT_TOPIC = Topic.RTDS_BINANCE

#: Declared settlement basis -> the raw topic carrying it. Item 15: the basis
#: is a property of the market, read from its persisted configuration.
_SETTLEMENT_TOPIC = {
    ("chainlink_twap", 30): Topic.RTDS_TWAP_30,
    ("chainlink_twap", 60): Topic.RTDS_TWAP_60,
}
_REFERENCE_TOPIC = Topic.RTDS_CHAINLINK

#: The only settlement rules this system knows how to reconstruct. Gate A.0
#: item 8: an unrecognized value must RAISE, not fall through to plain
#: Chainlink reference. Silently treating `"twap"` or a malformed string as
#: the reference series would pick the wrong price series for the label -
#: the same class of error as guessing a missing rule, just harder to see.
SUPPORTED_SETTLEMENT_KINDS = frozenset({"chainlink_twap", "chainlink_reference"})


class ProjectionError(RuntimeError):
    """Raised instead of inventing a value the capture does not contain."""


def settlement_topic_for(settlement_kind: str, twap_window_s: int | None) -> Topic:
    """The raw topic this market declares as its settlement reference."""
    if settlement_kind not in SUPPORTED_SETTLEMENT_KINDS:
        raise ProjectionError(
            f"unsupported settlement_kind {settlement_kind!r}; known values are "
            f"{sorted(SUPPORTED_SETTLEMENT_KINDS)}. Refusing to fall back to the plain "
            "Chainlink reference - that would silently choose the wrong price series "
            "as label truth."
        )
    if settlement_kind == "chainlink_twap":
        topic = _SETTLEMENT_TOPIC.get((settlement_kind, int(twap_window_s or 0)))
        if topic is None:
            raise ProjectionError(
                f"market declares chainlink_twap with window {twap_window_s}s, which is "
                "not a stream RTDS publishes (only 30 and 60); refusing to substitute another"
            )
        return topic
    return _REFERENCE_TOPIC


@dataclass
class ProjectionResult:
    """What was projected, and - just as importantly - what was not."""

    round_id: str
    provenance: DataProvenance
    p0: float
    settlement_topic: str
    start_ts: float
    end_ts: float
    counts: dict[str, int] = field(default_factory=dict)
    #: Raw events deliberately not projected, by reason. A non-empty value is
    #: information about the capture, not an error.
    skipped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: Phase 12C.2 item 3: the supervised target, kept OFF the causal event
    #: stream. "causal market events -> FeatureEngine, eventual RoundLabel ->
    #: supervised target". None when the round has no venue outcome yet.
    label: "RoundLabel | None" = None
    #: When the venue's outcome was actually observed, seconds. Strictly
    #: later than `end_ts` in every real capture.
    label_observed_at: float | None = None
    #: Tick sizes in force over the round, oldest first, as
    #: `(source_ts, tick)` - the causal record item 2 requires.
    tick_timeline: list[tuple[float, float]] = field(default_factory=list)

    @property
    def total_projected(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "provenance": self.provenance.value,
            "p0": self.p0,
            "settlement_topic": self.settlement_topic,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "counts": dict(self.counts),
            "skipped": dict(self.skipped),
            "total_projected": self.total_projected,
            "warnings": list(self.warnings),
            "label": None if self.label is None else {
                "outcome": self.label.outcome.value,
                "p0": self.label.p0,
                "final_reference": self.label.final_reference,
                "provenance": self.label.provenance.value,
            },
            "label_observed_at": self.label_observed_at,
            "tick_timeline": [list(t) for t in self.tick_timeline],
        }


def _provenance_block(e: RawEvent) -> dict:
    """Everything item 8 requires be preserved that the normalized `Event`
    has no column for."""
    return {
        "raw_event_id": [e.session_id, e.recorder_sequence],
        "topic": e.topic.value,
        "wire_event_type": e.event_type,
        "publisher_timestamp_ns": e.publisher_timestamp_ns,
        "recv_wall_timestamp_ns": e.recv_wall_timestamp_ns,
        "recv_monotonic_ns": e.recv_monotonic_ns,
        "condition_id": e.condition_id,
        "token_id": e.token_id,
        "normalized_side": e.normalized_side,
        "payload_sha256": hashlib.sha256(e.payload_json.encode("utf-8")).hexdigest(),
        "reconnect_generation": e.reconnect_generation,
        "provenance": DataProvenance.REAL_REPLAY.value,
    }


def _times(e: RawEvent) -> tuple[float, float] | None:
    """(source_ts, recv_ts) in seconds, or None when the raw event carries no
    source timestamp - which is exactly the case that must NOT be papered
    over by falling back to receive time."""
    if e.source_timestamp_ns is None:
        return None
    return (e.source_timestamp_ns / 1e9, e.recv_wall_timestamp_ns / 1e9)


#: Raw event types that record the moment the venue's resolution was
#: actually OBSERVED. `market_metadata_resolution` is written by the
#: post-capture resolution sweep; `label_reconstruction_resolved` by the
#: reconciliation that follows it.
_RESOLUTION_EVENT_TYPES = (
    "market_metadata_resolution",
    "clob_market_info_resolution",
    "label_reconstruction_resolved",
    "market_resolved_final",
)


def _label_status_for(raw: RawEventStore, round_id: str) -> str | None:
    """The `LabelStatus` the recorder computed for this round, if it recorded
    one. Read from the last `label_reconstruction*` control event so the
    rule-text cross-check (Gate A.0 item 3) participates here too."""
    status = None
    for e in raw.events(round_id=round_id, topics=[Topic.RECORDER_CONTROL]):
        if e.event_type in ("label_reconstruction", "label_reconstruction_resolved"):
            payload = e.payload
            status = payload.get("status") or status
    return status


def reference_events(raw: RawEventStore, topic: Topic) -> list[RawEvent]:
    """Every observation on a reference topic, across the WHOLE capture.

    Reference feeds (Chainlink, TWAP-30/60, Binance) are GLOBAL: one BTC
    price series, not a per-market one. The recorder tags each observation
    with a single `round_id` - whichever round was ACTIVE when it arrived -
    because a raw row has one round column, but that tag is an attribution
    convenience, not a scope.

    Filtering by it here would be a real error, and a quiet one. In a
    multi-round batch only the FIRST round is tagged with the pre-round
    lookback; rounds 2..N receive observations only inside their own active
    window. Measured on a captured 8-round batch: round 1 carried 782
    TWAP-60 observations spanning -531s to -2s, while round 3's earliest
    tagged observation was +0.0s and its latest was -2s relative to its own
    close. Selecting by `round_id` would therefore leave almost every round
    with no reference observation at or before its open (no `p0`) and none
    at or after its close (no end reference) - and the projection would
    correctly refuse to label rounds whose data is in fact present, just
    filed under a neighbour.

    Boundary selection is by TIMESTAMP, which is what the physical question
    ("what had the oracle observed by the round boundary?") actually asks.
    """
    return raw.events(topics=[topic])


def _resolution_observed_at(raw: RawEventStore, round_id: str) -> float | None:
    """When this system actually learned the venue's outcome, in seconds.

    Phase 12C.2 item 3: a label must never become causally visible merely
    because the five-minute clock ended. In the verified capture the round
    closed at t=0+300s but the outcome was not observed until t=+391.6s.
    Returns None when the capture contains no resolution observation at all -
    in which case there is no honest timestamp and nothing is emitted.
    """
    candidates = [
        e.recv_wall_timestamp_ns / 1e9
        for e in raw.events(round_id=round_id,
                            topics=[Topic.MARKET_METADATA, Topic.RECORDER_CONTROL])
        if e.event_type in _RESOLUTION_EVENT_TYPES
    ]
    return min(candidates) if candidates else None


def _in_window(events: list, lo: float, hi: float) -> list:
    """Reference observations whose SOURCE timestamp falls in [lo, hi]."""
    return [
        e for e in events
        if e.source_timestamp_ns is not None and lo <= e.source_timestamp_ns / 1e9 <= hi
    ]


def _levels(raw: list) -> list[list[float]]:
    """`[{"price": "0.44", "size": "100"}, ...]` -> `[[0.44, 100.0], ...]`,
    the shape `features/engine.py` and `replay/feeds.py` both expect."""
    return [[float(lvl["price"]), float(lvl["size"])] for lvl in raw]


def _reference_value(payload: dict) -> float | None:
    """Prefer the venue's full-accuracy value over the rounded float.

    For the Chainlink topics `full_accuracy_value` is an E18 fixed-point
    integer string; a five-minute BTC round can be flat to several decimal
    places, and the rounded `value` can straddle a tie the exact value
    resolves cleanly.
    """
    inner = payload.get("payload") or {}
    full = inner.get("full_accuracy_value")
    if isinstance(full, str) and full.isdigit():
        return int(full) / 10**18
    value = inner.get("value")
    return float(value) if value is not None else None


def project_round(
    raw: RawEventStore,
    round_id: str,
    out: EventStore,
    *,
    spot_topic: Topic = SPOT_TOPIC,
    include_settlement: bool = False,
) -> ProjectionResult:
    """Project one captured round into `out`.

    `out` must already be labelled `REAL_REPLAY`; refusing to write real
    observations into an unlabelled (therefore synthetic-by-default) store is
    the whole point of the provenance default.
    """
    if out.provenance is not DataProvenance.REAL_REPLAY:
        raise ProjectionError(
            f"destination store is labelled {out.provenance.value}; a real projection "
            "must be written into an EventStore(provenance=DataProvenance.REAL_REPLAY)"
        )

    row = raw.get_round(round_id)
    if row is None:
        raise ProjectionError(f"no round metadata recorded for {round_id}")
    for required in ("start_ts_ns", "end_ts_ns", "tick_size", "min_order_size",
                     "up_token_id", "down_token_id"):
        if row.get(required) is None:
            raise ProjectionError(
                f"{round_id}: recorded round metadata has no {required}; refusing to "
                "default an executable market parameter"
            )

    start_ts = row["start_ts_ns"] / 1e9
    end_ts = row["end_ts_ns"] / 1e9
    # Phase 12C.2 item 4: fail closed. This used to read
    # `row.get("settlement_kind") or "chainlink_reference"`, so a round whose
    # settlement rule was never captured would silently be projected against
    # the plain reference series - quietly choosing which price series counts
    # as truth for the label, which is the most consequential parameter in
    # the round. A missing rule now invalidates the projection.
    settlement_kind = row.get("settlement_kind")
    if not settlement_kind:
        raise ProjectionError(
            f"{round_id}: no settlement rule recorded in the round metadata. "
            "Refusing to guess one - a round whose settlement basis is unknown "
            "cannot be labelled or replayed."
        )
    twap_window_s = row.get("twap_window_s")
    twap_topic = settlement_topic_for(settlement_kind, twap_window_s)

    result = ProjectionResult(
        round_id=round_id,
        provenance=DataProvenance.REAL_REPLAY,
        p0=float("nan"),
        settlement_topic=twap_topic.value,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    batch: list[tuple] = []

    def emit(event_type: EventType, source_ts: float, recv_ts: float, payload: dict) -> None:
        batch.append((event_type, round_id, recv_ts, source_ts, payload))
        result.counts[event_type.value] = result.counts.get(event_type.value, 0) + 1

    def emit_raw(event_type: EventType, source_ts: float | None, recv_ts: float, payload: dict) -> None:
        """Same as `emit`, but allows `source_ts=None` for an observation that
        genuinely has no external source timestamp (item 3)."""
        batch.append((event_type, round_id, recv_ts, source_ts, payload))
        result.counts[event_type.value] = result.counts.get(event_type.value, 0) + 1

    def skip(reason: str) -> None:
        result.skipped[reason] = result.skipped.get(reason, 0) + 1

    # ---------------------------------------------------- settlement basis
    twap_events = reference_events(raw, twap_topic)
    if not twap_events:
        raise ProjectionError(
            f"{round_id}: no {twap_topic.value} observations captured, so the market's "
            f"declared settlement basis ({settlement_kind}, window {twap_window_s}) is "
            "absent. Refusing to substitute another series."
        )

    # p0 is the settlement reference at the round's open - a REAL observation
    # at or before start_ts, never a default (item 8).
    p0_candidates = []
    for e in twap_events:
        t = _times(e)
        if t is None:
            continue
        value = _reference_value(e.payload)
        if value is not None and t[0] <= start_ts:
            p0_candidates.append((t[0], value))
    if not p0_candidates:
        raise ProjectionError(
            f"{round_id}: no {twap_topic.value} observation at or before the round open "
            f"({start_ts}); p0 cannot be established from real data and must not be "
            "invented"
        )
    p0_ts, result.p0 = max(p0_candidates, key=lambda kv: kv[0])
    if start_ts - p0_ts > 5.0:
        result.warnings.append(
            f"p0 taken from an observation {start_ts - p0_ts:.1f}s before the round open"
        )

    # ------------------------------------------------------ MARKET_CONFIG
    # Emitted at the earliest captured instant so it is causally visible at
    # every decision point, and built ONLY from recorded values.
    fee_cfg = json.loads(row.get("fee_config_json") or "{}")
    fee_rate = (fee_cfg.get("feeSchedule") or {}).get("rate")
    if fee_rate is None:
        raise ProjectionError(
            f"{round_id}: recorded fee configuration has no schedule rate; refusing to "
            "default the fee rate"
        )
    if twap_window_s is None:
        raise ProjectionError(
            f"{round_id}: market declares no TWAP window; refusing to assume one"
        )

    # Phase 12C.2 item 3: the MARKET_CONFIG event's visibility timestamp is
    # the moment the recorder ACTUALLY received the market metadata, not a
    # fabricated `min(earliest, start_ts) - 1e-6`. That expression existed
    # only to force the config to sort first; inventing a source timestamp to
    # win a sort is exactly the kind of small fiction this phase removes.
    #
    # The REST metadata carries no external source timestamp, so
    # `source_ts = None` is the honest value - `Event.event_time` then falls
    # back to `recv_ts`, which is genuinely when this system learned it. In
    # the verified capture that is 473.5s before the round opens, so the
    # config is causally visible at every decision point on its own merits.
    metadata_events = [
        e for e in raw.events(round_id=round_id, topics=[Topic.MARKET_METADATA])
        if e.event_type == "market_metadata_discovered"
    ]
    if not metadata_events:
        raise ProjectionError(
            f"{round_id}: no market_metadata_discovered event recorded, so there is no "
            "real timestamp at which this system learned the market's parameters. "
            "Refusing to fabricate one."
        )
    config_recv_ts = metadata_events[0].recv_wall_timestamp_ns / 1e9
    emit_raw(EventType.MARKET_CONFIG, None, config_recv_ts, {
        "market_id": round_id,
        "up_token_id": row["up_token_id"],
        "down_token_id": row["down_token_id"],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "tick_size": float(row["tick_size"]),
        "min_order_size": float(row["min_order_size"]),
        "fee_rate": float(fee_rate),
        "taker_delay_ms": float(row.get("taker_delay_ms") or 0.0),
        "twap_window_seconds": int(twap_window_s),
        # Phase 12C.2 item 4: the settlement rule is a first-class market
        # parameter, carried Raw metadata -> MARKET_CONFIG ->
        # MarketConstraints -> ShadowRunner, so no layer re-derives or
        # defaults it.
        "settlement_kind": settlement_kind,
        PROVENANCE_KEY: {
            "source": "recorder rounds table + market_metadata_discovered",
            "condition_id": row.get("condition_id"),
            "resolution_source": row.get("resolution_source"),
            "raw_event_id": [metadata_events[0].session_id,
                             metadata_events[0].recorder_sequence],
            "provenance": DataProvenance.REAL_REPLAY.value,
        },
    })

    # ------------------------------------------- causal tick-size updates
    # Phase 12C.2 item 2. `tick_size_change` used to be counted as
    # "no_normalized_event_for" and dropped, leaving one immutable tick for
    # the whole replay. The verified capture contains a real change from 0.01
    # to 0.001 at t=+280.2s, so every late-round candidate would have been
    # priced, rounded and limit-capped on a grid the venue had already
    # replaced.
    #
    # Each change is projected as a later MARKET_CONFIG carrying the new tick
    # and every other constraint unchanged, so `ShadowRunner` can simply read
    # "the latest MARKET_CONFIG visible by decision time". The event keeps its
    # own source/receive timestamps, so it becomes visible exactly when it
    # genuinely arrived and no future tick leaks backward.
    #
    # A tick change is a MARKET-level fact that the venue announces once per
    # affected token; deduplicating on (source_ts, new_tick) avoids emitting
    # one redundant config per token for the same change.
    seen_ticks: set[tuple[int, float]] = set()
    current_tick = float(row["tick_size"])
    for e in raw.events(round_id=round_id, topics=[Topic.CLOB_MARKET]):
        if e.event_type != "tick_size_change":
            continue
        t = _times(e)
        new_tick = e.payload.get("new_tick_size") or e.payload.get("tick_size")
        if t is None or new_tick is None:
            skip("tick_size_change_without_timestamp_or_value")
            continue
        if float(new_tick) == current_tick:
            # The venue announces a change once per affected token, and the
            # two announcements can carry slightly different millisecond
            # timestamps. Re-emitting a config for a tick that did not
            # actually change would add noise to the causal record without
            # adding information.
            skip("tick_size_change_repeat_same_value")
            continue
        seen_ticks.add((e.source_timestamp_ns, float(new_tick)))
        current_tick = float(new_tick)
        emit(EventType.MARKET_CONFIG, t[0], t[1], {
            "market_id": round_id,
            "up_token_id": row["up_token_id"],
            "down_token_id": row["down_token_id"],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "tick_size": current_tick,
            "min_order_size": float(row["min_order_size"]),
            "fee_rate": float(fee_rate),
            "taker_delay_ms": float(row.get("taker_delay_ms") or 0.0),
            "twap_window_seconds": int(twap_window_s),
            "settlement_kind": settlement_kind,
            PROVENANCE_KEY: dict(
                _provenance_block(e),
                source="clob tick_size_change",
                old_tick_size=e.payload.get("old_tick_size"),
            ),
        })

    # --------------------------------------------------------------- TWAP
    # Trimmed to this round's own causal window: enough lookback for the
    # largest feature window plus the settle tail, not the whole batch.
    window_lo = start_ts - REFERENCE_LOOKBACK_S
    window_hi = end_ts + REFERENCE_TAIL_S
    for e in _in_window(twap_events, window_lo, window_hi):
        t = _times(e)
        if t is None:
            skip("twap_without_source_timestamp")
            continue
        value = _reference_value(e.payload)
        if value is None:
            skip("twap_without_value")
            continue
        emit(EventType.TWAP, t[0], t[1], {
            "value": value,
            "window_seconds": int(twap_window_s),
            PROVENANCE_KEY: _provenance_block(e),
        })

    # --------------------------------------------------------------- SPOT
    spot_events = reference_events(raw, spot_topic)
    if not spot_events:
        result.warnings.append(
            f"no {spot_topic.value} observations captured; every decision point will "
            "report MISSING_SPOT"
        )
    for e in _in_window(spot_events, window_lo, window_hi):
        t = _times(e)
        if t is None:
            skip("spot_without_source_timestamp")
            continue
        inner = e.payload.get("payload") or {}
        value = inner.get("value")
        if value is None:
            skip("spot_without_value")
            continue
        emit(EventType.SPOT, t[0], t[1], {
            "value": float(value),
            "provider": str(inner.get("symbol") or spot_topic.value),
            PROVENANCE_KEY: _provenance_block(e),
        })

    # --------------------------------------------------------------- BOOK
    for e in raw.events(round_id=round_id, topics=[Topic.CLOB_MARKET, Topic.CLOB_REST]):
        side = e.normalized_side
        if side not in ("UP", "DOWN"):
            skip("book_event_without_known_side")
            continue
        t = _times(e)
        if t is None:
            skip("book_event_without_source_timestamp")
            continue
        payload = e.payload

        if e.event_type in ("book", "book_snapshot_rest", "book_snapshot_rest_reconcile"):
            emit(EventType.BOOK_SNAPSHOT, t[0], t[1], {
                "side": side,
                "bids": _levels(payload.get("bids") or []),
                "asks": _levels(payload.get("asks") or []),
                "book_hash": payload.get("hash"),
                PROVENANCE_KEY: _provenance_block(e),
            })
        elif e.event_type == "price_change":
            # `side` on the wire names the side of the BOOK the level lives
            # on: BUY levels are bids, SELL levels are asks.
            wire_side = str(payload.get("side", "")).upper()
            if wire_side not in ("BUY", "SELL"):
                skip("price_change_without_book_side")
                continue
            emit(EventType.BOOK_DELTA, t[0], t[1], {
                "side": side,
                "book": "bids" if wire_side == "BUY" else "asks",
                "price": float(payload["price"]),
                "size": float(payload["size"]),
                "book_hash": payload.get("hash"),
                PROVENANCE_KEY: _provenance_block(e),
            })
        elif e.event_type == "tick_size_change":
            # Handled by its own pass below (Phase 12C.2 item 2), which emits
            # a MARKET_CONFIG update. Not a skip.
            continue
        else:
            # last_trade_price / best_bid_ask / control events have no
            # normalized counterpart the feature engine reads. They stay in
            # the raw log rather than being forced into a shape that would
            # misrepresent them.
            skip(f"no_normalized_event_for:{e.event_type}")

    # --------------------------------------------------------- SETTLEMENT
    # Phase 12C.2 item 3. This used to emit
    # `SETTLEMENT(source_ts=end_ts, recv_ts=end_ts)`, which made the venue's
    # outcome causally visible the instant the five-minute clock ran out. In
    # the verified capture the outcome was not learned until t=+391.6s - so
    # that stamp handed the feature/controller stream 92 seconds of
    # foreknowledge of the answer.
    #
    # It is now OFF by default: the supervised target belongs on the
    # `RoundLabel` path (`result.label`), not in the causal event stream that
    # feeds the feature engine. When a caller does want it in the store, its
    # visibility timestamp is the moment the resolution was actually
    # observed.
    if include_settlement:
        resolution_recv = _resolution_observed_at(raw, round_id)
        if resolution_recv is None:
            result.warnings.append(
                "settlement requested but no resolution-observation event was recorded; "
                "omitting SETTLEMENT rather than stamping it at the round end"
            )
        for res in raw.round_results():
            if res.get("round_id") != round_id or not res.get("reported_outcome"):
                continue
            if resolution_recv is None:
                continue
            emit_raw(EventType.SETTLEMENT, None, resolution_recv, {
                "outcome": res["reported_outcome"],
                "final_reference": res.get("end_reference_value"),
                "p0": result.p0,
                PROVENANCE_KEY: {
                    "source": "recorder round_results",
                    "reconstructed_outcome": res.get("reconstructed_outcome"),
                    "reconstruction_basis": res.get("reconstruction_basis"),
                    "label_agreement": res.get("label_agreement"),
                    "observed_at": resolution_recv,
                    "provenance": DataProvenance.REAL_REPLAY.value,
                },
            })

    # Phase 12C.2 item 2: the causal tick record, oldest first.
    result.tick_timeline = sorted(
        {(config_recv_ts, float(row["tick_size"]))}
        | {(ts / 1e9, tick) for ts, tick in seen_ticks}
    )

    # Phase 12C.2 item 3: the supervised target travels here, NOT in the
    # causal event stream, and carries the moment it was actually observed.
    result.label_observed_at = _resolution_observed_at(raw, round_id)
    for res in raw.round_results():
        if res.get("round_id") != round_id:
            continue
        from xamarinbot.portfolio.state import Side

        # Gate A.0 item 2. This used to emit a label whenever
        # `reported_outcome` was present, so a round with
        #     reported=UP, reconstructed=None, status=UNRESOLVED
        # still produced `RoundLabel(outcome=UP)` - a supervised target
        # taken purely on the venue's word, with our independent
        # reconstruction having failed or disagreed. The whole point of
        # reconstructing the settlement rule ourselves is to be able to
        # refuse in exactly that case.
        #
        # A supervised target is now emitted ONLY when the independent
        # reconstruction is CONFIRMED, and the value used is the
        # RECONSTRUCTED outcome (verified equal to the venue's), not the
        # venue's outcome copied over.
        reported = res.get("reported_outcome")
        reconstructed = res.get("reconstructed_outcome")
        agrees = res.get("label_agreement")
        status = _label_status_for(raw, round_id)

        if reported is None:
            result.warnings.append("no venue outcome recorded; no supervised label emitted")
            continue
        if reconstructed is None:
            result.warnings.append(
                "settlement label could not be independently reconstructed; no supervised "
                "label emitted (the venue's word alone is not a verified target)"
            )
            continue
        if agrees != 1 or reconstructed != reported:
            result.warnings.append(
                f"reconstruction ({reconstructed}) disagrees with the venue ({reported}); "
                "LABEL_AMBIGUOUS, no supervised label emitted"
            )
            continue
        if status is not None and status != "CONFIRMED":
            result.warnings.append(
                f"label status is {status}; no supervised label emitted"
            )
            continue

        result.label = RoundLabel(
            round_id=round_id,
            p0=result.p0,
            final_reference=res.get("end_reference_value") or float("nan"),
            outcome=Side.UP if reconstructed == "UP" else Side.DOWN,
            provenance=DataProvenance.REAL_REPLAY,
        )
        if result.label_observed_at is not None and result.label_observed_at <= end_ts:
            result.warnings.append(
                f"resolution observed at {result.label_observed_at} which is not after "
                f"the round end {end_ts}; treat this round's label timing as suspect"
            )

    out.append_many(batch)
    return result


def project_capture(
    raw: RawEventStore,
    out: EventStore,
    round_ids: list[str] | None = None,
    log=lambda msg: None,
) -> list[ProjectionResult]:
    """Project every (or a chosen subset of) captured round. A round that
    cannot be projected is REPORTED and skipped, not silently dropped and not
    partially faked."""
    results: list[ProjectionResult] = []
    for round_id in round_ids if round_ids is not None else raw.round_ids():
        try:
            res = project_round(raw, round_id, out)
        except ProjectionError as exc:
            log(f"[projection] {round_id}: SKIPPED - {exc}")
            continue
        log(
            f"[projection] {round_id}: {res.total_projected} events "
            f"({res.counts}) p0={res.p0}"
        )
        results.append(res)
    return results
