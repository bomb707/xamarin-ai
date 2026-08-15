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


class ProjectionError(RuntimeError):
    """Raised instead of inventing a value the capture does not contain."""


def settlement_topic_for(settlement_kind: str, twap_window_s: int | None) -> Topic:
    """The raw topic this market declares as its settlement reference."""
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
    include_settlement: bool = True,
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
    settlement_kind = row.get("settlement_kind") or "chainlink_reference"
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

    def skip(reason: str) -> None:
        result.skipped[reason] = result.skipped.get(reason, 0) + 1

    # ---------------------------------------------------- settlement basis
    twap_events = raw.events(round_id=round_id, topics=[twap_topic])
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

    earliest = min(
        (t[0] for e in raw.events(round_id=round_id) if (t := _times(e)) is not None),
        default=start_ts,
    )
    config_ts = min(earliest, start_ts) - 1e-6
    emit(EventType.MARKET_CONFIG, config_ts, config_ts, {
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
        PROVENANCE_KEY: {
            "source": "recorder rounds table",
            "condition_id": row.get("condition_id"),
            "settlement_kind": settlement_kind,
            "resolution_source": row.get("resolution_source"),
            "provenance": DataProvenance.REAL_REPLAY.value,
        },
    })

    # --------------------------------------------------------------- TWAP
    for e in twap_events:
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
    spot_events = raw.events(round_id=round_id, topics=[spot_topic])
    if not spot_events:
        result.warnings.append(
            f"no {spot_topic.value} observations captured; every decision point will "
            "report MISSING_SPOT"
        )
    for e in spot_events:
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
        else:
            # last_trade_price / best_bid_ask / tick_size_change / control
            # events have no normalized counterpart the feature engine reads.
            # They stay in the raw log rather than being forced into a shape
            # that would misrepresent them.
            skip(f"no_normalized_event_for:{e.event_type}")

    # --------------------------------------------------------- SETTLEMENT
    if include_settlement:
        for res in raw.round_results():
            if res.get("round_id") != round_id or not res.get("reported_outcome"):
                continue
            emit(EventType.SETTLEMENT, end_ts, end_ts, {
                "outcome": res["reported_outcome"],
                "final_reference": res.get("end_reference_value"),
                "p0": result.p0,
                PROVENANCE_KEY: {
                    "source": "recorder round_results",
                    "reconstructed_outcome": res.get("reconstructed_outcome"),
                    "reconstruction_basis": res.get("reconstruction_basis"),
                    "label_agreement": res.get("label_agreement"),
                    "provenance": DataProvenance.REAL_REPLAY.value,
                },
            })

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
