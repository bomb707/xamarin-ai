"""Incremental LIVE raw -> normalized projection (readiness audit item 3).

The gap this closes
-------------------
`replay/projection.py` turns a finished capture into a normalized
`EventStore`. It is a batch job over a completed round, and everything
downstream - `FeatureEngine`, `ShadowRunner` - reads that store.

That means the only path from real market data to a decision was

    raw capture -> normalized replay DB -> ReplayCursor -> decision

which is a REPLAY system. A live bot cannot wait for a round to finish
before deciding inside it.

This module applies the SAME mapping, one event at a time, as events arrive
off the wire. It is deliberately a thin translator, not a second feature
engine: the normalized payload contract is byte-identical to the offline
projection's, so `features/engine.compute` sees exactly the same input shape
live as in replay, and the same mathematics serve both.
"""
from __future__ import annotations

import json

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.raw_events import RawEvent, Topic

#: Raw topic -> the reference series it becomes. Only the round's DECLARED
#: settlement basis may become `EventType.TWAP`; mixing two different
#: quantities into one series would silently corrupt `gap_twap_bp`/`z_gap`.
SPOT_TOPIC = Topic.RTDS_BINANCE


class LiveProjector:
    """Translates live raw events into a normalized `EventStore`.

    One projector per round. `settlement_topic` is the round's own declared
    basis, read from its market metadata - never a constant.
    """

    def __init__(
        self,
        store: EventStore,
        round_id: str,
        settlement_topic: Topic,
        *,
        spot_topic: Topic = SPOT_TOPIC,
    ):
        if not store.provenance.is_real:
            raise ValueError(
                f"live projection refuses to write into a {store.provenance.value} "
                "store; real observations must never land in a synthetic-labelled DB"
            )
        self.store = store
        self.round_id = round_id
        self.settlement_topic = settlement_topic
        self.spot_topic = spot_topic
        self.counts: dict[str, int] = {}
        self.skipped: dict[str, int] = {}

    # ------------------------------------------------------------ helpers

    def _emit(self, kind: EventType, source_ns: int | None, recv_ns: int, payload: dict) -> None:
        self.store.append(
            kind, self.round_id,
            recv_ts=recv_ns / 1e9,
            source_ts=(source_ns / 1e9) if source_ns is not None else None,
            payload=payload,
        )
        self.counts[kind.name] = self.counts.get(kind.name, 0) + 1

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def emit_market_config(self, metadata, recv_ns: int) -> None:
        """MARKET_CONFIG carries the round's executable parameters.

        `source_ts=None` deliberately: REST metadata has no exchange clock,
        so its only honest timestamp is when we received it. That is also
        what makes it visible to the recv_ts gate at the right moment.
        """
        self._emit(EventType.MARKET_CONFIG, None, recv_ns, {
            "market_id": metadata.condition_id,
            "up_token_id": metadata.up_token_id,
            "down_token_id": metadata.down_token_id,
            "start_ts": metadata.start_ts,
            "end_ts": metadata.end_ts,
            "tick_size": metadata.tick_size,
            "min_order_size": metadata.min_order_size,
            "fee_rate": metadata.fee_config.rate if metadata.fee_config else 0.0,
            "taker_delay_ms": metadata.taker_delay_ms,
            "twap_window_seconds": metadata.twap_window_s or 60,
            "settlement_kind": metadata.settlement_kind,
        })

    # -------------------------------------------------------------- apply

    def apply(self, event: RawEvent) -> bool:
        """Project one live raw event. Returns True if it produced a
        normalized event.

        Events with no normalized counterpart (trades, control frames,
        non-declared reference series) are COUNTED as skipped rather than
        silently dropped - a projection that quietly discards is how a
        feature series ends up thinner than anyone realises.
        """
        if event.round_id is not None and event.round_id != self.round_id:
            # Reference feeds are global and are tagged to whichever round
            # was active; they are still this round's inputs. Book events
            # are per token and are filtered by token below.
            if event.topic in (Topic.CLOB_MARKET, Topic.CLOB_REST):
                self._skip("other_round_book")
                return False

        recv_ns = event.recv_wall_timestamp_ns
        source_ns = event.source_timestamp_ns
        payload = event.payload

        if event.topic is self.settlement_topic:
            value = self._reference_value(payload)
            if value is None:
                self._skip("reference_without_value")
                return False
            self._emit(EventType.TWAP, source_ns, recv_ns, {
                "value": value,
                "window_seconds": (payload.get("payload") or {}).get("window_s") or 60,
            })
            return True

        if event.topic is self.spot_topic:
            value = self._reference_value(payload)
            if value is None:
                self._skip("spot_without_value")
                return False
            self._emit(EventType.SPOT, source_ns, recv_ns,
                       {"value": value, "provider": "binance"})
            return True

        if event.topic in (Topic.RTDS_CHAINLINK, Topic.RTDS_TWAP_30, Topic.RTDS_TWAP_60):
            # A real reference series, but not THIS round's declared basis.
            # Kept out of EventType.TWAP on purpose - see module docstring.
            self._skip("non_declared_reference")
            return False

        if event.topic in (Topic.CLOB_MARKET, Topic.CLOB_REST):
            return self._apply_book(event, source_ns, recv_ns, payload)

        self._skip(f"no_normalized_counterpart:{event.topic.value}")
        return False

    def _reference_value(self, payload: dict) -> float | None:
        inner = payload.get("payload") if isinstance(payload, dict) else None
        raw = (inner or {}).get("value") if isinstance(inner, dict) else None
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _apply_book(self, event: RawEvent, source_ns, recv_ns, payload: dict) -> bool:
        side = event.normalized_side
        if side not in ("UP", "DOWN"):
            self._skip("book_without_side")
            return False

        if event.event_type in ("book", "book_snapshot_rest", "book_snapshot_rest_resync"):
            self._emit(EventType.BOOK_SNAPSHOT, source_ns, recv_ns, {
                "side": side,
                "bids": [[float(b["price"]), float(b["size"])]
                         for b in payload.get("bids") or []],
                "asks": [[float(a["price"]), float(a["size"])]
                         for a in payload.get("asks") or []],
                "token_id": event.token_id,
                "hash": payload.get("hash"),
            })
            return True

        if event.event_type == "price_change":
            emitted = False
            for change in payload.get("price_changes") or [payload]:
                try:
                    price = float(change["price"])
                    size = float(change["size"])
                except (KeyError, TypeError, ValueError):
                    self._skip("malformed_price_change")
                    continue
                venue_side = str(change.get("side", "")).upper()
                if venue_side not in ("BUY", "SELL"):
                    self._skip("price_change_without_side")
                    continue
                self._emit(EventType.BOOK_DELTA, source_ns, recv_ns, {
                    "side": side,
                    "book": "bids" if venue_side == "BUY" else "asks",
                    "price": price,
                    "size": size,
                    "token_id": event.token_id,
                    "hash": change.get("hash") or payload.get("hash"),
                })
                emitted = True
            return emitted

        if event.event_type == "tick_size_change":
            self._skip("tick_size_change_handled_by_market_config")
            return False

        self._skip(f"no_normalized_counterpart:{event.event_type}")
        return False
