"""Production Polymarket CLOB market-channel adapter (Phase 12C item 3).

Replaces the behavior of the previous `feeds/polymarket_clob.py`, whose
defects the brief enumerates. Each is fixed here, and each was CONFIRMED
against the live socket on 2026-08-15 rather than reasoned about:

  `_map_tokens_to_sides()` NotImplementedError
      Moved out of the feed entirely. Token->side is resolved once, up
      front, from explicit outcome labels by `discovery.py`, and this
      adapter is simply *given* the mapping.

  ISO timestamp parsing
      Not this module's job either (see `discovery.parse_iso8601`); the
      market channel's own `timestamp` is a millisecond epoch delivered as
      a JSON string (`"timestamp": "1786771382832"`), handled by `ms_to_ns`.

  `get_snapshot()` doing REST on every call
      `get_snapshot` now reads the in-memory book maintained from the
      WebSocket. REST is used for exactly three things: the bootstrap
      snapshot, resynchronization after a reconnect, and the periodic
      integrity check.

  `price_change` token routing
      THE critical bug. The old handler did `token_id = msg.get("asset_id")`
      for every message, but a live `price_change` frame has NO top-level
      `asset_id` - it carries a `price_changes` array whose ELEMENTS each
      have their own `asset_id`. `_side_for_token(None)` therefore returned
      None and the handler returned early, silently discarding every single
      book delta. Measured live: 8090 `price_change` frames in 60 seconds,
      i.e. the old adapter's book would have been frozen at its bootstrap
      snapshot forever while appearing to work. Each element is now applied
      independently, exactly as the brief requires.

  tick-size changes
      `tick_size_change` is handled on this same connection and updates the
      book's tick size, rather than raising NotImplementedError and telling
      the caller to open a second socket.

  reconnect/resync behavior
      Reconnect increments a `reconnect_generation` stamped onto every
      subsequent raw event, then re-bootstraps every token from REST before
      applying further deltas, so a gap can never be mistaken for a quiet
      book.

Live-verified subscription and event vocabulary
-----------------------------------------------
`{"assets_ids": [...], "type": "market", "custom_feature_enabled": true}`
was A/B tested against the plain form over 60s on the same tokens:

    plain    -> book, price_change, last_trade_price
    custom   -> book, price_change, last_trade_price, best_bid_ask, new_market

so `custom_feature_enabled` genuinely gates the extra streams and is
enabled by default here. `market_resolved` is subscribed for and handled;
it was not observed during the probe window because no round resolved
inside it.

All prices and sizes arrive as STRINGS and are converted at the boundary.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from xamarinbot.realtime.raw_events import RawEvent, RawEventBuilder, Topic, ms_to_ns

CLOB_REST_BASE_URL = "https://clob.polymarket.com"
CLOB_MARKET_WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

#: Event types the market channel emits (all confirmed live except
#: `market_resolved`, which is documented and handled but did not occur
#: during the probe window).
MARKET_EVENT_TYPES = (
    "book",
    "price_change",
    "last_trade_price",
    "tick_size_change",
    "best_bid_ask",
    "new_market",
    "market_resolved",
)


def _f(v) -> float:
    return float(v)


#: Text the venue sends when it closes the market channel because every
#: subscribed token has settled. Not an error - the end of the subscription.
_ALL_RESOLVED = "all subscribed assets resolved"


def _is_normal_close(exc: Exception) -> bool:
    """True for a clean WebSocket close (status 1000), which is the venue
    ending the subscription rather than a fault on our side."""
    return type(exc).__name__ in ("ConnectionClosedOK",) or "received 1000 (OK)" in str(exc)


def _all_assets_resolved(exc: Exception) -> bool:
    return _ALL_RESOLVED in str(exc)


def _is_settled_token_404(exc: Exception) -> bool:
    """A REST `/book` 404 means the token's market has settled and its book
    no longer exists - expected at end of round, not a fault."""
    return "404" in str(exc) and "/book" in str(exc)


@dataclass
class BookState:
    """In-memory order book for ONE token, maintained from the WebSocket."""

    token_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    tick_size: float | None = None
    book_hash: str | None = None
    #: Source (exchange) timestamp of the last applied update, ns.
    source_ts_ns: int | None = None
    #: Local wall clock of the last applied update, ns.
    recv_ts_ns: int | None = None
    last_trade_price: float | None = None
    #: True until a snapshot has ever been applied - a book that has only
    #: seen deltas is not a book, and must never be quoted from.
    awaiting_snapshot: bool = True
    #: Set when a delta arrived while awaiting a snapshot, i.e. a known gap.
    has_gap: bool = False
    applied_updates: int = 0

    def apply_snapshot(self, bids: Iterable[dict], asks: Iterable[dict], *, book_hash: str | None,
                       source_ts_ns: int | None, recv_ts_ns: int, tick_size: float | None = None) -> None:
        self.bids = {_f(b["price"]): _f(b["size"]) for b in bids if _f(b["size"]) > 0}
        self.asks = {_f(a["price"]): _f(a["size"]) for a in asks if _f(a["size"]) > 0}
        self.book_hash = book_hash
        self.source_ts_ns = source_ts_ns
        self.recv_ts_ns = recv_ts_ns
        if tick_size is not None:
            self.tick_size = tick_size
        self.awaiting_snapshot = False
        self.has_gap = False
        self.applied_updates += 1

    def apply_price_change(self, price: float, size: float, side: str, *, book_hash: str | None,
                           source_ts_ns: int | None, recv_ts_ns: int) -> None:
        """One element of a `price_changes` array. `side` is the venue's
        BUY/SELL, which refers to the side of the BOOK the level lives on:
        BUY levels are bids, SELL levels are asks. A size of 0 removes the
        level."""
        if self.awaiting_snapshot:
            # A delta with no snapshot underneath it cannot be applied to
            # anything meaningful. Record the gap instead of pretending.
            self.has_gap = True
            return
        book = self.bids if side.upper() == "BUY" else self.asks
        if size == 0:
            book.pop(price, None)
        else:
            book[price] = size
        if book_hash is not None:
            self.book_hash = book_hash
        self.source_ts_ns = source_ts_ns
        self.recv_ts_ns = recv_ts_ns
        self.applied_updates += 1

    @property
    def best_bid(self) -> tuple[float, float] | None:
        if not self.bids:
            return None
        p = max(self.bids)
        return (p, self.bids[p])

    @property
    def best_ask(self) -> tuple[float, float] | None:
        if not self.asks:
            return None
        p = min(self.asks)
        return (p, self.asks[p])

    @property
    def is_quotable(self) -> bool:
        """A book only counts as usable once it has a snapshot underneath
        it, has no known gap, and has both sides."""
        return (
            not self.awaiting_snapshot
            and not self.has_gap
            and bool(self.bids)
            and bool(self.asks)
        )

    def sorted_bids(self) -> list[tuple[float, float]]:
        return sorted(self.bids.items(), key=lambda kv: -kv[0])

    def sorted_asks(self) -> list[tuple[float, float]]:
        return sorted(self.asks.items(), key=lambda kv: kv[0])

    def crossed(self) -> bool:
        bb, ba = self.best_bid, self.best_ask
        return bb is not None and ba is not None and bb[0] >= ba[0]


@dataclass(frozen=True)
class IntegrityResult:
    token_id: str
    matched: bool
    detail: str
    ws_best_bid: float | None = None
    ws_best_ask: float | None = None
    rest_best_bid: float | None = None
    rest_best_ask: float | None = None


class PolymarketMarketStream:
    """One WebSocket connection serving every subscribed token.

    The reader loop does no persistence itself: it hands each raw event to
    `on_raw_event`, which the recorder implements as a non-blocking enqueue.
    """

    def __init__(
        self,
        token_ids: list[str],
        *,
        builder: RawEventBuilder,
        on_raw_event: Callable[[RawEvent], None],
        side_for_token: dict[str, str] | None = None,
        round_for_token: dict[str, str] | None = None,
        condition_for_token: dict[str, str] | None = None,
        ws_url: str = CLOB_MARKET_WSS_URL,
        rest_base_url: str = CLOB_REST_BASE_URL,
        request_timeout_s: float = 10.0,
        ping_interval_s: float = 10.0,
        #: Stall watchdog, for the same reason as `RTDSClient`'s: a socket
        #: that stops delivering without raising must not be sat on. The
        #: CLOB market channel was measured at ~130 messages/second, so 30s
        #: of total silence on a live round is unambiguous.
        stall_timeout_s: float = 30.0,
        custom_features: bool = True,
        on_parse_failure: Callable[[str, Exception], None] | None = None,
        on_reconnect: Callable[[int], None] | None = None,
        on_resnapshot: Callable[[str], None] | None = None,
    ):
        self._token_ids = list(token_ids)
        self._builder = builder
        self._emit = on_raw_event
        self._side_for_token = dict(side_for_token or {})
        self._round_for_token = dict(round_for_token or {})
        self._condition_for_token = dict(condition_for_token or {})
        self._ws_url = ws_url
        self._rest_base = rest_base_url
        self._timeout = request_timeout_s
        self._ping_interval = ping_interval_s
        self._stall_timeout = stall_timeout_s
        self._custom_features = custom_features
        self._on_parse_failure = on_parse_failure or (lambda raw, exc: None)
        self._on_reconnect = on_reconnect or (lambda gen: None)
        self._on_resnapshot = on_resnapshot or (lambda token_id: None)

        self._books: dict[str, BookState] = {t: BookState(t) for t in self._token_ids}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._resolved: dict[str, dict] = {}
        #: Set when the venue closed the subscription because every
        #: subscribed asset had settled. See `_run_forever`.
        self._closed_by_venue = False

    # ------------------------------------------------------------- tokens

    def add_tokens(self, token_ids: list[str], *, side_for_token: dict[str, str] | None = None,
                   round_for_token: dict[str, str] | None = None,
                   condition_for_token: dict[str, str] | None = None) -> None:
        """Track additional tokens (the next round's) without tearing down
        the connection - the market channel supports incremental
        subscription, and a reconnect between consecutive 5-minute rounds
        would guarantee a data gap exactly at the rollover."""
        with self._lock:
            new = [t for t in token_ids if t not in self._books]
            for t in new:
                self._books[t] = BookState(t)
                self._token_ids.append(t)
            self._side_for_token.update(side_for_token or {})
            self._round_for_token.update(round_for_token or {})
            self._condition_for_token.update(condition_for_token or {})
        if not new:
            return
        if self._closed_by_venue:
            # The reader exited because every previously-subscribed asset
            # had settled. These new tokens have not, so bring it back up
            # rather than leaving them subscribed to a dead thread.
            self._closed_by_venue = False
            self.start()
            return
        if self._connected.is_set():
            self._send(json.dumps({"assets_ids": new, "operation": "subscribe"}))
            for t in new:
                self.bootstrap_token(t)

    # --------------------------------------------------------------- REST

    def _rest_book(self, token_id: str) -> dict:
        from xamarinbot.feeds._live_deps import require_live_deps

        require_live_deps()
        import httpx

        resp = httpx.get(f"{self._rest_base}/book", params={"token_id": token_id}, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def bootstrap_token(self, token_id: str) -> None:
        """REST bootstrap / resync for one token. One of only three
        permitted REST uses (item 3)."""
        payload = self._rest_book(token_id)
        raw = json.dumps(payload, separators=(",", ":"))
        src_ns = ms_to_ns(payload.get("timestamp"))
        ev = self._build(
            Topic.CLOB_REST, "book_snapshot_rest", payload, token_id=token_id,
            source_ts_ns=src_ns, raw_json=raw,
        )
        with self._lock:
            self._books.setdefault(token_id, BookState(token_id)).apply_snapshot(
                payload.get("bids", []), payload.get("asks", []),
                book_hash=payload.get("hash"), source_ts_ns=src_ns,
                recv_ts_ns=ev.recv_wall_timestamp_ns,
                tick_size=_f(payload["tick_size"]) if payload.get("tick_size") else None,
            )
        self._emit(ev)
        self._on_resnapshot(token_id)

    def bootstrap_all(self) -> None:
        for token_id in list(self._books):
            try:
                self.bootstrap_token(token_id)
            except Exception as exc:
                if _is_settled_token_404(exc):
                    # LIVE FINDING (2026-08-15): `/book?token_id=…` returns
                    # 404 once the market has settled - the book no longer
                    # exists. Expected for a token whose round is over, and
                    # not a data-integrity failure; recorded as a control
                    # event instead of a parse failure so it cannot
                    # disqualify the capture.
                    self._emit(self._build(
                        Topic.RECORDER_CONTROL, "book_unavailable_settled",
                        {"token_id": token_id, "reason": str(exc)[:200]},
                        token_id=token_id,
                    ))
                    continue
                self._on_parse_failure(f"bootstrap:{token_id}", exc)

    def check_integrity(self, token_id: str) -> IntegrityResult:
        """Periodic in-memory-vs-REST verification (item 15).

        A mismatch does NOT silently continue: the caller marks the interval
        suspect, this method resnapshots the book from the REST response it
        just fetched, and a reconciliation event is written to the raw log.
        """
        try:
            payload = self._rest_book(token_id)
        except Exception as exc:
            self._on_parse_failure(f"integrity:{token_id}", exc)
            return IntegrityResult(token_id, False, f"REST fetch failed: {exc}")

        rest_bids = {_f(b["price"]): _f(b["size"]) for b in payload.get("bids", []) if _f(b["size"]) > 0}
        rest_asks = {_f(a["price"]): _f(a["size"]) for a in payload.get("asks", []) if _f(a["size"]) > 0}
        rest_bb = max(rest_bids) if rest_bids else None
        rest_ba = min(rest_asks) if rest_asks else None

        with self._lock:
            book = self._books.get(token_id)
            ws_bb = book.best_bid[0] if book and book.best_bid else None
            ws_ba = book.best_ask[0] if book and book.best_ask else None
            ws_hash = book.book_hash if book else None

        # Compare top of book rather than the full ladder: the REST snapshot
        # and the WS stream are read at genuinely different instants against
        # a book updating ~130 times/second, so a deep-level difference is
        # expected and is not evidence of corruption. A top-of-book
        # disagreement is the signal that actually matters for execution.
        matched = (ws_bb == rest_bb) and (ws_ba == rest_ba)
        detail = (
            "top-of-book agrees" if matched
            else f"ws=({ws_bb},{ws_ba}) rest=({rest_bb},{rest_ba}) ws_hash={ws_hash} rest_hash={payload.get('hash')}"
        )
        result = IntegrityResult(token_id, matched, detail, ws_bb, ws_ba, rest_bb, rest_ba)

        self._emit(self._build(
            Topic.RECORDER_CONTROL, "book_integrity_check",
            {
                "token_id": token_id, "matched": matched, "detail": detail,
                "ws_best_bid": ws_bb, "ws_best_ask": ws_ba,
                "rest_best_bid": rest_bb, "rest_best_ask": rest_ba,
            },
            token_id=token_id,
        ))
        if not matched:
            # Resnapshot from the response we already have rather than
            # issuing a second REST call at a third instant.
            src_ns = ms_to_ns(payload.get("timestamp"))
            ev = self._build(
                Topic.CLOB_REST, "book_snapshot_rest_reconcile", payload,
                token_id=token_id, source_ts_ns=src_ns,
                raw_json=json.dumps(payload, separators=(",", ":")),
            )
            with self._lock:
                self._books[token_id].apply_snapshot(
                    payload.get("bids", []), payload.get("asks", []),
                    book_hash=payload.get("hash"), source_ts_ns=src_ns,
                    recv_ts_ns=ev.recv_wall_timestamp_ns,
                )
            self._emit(ev)
            self._on_resnapshot(token_id)
        return result

    # ---------------------------------------------------------- WebSocket

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="clob-market-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _subscribe_message(self) -> str:
        msg: dict = {"assets_ids": list(self._books), "type": "market"}
        if self._custom_features:
            # Live-verified: gates best_bid_ask / new_market / market_resolved.
            msg["custom_feature_enabled"] = True
        return json.dumps(msg)

    def _send(self, text: str) -> None:
        ws = getattr(self, "_ws", None)
        if ws is not None:
            try:
                ws.send(text)
            except Exception:
                pass

    def _run_forever(self) -> None:
        from websockets.sync.client import connect

        backoff = 1.0
        first = True
        while not self._stop.is_set():
            try:
                with connect(self._ws_url, open_timeout=self._timeout, max_size=None) as ws:
                    self._ws = ws
                    if not first:
                        # A reconnect means an unknown gap. Bump the
                        # generation so every subsequent raw event is
                        # attributable to this connection, and mark every
                        # book as needing a snapshot before it is trusted
                        # again.
                        self._builder.reconnect_generation += 1
                        with self._lock:
                            for book in self._books.values():
                                book.awaiting_snapshot = True
                        self._on_reconnect(self._builder.reconnect_generation)
                        self._emit(self._build(
                            Topic.RECORDER_CONTROL, "reconnect",
                            {"generation": self._builder.reconnect_generation, "stream": "clob_market"},
                        ))
                    first = False
                    ws.send(self._subscribe_message())
                    self._connected.set()
                    # Resync from REST before trusting further deltas.
                    self.bootstrap_all()
                    backoff = 1.0
                    last_ping = time.monotonic()
                    last_data = time.monotonic()
                    while not self._stop.is_set():
                        try:
                            raw = ws.recv(timeout=1.0)
                        except TimeoutError:
                            raw = None
                        if time.monotonic() - last_ping >= self._ping_interval:
                            ws.send("PING")
                            last_ping = time.monotonic()
                        if raw:
                            if raw not in ("PONG", "pong"):
                                last_data = time.monotonic()
                            self.handle_message(raw)
                        if time.monotonic() - last_data > self._stall_timeout:
                            self._emit(self._build(
                                Topic.RECORDER_CONTROL, "stream_stalled",
                                {
                                    "stream": "clob_market",
                                    "silent_for_s": time.monotonic() - last_data,
                                    "stall_timeout_s": self._stall_timeout,
                                },
                            ))
                            break
            except Exception as exc:
                self._connected.clear()
                if self._stop.is_set():
                    return
                if _is_normal_close(exc):
                    # LIVE FINDING (2026-08-15): once every subscribed token
                    # has settled, Polymarket closes the market channel
                    # cleanly with
                    #   ConnectionClosedOK: received 1000 (OK)
                    #   "all subscribed assets resolved"
                    # That is the venue telling us the subscription is
                    # finished, not a data-integrity problem. Counting it as
                    # a parse failure disqualified an otherwise flawless
                    # capture (0 dropped events, 30/30 integrity checks) for
                    # model training, which is exactly backwards.
                    self._emit(self._build(
                        Topic.RECORDER_CONTROL, "stream_closed_by_venue",
                        {"stream": "clob_market", "reason": str(exc)},
                    ))
                    if _all_assets_resolved(exc):
                        # Nothing further can arrive on this subscription.
                        # Flagged rather than just returning, so `add_tokens`
                        # can restart the reader if a later round's tokens
                        # are subscribed after this point - otherwise the
                        # thread would be gone and the new subscription would
                        # silently receive nothing.
                        self._closed_by_venue = True
                        self._thread = None
                        return
                else:
                    self._on_parse_failure("ws_connection", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                self._ws = None
                self._connected.clear()

    # ------------------------------------------------------------ routing

    def _build(self, topic: Topic, event_type: str, payload, *, token_id: str | None = None,
               source_ts_ns: int | None = None, publisher_ts_ns: int | None = None,
               raw_json: str | None = None) -> RawEvent:
        return self._builder.build(
            topic, event_type, payload,
            round_id=self._round_for_token.get(token_id) if token_id else None,
            condition_id=self._condition_for_token.get(token_id) if token_id else None,
            token_id=token_id,
            source_timestamp_ns=source_ts_ns,
            publisher_timestamp_ns=publisher_ts_ns,
            normalized_side=self._side_for_token.get(token_id) if token_id else None,
            raw_json=raw_json,
        )

    def handle_message(self, raw: str) -> list[RawEvent]:
        """Parse one wire frame and apply it. Public so tests can drive the
        adapter with captured frames without a socket.

        Returns the raw events emitted, for test convenience; the recorder
        consumes them through `on_raw_event`.
        """
        if raw in ("PONG", "PING", ""):
            return []
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            self._on_parse_failure(raw, exc)
            return []

        # The channel delivers both single objects and arrays of them.
        messages = msg if isinstance(msg, list) else [msg]
        emitted: list[RawEvent] = []
        for m in messages:
            if not isinstance(m, dict):
                self._on_parse_failure(str(m), TypeError("non-object market message"))
                continue
            try:
                emitted.extend(self._route(m))
            except Exception as exc:
                self._on_parse_failure(json.dumps(m)[:500], exc)
        return emitted

    def _route(self, m: dict) -> list[RawEvent]:
        et = m.get("event_type")
        if et == "book":
            return [self._handle_book(m)]
        if et == "price_change":
            return self._handle_price_change(m)
        if et == "tick_size_change":
            return [self._handle_tick_size_change(m)]
        if et == "last_trade_price":
            return [self._handle_simple(m, et)]
        if et == "best_bid_ask":
            return [self._handle_simple(m, et)]
        if et == "market_resolved":
            return [self._handle_market_resolved(m)]
        if et == "new_market":
            # Recorded for discovery/audit even though it usually concerns
            # other assets; it is how the venue announces the next round.
            return [self._emit_and_return(self._build(
                Topic.CLOB_MARKET, "new_market", m,
                source_ts_ns=ms_to_ns(m.get("timestamp")),
                raw_json=json.dumps(m, separators=(",", ":")),
            ))]
        # Unknown event type: still recorded verbatim. An unrecognized
        # message is data, not noise - dropping it would make the raw log a
        # filtered view of the wire rather than the wire.
        return [self._emit_and_return(self._build(
            Topic.CLOB_MARKET, str(et or "unknown"), m,
            source_ts_ns=ms_to_ns(m.get("timestamp")),
            raw_json=json.dumps(m, separators=(",", ":")),
        ))]

    def _emit_and_return(self, ev: RawEvent) -> RawEvent:
        self._emit(ev)
        return ev

    def _handle_book(self, m: dict) -> RawEvent:
        token_id = m.get("asset_id")
        src_ns = ms_to_ns(m.get("timestamp"))
        ev = self._build(Topic.CLOB_MARKET, "book", m, token_id=token_id,
                         source_ts_ns=src_ns, raw_json=json.dumps(m, separators=(",", ":")))
        with self._lock:
            book = self._books.setdefault(token_id, BookState(token_id))
            book.apply_snapshot(
                m.get("bids", []), m.get("asks", []),
                book_hash=m.get("hash"), source_ts_ns=src_ns,
                recv_ts_ns=ev.recv_wall_timestamp_ns,
                tick_size=_f(m["tick_size"]) if m.get("tick_size") else None,
            )
            if m.get("last_trade_price") is not None:
                book.last_trade_price = _f(m["last_trade_price"])
        self._emit(ev)
        return ev

    def _handle_price_change(self, m: dict) -> list[RawEvent]:
        """A `price_change` frame carries MULTIPLE tokens' changes. Each is
        processed independently (item 3), and each becomes its own raw event
        so a per-token query returns exactly that token's deltas.

        Live shape:
            {"market": "0x..", "timestamp": "1786771382835",
             "event_type": "price_change",
             "price_changes": [{"asset_id": "...", "price": "0.01",
                                "size": "2315.29", "side": "BUY",
                                "hash": "...", "best_bid": "0.98",
                                "best_ask": "0.99"}, ...]}
        Note there is NO top-level `asset_id` - reading one was the old
        adapter's routing bug.
        """
        src_ns = ms_to_ns(m.get("timestamp"))
        changes = m.get("price_changes") or []
        out: list[RawEvent] = []
        for change in changes:
            token_id = change.get("asset_id")
            # Each change is stored as its own row, carrying the frame's
            # market/timestamp so nothing about the original grouping is lost.
            element = dict(change)
            element["market"] = m.get("market")
            element["timestamp"] = m.get("timestamp")
            element["_frame_change_count"] = len(changes)
            ev = self._build(
                Topic.CLOB_MARKET, "price_change", element, token_id=token_id,
                source_ts_ns=src_ns, raw_json=json.dumps(element, separators=(",", ":")),
            )
            with self._lock:
                book = self._books.setdefault(token_id, BookState(token_id))
                book.apply_price_change(
                    _f(change["price"]), _f(change["size"]), str(change.get("side", "")),
                    book_hash=change.get("hash"), source_ts_ns=src_ns,
                    recv_ts_ns=ev.recv_wall_timestamp_ns,
                )
            self._emit(ev)
            out.append(ev)
        return out

    def _handle_tick_size_change(self, m: dict) -> RawEvent:
        token_id = m.get("asset_id")
        src_ns = ms_to_ns(m.get("timestamp"))
        ev = self._build(Topic.CLOB_MARKET, "tick_size_change", m, token_id=token_id,
                         source_ts_ns=src_ns, raw_json=json.dumps(m, separators=(",", ":")))
        new_tick = m.get("new_tick_size") or m.get("tick_size")
        with self._lock:
            book = self._books.setdefault(token_id, BookState(token_id))
            if new_tick is not None:
                book.tick_size = _f(new_tick)
        self._emit(ev)
        return ev

    def _handle_simple(self, m: dict, event_type: str) -> RawEvent:
        token_id = m.get("asset_id")
        src_ns = ms_to_ns(m.get("timestamp"))
        ev = self._build(Topic.CLOB_MARKET, event_type, m, token_id=token_id,
                         source_ts_ns=src_ns, raw_json=json.dumps(m, separators=(",", ":")))
        if event_type == "last_trade_price" and token_id and m.get("price") is not None:
            with self._lock:
                self._books.setdefault(token_id, BookState(token_id)).last_trade_price = _f(m["price"])
        self._emit(ev)
        return ev

    def _handle_market_resolved(self, m: dict) -> RawEvent:
        condition_id = m.get("market") or m.get("condition_id")
        ev = self._build(Topic.CLOB_MARKET, "market_resolved", m,
                         source_ts_ns=ms_to_ns(m.get("timestamp")),
                         raw_json=json.dumps(m, separators=(",", ":")))
        if condition_id:
            with self._lock:
                self._resolved[str(condition_id)] = m
        self._emit(ev)
        return ev

    # ------------------------------------------------------------ reading

    def book(self, token_id: str) -> BookState | None:
        with self._lock:
            book = self._books.get(token_id)
            if book is None:
                return None
            # Return a shallow copy so a caller iterating the ladder cannot
            # race the reader thread mutating it.
            copy = BookState(
                token_id=book.token_id, bids=dict(book.bids), asks=dict(book.asks),
                tick_size=book.tick_size, book_hash=book.book_hash,
                source_ts_ns=book.source_ts_ns, recv_ts_ns=book.recv_ts_ns,
                last_trade_price=book.last_trade_price,
                awaiting_snapshot=book.awaiting_snapshot, has_gap=book.has_gap,
                applied_updates=book.applied_updates,
            )
        return copy

    def resolution_for(self, condition_id: str) -> dict | None:
        with self._lock:
            return self._resolved.get(condition_id)

    @property
    def reconnect_generation(self) -> int:
        return self._builder.reconnect_generation
