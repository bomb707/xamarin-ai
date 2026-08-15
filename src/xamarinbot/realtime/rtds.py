"""Shared Polymarket RTDS connection for BTC reference signals (item 4).

One connection serves all four streams the brief names:

    Chainlink BTC/USD reference price   topic crypto_prices_chainlink
    Binance BTCUSDT price               topic crypto_prices
    Chainlink BTC/USD 30-second TWAP    topic crypto_prices_twap_thirty
    Chainlink BTC/USD 60-second TWAP    topic crypto_prices_twap_sixty

LIVE VERIFICATION, 2026-08-15 - two findings that change the implementation
away from what the documentation implies:

1. **The dotted topic aliases do not exist on the live socket.** Subscribing
   to `prices.crypto.chainlink.twap` (with `{"symbol":"btc/usd","window":30}`)
   or to `prices.crypto.chainlink` was answered with
   `{"message": "Invalid request body", ...}` and the connection delivered
   nothing. Only the raw topic names above are accepted. The brief allowed
   for this - "or the corresponding raw RTDS topics" - and this module uses
   the raw names exclusively.

2. **`filters` suppresses live delivery.** A subscription carrying
   `"filters": "{\\"symbol\\":\\"btc/usd\\"}"` (the documented form) returned
   exactly ONE message - a `type: "subscribe"` backfill frame whose payload
   was a `data` array of ~56 one-second historical points - and then zero
   updates for the remainder of a 45-second window. The same subscription
   WITHOUT `filters` delivered 48-49 btc updates per topic in 50 seconds,
   about 1 Hz, which matches the publication rate. This adapter therefore
   subscribes UNFILTERED and filters by `payload.symbol` client-side. The
   cost is receiving other assets' ticks (roughly 8x the volume, still only
   ~30 messages/second total) and discarding them; the benefit is actually
   receiving BTC data at all.

   The backfill frame is still parsed and recorded when one arrives, but it
   is explicitly NOT relied on for pre-round history - item 7's requirement
   is met by starting the recorder before the round opens (see
   `lifecycle.py`), not by a subscribe-time dump that costs the live stream.

Observed payload shape (Chainlink TWAP-60, btc/usd):

    {"connection_id": "...",
     "payload": {"full_accuracy_value": "63086406212239293939712",
                 "symbol": "btc/usd",
                 "timestamp": 1786771253000,     <- Chainlink observation ms
                 "value": 63086.40621223929,
                 "window_s": 60},
     "timestamp": 1786771254918,                 <- RTDS publisher ms
     "topic": "crypto_prices_twap_sixty",
     "type": "update"}

The two `timestamp` fields are genuinely different clocks and are preserved
separately, together with local wall and monotonic receive times, per item
4's "Do not replace Chainlink observation time with local receive time."
Measured on the sample above, the observation-to-publish gap was ~1.9s while
publish-to-receive was well under 100ms - so collapsing them would have
attributed nearly two seconds of oracle latency to our own network.

Binance values arrive on the same socket with `symbol: "btcusdt"` and a
plain decimal `full_accuracy_value`; Chainlink values use an E18
fixed-point integer string. Both are preserved verbatim; `value` is the
float the venue itself computed.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable

from xamarinbot.realtime.attribution import Stream, StreamGapTracker
from xamarinbot.realtime.raw_events import RawEvent, RawEventBuilder, Topic, ms_to_ns

RTDS_WS_URL = "wss://ws-live-data.polymarket.com"

TOPIC_BINANCE = "crypto_prices"
TOPIC_CHAINLINK = "crypto_prices_chainlink"
TOPIC_TWAP_30 = "crypto_prices_twap_thirty"
TOPIC_TWAP_60 = "crypto_prices_twap_sixty"

#: Wire topic -> our raw-event topic.
_TOPIC_MAP = {
    TOPIC_BINANCE: Topic.RTDS_BINANCE,
    TOPIC_CHAINLINK: Topic.RTDS_CHAINLINK,
    TOPIC_TWAP_30: Topic.RTDS_TWAP_30,
    TOPIC_TWAP_60: Topic.RTDS_TWAP_60,
}

#: Symbol each topic uses for BTC. Binance uses exchange pair notation,
#: Chainlink uses the slash-delimited feed name; they are not interchangeable.
BTC_SYMBOLS = {
    TOPIC_BINANCE: "btcusdt",
    TOPIC_CHAINLINK: "btc/usd",
    TOPIC_TWAP_30: "btc/usd",
    TOPIC_TWAP_60: "btc/usd",
}

_E18 = 10 ** 18


def decode_full_accuracy(topic: str, value: str | None) -> float | None:
    """`full_accuracy_value` is E18 fixed-point for the Chainlink topics
    (`"63086406212239293939712"`) and a plain decimal for Binance
    (`"63151.96000000"`). Returns None rather than a wrong-by-1e18 number
    when the value is missing."""
    if value is None or value == "":
        return None
    try:
        if topic == TOPIC_BINANCE:
            return float(value)
        return int(value) / _E18
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ReferenceObservation:
    """One BTC reference observation, with all four clocks kept apart."""

    topic: str
    symbol: str
    value: float
    full_accuracy_value: float | None
    window_s: int | None
    #: Chainlink/Binance observation time (ns). The ONLY timestamp
    #: freshness may be judged against.
    source_ts_ns: int
    #: RTDS publisher time (ns).
    publisher_ts_ns: int | None
    #: Local wall clock at receive (ns).
    recv_wall_ns: int
    #: Local monotonic clock at receive (ns) - not an epoch.
    recv_monotonic_ns: int

    @property
    def source_ts(self) -> float:
        return self.source_ts_ns / 1e9


class RTDSClient:
    """One shared RTDS connection.

    Maintains the latest observation per topic for BTC in memory (for
    freshness and feature computation) while handing every raw frame to the
    recorder.
    """

    def __init__(
        self,
        *,
        builder: RawEventBuilder,
        on_raw_event: Callable[[RawEvent], None],
        symbols: dict[str, str] | None = None,
        topics: list[str] | None = None,
        ws_url: str = RTDS_WS_URL,
        ping_interval_s: float = 5.0,
        open_timeout_s: float = 15.0,
        stall_timeout_s: float = 30.0,
        on_parse_failure: Callable[[str, Exception], None] | None = None,
        on_data_gap: Callable[[object], None] | None = None,
        on_reconnect: Callable[[int], None] | None = None,
        on_observation: Callable[[ReferenceObservation], None] | None = None,
    ):
        self._builder = builder
        self._emit = on_raw_event
        self._symbols = dict(symbols or BTC_SYMBOLS)
        self._topics = list(topics or [TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30, TOPIC_TWAP_60])
        self._ws_url = ws_url
        self._ping_interval = ping_interval_s
        self._open_timeout = open_timeout_s
        # STALL WATCHDOG. Found by a real capture on 2026-08-15: the RTDS
        # socket stopped delivering after ~688 seconds and NEVER RAISED -
        # `recv` simply kept timing out on an apparently-healthy connection.
        # The CLOB stream on the same process ran to completion (1084s), so
        # this was specific to RTDS. Without a watchdog the recorder silently
        # captured zero reference data for the last two of three rounds while
        # reporting zero reconnects and zero errors, which is the worst
        # possible failure mode: a clean-looking capture with no settlement
        # reference in it.
        #
        # 30s is chosen against the measured ~1 Hz publication rate: half a
        # minute of total silence across FOUR independent symbol streams is
        # not a quiet market, it is a dead socket.
        self._stall_timeout = stall_timeout_s
        self._on_parse_failure = on_parse_failure or (lambda raw, exc: None)
        self._on_reconnect = on_reconnect or (lambda gen: None)
        self._on_observation = on_observation or (lambda obs: None)
        #: Gate A.0.2 item 1: RTDS outages as INTERVALS. A stall opens a gap
        #: at the last observation actually received; the first valid BTC
        #: observation after the reconnect closes it.
        self.gaps = StreamGapTracker(Stream.RTDS, on_gap=on_data_gap)

        self._latest: dict[str, ReferenceObservation] = {}
        self._history: dict[str, list[ReferenceObservation]] = {t: [] for t in self._topics}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: Bound on retained per-topic history. At ~1 Hz this is over an
        #: hour, comfortably more than item 7's "largest feature window plus
        #: a safety margin" while staying trivially small in memory.
        self.history_limit = 4096
        #: The round this client is currently attributing observations to.
        #: Reference feeds are global, not per-market, so the recorder tells
        #: the client which round's file they belong to.
        self.current_round_id: str | None = None

    # ------------------------------------------------------------ control

    def subscribe_message(self) -> str:
        """UNFILTERED subscriptions - see module docstring finding 2."""
        return json.dumps({
            "action": "subscribe",
            "subscriptions": [{"topic": t, "type": "update"} for t in self._topics],
        })

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="rtds-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run_forever(self) -> None:
        from websockets.sync.client import connect

        backoff = 1.0
        first = True
        while not self._stop.is_set():
            try:
                with connect(self._ws_url, open_timeout=self._open_timeout, max_size=None) as ws:
                    if not first:
                        self._builder.reconnect_generation += 1
                        self._on_reconnect(self._builder.reconnect_generation)
                        self._emit(self._builder.build(
                            Topic.RECORDER_CONTROL, "reconnect",
                            {"generation": self._builder.reconnect_generation, "stream": "rtds"},
                            round_id=self.current_round_id,
                        ))
                    first = False
                    ws.send(self.subscribe_message())
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
                            # PONG counts as liveness for the socket, but not
                            # as DATA - a server that answers pings while
                            # having silently dropped our subscriptions is
                            # exactly the failure this watchdog exists for.
                            if raw not in ("PONG", "pong"):
                                last_data = time.monotonic()
                            self.handle_message(raw)
                        if time.monotonic() - last_data > self._stall_timeout:
                            silent_for = time.monotonic() - last_data
                            # Gate A.0.2 item 1: open a real INTERVAL. The
                            # outage began at the last observation actually
                            # received, which is `silent_for` seconds ago -
                            # not now, when the watchdog happened to notice.
                            self.gaps.begin("stream_stalled")
                            self._emit(self._builder.build(
                                Topic.RECORDER_CONTROL, "stream_stalled",
                                {
                                    "stream": "rtds",
                                    "silent_for_s": silent_for,
                                    "stall_timeout_s": self._stall_timeout,
                                },
                                round_id=self.current_round_id,
                            ))
                            # Break out to the reconnect path rather than
                            # sitting on a dead socket. The gap stays OPEN
                            # and is closed by the first valid BTC
                            # observation on the new connection.
                            break
            except Exception as exc:
                if self._stop.is_set():
                    return
                self._on_parse_failure("rtds_connection", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    # ------------------------------------------------------------ routing

    def handle_message(self, raw: str) -> list[RawEvent]:
        """Public so tests can replay captured frames without a socket."""
        if raw in ("PONG", "PING", ""):
            return []
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            self._on_parse_failure(raw, exc)
            return []
        out: list[RawEvent] = []
        for m in (msg if isinstance(msg, list) else [msg]):
            if not isinstance(m, dict):
                continue
            if "message" in m and "topic" not in m:
                # e.g. {"message": "Invalid request body"} - a control-plane
                # rejection. Recorded so a silently-empty capture is
                # explainable after the fact.
                self._emit(self._builder.build(
                    Topic.RECORDER_CONTROL, "rtds_error", m, round_id=self.current_round_id,
                ))
                self._on_parse_failure(json.dumps(m), RuntimeError(str(m.get("message"))))
                continue
            ev = self._route(m)
            if ev is not None:
                out.append(ev)
        return out

    def _route(self, m: dict) -> RawEvent | None:
        wire_topic = m.get("topic")
        topic = _TOPIC_MAP.get(wire_topic)
        if topic is None:
            return None
        payload = m.get("payload") or {}

        # Subscribe-time backfill frame: payload is {"symbol", "data": [...]}.
        if isinstance(payload, dict) and "data" in payload:
            symbol = payload.get("symbol")
            if symbol is not None and not self._is_wanted(wire_topic, symbol):
                return None
            return self._emit_and_return(self._builder.build(
                topic, "backfill", m, round_id=self.current_round_id,
                publisher_timestamp_ns=ms_to_ns(m.get("timestamp")),
                raw_json=json.dumps(m, separators=(",", ":")),
            ))

        symbol = payload.get("symbol")
        if not self._is_wanted(wire_topic, symbol):
            # Client-side filtering: unfiltered subscription is the only one
            # that delivers, so other assets' ticks arrive and are dropped
            # here. They are NOT recorded - the capture is about BTC, and
            # storing every asset would multiply the log ~8x for no use.
            return None

        source_ns = ms_to_ns(payload.get("timestamp"))
        if source_ns is None:
            self._on_parse_failure(json.dumps(m), ValueError("reference update without source timestamp"))
            return None
        publisher_ns = ms_to_ns(m.get("timestamp"))
        ev = self._builder.build(
            topic, str(m.get("type") or "update"), m,
            round_id=self.current_round_id,
            source_timestamp_ns=source_ns,
            publisher_timestamp_ns=publisher_ns,
            raw_json=json.dumps(m, separators=(",", ":")),
        )
        try:
            value = float(payload["value"])
        except (KeyError, TypeError, ValueError) as exc:
            self._on_parse_failure(json.dumps(m), exc)
            self._emit(ev)
            return ev

        # Gate A.0.2 item 1: THIS is what ends an RTDS outage - a parsed,
        # wanted BTC observation with a usable value. Not the reconnect
        # (which only proves a socket opened), and not a PONG. Any open gap
        # is closed at this observation's own receive timestamp.
        self.gaps.note_data(ev.recv_wall_timestamp_ns)

        window = payload.get("window_s")
        obs = ReferenceObservation(
            topic=wire_topic,
            symbol=str(symbol),
            value=value,
            full_accuracy_value=decode_full_accuracy(wire_topic, payload.get("full_accuracy_value")),
            window_s=int(window) if window is not None else None,
            source_ts_ns=source_ns,
            publisher_ts_ns=publisher_ns,
            recv_wall_ns=ev.recv_wall_timestamp_ns,
            recv_monotonic_ns=ev.recv_monotonic_ns,
        )
        with self._lock:
            prev = self._latest.get(wire_topic)
            # Out-of-order arrival is possible; never let a stale
            # observation overwrite a newer one as "latest".
            if prev is None or obs.source_ts_ns >= prev.source_ts_ns:
                self._latest[wire_topic] = obs
            hist = self._history.setdefault(wire_topic, [])
            hist.append(obs)
            if len(hist) > self.history_limit:
                del hist[: len(hist) - self.history_limit]
        self._emit(ev)
        self._on_observation(obs)
        return ev

    def _emit_and_return(self, ev: RawEvent) -> RawEvent:
        self._emit(ev)
        return ev

    def _is_wanted(self, wire_topic: str, symbol) -> bool:
        want = self._symbols.get(wire_topic)
        return want is not None and str(symbol).lower() == want

    # ------------------------------------------------------------ reading

    def latest(self, wire_topic: str) -> ReferenceObservation | None:
        with self._lock:
            return self._latest.get(wire_topic)

    def history(self, wire_topic: str) -> list[ReferenceObservation]:
        with self._lock:
            return list(self._history.get(wire_topic, ()))

    def observation_at_or_before(self, wire_topic: str, ts_ns: int) -> ReferenceObservation | None:
        """The most recent observation whose SOURCE timestamp is at or
        before `ts_ns`. Used for settlement-label reconstruction, where the
        question is precisely "what had the oracle observed by the round
        boundary" - which local receive time cannot answer."""
        best: ReferenceObservation | None = None
        with self._lock:
            for obs in self._history.get(wire_topic, ()):
                if obs.source_ts_ns <= ts_ns and (best is None or obs.source_ts_ns > best.source_ts_ns):
                    best = obs
        return best

    def observation_at_or_after(self, wire_topic: str, ts_ns: int) -> ReferenceObservation | None:
        best: ReferenceObservation | None = None
        with self._lock:
            for obs in self._history.get(wire_topic, ()):
                if obs.source_ts_ns >= ts_ns and (best is None or obs.source_ts_ns < best.source_ts_ns):
                    best = obs
        return best
