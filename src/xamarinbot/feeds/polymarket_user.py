"""Real Polymarket authenticated user/order WSS adapter [P3].

Endpoint and message shapes pulled from docs.polymarket.com on 2026-08-13
(https://docs.polymarket.com/market-data/websocket/user-channel):

  WSS: wss://ws-subscriptions-clob.polymarket.com/ws/user
  Subscribe: {"auth": {"apiKey","secret","passphrase"}, "type": "user",
              "markets": [condition_id, ...]}
  order message: event_type=order, status in
              LIVE|MATCHED|DELAYED|UNMATCHED|CANCELED
  trade message: event_type=trade, status in
              MATCHED|MATCHED_NOT_BROADCASTED|MINED|CONFIRMED|RETRYING|FAILED,
              trader_side in TAKER|MAKER

CAVEAT: the REST endpoint/response schema for `open_orders()`/`reconcile()`
(listing currently-open orders on demand, needed right after a fresh
subscribe or after a reconnect gap) was not confirmed against current docs
at write time - only the WSS message shapes were. Confirm the REST path
(likely under https://clob.polymarket.com) against docs.polymarket.com
before relying on `open_orders()`/`reconcile()` in production; the WSS
handling below is the verified part of this adapter.

Never expose apiKey/secret/passphrase client-side; load them from a server
environment (env vars / secrets manager), never hardcode them.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable

from xamarinbot.feeds._live_deps import require_live_deps
from xamarinbot.feeds.base import UserOrderEvent, UserOrderFeed

USER_WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


@dataclass(frozen=True)
class PolymarketApiCreds:
    api_key: str
    secret: str
    passphrase: str


class PolymarketUserOrderFeed(UserOrderFeed):
    def __init__(
        self,
        creds: PolymarketApiCreds,
        market_ids: list[str] | None = None,
        ws_url: str = USER_WSS_URL,
        heartbeat_interval_s: float = 10.0,
    ):
        require_live_deps()
        self._creds = creds
        self._market_ids = market_ids or []
        self._ws_url = ws_url
        self._heartbeat_interval = heartbeat_interval_s
        self._orders: dict[str, UserOrderEvent] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback: Callable[[UserOrderEvent], None] = lambda e: None

    def open_orders(self) -> list[UserOrderEvent]:
        with self._lock:
            return [o for o in self._orders.values() if o.status in ("LIVE", "DELAYED")]

    def reconcile(self) -> list[UserOrderEvent]:
        with self._lock:
            return list(self._orders.values())

    def subscribe(self, callback: Callable[[UserOrderEvent], None]) -> None:
        self._callback = callback
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="polymarket-user-ws", daemon=True)
        self._thread.start()

    def reconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self.subscribe(self._callback)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run_forever(self) -> None:
        from websockets.sync.client import connect

        backoff = 1.0
        while not self._stop.is_set():
            try:
                with connect(self._ws_url, open_timeout=5.0) as ws:
                    ws.send(
                        json.dumps(
                            {
                                "auth": {
                                    "apiKey": self._creds.api_key,
                                    "secret": self._creds.secret,
                                    "passphrase": self._creds.passphrase,
                                },
                                "type": "user",
                                "markets": self._market_ids,
                            }
                        )
                    )
                    backoff = 1.0
                    last_ping = time.time()
                    while not self._stop.is_set():
                        try:
                            raw = ws.recv(timeout=self._heartbeat_interval)
                        except TimeoutError:
                            raw = None
                        if time.time() - last_ping >= self._heartbeat_interval:
                            ws.send("PING")
                            last_ping = time.time()
                        if raw is None:
                            continue
                        self._handle_message(raw)
            except Exception:
                if self._stop.is_set():
                    return
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_message(self, raw: str) -> None:
        if raw == "PONG":
            return
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        if msg.get("event_type") != "order":
            return  # "trade" messages carry fills; order status is the
            # source of truth for filled_size/remaining_size used here.

        order_id = msg["id"]
        original = float(msg.get("original_size", 0))
        matched = float(msg.get("size_matched", 0))
        event = UserOrderEvent(
            order_id=order_id,
            status=msg["status"],
            filled_size=matched,
            remaining_size=max(0.0, original - matched),
            ts=float(msg["timestamp"]) / 1000.0,
            recv_ts=time.time(),
        )
        with self._lock:
            self._orders[order_id] = event
        self._callback(event)
