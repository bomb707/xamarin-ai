"""Real Polymarket CLOB adapters: market metadata + order book (REST + WSS).

Endpoints and message shapes below were pulled directly from
docs.polymarket.com on 2026-08-13 (the same pages cited as [P2][P4][P5] in
the strategy doc's references) rather than guessed:

  - Market metadata (REST):  GET https://gamma-api.polymarket.com/markets/{id}
      https://docs.polymarket.com/api-reference/markets/get-market-by-id
  - Book snapshot (REST):    GET https://clob.polymarket.com/book?token_id=...
      https://docs.polymarket.com/trading/orderbook
  - Market channel (WSS):    wss://ws-subscriptions-clob.polymarket.com/ws/market
      https://docs.polymarket.com/market-data/websocket/market-channel

CAVEAT - confirm against a live response before trusting in production:
  - `secondsDelay` is the best-match field name for the taker-delay flag
    found in the market-by-id docs excerpt, but was described only as "a
    potential delay-related parameter" - cross-check it actually reflects
    the 250ms crypto/finance up-down delay described in the order-lifecycle
    doc [P3] before wiring it into risk logic.
  - `clobTokenIds` needs to be matched against the market's outcome
    labels to know which token is UP vs DOWN; the docs excerpt did not
    enumerate the exact outcomes field name, so `_map_tokens_to_sides`
    below raises rather than guessing a token order.
  - `makerBaseFee` / `takerBaseFee` / `feeSchedule` units were not fully
    specified (integer - likely bps); do not trust these over the
    documented fee formula's `feeRate=0.07` fallback until verified.

This module is one of the two Phase-1 pieces the user asked to implement
for real, as opposed to interface + mock-only (Chainlink TWAP and the BTC
spot composite, whose endpoints are less standardized, stay as
endpoint-required skeletons in chainlink_twap.py / spot_composite.py).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from xamarinbot.feeds._live_deps import require_live_deps
from xamarinbot.feeds.base import (
    BookFeed,
    BookLevel,
    BookSnapshot,
    MarketConfig,
    MarketConfigProvider,
)
from xamarinbot.portfolio.state import Side

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_REST_BASE_URL = "https://clob.polymarket.com"
CLOB_MARKET_WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_REQUIRED_MARKET_FIELDS = (
    "clobTokenIds",
    "orderPriceMinTickSize",
    "orderMinSize",
)


class MarketConfigFieldError(RuntimeError):
    """Raised when the Gamma market payload is missing a field this adapter
    depends on, instead of silently defaulting a financial parameter."""


class PolymarketMarketConfigProvider(MarketConfigProvider):
    def __init__(
        self,
        up_token_field: str = "up_token_id",
        down_token_field: str = "down_token_id",
        gamma_base_url: str = GAMMA_BASE_URL,
        clob_ws_url: str = CLOB_MARKET_WSS_URL,
        request_timeout_s: float = 5.0,
    ):
        require_live_deps()
        self._gamma_base_url = gamma_base_url
        self._clob_ws_url = clob_ws_url
        self._timeout = request_timeout_s

    def get_market_config(self, market_id: str) -> MarketConfig:
        import httpx

        resp = httpx.get(f"{self._gamma_base_url}/markets/{market_id}", timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()

        missing = [f for f in _REQUIRED_MARKET_FIELDS if f not in payload]
        if missing:
            raise MarketConfigFieldError(
                f"market {market_id} payload missing required fields {missing}; "
                "Gamma API schema may have changed - do not fall back silently."
            )

        up_token_id, down_token_id = self._map_tokens_to_sides(payload)

        return MarketConfig(
            market_id=market_id,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            start_ts=float(payload.get("gameStartTime") or payload.get("startDate") or 0.0),
            end_ts=float(payload.get("endDate") or 0.0),
            tick_size=float(payload["orderPriceMinTickSize"]),
            min_order_size=float(payload["orderMinSize"]),
            fee_rate=float(payload.get("takerBaseFee") or 0.07),
            taker_delay_ms=float(payload.get("secondsDelay") or 0) * 1000.0,
            twap_window_seconds=int(payload.get("twapWindowSeconds") or 30),
        )

    @staticmethod
    def _map_tokens_to_sides(payload: dict) -> tuple[str, str]:
        """Deliberately does not guess which clobTokenIds entry is UP vs
        DOWN - the docs excerpt available at write time didn't confirm the
        outcomes field name/order. Wire the correct mapping once verified
        against a live market payload (compare against `outcomes`)."""
        raise NotImplementedError(
            "UP/DOWN token mapping must be confirmed against a live "
            "Gamma market payload's outcomes field before use; see module "
            "docstring caveats."
        )

    def subscribe_tick_size_changes(self, market_id: str, callback: Callable[[float], None]) -> None:
        # tick_size_change arrives on the same market WSS channel as book
        # deltas; PolymarketBookFeed.subscribe_deltas is the single
        # connection owner. Expose this as a documented no-op here so
        # callers wire tick-size-change handling through that feed instead
        # of opening a second connection per the same asset.
        raise NotImplementedError(
            "Subscribe via PolymarketBookFeed's WSS connection (event_type "
            "'tick_size_change') rather than opening a second socket here."
        )


@dataclass
class _LiveBookState:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    ts: float = 0.0
    book_hash: str | None = None


class PolymarketBookFeed(BookFeed):
    """REST snapshot + WSS market-channel deltas, per [P4][P5]."""

    def __init__(
        self,
        token_id_for_side: dict[Side, str],
        clob_rest_base_url: str = CLOB_REST_BASE_URL,
        clob_ws_url: str = CLOB_MARKET_WSS_URL,
        request_timeout_s: float = 5.0,
        heartbeat_interval_s: float = 10.0,
    ):
        require_live_deps()
        self._token_id_for_side = token_id_for_side
        self._rest_base = clob_rest_base_url
        self._ws_url = clob_ws_url
        self._timeout = request_timeout_s
        self._heartbeat_interval = heartbeat_interval_s
        self._state: dict[str, _LiveBookState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def get_snapshot(self, round_id: str, side: Side) -> BookSnapshot | None:
        return self.resnapshot(round_id, side)

    def resnapshot(self, round_id: str, side: Side) -> BookSnapshot | None:
        """Safety resnapshot via REST, per Roadmap Phase 1: "periodic/safety
        resnapshot"."""
        import httpx

        token_id = self._token_id_for_side[side]
        resp = httpx.get(
            f"{self._rest_base}/book", params={"token_id": token_id}, timeout=self._timeout
        )
        resp.raise_for_status()
        payload = resp.json()
        recv_ts = time.time()
        bids = tuple(BookLevel(float(b["price"]), float(b["size"])) for b in payload.get("bids", []))
        asks = tuple(BookLevel(float(a["price"]), float(a["size"])) for a in payload.get("asks", []))
        snap = BookSnapshot(
            side=side,
            bids=bids,
            asks=asks,
            ts=float(payload["timestamp"]) / 1000.0,
            recv_ts=recv_ts,
            book_hash=payload.get("hash"),
        )
        with self._lock:
            self._state[token_id] = _LiveBookState(
                bids={lvl.price: lvl.size for lvl in bids},
                asks={lvl.price: lvl.size for lvl in asks},
                ts=snap.ts,
                book_hash=snap.book_hash,
            )
        return snap

    def subscribe_deltas(
        self, round_id: str, side: Side, callback: Callable[[BookSnapshot], None]
    ) -> None:
        if self._thread is not None:
            return  # single connection serves both sides already
        self._callback = callback
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="polymarket-market-ws", daemon=True)
        self._thread.start()

    def reconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self.subscribe_deltas("", Side.UP, getattr(self, "_callback", lambda snap: None))

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
                with connect(self._ws_url, open_timeout=self._timeout) as ws:
                    ws.send(
                        json.dumps(
                            {"assets_ids": list(self._token_id_for_side.values()), "type": "market"}
                        )
                    )
                    backoff = 1.0
                    last_ping = time.time()
                    ws.settimeout = getattr(ws, "settimeout", None)
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

        event_type = msg.get("event_type")
        token_id = msg.get("asset_id")
        side = self._side_for_token(token_id)
        if side is None:
            return

        with self._lock:
            state = self._state.setdefault(token_id, _LiveBookState())
            if event_type == "book":
                state.bids = {float(b["price"]): float(b["size"]) for b in msg.get("bids", [])}
                state.asks = {float(a["price"]): float(a["size"]) for a in msg.get("asks", [])}
                state.book_hash = msg.get("hash")
                state.ts = float(msg["timestamp"]) / 1000.0
            elif event_type == "price_change":
                for change in msg.get("price_changes", []):
                    book = state.bids if change.get("side") == "BUY" else state.asks
                    price, size = float(change["price"]), float(change["size"])
                    if size == 0:
                        book.pop(price, None)
                    else:
                        book[price] = size
                    state.book_hash = change.get("hash", state.book_hash)
                state.ts = float(msg["timestamp"]) / 1000.0
            else:
                return  # tick_size_change etc. handled elsewhere

            snap = self._snapshot_from_state(side, state)

        self._callback(snap)

    def _side_for_token(self, token_id: str | None) -> Side | None:
        for side, tid in self._token_id_for_side.items():
            if tid == token_id:
                return side
        return None

    @staticmethod
    def _snapshot_from_state(side: Side, state: _LiveBookState) -> BookSnapshot:
        bids = tuple(BookLevel(p, s) for p, s in sorted(state.bids.items(), key=lambda kv: -kv[0]))
        asks = tuple(BookLevel(p, s) for p, s in sorted(state.asks.items(), key=lambda kv: kv[0]))
        return BookSnapshot(
            side=side, bids=bids, asks=asks, ts=state.ts, recv_ts=time.time(), book_hash=state.book_hash
        )
