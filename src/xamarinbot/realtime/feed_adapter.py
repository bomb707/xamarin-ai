"""THE canonical real-market Phase-1 feed adapter.

Phase 12C.1 item 7: there must not be two apparent production sources for
the same market datum. This module was `feeds/polymarket_clob.py`; it now
lives under `realtime/` because `realtime/*` is the real-market
single-source-of-truth. Its two superseded siblings were deleted outright
rather than left as a second apparent source:

  * `feeds/chainlink_twap.py` - direct Chainlink Data Streams client.
    Superseded by `realtime/rtds.py`, which is what Polymarket's own
    documentation recommends for production Chainlink TWAP and which needs
    no credentials. The deleted module's auth/subscription handshake was
    never verified against a live endpoint.
  * `feeds/spot_composite.py` - Coinbase/Binance REST polling composite.
    Superseded by the RTDS Binance stream, which arrives on the same socket
    as the reference feeds and carries a provider timestamp.

Both had zero callers. `feeds/polymarket_user.py` is NOT superseded (there
is no `realtime/` equivalent of the authenticated user/order stream) and is
retained, deprecated, for Phase 13.

This module used to contain its own REST+WSS implementation. That
implementation had the defects Phase 12C item 3 enumerates - a
NotImplementedError token mapping, `float()` applied to ISO-8601 date
strings, a REST call on every `get_snapshot()`, and a `price_change` router
that read a top-level `asset_id` that live frames do not have, silently
discarding every book delta. Rather than patch a second copy of the logic,
the real behavior now lives in `xamarinbot.realtime` (discovery.py,
clob_ws.py), and this module is the thin `MarketConfigProvider` / `BookFeed`
adapter that exposes it through the Phase-1 interfaces the controller,
`ShadowRunner` and the replay feeds all share.

Concretely:

  `PolymarketMarketConfigProvider.get_market_config`
      delegates to `realtime.discovery`, so ISO timestamps parse, the round
      window comes from `eventStartTime`/`endDate` rather than the
      row-creation `startDate`, UP/DOWN comes from explicit outcome labels,
      and tick size / min order size / fees / taker delay come from CLOB
      market-info.

  `PolymarketBookFeed`
      wraps `realtime.clob_ws.PolymarketMarketStream`. `get_snapshot` reads
      the in-memory book maintained from the WebSocket - no REST per call -
      while `resnapshot` is the explicit REST resync.

  `subscribe_tick_size_changes`
      is now genuinely implemented on the single shared connection instead
      of raising and telling the caller to open a second socket.
"""
from __future__ import annotations

import threading
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
from xamarinbot.realtime.clob_ws import (
    CLOB_MARKET_WSS_URL,
    CLOB_REST_BASE_URL,
    PolymarketMarketStream,
)
from xamarinbot.realtime.discovery import (
    GAMMA_BASE_URL,
    MarketDiscovery,
    MarketDiscoveryError,
    RealMarketMetadata,
)
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic

__all__ = [
    "GAMMA_BASE_URL",
    "CLOB_REST_BASE_URL",
    "CLOB_MARKET_WSS_URL",
    "MarketConfigFieldError",
    "PolymarketMarketConfigProvider",
    "PolymarketBookFeed",
]


class MarketConfigFieldError(RuntimeError):
    """Raised when the market payload is missing a field this adapter
    depends on, instead of silently defaulting a financial parameter.

    Kept as a distinct name for existing callers; `MarketDiscoveryError` is
    the underlying cause and is chained.
    """


def market_config_from_metadata(meta: RealMarketMetadata) -> MarketConfig:
    """Project the full Phase 12C metadata onto the Phase-1 `MarketConfig`.

    `twap_window_seconds` is the market's DECLARED window when it declares
    one. It is not defaulted to 30 - the previous code's
    `int(payload.get("twapWindowSeconds") or 30)` invented a window for
    every market, which is exactly the "never assume every BTC 5-minute
    market uses the same window" failure the strategy doc warns about. Every
    live BTC 5-minute market observed on 2026-08-15 declared 60.
    """
    if meta.up_token_id is None or meta.down_token_id is None:
        raise MarketConfigFieldError(
            f"{meta.slug}: UP/DOWN token ids are not both known "
            f"(source={meta.outcome_label_source}); refusing to build a MarketConfig"
        )
    if meta.twap_window_s is None:
        raise MarketConfigFieldError(
            f"{meta.slug}: market declares no TWAP window (settlement_kind="
            f"{meta.settlement_kind}); refusing to assume one"
        )
    return MarketConfig(
        market_id=meta.market_id,
        up_token_id=meta.up_token_id,
        down_token_id=meta.down_token_id,
        start_ts=meta.start_ts,
        end_ts=meta.end_ts,
        tick_size=meta.tick_size,
        min_order_size=meta.min_order_size,
        fee_rate=meta.fees.effective_rate,
        taker_delay_ms=meta.taker_delay_ms,
        twap_window_seconds=meta.twap_window_s,
    )


class PolymarketMarketConfigProvider(MarketConfigProvider):
    """Market metadata from Gamma + CLOB market-info, via `realtime.discovery`."""

    def __init__(
        self,
        gamma_base_url: str = GAMMA_BASE_URL,
        clob_base_url: str = CLOB_REST_BASE_URL,
        request_timeout_s: float = 10.0,
        discovery: MarketDiscovery | None = None,
    ):
        require_live_deps()
        self._discovery = discovery or MarketDiscovery(
            gamma_base_url=gamma_base_url,
            clob_base_url=clob_base_url,
            request_timeout_s=request_timeout_s,
        )
        self._tick_callbacks: dict[str, list[Callable[[float], None]]] = {}
        self._book_feed: PolymarketBookFeed | None = None

    def get_metadata(self, market_slug: str) -> RealMarketMetadata:
        """Full Phase 12C metadata (everything item 2 requires persisted),
        of which `MarketConfig` is a lossy projection."""
        gamma = self._discovery.fetch_gamma_market(market_slug)
        if gamma is None:
            raise MarketConfigFieldError(f"no Gamma market for slug {market_slug}")
        condition_id = gamma.get("conditionId")
        if not condition_id:
            raise MarketConfigFieldError(f"{market_slug}: Gamma payload carries no conditionId")
        clob = self._discovery.fetch_clob_market(str(condition_id))
        from xamarinbot.realtime.discovery import build_metadata

        try:
            return build_metadata(gamma, clob)
        except MarketDiscoveryError as exc:
            raise MarketConfigFieldError(str(exc)) from exc

    def get_market_config(self, market_id: str) -> MarketConfig:
        """`market_id` is the market SLUG (e.g. `btc-updown-5m-1786772100`),
        which is what identifies a round in this series."""
        return market_config_from_metadata(self.get_metadata(market_id))

    def bind_book_feed(self, book_feed: "PolymarketBookFeed") -> None:
        """Register the feed that owns the single market-channel
        connection, so tick-size changes can be delivered from it."""
        self._book_feed = book_feed
        book_feed.add_tick_size_listener(self._dispatch_tick_size)

    def subscribe_tick_size_changes(self, market_id: str, callback: Callable[[float], None]) -> None:
        """`tick_size_change` arrives on the same market channel as book
        deltas, so this registers a listener on the shared connection rather
        than opening a second socket (which is what the old implementation
        raised NotImplementedError to avoid)."""
        self._tick_callbacks.setdefault(market_id, []).append(callback)
        if self._book_feed is None:
            raise MarketConfigFieldError(
                "call bind_book_feed(PolymarketBookFeed) before subscribing to "
                "tick-size changes - they are delivered on the book feed's "
                "market-channel connection"
            )

    def _dispatch_tick_size(self, token_id: str, new_tick_size: float) -> None:
        for callbacks in self._tick_callbacks.values():
            for cb in callbacks:
                cb(new_tick_size)


class PolymarketBookFeed(BookFeed):
    """Live order book from the CLOB market channel.

    One connection serves both sides. `get_snapshot` is a pure in-memory
    read; REST is used only for bootstrap, resync and integrity checks.
    """

    def __init__(
        self,
        token_id_for_side: dict[Side, str],
        clob_rest_base_url: str = CLOB_REST_BASE_URL,
        clob_ws_url: str = CLOB_MARKET_WSS_URL,
        request_timeout_s: float = 10.0,
        heartbeat_interval_s: float = 10.0,
        round_id: str = "",
        condition_id: str | None = None,
    ):
        require_live_deps()
        self._token_id_for_side = dict(token_id_for_side)
        self._side_for_token = {tid: side for side, tid in token_id_for_side.items()}
        self._round_id = round_id
        self._builder = RawEventBuilder(session_id=f"bookfeed-{round_id or 'adhoc'}")
        self._callback: Callable[[BookSnapshot], None] = lambda snap: None
        self._tick_listeners: list[Callable[[str, float], None]] = []
        self._lock = threading.Lock()
        self._stream = PolymarketMarketStream(
            list(token_id_for_side.values()),
            builder=self._builder,
            on_raw_event=self._on_raw_event,
            side_for_token={tid: side.value for side, tid in token_id_for_side.items()},
            round_for_token={tid: round_id for tid in token_id_for_side.values()},
            condition_for_token={tid: condition_id for tid in token_id_for_side.values()},
            ws_url=clob_ws_url,
            rest_base_url=clob_rest_base_url,
            request_timeout_s=request_timeout_s,
            ping_interval_s=heartbeat_interval_s,
        )

    # ---------------------------------------------------------- plumbing

    def add_tick_size_listener(self, listener: Callable[[str, float], None]) -> None:
        self._tick_listeners.append(listener)

    def _on_raw_event(self, event) -> None:
        """Bridge the raw-event stream to the Phase-1 callback contract.

        Only book-affecting events produce a `BookSnapshot` callback;
        everything else (trades, best_bid_ask, control events) is still
        recorded by whatever recorder wraps this feed, but does not
        masquerade as a book update.
        """
        if event.topic is Topic.CLOB_MARKET and event.event_type == "tick_size_change":
            payload = event.payload
            new_tick = payload.get("new_tick_size") or payload.get("tick_size")
            if new_tick is not None and event.token_id:
                for listener in self._tick_listeners:
                    listener(event.token_id, float(new_tick))
            return
        if event.event_type not in ("book", "price_change", "book_snapshot_rest",
                                    "book_snapshot_rest_reconcile"):
            return
        side = self._side_for_token.get(event.token_id)
        if side is None:
            return
        snap = self.get_snapshot(self._round_id, side)
        if snap is not None:
            self._callback(snap)

    def _snapshot_for(self, side: Side) -> BookSnapshot | None:
        token_id = self._token_id_for_side.get(side)
        if token_id is None:
            return None
        book = self._stream.book(token_id)
        if book is None or book.awaiting_snapshot:
            # A book that has never had a snapshot applied is not a book.
            # Returning None makes the caller's freshness/InvalidFeatureState
            # path fire rather than quoting from a half-built ladder.
            return None
        import time as _time

        return BookSnapshot(
            side=side,
            bids=tuple(BookLevel(p, s) for p, s in book.sorted_bids()),
            asks=tuple(BookLevel(p, s) for p, s in book.sorted_asks()),
            ts=(book.source_ts_ns / 1e9) if book.source_ts_ns is not None else 0.0,
            recv_ts=(book.recv_ts_ns / 1e9) if book.recv_ts_ns is not None else _time.time(),
            book_hash=book.book_hash,
        )

    # ------------------------------------------------------ BookFeed API

    def get_snapshot(self, round_id: str, side: Side) -> BookSnapshot | None:
        """In-memory read of the WebSocket-maintained book. No REST."""
        return self._snapshot_for(side)

    def resnapshot(self, round_id: str, side: Side) -> BookSnapshot | None:
        """Explicit REST resync - one of the three permitted REST uses."""
        token_id = self._token_id_for_side.get(side)
        if token_id is None:
            return None
        self._stream.bootstrap_token(token_id)
        return self._snapshot_for(side)

    def subscribe_deltas(self, round_id: str, side: Side, callback: Callable[[BookSnapshot], None]) -> None:
        self._callback = callback
        self._stream.start()

    def reconnect(self) -> None:
        """Force a reconnect. The stream increments its reconnect
        generation and re-bootstraps every token from REST before applying
        further deltas, so a gap can never be mistaken for a quiet book."""
        self._stream.stop()
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()

    # ------------------------------------------------------ diagnostics

    def check_integrity(self, side: Side):
        token_id = self._token_id_for_side.get(side)
        return self._stream.check_integrity(token_id) if token_id else None

    @property
    def reconnect_generation(self) -> int:
        return self._stream.reconnect_generation
