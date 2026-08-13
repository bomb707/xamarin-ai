"""Mock/replay feed adapters satisfying every Phase-1 interface, driven by
an EventStore. These are the adapters used by replay and tests; real
adapters (polymarket_clob.py etc.) satisfy the same interfaces so the
controller code never has to know whether it's live or replaying.

Each adapter shares a MockFeedCursor: "now" advances as the replay clock
advances, and get_latest()/get_snapshot() only ever return data with
event_time <= now - the same causality guarantee as the live system, just
driven by a stored log instead of a socket.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import Event, EventType
from xamarinbot.feeds.base import (
    BookFeed,
    BookLevel,
    BookSnapshot,
    MarketConfig,
    MarketConfigProvider,
    SpotFeed,
    SpotObservation,
    TWAPFeed,
    TWAPObservation,
    UserOrderEvent,
    UserOrderFeed,
)
from xamarinbot.portfolio.state import Side


@dataclass
class MockFeedCursor:
    """Shared replay cursor: all mock feeds for a round read through this.

    Loads the round's events from the store once at construction time (the
    round is already fully written by the time replay starts - append-only
    events aren't added mid-replay here) and filters/groups them in memory,
    rather than re-querying SQLite and re-sorting on every single
    events_up_to() call, which is what every mock feed does at every
    decision point.
    """

    store: EventStore
    round_id: str
    now: float = float("-inf")
    preloaded: list[Event] | None = None
    _by_type: dict[EventType, list[Event]] = field(default_factory=dict, repr=False, init=False)
    _times_by_type: dict[EventType, list[float]] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        events = self.preloaded if self.preloaded is not None else self.store.all_events(self.round_id)
        for e in events:
            self._by_type.setdefault(e.event_type, []).append(e)
        for event_type, evs in self._by_type.items():
            self._times_by_type[event_type] = [e.event_time for e in evs]

    def advance_to(self, t: float) -> None:
        self.now = t

    def events_up_to(self, event_type: EventType) -> list[Event]:
        import bisect

        times = self._times_by_type.get(event_type, [])
        idx = bisect.bisect_right(times, self.now)
        return self._by_type[event_type][:idx] if idx else []

    def all_of_type(self, event_type: EventType) -> list[Event]:
        """Every event of this type in the round, regardless of `now` -
        for adapters (MockBookFeed) that want to build a static merged/
        sorted view once and then apply a moving prefix incrementally."""
        return self._by_type.get(event_type, [])


class MockMarketConfigProvider(MarketConfigProvider):
    def __init__(self, cursor: MockFeedCursor):
        self._cursor = cursor
        self._tick_subscribers: list[Callable[[float], None]] = []

    def get_market_config(self, market_id: str) -> MarketConfig:
        events = self._cursor.events_up_to(EventType.MARKET_CONFIG)
        if not events:
            raise LookupError(f"no MARKET_CONFIG event visible yet for {market_id}")
        latest = events[-1].payload
        return MarketConfig(**latest)

    def subscribe_tick_size_changes(self, market_id: str, callback: Callable[[float], None]) -> None:
        self._tick_subscribers.append(callback)
        events = self._cursor.events_up_to(EventType.MARKET_CONFIG)
        prev_tick: float | None = None
        for e in events:
            tick = e.payload["tick_size"]
            if prev_tick is not None and tick != prev_tick:
                callback(tick)
            prev_tick = tick


class MockTWAPFeed(TWAPFeed):
    def __init__(self, cursor: MockFeedCursor):
        self._cursor = cursor

    def get_latest(self, round_id: str) -> TWAPObservation | None:
        events = self._cursor.events_up_to(EventType.TWAP)
        if not events:
            return None
        e = events[-1]
        return TWAPObservation(
            value=e.payload["value"],
            window_seconds=e.payload["window_seconds"],
            observation_ts=e.source_ts if e.source_ts is not None else e.recv_ts,
            recv_ts=e.recv_ts,
        )

    def subscribe(self, round_id: str, callback: Callable[[TWAPObservation], None]) -> None:
        for e in self._cursor.events_up_to(EventType.TWAP):
            callback(
                TWAPObservation(
                    value=e.payload["value"],
                    window_seconds=e.payload["window_seconds"],
                    observation_ts=e.source_ts if e.source_ts is not None else e.recv_ts,
                    recv_ts=e.recv_ts,
                )
            )

    def reconnect(self) -> None:
        pass  # replay has no connection to lose


class MockSpotFeed(SpotFeed):
    def __init__(self, cursor: MockFeedCursor):
        self._cursor = cursor

    def get_latest(self, round_id: str) -> SpotObservation | None:
        events = self._cursor.events_up_to(EventType.SPOT)
        if not events:
            return None
        e = events[-1]
        return SpotObservation(
            value=e.payload["value"],
            source_ts=e.source_ts if e.source_ts is not None else e.recv_ts,
            recv_ts=e.recv_ts,
            provider=e.payload.get("provider", "mock"),
        )

    def subscribe(self, round_id: str, callback: Callable[[SpotObservation], None]) -> None:
        for e in self._cursor.events_up_to(EventType.SPOT):
            callback(
                SpotObservation(
                    value=e.payload["value"],
                    source_ts=e.source_ts if e.source_ts is not None else e.recv_ts,
                    recv_ts=e.recv_ts,
                    provider=e.payload.get("provider", "mock"),
                )
            )

    def reconnect(self) -> None:
        pass


def _levels(raw: list[list[float]]) -> tuple[BookLevel, ...]:
    return tuple(BookLevel(price=p, size=s) for p, s in raw)


@dataclass
class _BookSideCache:
    """Incremental reconstruction state for one side, so a replay that
    calls get_snapshot() at every decision point (as run_baseline_replay.py
    does) applies each BOOK_DELTA once, not once per call."""

    merged: list[Event] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    book_hash: str | None = None
    processed: int = 0
    snapshot_at_idx: int = -1
    snapshot: BookSnapshot | None = None


class MockBookFeed(BookFeed):
    """Reconstructs book state from BOOK_SNAPSHOT (full replace) and
    BOOK_DELTA (level upsert/remove) events, per side, up to the cursor."""

    def __init__(self, cursor: MockFeedCursor):
        self._cursor = cursor
        self._side_cache: dict[Side, _BookSideCache] = {}

    def _cache_for(self, side: Side) -> _BookSideCache:
        cache = self._side_cache.get(side)
        if cache is None:
            events = [
                e
                for e in self._cursor.all_of_type(EventType.BOOK_SNAPSHOT)
                + self._cursor.all_of_type(EventType.BOOK_DELTA)
                if e.payload.get("side") == side.value
            ]
            events.sort(key=lambda e: e.sort_key)
            cache = _BookSideCache(merged=events, times=[e.event_time for e in events])
            self._side_cache[side] = cache
        return cache

    def _reconstruct(self, side: Side) -> BookSnapshot | None:
        import bisect

        cache = self._cache_for(side)
        idx = bisect.bisect_right(cache.times, self._cursor.now)
        if idx == 0:
            return None
        if idx < cache.processed:
            # cursor moved backward relative to this feed's cached state -
            # rebuild from scratch rather than return a stale snapshot.
            cache.bids.clear()
            cache.asks.clear()
            cache.processed = 0
            cache.snapshot_at_idx = -1
        if idx == cache.snapshot_at_idx:
            return cache.snapshot

        for e in cache.merged[cache.processed : idx]:
            if e.event_type is EventType.BOOK_SNAPSHOT:
                cache.bids = {p: s for p, s in e.payload["bids"]}
                cache.asks = {p: s for p, s in e.payload["asks"]}
            else:  # BOOK_DELTA: size 0 => remove level
                book = cache.bids if e.payload["book"] == "bids" else cache.asks
                price, size = e.payload["price"], e.payload["size"]
                if size == 0:
                    book.pop(price, None)
                else:
                    book[price] = size
            cache.book_hash = e.payload.get("book_hash", cache.book_hash)
        cache.processed = idx

        last = cache.merged[idx - 1]
        sorted_bids = tuple(BookLevel(p, s) for p, s in sorted(cache.bids.items(), key=lambda kv: -kv[0]))
        sorted_asks = tuple(BookLevel(p, s) for p, s in sorted(cache.asks.items(), key=lambda kv: kv[0]))
        snapshot = BookSnapshot(
            side=side, bids=sorted_bids, asks=sorted_asks, ts=last.event_time, recv_ts=last.recv_ts,
            book_hash=cache.book_hash,
        )
        cache.snapshot_at_idx = idx
        cache.snapshot = snapshot
        return snapshot

    def get_snapshot(self, round_id: str, side: Side) -> BookSnapshot | None:
        return self._reconstruct(side)

    def subscribe_deltas(self, round_id: str, side: Side, callback: Callable[[BookSnapshot], None]) -> None:
        snap = self._reconstruct(side)
        if snap is not None:
            callback(snap)

    def resnapshot(self, round_id: str, side: Side) -> BookSnapshot | None:
        return self._reconstruct(side)

    def reconnect(self) -> None:
        pass


class MockUserOrderFeed(UserOrderFeed):
    def __init__(self, cursor: MockFeedCursor):
        self._cursor = cursor

    def _latest_by_order(self) -> dict[str, UserOrderEvent]:
        events = self._cursor.events_up_to(EventType.ORDER_STATUS)
        events.sort(key=lambda e: e.sort_key)
        latest: dict[str, UserOrderEvent] = {}
        for e in events:
            latest[e.payload["order_id"]] = UserOrderEvent(
                order_id=e.payload["order_id"],
                status=e.payload["status"],
                filled_size=e.payload.get("filled_size", 0.0),
                remaining_size=e.payload.get("remaining_size", 0.0),
                ts=e.event_time,
                recv_ts=e.recv_ts,
            )
        return latest

    def open_orders(self) -> list[UserOrderEvent]:
        return [o for o in self._latest_by_order().values() if o.status in ("OPEN", "PARTIALLY_FILLED")]

    def subscribe(self, callback: Callable[[UserOrderEvent], None]) -> None:
        for o in self._latest_by_order().values():
            callback(o)

    def reconcile(self) -> list[UserOrderEvent]:
        return list(self._latest_by_order().values())

    def reconnect(self) -> None:
        pass
