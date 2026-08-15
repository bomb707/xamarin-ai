"""Causal market-data adapter interfaces (Roadmap Phase 1).

Every adapter (replay or real) implements these interfaces so the
controller can run unchanged against replay data or live feeds. Real
adapters live in realtime/feed_adapter.py (the canonical real-market
implementation, backed by realtime/clob_ws.py + realtime/rtds.py) and
feeds/polymarket_user.py; replay/feeds.py provides fully-working
replay-driven implementations over a recorded EventStore.

"Never infer TWAP window from update cadence."
"Do not submit orders when state freshness is uncertain."
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from xamarinbot.portfolio.state import Side


@dataclass(frozen=True)
class MarketConfig:
    """Runtime market metadata (Roadmap Phase 1 deliverable: "MarketConfig
    runtime object"). Nothing here may be hardcoded by callers - it must be
    fetched/subscribed at market start."""

    market_id: str
    up_token_id: str
    down_token_id: str
    start_ts: float
    end_ts: float
    tick_size: float
    min_order_size: float
    fee_rate: float
    taker_delay_ms: float  # 0 if the market has no taker delay
    twap_window_seconds: int  # 30 or 60, per market configuration [P6]
    #: How this market settles, as the market itself declares it:
    #: "chainlink_twap" or "chainlink_reference". Phase 12C.2 item 4 made
    #: this a REQUIRED field with no default - a settlement rule that was
    #: never recorded must fail the projection closed, not be guessed. It
    #: travels Raw round metadata -> MARKET_CONFIG -> MarketConstraints ->
    #: ShadowRunner so no layer has to re-derive it.
    settlement_kind: str


@dataclass(frozen=True)
class TWAPObservation:
    """One Chainlink TWAP observation. `observation_ts` is the Chainlink
    observation timestamp - freshness must be judged from this, never from
    local receive cadence [P6]."""

    value: float
    window_seconds: int
    observation_ts: float
    recv_ts: float


@dataclass(frozen=True)
class SpotObservation:
    value: float
    source_ts: float
    recv_ts: float
    provider: str


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class BookSnapshot:
    side: Side
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    ts: float
    recv_ts: float
    book_hash: str | None = None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None


@dataclass(frozen=True)
class UserOrderEvent:
    order_id: str
    status: str  # OPEN / PARTIALLY_FILLED / MATCHED / CONFIRMED / CANCELED
    filled_size: float
    remaining_size: float
    ts: float
    recv_ts: float


class MarketConfigProvider(ABC):
    """Fetches market metadata at round start and reports runtime changes
    (e.g. tick_size_change) [P2][P5]."""

    @abstractmethod
    def get_market_config(self, market_id: str) -> MarketConfig: ...

    @abstractmethod
    def subscribe_tick_size_changes(
        self, market_id: str, callback: Callable[[float], None]
    ) -> None: ...


class TWAPFeed(ABC):
    """Official Chainlink TWAP RTDS client [P6]."""

    @abstractmethod
    def get_latest(self, round_id: str) -> TWAPObservation | None: ...

    @abstractmethod
    def subscribe(self, round_id: str, callback: Callable[[TWAPObservation], None]) -> None: ...

    @abstractmethod
    def reconnect(self) -> None: ...


class SpotFeed(ABC):
    """Current BTC leading-price feed."""

    @abstractmethod
    def get_latest(self, round_id: str) -> SpotObservation | None: ...

    @abstractmethod
    def subscribe(self, round_id: str, callback: Callable[[SpotObservation], None]) -> None: ...

    @abstractmethod
    def reconnect(self) -> None: ...


class BookFeed(ABC):
    """CLOB market-channel book deltas plus periodic/safety resnapshot [P4]."""

    @abstractmethod
    def get_snapshot(self, round_id: str, side: Side) -> BookSnapshot | None: ...

    @abstractmethod
    def subscribe_deltas(
        self, round_id: str, side: Side, callback: Callable[[BookSnapshot], None]
    ) -> None: ...

    @abstractmethod
    def resnapshot(self, round_id: str, side: Side) -> BookSnapshot | None: ...

    @abstractmethod
    def reconnect(self) -> None: ...


class UserOrderFeed(ABC):
    """Authenticated user/order stream for open orders, partial fills,
    matched/confirmed transitions, and reconciliation [P3]."""

    @abstractmethod
    def open_orders(self) -> list[UserOrderEvent]: ...

    @abstractmethod
    def subscribe(self, callback: Callable[[UserOrderEvent], None]) -> None: ...

    @abstractmethod
    def reconcile(self) -> list[UserOrderEvent]: ...

    @abstractmethod
    def reconnect(self) -> None: ...
