"""Real current-BTC leading-price composite feed.

Roadmap Phase 1: "Implement current BTC leading-price feed. Prefer a
low-latency robust source/composite; record provider timestamp and receive
timestamp." Strategy doc SS22: "If current BTC feed diverges materially
across providers, degrade confidence rather than choosing the most
favorable source."

Polymarket does not publish an official "current BTC spot" endpoint (that's
exactly why this is a leading/composite signal rather than the settlement
source of truth) - the providers below are standard public exchange REST
endpoints, confirmed from prior knowledge of their stable public APIs:

  Coinbase: GET https://api.coinbase.com/v2/prices/BTC-USD/spot
  Binance:  GET https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT

Both are simple, unauthenticated, well-established public endpoints, unlike
the Chainlink Data Streams / Polymarket CLOB internals elsewhere in this
package - lower integration risk, but still verify response shape against a
live call before production use, and add/remove providers as needed.
"""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from xamarinbot.feeds._live_deps import require_live_deps
from xamarinbot.feeds.base import SpotFeed, SpotObservation


class SpotPriceProvider(Protocol):
    name: str

    def fetch(self, timeout_s: float) -> float: ...


@dataclass
class CoinbaseSpotProvider:
    name: str = "coinbase"
    url: str = "https://api.coinbase.com/v2/prices/BTC-USD/spot"

    def fetch(self, timeout_s: float) -> float:
        import httpx

        resp = httpx.get(self.url, timeout=timeout_s)
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])


@dataclass
class BinanceSpotProvider:
    name: str = "binance"
    url: str = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    def fetch(self, timeout_s: float) -> float:
        import httpx

        resp = httpx.get(self.url, timeout=timeout_s)
        resp.raise_for_status()
        return float(resp.json()["price"])


class SpotFeedDivergenceError(RuntimeError):
    """Raised when providers disagree beyond `max_divergence_bp` - degrade
    confidence rather than silently picking the most favorable source."""


class CompositeSpotFeed(SpotFeed):
    def __init__(
        self,
        providers: list[SpotPriceProvider] | None = None,
        max_divergence_bp: float = 15.0,
        poll_interval_s: float = 1.0,
        request_timeout_s: float = 3.0,
    ):
        require_live_deps()
        self._providers = providers or [CoinbaseSpotProvider(), BinanceSpotProvider()]
        self._max_divergence_bp = max_divergence_bp
        self._poll_interval = poll_interval_s
        self._timeout = request_timeout_s
        self._latest: dict[str, SpotObservation] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback: Callable[[SpotObservation], None] = lambda obs: None

    def get_latest(self, round_id: str) -> SpotObservation | None:
        with self._lock:
            return self._latest.get(round_id)

    def poll_once(self) -> SpotObservation:
        prices: dict[str, float] = {}
        for provider in self._providers:
            try:
                prices[provider.name] = provider.fetch(self._timeout)
            except Exception:
                continue
        if not prices:
            raise SpotFeedDivergenceError("no spot providers returned a price")

        values = list(prices.values())
        median = statistics.median(values)
        max_dev_bp = max(abs(v - median) / median * 10_000 for v in values)
        if len(values) > 1 and max_dev_bp > self._max_divergence_bp:
            raise SpotFeedDivergenceError(
                f"providers diverge {max_dev_bp:.1f}bp > {self._max_divergence_bp}bp: {prices}"
            )

        now = time.time()
        return SpotObservation(value=median, source_ts=now, recv_ts=now, provider="+".join(prices))

    def subscribe(self, round_id: str, callback: Callable[[SpotObservation], None]) -> None:
        self._callback = callback
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_forever, args=(round_id,), name="spot-composite-poll", daemon=True
        )
        self._thread.start()

    def reconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def _poll_forever(self, round_id: str) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                obs = self.poll_once()
                with self._lock:
                    self._latest[round_id] = obs
                self._callback(obs)
                backoff = 1.0
            except Exception:
                backoff = min(backoff * 2, 30.0)
            self._stop.wait(self._poll_interval)
