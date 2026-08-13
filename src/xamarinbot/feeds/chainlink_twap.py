"""Real Chainlink TWAP client skeleton [P6].

Roadmap Phase 1: "Implement official Chainlink TWAP RTDS client using
observation timestamp, reconnect/resubscribe logic, and configurable 30/60
second window mapping." Strategy doc: "never assume every BTC 5-minute
market uses the same window."

From docs.polymarket.com/market-data/chainlink-twap (fetched 2026-08-13):
  REST base: https://api.dataengine.chain.link
  WSS base:  wss://ws.dataengine.chain.link
  Report fields: feedID, decoded.price (signed E18 fixed-point),
    report.observationsTimestamp (unix seconds), decoded.expiresAt.
  "Reports do not include symbol or window labels. Maintain that mapping
  yourself" - i.e. you must look up the feed ID for "BTC / USD - TWAP: 30s"
  vs "...: 60s" in the Chainlink feed catalog and pass the right one in.

CAVEAT - genuinely unverified, do not treat as production-ready: this is
Chainlink's Data Streams infrastructure, a different vendor from Polymarket,
and the fetched docs page did not specify the authentication/subscription
handshake (Chainlink Data Streams normally requires HMAC-signed requests
with a client ID/secret). `auth_headers` below is intentionally a pluggable
caller-supplied dict rather than a guessed signing implementation - wire it
up per Chainlink's own Data Streams API docs before connecting for real.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable

from xamarinbot.feeds._live_deps import require_live_deps
from xamarinbot.feeds.base import TWAPFeed, TWAPObservation

CHAINLINK_REST_BASE_URL = "https://api.dataengine.chain.link"
CHAINLINK_WS_URL = "wss://ws.dataengine.chain.link"

_E18 = 10**18


def decode_e18_price(raw_price: int | str) -> float:
    """decoded.price is a signed E18 fixed-point integer per the docs."""
    return int(raw_price) / _E18


class ChainlinkTWAPFeed(TWAPFeed):
    def __init__(
        self,
        feed_id: str,
        window_seconds: int,
        auth_headers: dict[str, str] | None = None,
        rest_base_url: str = CHAINLINK_REST_BASE_URL,
        ws_url: str = CHAINLINK_WS_URL,
        request_timeout_s: float = 5.0,
    ):
        require_live_deps()
        if window_seconds not in (30, 60):
            raise ValueError("Polymarket currently supports 30s or 60s TWAP windows [P6]")
        self._feed_id = feed_id
        self._window_seconds = window_seconds
        self._auth_headers = auth_headers or {}
        self._rest_base_url = rest_base_url
        self._ws_url = ws_url
        self._timeout = request_timeout_s
        self._latest: dict[str, TWAPObservation] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback: Callable[[TWAPObservation], None] = lambda obs: None

    def get_latest(self, round_id: str) -> TWAPObservation | None:
        with self._lock:
            return self._latest.get(round_id)

    def fetch_latest_report(self) -> TWAPObservation:
        """One-shot REST fetch - the confirmed-shape part of this adapter.
        Requires `auth_headers` populated per Chainlink's own API docs."""
        import httpx

        resp = httpx.get(
            f"{self._rest_base_url}/reports/latest",
            params={"feedID": self._feed_id},
            headers=self._auth_headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        decoded = payload["decoded"]
        report = payload["report"]
        return TWAPObservation(
            value=decode_e18_price(decoded["price"]),
            window_seconds=self._window_seconds,
            observation_ts=float(report["observationsTimestamp"]),
            recv_ts=time.time(),
        )

    def subscribe(self, round_id: str, callback: Callable[[TWAPObservation], None]) -> None:
        self._callback = callback
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_forever, args=(round_id,), name="chainlink-twap-poll", daemon=True
        )
        self._thread.start()

    def reconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def _poll_forever(self, round_id: str) -> None:
        """Polling fallback until the WSS subscription handshake is
        confirmed against Chainlink's own docs; safe default since TWAP
        does not need sub-second freshness for a 5-minute round."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                obs = self.fetch_latest_report()
                with self._lock:
                    self._latest[round_id] = obs
                self._callback(obs)
                backoff = 1.0
                self._stop.wait(1.0)
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
