"""Phase 12C - Real Market Recorder and Event-Level Shadow Foundation.

Everything in this package talks to the *real* Polymarket/Chainlink/Binance
public data plane. Nothing in it places, cancels, or replaces an order, and
nothing in it requires a private key or an authenticated session - Phase 12C
is a recorder plus paper shadow only (item 14).

Module map:

  discovery.py     BTC 5-minute UP/DOWN market discovery + full per-round
                   metadata, from Gamma + CLOB market-info (item 2).
  clob_ws.py       Production CLOB market-channel adapter: real in-memory
                   order book from the WebSocket, REST only for bootstrap /
                   resync / integrity checks (item 3).
  rtds.py          One shared RTDS connection for Chainlink BTC/USD
                   reference, Binance BTCUSDT, and Chainlink 30s/60s TWAP
                   (item 4).
  raw_events.py    The immutable raw-event record and its topic vocabulary
                   (item 6).
  raw_store.py     WAL-backed, batched SQLite writer for raw events.
  recorder.py      Bounded async ingestion queue + writer thread + health
                   metrics (item 6).
  metrics.py       Recorder health counters and latency statistics.
  lifecycle.py     Explicit round lifecycle: DISCOVERED -> PRE_ROUND ->
                   ACTIVE -> ENDED -> RESOLVED -> FINALIZED (item 8).
  label.py         Independent settlement-label reconstruction and
                   agreement checking against Polymarket's own resolution
                   (items 5 and 8).
  freshness.py     Real freshness computed from actual source timestamps,
                   replacing the hardcoded `is_fresh=True` (item 10).
  counterfactual.py
                   Per-hypothetical-maker-quote counterfactual capture, so
                   fill probability can LATER be estimated from real data
                   instead of the synthetic Bernoulli (item 11).
  service.py       The orchestrator that wires the above into one running
                   recorder for N consecutive rounds.
  report.py        The captured-sample integrity report (item 15).
"""
