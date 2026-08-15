"""Shadow-run record types (Roadmap Phase 12 deliverable: "Record desired
order, hypothetical submit time, expected fill and real subsequent book
evolution.")."""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.portfolio.state import PortfolioState


@dataclass(frozen=True)
class ShadowDecisionRecord:
    round_id: str
    decision_ts: float
    action_id: str
    mode: str
    side: str | None
    price: float | None
    qty: float
    expected_fill: float
    delta_ev: float
    g_after: float
    decide_elapsed_ms: float
    missed_deadline: bool  # True if decide_elapsed_ms exceeded the configured deadline (SS21 Latency gate)
    reconnected: bool  # True if a simulated feed outage was recovered from just before this decision
    # Phase 12C item 10: real freshness, computed from the actual source
    # timestamps of the book / Chainlink reference / Chainlink TWAP /
    # Binance inputs - never the hardcoded `True` this runner used to send.
    is_fresh: bool = True
    #: Explicit per-feed reason when `is_fresh` is False, e.g.
    #: "book:STALE age=7.20s>6.00s; binance:MISSING". None when fresh.
    freshness_reason: str | None = None
    #: Set when `compute()` returned an `InvalidFeatureState` instead of a
    #: `FeatureVector` - the decision point was reached but no features
    #: could be computed. Previously such points vanished from the record
    #: entirely (`continue`), so a round could show a clean decision stream
    #: while silently having skipped half its decision points.
    invalid_reason: str | None = None
    #: True when the decision point produced no new ALPHA candidate because
    #: an input was missing or stale. Distinct from `missed_deadline`
    #: (a latency failure) and from a genuine WAIT (an economic choice).
    suppressed_by_freshness: bool = False


@dataclass(frozen=True)
class ShadowRoundResult:
    round_id: str
    records: tuple[ShadowDecisionRecord, ...]
    final_portfolio: PortfolioState  # caller derives hypothetical PnL against the real settlement outcome
    n_reconnects: int
    n_missed_deadlines: int
    # Phase 12C item 10 health counters, so a shadow round reports WHY it
    # was quiet rather than leaving "few actions" indistinguishable between
    # "no edge" and "the feeds were down".
    n_freshness_failures: int = 0
    n_invalid_feature_states: int = 0
    n_orders_reviewed_while_stale: int = 0
