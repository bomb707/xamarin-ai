"""Decision/fill journal schema v1 (Roadmap SS20 "Data Schema and
Journaling").

market_config, feed_event, portfolio_state, order_event, fill, settlement,
and audit are implemented and populated starting Phase 0-2. feature_state
(Phase 4), model_output (Phase 5) and candidate_action (Phase 8) are
declared now, matching SS20's full entity list, but stay unused until those
phases are built - see docs/PHASE_STATUS.md.

regime_transition (Phase 6) is an addition beyond SS20's original table -
the Roadmap's "Transition statistics report" deliverable needs somewhere to
persist RegimeTransition events, and SS20 doesn't name a slot for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarketConfigRecord:
    round_id: str
    up_token_id: str
    down_token_id: str
    start_ts: float
    end_ts: float
    tick_size: float
    min_order_size: float
    fee_rate: float
    taker_delay_ms: float
    twap_window_seconds: int


@dataclass(frozen=True)
class FeedEventRecord:
    round_id: str
    source: str
    source_ts: float | None
    recv_ts: float
    sequence: int
    value_or_hash: str


@dataclass(frozen=True)
class FeatureStateRecord:
    """Phase 4 (Feature Engineering) - declared, not yet populated."""

    round_id: str
    decision_ts: float
    feature_version: str
    features: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutputRecord:
    """Phase 5 (Probability Model) - declared, not yet populated."""

    round_id: str
    decision_ts: float
    model_version: str
    q: float
    uncertainty: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioStateRecord:
    round_id: str
    decision_ts: float
    U: float
    D: float
    C: float
    Pi_U: float
    Pi_D: float
    G: float
    R: float


@dataclass(frozen=True)
class CandidateActionRecord:
    """Phase 8 (One-Step Optimizer) - declared, not yet populated."""

    round_id: str
    decision_ts: float
    action_id: str
    purpose: str
    mode: str
    side: str
    price: float
    qty: float
    ev_after: float
    g_after: float
    pi_u_after: float
    pi_d_after: float


@dataclass(frozen=True)
class OrderEventRecord:
    round_id: str
    order_id: str
    decision_ts: float
    side: str
    role: str
    price: float
    quantity: float
    event: str  # SUBMIT / CANCEL / REPLACE / STATUS
    status: str | None = None


@dataclass(frozen=True)
class FillRecord:
    round_id: str
    order_id: str
    fill_ts: float
    side: str
    price: float
    size: float
    fee: float
    liquidity_role: str


@dataclass(frozen=True)
class SettlementRecord:
    round_id: str
    outcome: str  # UP / DOWN
    payout: float
    realized_pnl: float


@dataclass(frozen=True)
class RegimeTransitionRecord:
    """Phase 6 (Seed Regime State Machine) - populated by
    regime/classifier.py. from_state is None for a round's first
    observation."""

    round_id: str
    transition_ts: float
    from_state: dict | None
    to_state: dict
    seed_action: str
    dwell_time_s: float | None


@dataclass(frozen=True)
class SupervisorDecisionRecord:
    """Phase 9 (OrderSupervisor) - populated by supervisor/supervisor.py.
    Another addition beyond SS20's original table, same rationale as
    regime_transition."""

    round_id: str
    order_id: str
    decision_ts: float
    action: str  # HOLD / CANCEL / REPLACE
    reason: str | None
    side: str
    price: float


@dataclass(frozen=True)
class AuditRecord:
    round_id: str
    decision_ts: float
    strategy_version: str
    config_hash: str
    skip_reason: str | None
    model_version: str | None = None
    diagnostics: dict = field(default_factory=dict)
