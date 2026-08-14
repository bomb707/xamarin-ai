"""OrderSupervisor types (Roadmap Phase 9).

"Attach origin_state_id, purpose, q_at_submit, fair_value, G_after_if_fill,
expiry policy and cancel predicates to every order." The trigger vocabulary
below is Strategy doc SS16's table almost verbatim: edge failure, regime
flip, risk breach, time compression, book displacement, feed freshness
failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xamarinbot.execution.order_state import OrderState
from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.regime.types import RegimeState


class CancelReason(str, Enum):
    EDGE_FAILURE = "EDGE_FAILURE"
    REGIME_FLIP = "REGIME_FLIP"
    RISK_BREACH = "RISK_BREACH"
    TIME_COMPRESSION = "TIME_COMPRESSION"
    FEED_STALE = "FEED_STALE"
    RATE_LIMITED_HOLD = "RATE_LIMITED_HOLD"  # not a real trigger; used only in diagnostics when a trigger fired but was suppressed


class SupervisorActionType(str, Enum):
    HOLD = "HOLD"
    CANCEL = "CANCEL"
    REPLACE = "REPLACE"


@dataclass
class TrackedOrder:
    """An open order plus the thesis that justified it, so the supervisor
    can tell when that thesis stops holding. `order_state` is Phase 7's
    lifecycle state machine - this module only ever reads/cancels it, never
    reimplements fill/cancel mechanics."""

    order_state: OrderState
    origin_regime_state: RegimeState
    purpose: OrderPurpose
    q_at_submit: float
    fair_value_at_submit: float  # SS9: FairValue_UP = q_t, FairValue_DOWN = 1-q_t
    g_after_if_fill_at_submit: float
    ev_at_submit: float
    ttl_s: float | None
    submit_ts: float
    last_action_ts: float

    @property
    def order_id(self) -> str:
        return self.order_state.order_id


@dataclass(frozen=True)
class SupervisorDecision:
    order_id: str
    action: SupervisorActionType
    reason: CancelReason | None = None
    detail: str = ""
