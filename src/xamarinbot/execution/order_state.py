"""Order lifecycle state machine (Roadmap Phase 7: "Implement cancel and
replace with order-state reconciliation.")

This is the exchange-simulation *mechanics* only: given an order's current
state and a cancel/replace/fill event, what happens. *Deciding* when to
cancel or replace (regime flip, edge failure, etc.) is Roadmap Phase 9's
OrderSupervisor, not built yet - this module has no opinion on strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from xamarinbot.portfolio.state import LiquidityRole, Side

_TERMINAL_STATES_NAME = ("FILLED", "CANCELED", "REJECTED")


class OrderLifecycleState(str, Enum):
    PENDING_DELAY = "PENDING_DELAY"  # taker order in the 250ms window, cannot be canceled
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class CancelResult:
    accepted: bool
    reason: str | None = None


@dataclass
class OrderState:
    order_id: str
    side: Side
    role: LiquidityRole
    limit_price: float
    requested_shares: float
    submit_ts: float
    matched_ts: float | None = None  # when a PENDING_DELAY order becomes cancelable/live
    filled_shares: float = 0.0
    state: OrderLifecycleState = OrderLifecycleState.OPEN
    fill_events: list[tuple[float, float]] = field(default_factory=list)  # (ts, shares)

    def __post_init__(self) -> None:
        if self.role is LiquidityRole.TAKER and self.matched_ts is not None and self.matched_ts > self.submit_ts:
            self.state = OrderLifecycleState.PENDING_DELAY

    @property
    def remaining_shares(self) -> float:
        return max(0.0, self.requested_shares - self.filled_shares)

    def is_terminal(self) -> bool:
        return self.state.value in _TERMINAL_STATES_NAME

    def reconcile_fill(self, ts: float, shares: float) -> None:
        """Applies a FILL event (Roadmap Phase 2 causal event type). Safe
        to call on a PENDING_DELAY order - a fill event at/after matched_ts
        naturally transitions it out of pending."""
        if self.is_terminal():
            return
        self.fill_events.append((ts, shares))
        self.filled_shares = min(self.requested_shares, self.filled_shares + shares)
        if self.filled_shares >= self.requested_shares - 1e-9:
            self.state = OrderLifecycleState.FILLED
        else:
            self.state = OrderLifecycleState.PARTIALLY_FILLED

    def can_cancel(self, now_ts: float) -> bool:
        if self.is_terminal():
            return False
        if self.state is OrderLifecycleState.PENDING_DELAY and self.matched_ts is not None and now_ts < self.matched_ts:
            # Strategy doc SS2.2/SS22: "the order is pending and cannot be
            # canceled"; SS22: "do not assume cancellation is available
            # until the order returns."
            return False
        return True

    def cancel(self, now_ts: float) -> CancelResult:
        if not self.can_cancel(now_ts):
            reason = "pending_taker_delay" if self.state is OrderLifecycleState.PENDING_DELAY else f"terminal_state:{self.state.value}"
            return CancelResult(accepted=False, reason=reason)
        self.state = OrderLifecycleState.CANCELED
        return CancelResult(accepted=True)


@dataclass(frozen=True)
class ReplaceResult:
    cancel_result: CancelResult
    new_order: OrderState | None  # None if the cancel leg was rejected - no replace without a successful cancel


def replace_order(order: OrderState, now_ts: float, new_order_id: str, new_limit_price: float, new_requested_shares: float) -> ReplaceResult:
    """Strategy doc SS16 calls this a strategic *decision* (Phase 9); here
    it's just the exchange mechanic: cancel the remainder, then submit a
    fresh order for the caller-supplied new terms. Fails atomically if the
    cancel leg is rejected (e.g. the original order is still pending its
    taker delay) - never partially replaces."""
    cancel_result = order.cancel(now_ts)
    if not cancel_result.accepted:
        return ReplaceResult(cancel_result=cancel_result, new_order=None)

    new_order = OrderState(
        order_id=new_order_id,
        side=order.side,
        role=order.role,
        limit_price=new_limit_price,
        requested_shares=new_requested_shares,
        submit_ts=now_ts,
    )
    return ReplaceResult(cancel_result=cancel_result, new_order=new_order)
