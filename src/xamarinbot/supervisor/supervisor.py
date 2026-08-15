"""OrderSupervisor v1 (Roadmap Phase 9 deliverable).

"Recompute open-order value on every relevant event." This module doesn't
compute EV/G itself - the caller re-evaluates each tracked order's current
economics (via optimizer/candidates.py, Phase 8) and passes the results in,
keeping this module a thin, testable policy layer over Phase 7's OrderState
mechanics rather than a second copy of the EV math.

"Open orders always have a currently valid thesis or are canceled" (Phase 9
exit gate) is enforced by `review_order` running every trigger on every
call - there's no persistent "already checked, skip" shortcut that could
let a stale thesis survive unnoticed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.execution.order_state import CancelResult, ReplaceResult, replace_order
from xamarinbot.regime.types import RegimeState
from xamarinbot.supervisor.config import SupervisorConfig
from xamarinbot.supervisor.predicates import feed_stale, hold_eligible, regime_flip, risk_breach, time_compression, value_cancel, value_hold, value_replace
from xamarinbot.supervisor.types import CancelReason, SupervisorActionType, SupervisorDecision, TrackedOrder


@dataclass
class OrderSupervisor:
    cfg: SupervisorConfig
    orders: dict[str, TrackedOrder] = field(default_factory=dict)

    def register(self, tracked: TrackedOrder) -> None:
        self.orders[tracked.order_id] = tracked

    def review_order(
        self,
        tracked: TrackedOrder,
        now_ts: float,
        current_regime_state: RegimeState,
        current_delta_ev: float,
        current_g_after_if_fill: float,
        tau: float,
        is_fresh: bool,
        current_optimal_ev: float | None = None,
    ) -> SupervisorDecision:
        """Phase 12B Tranche 2.1 item 1: the two hard-safety triggers (stale
        data, risk breach) are checked FIRST, unconditionally, ahead of the
        rate limiter - these are genuine safety constraints, not ordinary
        economic churn, and must fire immediately even if the previous
        action was inside `min_action_interval_s` (a stale feed or a risk
        breach 100ms after the last action still needs an immediate
        CANCEL, not a wait for the rate-limit window to clear). The rate
        limiter only throttles ordinary economic HOLD/CANCEL/REPLACE
        churn - regime flip, edge failure, time compression, book
        displacement - which feed V_hold/V_cancel/V_replace, and the
        action taken is whichever scores highest - `argmax(V_hold +
        hysteresis_margin, V_cancel, V_replace)`. This is what makes a
        regime flip alone insufficient to force a cancel: a still
        strongly profitable order can outscore V_cancel even after the
        flip's penalty is applied."""
        if feed_stale(is_fresh):
            return SupervisorDecision(tracked.order_id, SupervisorActionType.CANCEL, CancelReason.FEED_STALE)
        if risk_breach(current_g_after_if_fill, self.cfg):
            return SupervisorDecision(tracked.order_id, SupervisorActionType.CANCEL, CancelReason.RISK_BREACH)

        if now_ts - tracked.last_action_ts < self.cfg.min_action_interval_s:
            return SupervisorDecision(tracked.order_id, SupervisorActionType.HOLD, None, "rate_limited")

        v_hold = value_hold(current_delta_ev, tracked.origin_regime_state, current_regime_state, tau, self.cfg)
        v_cancel = value_cancel(self.cfg)
        v_replace = value_replace(current_optimal_ev, self.cfg)

        # Phase 12B Tranche 2.1 item 10: edge_min/hysteresis_margin gate
        # whether HOLD is even a live option (via hold_eligible's widened
        # threshold), rather than being folded additively into v_hold's
        # magnitude - the latter would also (wrongly) tilt the
        # HOLD-vs-REPLACE comparison below, which has nothing to do with
        # the edge floor. Once HOLD is ineligible, the only live choice is
        # CANCEL vs REPLACE - an order not worth holding as-is may still
        # be worth repricing rather than simply abandoning.
        scores: dict[SupervisorActionType, float] = {}
        if hold_eligible(v_hold, self.cfg):
            scores[SupervisorActionType.HOLD] = v_hold
        else:
            scores[SupervisorActionType.CANCEL] = v_cancel
        if v_replace is not None:
            scores[SupervisorActionType.REPLACE] = v_replace
        best_action = max(scores, key=scores.get)

        if best_action is SupervisorActionType.HOLD:
            return SupervisorDecision(tracked.order_id, SupervisorActionType.HOLD, None)
        if best_action is SupervisorActionType.REPLACE:
            return SupervisorDecision(tracked.order_id, SupervisorActionType.REPLACE)
        return SupervisorDecision(tracked.order_id, SupervisorActionType.CANCEL, self._cancel_reason(tracked, current_regime_state, tau))

    def _cancel_reason(self, tracked: TrackedOrder, current_regime_state: RegimeState, tau: float) -> CancelReason:
        """Diagnostic attribution only (journaling/reports) - the argmax
        above has already decided CANCEL; this just names the most likely
        contributing predicate for `SupervisorDecisionRecord.reason`."""
        if regime_flip(tracked.origin_regime_state, current_regime_state):
            return CancelReason.REGIME_FLIP
        if time_compression(tau, self.cfg):
            return CancelReason.TIME_COMPRESSION
        return CancelReason.EDGE_FAILURE

    def apply_cancel(self, decision: SupervisorDecision, now_ts: float) -> CancelResult | None:
        tracked = self.orders.get(decision.order_id)
        if tracked is None or decision.action is not SupervisorActionType.CANCEL:
            return None
        result = tracked.order_state.cancel(now_ts)
        tracked.last_action_ts = now_ts
        if tracked.order_state.is_terminal():
            # covers both "cancel accepted" and "a fill beat the cancel
            # attempt" (Roadmap Phase 9 verification: "Partial fill then
            # cancel") - either way the order is no longer open to track.
            del self.orders[decision.order_id]
        return result

    def apply_replace(
        self, decision: SupervisorDecision, now_ts: float, new_order_id: str, new_price: float, new_qty: float
    ) -> ReplaceResult | None:
        tracked = self.orders.get(decision.order_id)
        if tracked is None or decision.action is not SupervisorActionType.REPLACE:
            return None
        result = replace_order(tracked.order_state, now_ts, new_order_id, new_price, new_qty)
        tracked.last_action_ts = now_ts
        if tracked.order_state.is_terminal():
            del self.orders[decision.order_id]
        return result

    def open_order_ids(self) -> list[str]:
        return list(self.orders.keys())
