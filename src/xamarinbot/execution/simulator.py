"""ExecutionSimulator v1 (Roadmap Phase 7 deliverable) - ties taker.py,
maker.py, and order_state.py into one cohesive API.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.events.replay import seeded_random
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.maker import MakerPlacement, fill_probability, optimal_maker_price, q_fill
from xamarinbot.execution.order_state import OrderLifecycleState, OrderState
from xamarinbot.execution.taker import TakerOrderResult, simulate_taker_order
from xamarinbot.feeds.base import BookLevel
from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, Side


@dataclass(frozen=True)
class MakerFillDraw:
    """Reproducible stochastic maker fill outcome (Roadmap Phase 2:
    "reproducible random seeds for stochastic fill models used only when
    historical queue data is insufficient" - exactly this use case, no
    real historical fill data exists yet)."""

    filled: bool
    fill_probability: float
    draw: float


class ExecutionSimulator:
    def __init__(self, round_id: str, fee_config: FeeConfig, cfg: ExecutionConfig):
        self.round_id = round_id
        self.fee_config = fee_config
        self.cfg = cfg

    def submit_taker_order(
        self,
        order_id: str,
        side: Side,
        requested_shares: float,
        limit_price: float,
        asks_at_submission: tuple[BookLevel, ...],
        submit_ts: float,
        taker_delay_ms: float | None = None,
        asks_at_revalidation: tuple[BookLevel, ...] | None = None,
    ) -> tuple[OrderState, TakerOrderResult]:
        """Returns the order in its correct *as-of-submit_ts* state
        (PENDING_DELAY if the market delays takers, else already resolved)
        plus the full TakerOrderResult describing what the fill will
        eventually be. The order is deliberately left unfilled here even
        though the eventual outcome is already computable in a backtest -
        call `resolve_pending` (or `order.reconcile_fill` directly) once
        the caller's simulated clock reaches `result.matched_ts`, so
        `order.can_cancel(now_ts)` stays meaningful for any `now_ts` in
        between (Roadmap Phase 7 verification: "Pending-delay no-cancel
        test"). Resolving immediately here would make every delayed order
        FILLED at submission, silently defeating that test."""
        delay = self.cfg.taker_delay_ms if taker_delay_ms is None else taker_delay_ms
        result = simulate_taker_order(
            asks_at_submission, requested_shares, limit_price, self.fee_config, submit_ts, delay, asks_at_revalidation
        )
        order = OrderState(
            order_id=order_id,
            side=side,
            role=LiquidityRole.TAKER,
            limit_price=limit_price,
            requested_shares=requested_shares,
            submit_ts=submit_ts,
            matched_ts=result.matched_ts if result.was_delayed else None,
        )
        if not result.was_delayed and result.walk.filled_shares > 0:
            # no delay - the fill is effectively immediate, nothing to wait on
            order.reconcile_fill(result.matched_ts, result.walk.filled_shares)
        return order, result

    def execute_taker(
        self,
        order_id: str,
        side: Side,
        requested_shares: float,
        limit_price: float,
        asks_at_submission: tuple[BookLevel, ...],
        submit_ts: float,
        revalidation_asks: tuple[BookLevel, ...] | None = None,
    ) -> tuple[OrderState, TakerOrderResult]:
        """Phase 12B audit items 13/E/L: the one common chronological
        taker execution path every backtest/ablation/shadow/baseline
        caller must go through - submit -> (delay/revalidation, if this
        market's `taker_delay_ms` is nonzero) -> resolve at matched_ts.
        Replaces every caller's previous shortcut of directly converting
        a chosen candidate's own pre-evaluation walk estimate into a
        `Fill` without ever actually submitting an order.

        `revalidation_asks`, if given, is the actual causal book at
        `matched_ts = submit_ts + self.cfg.taker_delay_ms/1000` - a
        backtest caller can obtain this immediately (ahead of its own
        `decision_ts` reaching that time) by querying replay data at that
        future timestamp, since it is the *simulated exchange*, not the
        strategy, that is allowed to know it (see
        `execution/taker.py::simulate_taker_order`'s docstring - this does
        not violate the strategy's own causal decision boundary, which
        never sees anything past `submit_ts`). Ignored when
        `taker_delay_ms<=0` (`matched_ts==submit_ts`, nothing to
        revalidate against).

        At the default `taker_delay_ms=0.0`, this resolves synchronously
        and produces the identical fill a direct-apply would have (same
        book, same walk).

        Phase 12B Tranche 1.1 item 7: for a genuinely delayed order
        (`result.was_delayed`), this call deliberately does NOT resolve
        the order here, even though the eventual fill is already fully
        computable from `revalidation_asks` - the order is returned in
        `PENDING_DELAY` state, unfilled. The caller must call
        `resolve_pending(order, result, now_ts)` once its own replay/event
        clock actually reaches `result.matched_ts`, so no caller ever
        mutates portfolio state for a fill whose `matched_ts` is still in
        the future (this was the bug: the previous version resolved and
        applied the fill synchronously at `submit_ts` regardless of
        delay, which - combined with no caller ever supplying
        `revalidation_asks` - made a delayed order behave identically to
        an immediate one except for a `matched_ts` stamp nothing read)."""
        return self.submit_taker_order(
            order_id, side, requested_shares, limit_price, asks_at_submission, submit_ts,
            asks_at_revalidation=revalidation_asks,
        )

    def resolve_pending(self, order: OrderState, result: TakerOrderResult, now_ts: float) -> bool:
        """Applies a delayed taker order's fill once `now_ts` reaches
        `result.matched_ts`. Returns True if resolved, False if it's not
        time yet (order stays PENDING_DELAY, uncancelable)."""
        if now_ts < result.matched_ts:
            return False
        if result.walk.filled_shares > 0:
            order.reconcile_fill(result.matched_ts, result.walk.filled_shares)
        elif order.state is OrderLifecycleState.PENDING_DELAY:
            # fully missed the book at revalidation - nothing filled, order
            # simply expires (FAK: no resting remainder).
            order.state = OrderLifecycleState.CANCELED
        return True

    def submit_maker_order(
        self, order_id: str, side: Side, requested_shares: float, limit_price: float, submit_ts: float
    ) -> OrderState:
        return OrderState(
            order_id=order_id,
            side=side,
            role=LiquidityRole.MAKER,
            limit_price=limit_price,
            requested_shares=requested_shares,
            submit_ts=submit_ts,
        )

    def evaluate_maker_placement(
        self,
        candidate_prices: list[float],
        distances_to_touch_ticks: list[float],
        queue_ahead_shares: list[float],
        quantity: float,
        horizon_s: float,
        q: float,
        side: Side,
    ) -> MakerPlacement:
        return optimal_maker_price(
            candidate_prices, distances_to_touch_ticks, queue_ahead_shares, quantity, horizon_s, q, side, self.cfg.maker
        )

    def draw_maker_fill(
        self, order: OrderState, distance_to_touch_ticks: float, queue_ahead_shares: float, horizon_s: float
    ) -> MakerFillDraw:
        """One reproducible Bernoulli draw for "does this maker order fill
        within horizon_s", using rho as the fill probability. Keyed by
        (round_id, order_id) so replaying the same round/order always
        produces the same outcome."""
        rho = fill_probability(distance_to_touch_ticks, queue_ahead_shares, horizon_s, self.cfg.maker)
        rng = seeded_random(self.round_id, order.order_id)
        draw = rng.random()
        return MakerFillDraw(filled=draw < rho, fill_probability=rho, draw=draw)

    def q_fill_for(self, q: float, side: Side) -> float:
        return q_fill(q, side, self.cfg.maker)
