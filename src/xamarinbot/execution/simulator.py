"""ExecutionSimulator v1 (Roadmap Phase 7 deliverable) - ties taker.py,
maker.py, and order_state.py into one cohesive API.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.events.replay import seeded_random
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.maker import MakerPlacement, fill_probability, optimal_maker_price, q_fill
from xamarinbot.execution.order_state import OrderLifecycleState, OrderState
from xamarinbot.execution.taker import DepthWalkResult, TakerOrderResult, walk_at_submission, walk_depth
from xamarinbot.feeds.base import BookLevel
from xamarinbot.portfolio.exposure import PendingExposure, exposure_from_pending_takers
from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, Side


@dataclass
class PendingTakerOrder:
    """Returned by `submit_taker()` - carries only what is legitimately
    knowable at `submit_ts`. For a delayed order (`was_delayed=True`), NO
    future fill information exists here at all: no filled quantity, no
    average price, no fee, no repriced flag - those only come into
    existence once `resolve_taker()` is called with the actual causal
    book at `matched_ts`. `walk_at_submission` is diagnostic-only (what
    WOULD fill against the book already visible at `submit_ts`) and is
    never used to determine the real fill for a delayed order.

    Phase 12B Tranche 1.2 item 2: this replaces the previous
    `submit_taker_order()`/`execute_taker()` design, where a caller could
    supply `asks_at_revalidation` (the actual future book) at submission
    time - the eventual fill was then fully computed and stored inside
    the returned `TakerOrderResult` even though the order itself stayed
    `PENDING_DELAY`, "dangerous by construction" since any caller reading
    `.walk` immediately (as one caller already did) silently bypassed the
    delay entirely, with nothing in the type system or the API shape to
    stop it. Under this design that bypass is structurally impossible:
    the only way to obtain a `TakerOrderResult` for a delayed order is to
    call `resolve_taker()`, which requires `now_ts >= matched_ts`."""

    order_state: OrderState
    side: Side
    requested_shares: float
    limit_price: float
    submit_ts: float
    matched_ts: float
    was_delayed: bool
    asks_at_submission: tuple[BookLevel, ...]
    walk_at_submission: DepthWalkResult
    _resolved_result: TakerOrderResult | None = field(default=None, repr=False)

    @property
    def is_resolved(self) -> bool:
        return self._resolved_result is not None


@dataclass
class TakerOrderQueue:
    """Wraps the `submit_taker`/`resolve_taker` pending-order lifecycle
    for one round's decision loop (Phase 12B Tranche 1.2 items 1 & 5) -
    every execution-oriented backtest/shadow/baseline path shares this one
    helper instead of six independent copies of "track pending orders,
    resolve at matched_ts, never mutate portfolio earlier."

    Also enforces the conservative Tranche 1.2 item 5 admission gate: at
    most one `PENDING_DELAY` taker order outstanding at a time. This
    codebase does not yet have an aggregate exposure/reservation model
    (`portfolio/exposure.py::PendingExposure` covers only sizing
    admission so far, not a second-submission gate), so a second delayed
    submission while one is still outstanding is deferred rather than
    risked - correctness over concurrency, since the taker delay window
    is short (Strategy doc SS2.2's 250ms) and the risk of two pending
    fills jointly breaching `g_min`/`spend_cap` is real (see
    `test_second_delayed_taker_is_deferred_while_one_is_pending`)."""

    sim: ExecutionSimulator
    _pending: list[PendingTakerOrder] = field(default_factory=list)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    @property
    def exposure(self) -> PendingExposure:
        """The aggregate `PendingExposure` (Phase 12B Tranche 1.2 item 5)
        from every order this queue is still tracking - `worst_case_qty`
        callers can add on top of confirmed `PortfolioState` for a
        risk-admission check. With the conservative single-pending gate
        in `try_submit` below, this sums over at most one order today,
        but the accounting itself generalizes."""
        return exposure_from_pending_takers(self._pending)

    def try_submit(
        self,
        order_id: str,
        side: Side,
        requested_shares: float,
        limit_price: float,
        asks_at_submission: tuple[BookLevel, ...],
        submit_ts: float,
    ) -> PendingTakerOrder | None:
        """Returns `None` (submission deferred, nothing changed) if a
        delayed taker order is already outstanding; otherwise submits and
        returns the `PendingTakerOrder` (queuing it internally if it came
        back delayed)."""
        if self._pending:
            return None
        pending = self.sim.submit_taker(order_id, side, requested_shares, limit_price, asks_at_submission, submit_ts)
        if pending.was_delayed:
            self._pending.append(pending)
        return pending

    def resolve_ready(self, now_ts: float, book_at_fn) -> list[tuple[PendingTakerOrder, TakerOrderResult]]:
        """Resolves every pending order whose `matched_ts` has arrived,
        fetching each one's actual causal book only now, via
        `book_at_fn(pending) -> asks_at_match`, never at submission time."""
        resolved: list[tuple[PendingTakerOrder, TakerOrderResult]] = []
        still_pending: list[PendingTakerOrder] = []
        for pending in self._pending:
            if now_ts >= pending.matched_ts:
                asks_at_match = book_at_fn(pending)
                result = self.sim.resolve_taker(pending, asks_at_match, now_ts)
                resolved.append((pending, result))
            else:
                still_pending.append(pending)
        self._pending = still_pending
        return resolved


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

    def submit_taker(
        self,
        order_id: str,
        side: Side,
        requested_shares: float,
        limit_price: float,
        asks_at_submission: tuple[BookLevel, ...],
        submit_ts: float,
        taker_delay_ms: float | None = None,
    ) -> PendingTakerOrder:
        """Phase 12B audit items 13/E/L + Tranche 1.2 item 2: the one
        common chronological taker submission path every backtest/
        ablation/shadow/baseline caller must go through. Computes only
        what is legitimately knowable at `submit_ts` - `walk_at_submission`
        (the book already visible now) and `matched_ts` - and returns a
        `PendingTakerOrder`, never a resolved fill, for a genuinely
        delayed order. The order is left `PENDING_DELAY` (unfilled) so
        `order.can_cancel(now_ts)` stays meaningful for any `now_ts` in
        between (Roadmap Phase 7 verification: "Pending-delay no-cancel
        test").

        At `taker_delay_ms<=0` (the market default), there is nothing to
        wait for - the submission book *is* the resolution book, so this
        resolves immediately and the returned `PendingTakerOrder` already
        carries its final result (`is_resolved` is True); calling
        `resolve_taker` on it just returns that cached result."""
        delay = self.cfg.taker_delay_ms if taker_delay_ms is None else taker_delay_ms
        was_delayed = delay > 0
        matched_ts = submit_ts + delay / 1000.0 if was_delayed else submit_ts
        walk_now = walk_at_submission(asks_at_submission, requested_shares, limit_price, self.fee_config)

        order = OrderState(
            order_id=order_id,
            side=side,
            role=LiquidityRole.TAKER,
            limit_price=limit_price,
            requested_shares=requested_shares,
            submit_ts=submit_ts,
            matched_ts=matched_ts if was_delayed else None,
        )
        pending = PendingTakerOrder(
            order_state=order, side=side, requested_shares=requested_shares, limit_price=limit_price,
            submit_ts=submit_ts, matched_ts=matched_ts, was_delayed=was_delayed,
            asks_at_submission=asks_at_submission, walk_at_submission=walk_now,
        )
        if not was_delayed:
            # no delay - the submission book IS the resolution book, no
            # future information is involved in resolving this now.
            if walk_now.filled_shares > 0:
                order.reconcile_fill(matched_ts, walk_now.filled_shares)
            pending._resolved_result = TakerOrderResult(
                submit_ts=submit_ts, matched_ts=matched_ts, was_delayed=False,
                walk=walk_now, repriced=False, walk_at_submission=None,
            )
        return pending

    def resolve_taker(
        self, pending: PendingTakerOrder, asks_at_match: tuple[BookLevel, ...] | None = None, now_ts: float | None = None
    ) -> TakerOrderResult:
        """Resolves a `PendingTakerOrder` against `asks_at_match` - the
        actual causal book at `pending.matched_ts`, obtained by the caller
        from replay data once its own simulated clock has genuinely
        reached that timestamp (never earlier). This is the ONLY place a
        delayed order's real fill (`.walk`, `.repriced`, price, fee) is
        ever computed (Phase 12B Tranche 1.2 item 2) - `submit_taker`
        never sees or is given this book.

        Idempotent for an already-resolved order (the `taker_delay_ms<=0`
        case, or a second call after a delayed order was already
        resolved) - `asks_at_match`/`now_ts` are ignored and the cached
        result is returned, so callers don't need to branch on
        `pending.was_delayed` before calling this.

        Raises `ValueError` if called on a still-pending delayed order
        before `now_ts` has actually reached `matched_ts`, or without the
        book to resolve against - resolving early or from the wrong book
        is exactly the bug this redesign closes, so this fails loudly
        rather than silently reusing stale/submission-time data."""
        if pending._resolved_result is not None:
            return pending._resolved_result
        if asks_at_match is None or now_ts is None:
            raise ValueError("resolving a delayed taker order requires the actual causal book (asks_at_match) and now_ts")
        if now_ts < pending.matched_ts:
            raise ValueError(f"cannot resolve before matched_ts ({pending.matched_ts}); now_ts={now_ts}")

        walk_final = walk_depth(asks_at_match, pending.requested_shares, pending.limit_price, self.fee_config)
        repriced = asks_at_match != pending.asks_at_submission
        if walk_final.filled_shares > 0:
            pending.order_state.reconcile_fill(pending.matched_ts, walk_final.filled_shares)
        elif pending.order_state.state is OrderLifecycleState.PENDING_DELAY:
            # fully missed the book at resolution - nothing filled, order
            # simply expires (FAK: no resting remainder).
            pending.order_state.state = OrderLifecycleState.CANCELED

        result = TakerOrderResult(
            submit_ts=pending.submit_ts, matched_ts=pending.matched_ts, was_delayed=True,
            walk=walk_final, repriced=repriced, walk_at_submission=pending.walk_at_submission,
        )
        pending._resolved_result = result
        return result

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
