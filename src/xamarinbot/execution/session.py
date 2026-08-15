"""TradingSession - shared paper-trading/session execution component
(Phase 12B Tranche 2.2 item 3).

Before this module existed, `walkforward/ablations.py::_run_controller_round`,
`scripts/run_order_supervisor_demo.py`, and `shadow/runner.py` each carried
their own copy of the same state-mutation logic: track a confirmed
`PortfolioState`, run taker orders through `TakerOrderQueue`'s submit/
delay/resolve lifecycle, track open maker orders via `OrderSupervisor`,
build a `RiskView` from all of that and use it to gate every submission,
and re-evaluate/cancel/replace open orders on every later decision point.
`ShadowRunner` in particular had fallen behind - no `RiskView`, no
aggregate maker exposure, no `OrderSupervisor`, makers resolved via an
immediate Bernoulli draw instead of resting and being tracked. Since real
market data is about to be attached to a shadow service built on this
runner, "a simplified alternate execution engine" is no longer acceptable
there.

`TradingSession` is the one place that OWNS this state and mutates it.
Strategy/controller code (`OneStepController`, `MPCController`) only ever
PRODUCES `CandidateAction`s; this class is what turns a chosen candidate
into simulated fills, tracks resting orders, and reviews/cancels/replaces
them later - the shared engine `ShadowRunner` and
`scripts/run_order_supervisor_demo.py` both now use, and the intended
target for `walk-forward` replay's own arms as well. (`walkforward/
ablations.py`'s `_run_controller_round` already routes every state
mutation through the exact same underlying primitives this class wraps -
`ExecutionSimulator`, `TakerOrderQueue`, `OrderSupervisor`, `RiskView`,
`evaluate_replacement_plan` - so no execution/risk logic is duplicated
there either, even though its own loop has not been rewritten onto this
specific class in this pass; see docs/PHASE_12B_AUDIT.md's Tranche 2.2
section for that scoping note.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.simulator import ExecutionSimulator, TakerOrderQueue
from xamarinbot.optimizer.candidates import (
    candidate_exposure,
    evaluate_maker_candidate,
    evaluate_replacement_plan,
    is_recovery_purpose,
)
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.types import CandidateAction, OrderMode
from xamarinbot.portfolio.exposure import RiskView, exposure_from_open_maker_orders
from xamarinbot.portfolio.state import FeeConfig, Fill, LiquidityRole, PortfolioState, Side, apply_fill
from xamarinbot.regime.types import RegimeState
from xamarinbot.supervisor.config import SupervisorConfig
from xamarinbot.supervisor.supervisor import OrderSupervisor
from xamarinbot.supervisor.types import SupervisorActionType, TrackedOrder


@dataclass
class TradingSession:
    """Owns confirmed `PortfolioState`, the taker order queue, open maker
    orders (via `OrderSupervisor`), and every `RiskView`-gated submit/
    cancel/replace/expire operation for one round.

    `supervisor_cfg` defaults to one derived from `cfg` (g_min/edge_min) -
    matching the convention every existing caller (`ablations.py`,
    `run_order_supervisor_demo.py`) already used - but can be overridden
    for a caller that wants supervisor-specific economics (cancel_cost,
    hysteresis_margin, etc.) distinct from the controller's own config.
    """

    round_id: str
    fee_config: FeeConfig
    exec_cfg: ExecutionConfig
    cfg: OneStepConfig
    supervisor_cfg: SupervisorConfig | None = None
    portfolio: PortfolioState = field(default_factory=PortfolioState)
    n_actions: int = 0
    n_attempts: int = 0
    n_maker_placed: int = 0
    n_maker_expired_filled: int = 0
    n_maker_expired_unfilled: int = 0
    sim: ExecutionSimulator = field(init=False)
    queue: TakerOrderQueue = field(init=False)
    supervisor: OrderSupervisor = field(init=False)
    _order_seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.sim = ExecutionSimulator(self.round_id, self.fee_config, self.exec_cfg)
        self.queue = TakerOrderQueue(self.sim)
        sup_cfg = self.supervisor_cfg or SupervisorConfig(g_min=self.cfg.g_min, edge_min=self.cfg.edge_min)
        self.supervisor = OrderSupervisor(sup_cfg)

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"{self.round_id}-o{self._order_seq}"

    def risk_view(self) -> RiskView:
        """The aggregate view every submission this session makes is
        gated against - confirmed portfolio + every pending (delayed)
        taker + every open maker order currently tracked."""
        return RiskView(
            self.portfolio,
            pending_taker_exposure=self.queue.exposure,
            open_maker_exposure=exposure_from_open_maker_orders(list(self.supervisor.orders.values()), self.fee_config),
        )

    def resolve_ready_takers(self, now_ts: float, book_at_fn) -> None:
        """Applies a fill for every pending delayed taker order whose
        `matched_ts` has arrived - never before. `book_at_fn(pending) ->
        asks_at_match` fetches the real causal book at resolve time only,
        mirroring every prior caller's own `_book_at` helper."""
        for pending, taker_result in self.queue.resolve_ready(now_ts, book_at_fn):
            if taker_result.walk.filled_shares > 0:
                self.portfolio = apply_fill(
                    self.portfolio,
                    Fill(pending.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee),
                )
                self.n_actions += 1

    def review_open_orders(self, decision_ts: float, regime_state: RegimeState, q: float, book_up, book_down, tau: float, is_fresh: bool, tick_size: float, on_decision=None) -> None:
        """Runs `OrderSupervisor.review_order` against every currently
        open maker order's re-evaluated economics, expiring TTL-lapsed
        orders via a stochastic fill draw, canceling/replacing per the
        supervisor's own argmax(V_hold, V_cancel, V_replace) policy.
        REPLACE is only executed once the replacement's own exposure
        clears a `RiskView` built EXCLUDING the order being replaced.

        `on_decision(order_id, decision)`, if given, is called for every
        order actually reviewed (not TTL-expiries, which aren't a
        `SupervisorDecision`) - callers that journal cancel/replace
        analytics (e.g. `scripts/run_order_supervisor_demo.py`) hook in
        here rather than this class owning journaling itself."""
        for order_id in list(self.supervisor.open_order_ids()):
            tracked = self.supervisor.orders[order_id]
            side = tracked.order_state.side
            book = book_up if side is Side.UP else book_down
            if book is None or not book.best_bid or not book.best_ask:
                continue

            horizon = tracked.ttl_s or self.cfg.maker_horizon_s
            if decision_ts - tracked.submit_ts >= horizon:
                self._expire_maker(order_id, tracked, decision_ts, horizon)
                continue

            current = evaluate_maker_candidate(
                f"review-{order_id}", side, tracked.purpose, tracked.order_state.limit_price,
                tracked.order_state.remaining_shares, distance_to_touch_ticks=0.0, queue_ahead_shares=0.0,
                horizon_s=horizon, portfolio=self.portfolio, q=q, exec_cfg=self.exec_cfg, cfg=self.cfg,
            )
            replacement = evaluate_replacement_plan(
                side, tracked.order_state.remaining_shares, book.best_bid.price, book.best_ask.price,
                tick_size, self.cfg.maker_price_offsets_ticks, horizon, self.portfolio, q, self.exec_cfg, self.fee_config, self.cfg,
            )
            current_optimal_ev = replacement.delta_ev if replacement is not None else None
            decision = self.supervisor.review_order(tracked, decision_ts, regime_state, current.delta_ev, current.g_after, tau, is_fresh, current_optimal_ev)
            if on_decision is not None:
                on_decision(order_id, decision)

            if decision.action is SupervisorActionType.CANCEL:
                self.supervisor.apply_cancel(decision, decision_ts)
            elif decision.action is SupervisorActionType.REPLACE and replacement is not None:
                other_makers = [t for oid, t in self.supervisor.orders.items() if oid != order_id]
                replace_risk_view = RiskView(
                    self.portfolio, pending_taker_exposure=self.queue.exposure,
                    open_maker_exposure=exposure_from_open_maker_orders(other_makers, self.fee_config),
                )
                if replace_risk_view.admits(replacement.exposure, self.cfg.g_min, self.cfg.spend_cap, self.cfg.position_limit, is_recovery_candidate=is_recovery_purpose(tracked.purpose)):
                    result = self.supervisor.apply_replace(decision, decision_ts, self._next_order_id(), replacement.price, replacement.qty)
                    if result and result.new_order is not None:
                        # Phase 12C item 12: a replacement is a NEW order
                        # with a NEW thesis. Every field below comes from
                        # the replacement's own re-evaluation at
                        # `decision_ts`; none is inherited from the order
                        # that was just canceled. Previously
                        # `fair_value_at_submit` was copied straight off
                        # `tracked` (the canceled order's stale fair value,
                        # priced against a q that had since moved - which
                        # is exactly the drift that triggered the REPLACE)
                        # and `g_after_if_fill_at_submit` was `current.g_after`,
                        # the re-evaluation of the OLD order's price/size
                        # rather than the replacement's. Every downstream
                        # cancel predicate the supervisor evaluates - edge
                        # failure against `ev_at_submit`, risk breach
                        # against `g_after_if_fill_at_submit`, fair-value
                        # drift - was therefore being judged against a
                        # thesis the new order never had.
                        self.supervisor.register(TrackedOrder(
                            result.new_order, regime_state, tracked.purpose,
                            replacement.q, replacement.fair_value,
                            replacement.g_after_if_fill, replacement.delta_ev,
                            replacement.ttl_s, decision_ts, decision_ts,
                            expected_delta_g_at_submit=replacement.expected_delta_g,
                        ))
                # else: the replacement itself would breach aggregate risk
                # - hold the original order rather than tear it up with
                # nothing safe to replace it with.

    def _expire_maker(self, order_id: str, tracked: TrackedOrder, decision_ts: float, horizon: float) -> None:
        draw = self.sim.draw_maker_fill(tracked.order_state, 0.0, 0.0, horizon)
        self.n_attempts += 1
        if draw.filled:
            shares = tracked.order_state.remaining_shares
            tracked.order_state.reconcile_fill(decision_ts, shares)
            self.portfolio = apply_fill(self.portfolio, Fill(tracked.order_state.side, tracked.order_state.limit_price, shares, LiquidityRole.MAKER, 0.0))
            self.n_actions += 1
            self.n_maker_expired_filled += 1
        else:
            tracked.order_state.cancel(decision_ts)
            self.n_maker_expired_unfilled += 1
        del self.supervisor.orders[order_id]

    def dispatch(self, chosen: CandidateAction, decision_ts: float, regime_state: RegimeState, q: float, book_up, book_down) -> None:
        """Submits `chosen` (from `OneStepController`/`MPCController`)
        through the RiskView-gated taker/maker path - the only place a
        candidate actually becomes a live (paper) order. Immediate
        (non-delayed) taker fills mutate `portfolio` synchronously; a
        genuinely delayed taker is only resolved later, via
        `resolve_ready_takers`, against the real causal book at its own
        `matched_ts` - never here. A maker candidate is registered with
        `OrderSupervisor` and stays open (tracked, reviewable, cancelable)
        until it fills, expires, or is replaced - never an immediate
        Bernoulli draw at submission time."""
        if chosen.mode is OrderMode.WAIT or chosen.side is None or chosen.qty <= 0:
            return

        risk_view = self.risk_view()
        exposure = candidate_exposure(chosen, self.fee_config)
        if exposure is None:
            return
        if not risk_view.admits(exposure, self.cfg.g_min, self.cfg.spend_cap, self.cfg.position_limit, is_recovery_candidate=is_recovery_purpose(chosen.purpose)):
            return

        if chosen.mode is OrderMode.FAK and not self.queue.has_pending:
            self.n_attempts += 1
            limit_price = chosen.max_execution_price if chosen.max_execution_price is not None else chosen.price
            asks = book_up.asks if chosen.side is Side.UP else book_down.asks
            pending = self.queue.try_submit(self._next_order_id(), chosen.side, chosen.qty, limit_price, asks, decision_ts)
            if pending is not None and not pending.was_delayed:
                taker_result = self.sim.resolve_taker(pending)
                if taker_result.walk.filled_shares > 0:
                    self.portfolio = apply_fill(self.portfolio, Fill(chosen.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))
                    self.n_actions += 1
        elif chosen.mode is OrderMode.POST_ONLY:
            order_state = self.sim.submit_maker_order(self._next_order_id(), chosen.side, chosen.qty, chosen.price, decision_ts)
            fair_value = q if chosen.side is Side.UP else (1.0 - q)
            self.supervisor.register(TrackedOrder(
                order_state, regime_state, chosen.purpose, q, fair_value,
                chosen.g_after, chosen.delta_ev, chosen.ttl_s or self.cfg.maker_horizon_s, decision_ts, decision_ts,
                expected_delta_g_at_submit=chosen.expected_delta_g,
            ))
            self.n_maker_placed += 1
