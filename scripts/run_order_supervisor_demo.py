#!/usr/bin/env python3
"""Phase 9 end-to-end demo: runs the OneStepController (Phase 8) to place
maker orders, tracks them with OrderSupervisor, re-evaluates every open
order at every decision point, and applies cancel/replace per Strategy doc
SS16's trigger table. Prints the cancel/replace analytics report.

Usage: python scripts/run_order_supervisor_demo.py [n_eval_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.simulator import ExecutionSimulator, TakerOrderQueue
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.feeds.mock import MockBookFeed, MockFeedCursor
from xamarinbot.journal.schema import SupervisorDecisionRecord
from xamarinbot.journal.writer import JournalWriter
from xamarinbot.model.calibrated import fit_calibrated_model
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG, design_vector
from xamarinbot.model.walkforward import round_ordered_split
from xamarinbot.optimizer.candidates import evaluate_maker_candidate, maker_price_grid
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import OrderMode
from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.portfolio.state import Fill, FeeConfig, LiquidityRole, PortfolioState, Side, apply_fill
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.reports.supervisor_report import build_supervisor_report, format_supervisor_report
from xamarinbot.supervisor.config import SupervisorConfig
from xamarinbot.supervisor.supervisor import OrderSupervisor
from xamarinbot.supervisor.types import SupervisorActionType, TrackedOrder
from xamarinbot.synthetic.rounds import generate_synthetic_dataset

HEARTBEAT_S = 10.0
N_TRAIN_ROUNDS = 15


def train_q_model(feature_cfg: FeatureConfig):
    # Phase 12B audit item 5/C: fit on train, calibrate (Platt) on a
    # disjoint validation slice.
    train_store = EventStore(":memory:")
    train_results = generate_synthetic_dataset(train_store, n_rounds=N_TRAIN_ROUNDS)
    by_fs = build_examples_multi(train_store, train_results, feature_cfg, [COMBINED_LEAD_LAG], heartbeat_s=HEARTBEAT_S)
    examples = by_fs[COMBINED_LEAD_LAG.name]
    split = round_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    return fit_calibrated_model(split.train, split.validation, COMBINED_LEAD_LAG)


def run_round(store, round_id, p0, feature_cfg, model, controller_cfg, supervisor_cfg, exec_cfg, fee_config, journal) -> tuple[PortfolioState, dict]:
    events = store.all_events(round_id)
    clock = ReplayClock(store, round_id)
    cursor = MockFeedCursor(store, round_id, preloaded=events)
    book_feed = MockBookFeed(cursor)
    # Dedicated cursor/book_feed for fetching the actual causal book at a
    # delayed taker order's matched_ts, only at resolve time (Phase 12B
    # Tranche 1.2 items 1/2).
    revalidation_cursor = MockFeedCursor(store, round_id, preloaded=events)
    revalidation_book_feed = MockBookFeed(revalidation_cursor)
    regime_clf = RegimeClassifier(round_id=round_id)
    controller = OneStepController(controller_cfg, exec_cfg, fee_config)
    supervisor = OrderSupervisor(supervisor_cfg)
    sim = ExecutionSimulator(round_id, fee_config, exec_cfg)
    queue = TakerOrderQueue(sim)
    portfolio = PortfolioState()
    stats = {"placed": 0, "expired_filled": 0, "expired_unfilled": 0}

    market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    tick_size = market_config["tick_size"]
    recompute_cfg = OneStepConfig(g_min=supervisor_cfg.g_min, edge_min=supervisor_cfg.edge_min)
    order_seq = 0

    def _book_at(pending) -> tuple:
        revalidation_cursor.advance_to(pending.matched_ts)
        book = revalidation_book_feed.get_snapshot(round_id, pending.side)
        return book.asks if book is not None else ()

    for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
        cursor.advance_to(decision_ts)

        for pending, taker_result in queue.resolve_ready(decision_ts, _book_at):
            if taker_result.walk.filled_shares > 0:
                portfolio = apply_fill(portfolio, Fill(pending.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))

        fv = compute(events, round_id, decision_ts, p0, feature_cfg)
        if not isinstance(fv, FeatureVector):
            continue
        snapshot = regime_clf.observe(fv)
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)
        vec = design_vector(fv, COMBINED_LEAD_LAG)
        q = model.predict_proba(vec) if vec is not None else 0.5

        # 1) review every open order against current conditions
        for order_id in list(supervisor.open_order_ids()):
            tracked = supervisor.orders[order_id]
            side = tracked.order_state.side
            book = book_up if side is Side.UP else book_down
            if book is None or not book.best_bid or not book.best_ask:
                continue

            current = evaluate_maker_candidate(
                f"review-{order_id}", side, tracked.purpose, tracked.order_state.limit_price,
                tracked.order_state.remaining_shares, distance_to_touch_ticks=0.0, queue_ahead_shares=0.0,
                horizon_s=tracked.ttl_s or controller_cfg.maker_horizon_s, portfolio=portfolio, q=q,
                exec_cfg=exec_cfg, cfg=recompute_cfg,
            )
            decision = supervisor.review_order(
                tracked, decision_ts, snapshot.state, current.delta_ev, current.g_after, fv.tau, is_fresh=True,
            )
            journal.write(SupervisorDecisionRecord(
                round_id=round_id, order_id=order_id, decision_ts=decision_ts, action=decision.action.value,
                reason=decision.reason.value if decision.reason else None, side=side.value, price=tracked.order_state.limit_price,
            ))

            if decision.action is SupervisorActionType.CANCEL:
                supervisor.apply_cancel(decision, decision_ts)
            elif decision.action is SupervisorActionType.REPLACE:
                grid = maker_price_grid(book.best_bid.price, book.best_ask.price, tick_size, controller_cfg.maker_price_offsets_ticks)
                if grid:
                    new_price, _ = grid[0]
                    order_seq += 1
                    result = supervisor.apply_replace(decision, decision_ts, f"{round_id}-o{order_seq}", new_price, tracked.order_state.remaining_shares)
                    if result and result.new_order is not None:
                        supervisor.register(TrackedOrder(
                            result.new_order, snapshot.state, tracked.purpose, q, tracked.fair_value_at_submit,
                            current.g_after, current.delta_ev, tracked.ttl_s, decision_ts, decision_ts,
                        ))

        # 2) resolve TTL-expired orders (stochastic maker fill, Phase 7)
        for order_id in list(supervisor.open_order_ids()):
            tracked = supervisor.orders[order_id]
            horizon = tracked.ttl_s or controller_cfg.maker_horizon_s
            if decision_ts - tracked.submit_ts < horizon:
                continue
            draw = sim.draw_maker_fill(tracked.order_state, distance_to_touch_ticks=0.0, queue_ahead_shares=0.0, horizon_s=horizon)
            if draw.filled:
                shares = tracked.order_state.remaining_shares
                tracked.order_state.reconcile_fill(decision_ts, shares)
                portfolio = apply_fill(portfolio, Fill(tracked.order_state.side, tracked.order_state.limit_price, shares, LiquidityRole.MAKER, 0.0))
                stats["expired_filled"] += 1
            else:
                tracked.order_state.cancel(decision_ts)
                stats["expired_unfilled"] += 1
            del supervisor.orders[order_id]

        # 3) place a new order if the one-step controller wants a maker candidate
        decision = controller.decide(round_id, decision_ts, portfolio, q, snapshot.permitted_actions, book_up, book_down, tick_size, is_fresh=True)
        if decision.chosen.mode is OrderMode.FAK and decision.chosen.qty > 0 and not queue.has_pending:
            # Phase 12B audit items 13/E/L, Tranche 1.2 items 1/2/5: real
            # submit->(delay)->resolve lifecycle via the shared
            # TakerOrderQueue - never mutated before matched_ts, and at
            # most one PENDING_DELAY taker outstanding at a time.
            order_seq += 1
            asks = book_up.asks if decision.chosen.side is Side.UP else book_down.asks
            limit_price = decision.chosen.max_execution_price if decision.chosen.max_execution_price is not None else decision.chosen.price
            new_pending = queue.try_submit(f"{round_id}-o{order_seq}", decision.chosen.side, decision.chosen.qty, limit_price, asks, decision_ts)
            if new_pending is not None and not new_pending.was_delayed:
                taker_result = sim.resolve_taker(new_pending)
                if taker_result.walk.filled_shares > 0:
                    portfolio = apply_fill(portfolio, Fill(decision.chosen.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))
        elif decision.chosen.mode is OrderMode.POST_ONLY:
            order_seq += 1
            order_id = f"{round_id}-o{order_seq}"
            order_state = sim.submit_maker_order(order_id, decision.chosen.side, decision.chosen.qty, decision.chosen.price, decision_ts)
            fair_value = q if decision.chosen.side is Side.UP else (1.0 - q)
            supervisor.register(TrackedOrder(
                order_state, snapshot.state, OrderPurpose.ALPHA, q, fair_value,
                decision.chosen.g_after, decision.chosen.delta_ev, decision.chosen.ttl_s or controller_cfg.maker_horizon_s,
                decision_ts, decision_ts,
            ))
            stats["placed"] += 1

    return portfolio, stats


def main() -> None:
    n_eval_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    feature_cfg = FeatureConfig()
    fee_config = FeeConfig()
    controller_cfg = OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0, edge_min=0.5)
    supervisor_cfg = SupervisorConfig(g_min=-100.0, edge_min=0.5, min_tau_for_passive_s=15.0, churn_threshold=0.5, min_action_interval_s=1.0)
    exec_cfg = ExecutionConfig()

    print(f"Training q model on {N_TRAIN_ROUNDS} separate rounds...")
    model = train_q_model(feature_cfg)

    print(f"Running OrderSupervisor over {n_eval_rounds} rounds...")
    eval_store = EventStore(":memory:")
    # id_offset=N_TRAIN_ROUNDS: disjoint from training rounds (Phase 12B
    # audit Addendum A).
    eval_results = generate_synthetic_dataset(eval_store, n_rounds=n_eval_rounds, id_offset=N_TRAIN_ROUNDS)
    journal = JournalWriter(":memory:")

    total_pnl = 0.0
    agg_stats = {"placed": 0, "expired_filled": 0, "expired_unfilled": 0}
    for result in eval_results:
        portfolio, stats = run_round(eval_store, result.round_id, result.p0, feature_cfg, model, controller_cfg, supervisor_cfg, exec_cfg, fee_config, journal)
        pnl = portfolio.Pi_U if result.outcome is Side.UP else portfolio.Pi_D
        total_pnl += pnl
        for k in agg_stats:
            agg_stats[k] += stats[k]

    print(f"\nOrders placed: {agg_stats['placed']}  |  expired filled: {agg_stats['expired_filled']}  |  expired unfilled: {agg_stats['expired_unfilled']}")
    print(f"Net PnL across {n_eval_rounds} rounds: {total_pnl:.3f}")
    print()
    print(format_supervisor_report(build_supervisor_report(journal, eval_store)))


if __name__ == "__main__":
    main()
