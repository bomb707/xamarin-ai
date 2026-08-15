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
from xamarinbot.execution.session import TradingSession
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
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.reports.supervisor_report import build_supervisor_report, format_supervisor_report
from xamarinbot.supervisor.config import SupervisorConfig
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
    # Phase 12B Tranche 2.2 item 3: the same shared execution/session
    # engine ShadowRunner now uses - RiskView-gated dispatch, open makers
    # tracked via OrderSupervisor, delayed taker resolution, cancel/
    # replace, all owned by TradingSession rather than duplicated here.
    session = TradingSession(round_id, fee_config, exec_cfg, controller_cfg, supervisor_cfg=supervisor_cfg)

    market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    tick_size = market_config["tick_size"]

    def _book_at(pending) -> tuple:
        revalidation_cursor.advance_to(pending.matched_ts)
        book = revalidation_book_feed.get_snapshot(round_id, pending.side)
        return book.asks if book is not None else ()

    def _journal_decision(order_id, decision) -> None:
        # The callback fires before apply_cancel/apply_replace, so the
        # order is still tracked at this point.
        order_state = session.supervisor.orders[order_id].order_state
        journal.write(SupervisorDecisionRecord(
            round_id=round_id, order_id=order_id, decision_ts=decision_ts, action=decision.action.value,
            reason=decision.reason.value if decision.reason else None,
            side=order_state.side.value, price=order_state.limit_price,
        ))

    for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
        cursor.advance_to(decision_ts)

        session.resolve_ready_takers(decision_ts, _book_at)

        fv = compute(events, round_id, decision_ts, p0, feature_cfg)
        if not isinstance(fv, FeatureVector):
            continue
        snapshot = regime_clf.observe(fv)
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)
        vec = design_vector(fv, COMBINED_LEAD_LAG)
        q = model.predict_proba(vec) if vec is not None else 0.5

        # 1) review every open order against current conditions (cancel/
        # replace/expire), journaling each real supervisor decision.
        session.review_open_orders(decision_ts, snapshot.state, q, book_up, book_down, fv.tau, True, tick_size, on_decision=_journal_decision)

        # 2) place a new order if the one-step controller wants one -
        # gated by the same aggregate RiskView as every other dispatch.
        risk_view = session.risk_view()
        decision = controller.decide(round_id, decision_ts, session.portfolio, q, snapshot.permitted_actions, book_up, book_down, tick_size, is_fresh=True, risk_view=risk_view)
        session.dispatch(decision.chosen, decision_ts, snapshot.state, q, book_up, book_down)

    stats = {"placed": session.n_maker_placed, "expired_filled": session.n_maker_expired_filled, "expired_unfilled": session.n_maker_expired_unfilled}
    return session.portfolio, stats

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
