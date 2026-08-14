#!/usr/bin/env python3
"""Phase 7 end-to-end demo: simulate taker orders (depth walking, FAK,
some with the 250ms delay + revalidation) and maker orders (fill
probability + reproducible stochastic fill draw) against the synthetic
order book, then print the taker slippage/delay report.

Usage: python scripts/run_execution_simulator_demo.py [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.simulator import ExecutionSimulator
from xamarinbot.feeds.mock import MockBookFeed, MockFeedCursor
from xamarinbot.portfolio.state import FeeConfig, Side
from xamarinbot.reports.execution_report import build_execution_report, format_execution_report
from xamarinbot.synthetic.rounds import generate_synthetic_dataset

HEARTBEAT_S = 10.0
ORDER_SIZE = 750.0  # > one book level (500 shares), forces depth walking
DELAY_MS = 250.0


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)
    fee_config = FeeConfig()
    exec_cfg = ExecutionConfig()

    all_taker_results = []
    n_cancel_rejected_pending = 0
    n_maker_fills = 0
    n_maker_orders = 0

    for i, result in enumerate(results):
        round_events = store.all_events(result.round_id)
        clock = ReplayClock(store, result.round_id)
        cursor = MockFeedCursor(store, result.round_id, preloaded=round_events)
        book_feed = MockBookFeed(cursor)
        sim = ExecutionSimulator(result.round_id, fee_config, exec_cfg)

        decision_points = clock.decision_points(heartbeat=HEARTBEAT_S)
        for j, decision_ts in enumerate(decision_points):
            cursor.advance_to(decision_ts)
            book = book_feed.get_snapshot(result.round_id, Side.UP)
            if book is None or not book.asks:
                continue

            use_delay = j % 2 == 0  # alternate delayed/instant orders across decision points
            order_id = f"{result.round_id}-taker-{j}"

            if use_delay:
                # Phase 12B Tranche 1.2 item 2: submit_taker() never sees
                # the future book - it only computes what's knowable at
                # decision_ts. The actual causal book at matched_ts is
                # fetched separately, only once resolving, and handed to
                # resolve_taker() explicitly - the eventual fill does not
                # exist anywhere before that call.
                pending = sim.submit_taker(
                    order_id, Side.UP, ORDER_SIZE, limit_price=0.99,
                    asks_at_submission=book.asks, submit_ts=decision_ts, taker_delay_ms=DELAY_MS,
                )
                # pending-delay no-cancel check, before resolving
                cancel_attempt = pending.order_state.cancel(decision_ts + 0.01)
                if not cancel_attempt.accepted:
                    n_cancel_rejected_pending += 1
                cursor.advance_to(pending.matched_ts)
                book_later = book_feed.get_snapshot(result.round_id, Side.UP)
                asks_later = book_later.asks if book_later else ()
                cursor.advance_to(decision_ts)  # restore - caller (this loop) owns the cursor
                taker_result = sim.resolve_taker(pending, asks_later, pending.matched_ts)
            else:
                pending = sim.submit_taker(
                    order_id, Side.UP, ORDER_SIZE, limit_price=0.99,
                    asks_at_submission=book.asks, submit_ts=decision_ts, taker_delay_ms=0.0,
                )
                taker_result = sim.resolve_taker(pending)  # already resolved at submission - no delay to wait on
            all_taker_results.append(taker_result)

            # maker order demo: place at the best bid, check reproducible fill draw
            if book.best_bid is not None:
                maker_order = sim.submit_maker_order(f"{result.round_id}-maker-{j}", Side.UP, 20.0, book.best_bid.price, decision_ts)
                draw = sim.draw_maker_fill(maker_order, distance_to_touch_ticks=1.0, queue_ahead_shares=100.0, horizon_s=HEARTBEAT_S)
                n_maker_orders += 1
                if draw.filled:
                    maker_order.reconcile_fill(decision_ts + HEARTBEAT_S / 2, 20.0)
                    n_maker_fills += 1

    print(f"Rounds: {n_rounds}  |  taker orders simulated: {len(all_taker_results)}")
    print(f"Pending-delay cancel attempts correctly rejected: {n_cancel_rejected_pending}")
    maker_fill_rate = (n_maker_fills / n_maker_orders) if n_maker_orders else 0.0
    print(f"Maker orders: {n_maker_orders}  |  maker fills (reproducible draws): {n_maker_fills} ({maker_fill_rate:.1%})")
    print()
    print(format_execution_report(build_execution_report(all_taker_results)))


if __name__ == "__main__":
    main()
