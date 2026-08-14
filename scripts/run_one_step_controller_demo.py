#!/usr/bin/env python3
"""Phase 8 end-to-end demo: run the OneStepController and the Phase 0
baseline strategy over the *same* causal replay of the same rounds side by
side (Roadmap Phase 8 verification: "Compare against baseline on identical
replay"), print a candidate diagnostics table for a sample decision, and
report net PnL / trade counts for both.

q for the one-step controller comes from a quickly-trained Phase 5 combined
lead-lag model, fit on a *separate* training dataset generated first (not
the evaluation rounds) to avoid the obvious lookahead bias of fitting and
evaluating on the same data.

Usage: python scripts/run_one_step_controller_demo.py [n_eval_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.baseline.config import BaselineConfig
from xamarinbot.baseline.inputs import elapsed_t
from xamarinbot.baseline.strategy import BaselineInputs, decide as baseline_decide
from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.simulator import ExecutionSimulator
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.feeds.mock import MockBookFeed, MockFeedCursor, MockSpotFeed, MockTWAPFeed
from xamarinbot.model.calibrated import fit_calibrated_model
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG
from xamarinbot.model.walkforward import time_ordered_split
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import OrderMode
from xamarinbot.portfolio.state import Fill, FeeConfig, LiquidityRole, PortfolioState, Side, apply_fill
from xamarinbot.regime.classifier import RegimeClassifier
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
    split = time_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    return fit_calibrated_model(split.train, split.validation, COMBINED_LEAD_LAG)


def run_baseline_round(store, round_id, p0, cfg: BaselineConfig, fee_config: FeeConfig, exec_cfg: ExecutionConfig) -> PortfolioState:
    """Phase 12B Tranche 1: this local copy had the exact same two bugs
    fixed in `walkforward/ablations.py::_run_baseline_round` (spot_prev
    always equal to spot; absolute decision_ts passed as elapsed round
    time) plus the same full-fill-at-limit-price execution shortcut -
    discovered while smoke-testing that fix's demo output showed this
    script's baseline still trading zero rounds. Fixed identically here,
    via the same shared `elapsed_t()` helper and `ExecutionSimulator.execute_taker`."""
    events = store.all_events(round_id)
    clock = ReplayClock(store, round_id)
    cursor = MockFeedCursor(store, round_id, preloaded=events)
    twap_feed, spot_feed, book_feed = MockTWAPFeed(cursor), MockSpotFeed(cursor), MockBookFeed(cursor)
    portfolio = PortfolioState()
    prev_clob_cursor = MockFeedCursor(store, round_id, preloaded=events)
    prev_book_feed = MockBookFeed(prev_clob_cursor)
    prev_spot_cursor = MockFeedCursor(store, round_id, preloaded=events)
    prev_spot_feed = MockSpotFeed(prev_spot_cursor)
    sim = ExecutionSimulator(round_id, fee_config, exec_cfg)
    order_seq = 0

    market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    round_start_ts = market_config["start_ts"]

    for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
        cursor.advance_to(decision_ts)
        twap_obs, spot_obs = twap_feed.get_latest(round_id), spot_feed.get_latest(round_id)
        if twap_obs is None or spot_obs is None:
            continue
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)
        if book_up is None or not book_up.best_bid or not book_up.best_ask:
            continue
        prev_clob_cursor.advance_to(decision_ts - cfg.clob_lookback_s)
        prev_book_up = prev_book_feed.get_snapshot(round_id, Side.UP)
        mid = (book_up.best_bid.price + book_up.best_ask.price) / 2.0
        mid_prev = ((prev_book_up.best_bid.price + prev_book_up.best_ask.price) / 2.0) if prev_book_up and prev_book_up.best_bid and prev_book_up.best_ask else mid

        prev_spot_cursor.advance_to(decision_ts - cfg.spot_lookback_s)
        prev_spot_obs = prev_spot_feed.get_latest(round_id)
        spot_prev = prev_spot_obs.value if prev_spot_obs is not None else spot_obs.value

        inputs = BaselineInputs(
            t=elapsed_t(decision_ts, round_start_ts), p0=p0, twap=twap_obs.value, clob_mid=mid, clob_mid_prev=mid_prev,
            spot=spot_obs.value, spot_prev=spot_prev,
            best_ask_up=book_up.best_ask.price, best_ask_down=book_down.best_ask.price if book_down and book_down.best_ask else None,
            is_fresh=True,
        )
        result = baseline_decide(inputs, portfolio.U, portfolio.D, portfolio.C, cfg)
        if result.order is not None:
            order_seq += 1
            book = book_up if result.order.side is Side.UP else book_down
            asks = book.asks if book is not None else ()
            _, taker_result = sim.execute_taker(f"{round_id}-o{order_seq}", result.order.side, result.order.quantity, result.order.price, asks, decision_ts)
            if taker_result.walk.filled_shares > 0:
                fill = Fill(result.order.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, result.order.role, taker_result.walk.total_fee)
                portfolio = apply_fill(portfolio, fill)
    return portfolio


def run_one_step_round(store, round_id, p0, feature_cfg, model, one_step_cfg, exec_cfg, fee_config, print_sample=False) -> tuple[PortfolioState, int]:
    events = store.all_events(round_id)
    clock = ReplayClock(store, round_id)
    cursor = MockFeedCursor(store, round_id, preloaded=events)
    book_feed = MockBookFeed(cursor)
    # Dedicated cursor/book_feed for fetching the actual causal book at a
    # delayed taker order's matched_ts (Phase 12B Tranche 1.1 item 7) -
    # kept separate from the main decision-time cursor above since it
    # advances to a different timestamp, mirroring the prev_cursor pattern
    # already used for baseline lookback in this same file.
    revalidation_cursor = MockFeedCursor(store, round_id, preloaded=events)
    revalidation_book_feed = MockBookFeed(revalidation_cursor)
    regime_clf = RegimeClassifier(round_id=round_id)
    controller = OneStepController(one_step_cfg, exec_cfg, fee_config)
    sim = ExecutionSimulator(round_id, fee_config, exec_cfg)
    portfolio = PortfolioState()
    n_actions = 0
    printed = False
    order_seq = 0
    pending: list = []  # (OrderState, TakerOrderResult) awaiting their matched_ts

    market_config = None
    for e in events:
        if e.event_type.value == "MARKET_CONFIG":
            market_config = e.payload
            break
    tick_size = market_config["tick_size"] if market_config else 0.01

    for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
        cursor.advance_to(decision_ts)

        # Resolve any delayed taker orders whose matched_ts has arrived -
        # never mutate portfolio for a fill whose matched_ts is still in
        # the future (Phase 12B Tranche 1.1 item 7).
        still_pending = []
        for order, result in pending:
            if decision_ts >= result.matched_ts:
                sim.resolve_pending(order, result, decision_ts)
                if order.filled_shares > 0:
                    portfolio = apply_fill(portfolio, Fill(order.side, result.walk.avg_price, order.filled_shares, LiquidityRole.TAKER, result.walk.total_fee))
            else:
                still_pending.append((order, result))
        pending = still_pending

        fv = compute(events, round_id, decision_ts, p0, feature_cfg)
        if not isinstance(fv, FeatureVector):
            continue
        snapshot = regime_clf.observe(fv)
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)

        from xamarinbot.model.features import design_vector

        vec = design_vector(fv, COMBINED_LEAD_LAG)
        q = model.predict_proba(vec) if vec is not None else 0.5

        decision = controller.decide(round_id, decision_ts, portfolio, q, snapshot.permitted_actions, book_up, book_down, tick_size, is_fresh=True)

        if print_sample and not printed and len(decision.candidates) > 2:
            printed = True
            print(f"\n--- Sample candidate table: {round_id} @ t={decision_ts:.0f}s (regime={snapshot.state}, q={q:.3f}) ---")
            print(f"{'action_id':<14}{'mode':<11}{'side':<6}{'price':>8}{'qty':>9}{'delta_ev':>11}{'g_after':>11}  valid")
            for c in decision.candidates:
                price_s = f"{c.price:.3f}" if c.price is not None else "-"
                side_s = c.side.value if c.side else "-"
                print(f"{c.action_id:<14}{c.mode.value:<11}{side_s:<6}{price_s:>8}{c.qty:>9.1f}{c.delta_ev:>11.3f}{c.g_after:>11.3f}  {c.is_valid}")
            print(f"CHOSEN: {decision.chosen.action_id} ({decision.chosen.mode.value})")

        chosen = decision.chosen
        if chosen.mode is OrderMode.WAIT:
            continue
        n_actions += 1
        if chosen.mode is OrderMode.FAK and chosen.qty > 0:
            # Phase 12B Tranche 1.1 item 6: route through the shared
            # submit->(delay/revalidation)->resolve execution lifecycle
            # (ExecutionSimulator.execute_taker), the same path every
            # other ablation/backtest arm already uses, instead of
            # directly converting the chosen candidate's own
            # pre-evaluation walk estimate into a Fill.
            order_seq += 1
            asks = book_up.asks if chosen.side is Side.UP else book_down.asks
            limit_price = chosen.max_execution_price if chosen.max_execution_price is not None else chosen.price
            revalidation_asks = None
            if exec_cfg.taker_delay_ms > 0:
                matched_ts = decision_ts + exec_cfg.taker_delay_ms / 1000.0
                revalidation_cursor.advance_to(matched_ts)
                revalidation_book = revalidation_book_feed.get_snapshot(round_id, chosen.side)
                revalidation_asks = revalidation_book.asks if revalidation_book is not None else ()
            order, taker_result = sim.execute_taker(
                f"{round_id}-o{order_seq}", chosen.side, chosen.qty, limit_price, asks, decision_ts,
                revalidation_asks=revalidation_asks,
            )
            if taker_result.was_delayed:
                pending.append((order, taker_result))
            elif taker_result.walk.filled_shares > 0:
                portfolio = apply_fill(portfolio, Fill(chosen.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))
        elif chosen.mode is OrderMode.POST_ONLY:
            order = sim.submit_maker_order(chosen.action_id, chosen.side, chosen.qty, chosen.price, decision_ts)
            draw = sim.draw_maker_fill(order, distance_to_touch_ticks=0.0, queue_ahead_shares=0.0, horizon_s=chosen.ttl_s or one_step_cfg.maker_horizon_s)
            if draw.filled:
                fill = Fill(chosen.side, chosen.price, chosen.qty, LiquidityRole.MAKER, 0.0)
                portfolio = apply_fill(portfolio, fill)

    return portfolio, n_actions


def main() -> None:
    n_eval_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    feature_cfg = FeatureConfig()
    fee_config = FeeConfig()
    baseline_cfg = BaselineConfig()
    one_step_cfg = OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0, edge_min=0.5)
    exec_cfg = ExecutionConfig()

    print(f"Training q model on {N_TRAIN_ROUNDS} separate rounds...")
    model = train_q_model(feature_cfg)

    print(f"Evaluating baseline vs one-step controller on {n_eval_rounds} held-out rounds...")
    eval_store = EventStore(":memory:")
    # id_offset=N_TRAIN_ROUNDS: genuinely disjoint from training rounds
    # (Phase 12B audit Addendum A - generate_synthetic_dataset always
    # restarted at index 0 before this, so "held-out" rounds were
    # frequently identical to training rounds).
    eval_results = generate_synthetic_dataset(eval_store, n_rounds=n_eval_rounds, round_length_s=300.0, id_offset=N_TRAIN_ROUNDS)

    baseline_pnls, one_step_pnls = [], []
    baseline_trades, one_step_trades = [], []
    g_min_breaches = 0

    for i, result in enumerate(eval_results):
        baseline_portfolio = run_baseline_round(eval_store, result.round_id, result.p0, baseline_cfg, fee_config, exec_cfg)
        one_step_portfolio, n_actions = run_one_step_round(
            eval_store, result.round_id, result.p0, feature_cfg, model, one_step_cfg, exec_cfg, fee_config, print_sample=(i == 0)
        )

        outcome = result.outcome
        b_pnl = baseline_portfolio.Pi_U if outcome is Side.UP else baseline_portfolio.Pi_D
        o_pnl = one_step_portfolio.Pi_U if outcome is Side.UP else one_step_portfolio.Pi_D
        baseline_pnls.append(b_pnl)
        one_step_pnls.append(o_pnl)
        baseline_trades.append(1 if (baseline_portfolio.U > 0 or baseline_portfolio.D > 0) else 0)
        one_step_trades.append(n_actions)
        if one_step_portfolio.G < one_step_cfg.g_min - 1e-6:
            g_min_breaches += 1

    print("\n=== Baseline vs One-Step Controller (identical replay, Phase 8) ===")
    print(f"Rounds: {n_eval_rounds}")
    print(f"Baseline:  rounds with a position: {sum(baseline_trades)}  |  net PnL: {sum(baseline_pnls):.3f}")
    print(f"One-step:  total actions taken: {sum(one_step_trades)}  |  net PnL: {sum(one_step_pnls):.3f}")
    print(f"G_min violations in final one-step portfolio state (must be 0): {g_min_breaches}")


if __name__ == "__main__":
    main()
