#!/usr/bin/env python3
"""Phase 10 end-to-end demo: builds a GapRegime transition model from a
causal replay of the Phase 6 regime classifier over a training dataset,
then runs MPCController against OneStepController on held-out rounds,
reporting how often the lookahead changes the chosen action, the latency
benchmark, and the fallback rate.

Uses a small nonzero `churn_penalty` (Phase 8's SS18-motivated parameter)
deliberately: under zero churn/time cost, "buy N shares now" and "buy N/2
now, N/2 later" have provably identical total EV (no time-discounting in
this V1 model), so MPC's lookahead has nothing to prefer between them. A
nonzero churn cost is exactly what the Phase 10 exit gate names -
"measurable timing improvement after churn/latency costs" - and is what
makes consolidating into fewer actions actually score better. See
docs/PHASE_STATUS.md for the full finding.

Usage: python scripts/run_mpc_controller_demo.py [n_eval_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.feeds.mock import MockBookFeed, MockFeedCursor
from xamarinbot.model.calibrated import fit_calibrated_model
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG, design_vector
from xamarinbot.model.walkforward import time_ordered_split
from xamarinbot.mpc.config import MPCConfig
from xamarinbot.mpc.controller import MPCController
from xamarinbot.mpc.scenario import build_transition_model
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.reports.mpc_report import build_latency_benchmark, format_latency_benchmark
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


def train_transition_model(feature_cfg: FeatureConfig):
    """"Estimate transition probabilities from historical state
    transitions" (Roadmap Phase 10) - here, a causal replay of Phase 6's
    RegimeClassifier over a training dataset stands in for real history."""
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=N_TRAIN_ROUNDS)
    all_transitions = []
    for result in results:
        events = store.all_events(result.round_id)
        clock = ReplayClock(store, result.round_id)
        classifier = RegimeClassifier(round_id=result.round_id)
        for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
            fv = compute(events, result.round_id, decision_ts, result.p0, feature_cfg)
            if isinstance(fv, FeatureVector):
                classifier.observe(fv)
        all_transitions.extend(classifier.transitions)
    return build_transition_model(all_transitions), all_transitions


def main() -> None:
    n_eval_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    feature_cfg = FeatureConfig()
    fee_config = FeeConfig()
    exec_cfg = ExecutionConfig()
    # churn_penalty is small relative to typical candidate EV here (median
    # ~1.0 in this dataset) - large enough to matter for tie-breaking
    # between similar options, not large enough to suppress trading
    # entirely (an earlier value of 5.0 did exactly that - see
    # docs/PHASE_STATUS.md).
    one_step_cfg = OneStepConfig(g_min=-200.0, spend_cap=400.0, position_limit=400.0, edge_min=0.0, churn_penalty=0.3)
    mpc_cfg = MPCConfig(horizon_steps=2, transition_top_k=2, time_budget_ms=50.0)

    print(f"Training q model on {N_TRAIN_ROUNDS} separate rounds...")
    model = train_q_model(feature_cfg)

    print(f"Estimating GapRegime transition model from {N_TRAIN_ROUNDS} separate rounds...")
    transition_model, transitions = train_transition_model(feature_cfg)
    print(f"Observed transitions: {len(transitions)}  |  states with outgoing data: {len(transition_model.probabilities)}")
    for from_state, row in list(transition_model.probabilities.items())[:3]:
        top = sorted(row.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {from_state.value}: " + ", ".join(f"{s.value}={p:.2f}" for s, p in top))

    one_step = OneStepController(one_step_cfg, exec_cfg, fee_config)
    mpc = MPCController(mpc_cfg, transition_model, one_step)

    print(f"\nEvaluating MPC vs one-step on {n_eval_rounds} held-out rounds...")
    eval_store = EventStore(":memory:")
    # id_offset=N_TRAIN_ROUNDS: disjoint from both the q-model and
    # transition-model training pools, which both span [0, N_TRAIN_ROUNDS)
    # (Phase 12B audit Addendum A).
    eval_results = generate_synthetic_dataset(eval_store, n_rounds=n_eval_rounds, id_offset=N_TRAIN_ROUNDS)

    n_decisions = 0
    n_differs = 0
    elapsed_samples: list[float] = []
    fallback_flags: list[bool] = []

    for result in eval_results:
        events = eval_store.all_events(result.round_id)
        clock = ReplayClock(eval_store, result.round_id)
        cursor = MockFeedCursor(eval_store, result.round_id, preloaded=events)
        book_feed = MockBookFeed(cursor)
        regime_clf = RegimeClassifier(round_id=result.round_id)
        portfolio = PortfolioState()

        market_config = next(e.payload for e in events if e.event_type.value == "MARKET_CONFIG")
        tick_size = market_config["tick_size"]

        for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
            cursor.advance_to(decision_ts)
            fv = compute(events, result.round_id, decision_ts, result.p0, feature_cfg)
            if not isinstance(fv, FeatureVector):
                continue
            snapshot = regime_clf.observe(fv)
            book_up = book_feed.get_snapshot(result.round_id, Side.UP)
            book_down = book_feed.get_snapshot(result.round_id, Side.DOWN)
            vec = design_vector(fv, COMBINED_LEAD_LAG)
            q = model.predict_proba(vec) if vec is not None else 0.5

            mpc_decision = mpc.decide(result.round_id, decision_ts, portfolio, q, snapshot.state, book_up, book_down, tick_size, is_fresh=True)
            one_step_decision = one_step.decide(result.round_id, decision_ts, portfolio, q, snapshot.permitted_actions, book_up, book_down, tick_size, is_fresh=True)

            n_decisions += 1
            if mpc_decision.chosen.action_id != one_step_decision.chosen.action_id:
                n_differs += 1
            elapsed_samples.append(mpc_decision.elapsed_ms)
            fallback_flags.append(mpc_decision.used_fallback)
            # portfolio deliberately not advanced here - this demo measures
            # decision-quality/latency, not a full backtest (see Phase 8/9's
            # demos for that).

    differ_rate = (n_differs / n_decisions) if n_decisions else 0.0
    print(f"\nDecisions compared: {n_decisions}  |  MPC chose differently than one-step: {n_differs} ({differ_rate:.1%})")
    print()
    print(format_latency_benchmark(build_latency_benchmark(elapsed_samples, fallback_flags)))


if __name__ == "__main__":
    main()
