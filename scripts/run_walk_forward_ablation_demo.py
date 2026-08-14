#!/usr/bin/env python3
"""Phase 11 end-to-end demo: runs the 8 mandatory ablations (Strategy doc
SS20.1) across rolling walk-forward windows, with bootstrap confidence
intervals on realized PnL and a parameter sensitivity/stability sweep on
`edge_min` across those same windows.

Ablations 6-8 (portfolio repair + taker-only / cancel-replace / MPC) are
expected to show zero non-wait actions on this synthetic dataset - two
distinct, verified findings, not bugs:
  - #6 (taker-only): this dataset's book-depth-sized taker quantities all
    breach the g_min=-100 risk floor, while maker's small fixed clip size
    (20 shares) does not, so taker-only execution has nothing it can size
    down to and trade.
  - #7/#8 (cancel-replace / MPC, both with OrderSupervisor): maker orders
    get registered but are cancelled via REGIME_FLIP before they ever
    reach TTL resolution, reproducing Phase 9's own documented finding
    (regime flips faster than any heartbeat can outrun) in this context.
See docs/PHASE_STATUS.md for the full diagnostic trail.

Usage: python scripts/run_walk_forward_ablation_demo.py [n_rounds]
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
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG
from xamarinbot.model.logistic import fit_logistic_regression
from xamarinbot.mpc.scenario import build_transition_model
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.portfolio.state import FeeConfig
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.reports.walkforward_report import format_ablation_matrix, format_sensitivity, format_stability, summarize_ablation
from xamarinbot.synthetic.rounds import generate_synthetic_dataset
from xamarinbot.walkforward.ablations import MANDATORY_ABLATIONS, run_ablation_round
from xamarinbot.walkforward.sensitivity import parameter_stability_across_windows, sweep_parameter
from xamarinbot.walkforward.windows import rolling_windows

HEARTBEAT_S = 10.0
N_TRAIN_ROUNDS = 15


def train_q_model(feature_cfg: FeatureConfig):
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=N_TRAIN_ROUNDS)
    by_fs = build_examples_multi(store, results, feature_cfg, [COMBINED_LEAD_LAG], heartbeat_s=HEARTBEAT_S)
    examples = by_fs[COMBINED_LEAD_LAG.name]
    return fit_logistic_regression([e.x for e in examples], [e.y for e in examples], COMBINED_LEAD_LAG.name, COMBINED_LEAD_LAG.column_names)


def train_transition_model(feature_cfg: FeatureConfig):
    """Same causal-replay-of-RegimeClassifier approach as Phase 10's demo
    (run_mpc_controller_demo.py) - real transition history doesn't exist
    yet, so a training-round replay stands in for it."""
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
    return build_transition_model(all_transitions)


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    feature_cfg = FeatureConfig()
    fee_config = FeeConfig()
    exec_cfg = ExecutionConfig()

    print(f"Training q model on {N_TRAIN_ROUNDS} separate rounds...")
    model = train_q_model(feature_cfg)
    print(f"Estimating GapRegime transition model from {N_TRAIN_ROUNDS} separate rounds...")
    transition_model = train_transition_model(feature_cfg)

    print(f"Generating {n_rounds}-round synthetic evaluation dataset (SYNTHETIC - not real Polymarket history)...")
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)

    print(f"\nRunning {len(MANDATORY_ABLATIONS)} mandatory ablations (SS20.1) across all {n_rounds} rounds...")
    summaries = []
    for spec in MANDATORY_ABLATIONS:
        round_results = [
            run_ablation_round(spec, store, r.round_id, r.p0, r.outcome, feature_cfg, fee_config, exec_cfg, model, transition_model)
            for r in results
        ]
        summaries.append(summarize_ablation(spec.name, spec.description, round_results))
    print()
    print(format_ablation_matrix(summaries))

    print("\nBuilding rolling walk-forward windows (n_train=4, n_validate=3, n_test=3)...")
    round_ids = [r.round_id for r in results]
    windows = rolling_windows(round_ids, n_train=4, n_validate=3, n_test=3)
    print(f"Windows: {len(windows)}")

    print("\nSweeping edge_min on ablation #5's config, evaluated on window 0's validate rounds only...")
    base_spec = next(s for s in MANDATORY_ABLATIONS if s.name == "5_lead_lag_clob_no_repair")
    if windows:
        validate_rounds = [r for r in results if r.round_id in set(windows[0].validate_round_ids)]
        sens = sweep_parameter("edge_min", [0.0, 0.5, 2.0, 10.0], base_spec.one_step_cfg, base_spec.feature_set, store, validate_rounds, feature_cfg, fee_config, exec_cfg, model)
        print()
        print(format_sensitivity(sens))

        print("\nChecking parameter stability across all windows (validate rounds only per window)...")
        stab = parameter_stability_across_windows("edge_min", [0.0, 0.5, 2.0, 10.0], base_spec.one_step_cfg, base_spec.feature_set, windows, store, results, feature_cfg, fee_config, exec_cfg, model)
        print()
        print(format_stability(stab))
    else:
        print("Not enough rounds to form a single walk-forward window at this size; skipping sensitivity/stability.")


if __name__ == "__main__":
    main()
