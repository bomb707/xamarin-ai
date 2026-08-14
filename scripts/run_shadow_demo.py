#!/usr/bin/env python3
"""Phase 12 end-to-end demo: runs ShadowRunner (Phase 12) over several
rounds under a strictly recv_ts-gated "live" view, with a simulated feed
outage injected mid-round (24/7 reconnect stability) and a very tight
decision deadline on one round (SS21 Latency gate fallback), then compares
every shadow decision against offline replay for the same timestamps
(live-vs-replay parity) and prints the daily shadow report + parity
report.

No real orders are ever submitted - this is Phase 1's same pluggable
mock/replay feeds, just consumed under the stricter live-arrival gate
instead of Phase 2's default.

Expect the parity report to show shadow near-permanently at WAIT: this
synthetic generator ticks on whole seconds, exactly matching
`FeatureConfig.canonical_horizon_s` (1.0s), so under the strict recv_ts
gate the "now" and "1-horizon-ago" spot lookups collapse onto the same
tick at every whole-second decision point, zeroing Z_spot and locking
`spot_direction` at FLAT for the whole round (regime rule #1: any FLAT
leg -> WAIT). Confirmed root cause, not a bug in the live-arrival gate
itself - see docs/PHASE_STATUS.md's Phase 12 finding for the full trace.

Usage: python scripts/run_shadow_demo.py [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.calibrated import fit_calibrated_model
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG
from xamarinbot.model.walkforward import time_ordered_split
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.portfolio.state import FeeConfig
from xamarinbot.reports.shadow_report import build_daily_shadow_report, format_daily_shadow_report, format_parity_report
from xamarinbot.shadow.config import ShadowConfig
from xamarinbot.shadow.parity import compare_live_vs_replay
from xamarinbot.shadow.runner import FaultInjector, ShadowRunner
from xamarinbot.synthetic.rounds import generate_synthetic_dataset

HEARTBEAT_S = 10.0
N_TRAIN_ROUNDS = 15


def train_q_model(feature_cfg: FeatureConfig):
    # Phase 12B audit item 5/C: fit on train, calibrate (Platt) on a
    # disjoint validation slice - the controller must consume q_calibrated,
    # never the raw logistic score, per Phase 5's own exit gate.
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=N_TRAIN_ROUNDS)
    by_fs = build_examples_multi(store, results, feature_cfg, [COMBINED_LEAD_LAG], heartbeat_s=HEARTBEAT_S)
    examples = by_fs[COMBINED_LEAD_LAG.name]
    split = time_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    return fit_calibrated_model(split.train, split.validation, COMBINED_LEAD_LAG)


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    feature_cfg = FeatureConfig()
    fee_config = FeeConfig()
    exec_cfg = ExecutionConfig()
    one_step_cfg = OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0, edge_min=0.0)

    print(f"Training q model on {N_TRAIN_ROUNDS} separate rounds...")
    model = train_q_model(feature_cfg)

    print(f"Generating {n_rounds}-round synthetic evaluation dataset (SYNTHETIC - not a real live feed)...")
    store = EventStore(":memory:")
    # id_offset=N_TRAIN_ROUNDS: disjoint from training rounds (Phase 12B
    # audit Addendum A).
    results = generate_synthetic_dataset(store, n_rounds=n_rounds, id_offset=N_TRAIN_ROUNDS)

    print("\nRunning ShadowRunner over each round under the recv_ts-gated live view...")
    round_results = []
    parity_reports = []
    for i, r in enumerate(results):
        # Round 0: inject a simulated feed outage partway through, to
        # exercise 24/7 reconnect stability rather than just assume it.
        # Round 1: an unrealistically tight deadline, to exercise the SS21
        # latency-gate fallback. Everything else: normal operation.
        fault = FaultInjector()
        cfg = ShadowConfig(decision_deadline_ms=50.0)
        if i == 0:
            probe = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, model, COMBINED_LEAD_LAG, cfg)
            probe_result = probe.run()
            all_ts = [rec.decision_ts for rec in probe_result.records]
            mid = len(all_ts) // 2
            fault = FaultInjector(disconnect_at=frozenset(all_ts[mid : mid + 3]))
        elif i == 1:
            cfg = ShadowConfig(decision_deadline_ms=0.001)

        runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, model, COMBINED_LEAD_LAG, cfg, fault=fault)
        result = runner.run()
        round_results.append(result)

        parity_reports.append(
            compare_live_vs_replay(result.records, store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, model, COMBINED_LEAD_LAG)
        )

    print()
    print(format_daily_shadow_report(build_daily_shadow_report(round_results)))
    print()
    print(format_parity_report(parity_reports))


if __name__ == "__main__":
    main()
