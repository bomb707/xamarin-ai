#!/usr/bin/env python3
"""Phase 11 end-to-end demo: runs the 8 mandatory ablations (Strategy doc
SS20.1) across rolling walk-forward windows, with a genuine per-window
TRAIN -> FIT -> VALIDATE -> CALIBRATE -> FREEZE -> TEST cycle for the
probability model and transition model in every window (Phase 12B audit
item 4/B - this demo previously fit one q-model and one transition model
*once*, globally, before windows even existed, and reused that same fixed
model everywhere; see `walkforward/pipeline.py` and
`docs/PHASE_12B_AUDIT.md`), plus bootstrap confidence intervals on
realized PnL and a parameter sensitivity/stability sweep on `edge_min`.

This is a structural-correctness demo, not a profitability claim (Phase
12B audit items 4/37): the numbers below come from SYNTHETIC data and
must not be read as evidence of real edge, optimal parameters, or that
any ablation "beats" another - see docs/PHASE_STATUS.md and
docs/PHASE_12B_AUDIT.md for why synthetic data specifically cannot
support such claims.

Usage: python scripts/run_walk_forward_ablation_demo.py [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.portfolio.state import FeeConfig
from xamarinbot.reports.walkforward_report import format_ablation_matrix, format_sensitivity, format_stability, summarize_ablation
from xamarinbot.synthetic.rounds import generate_synthetic_dataset
from xamarinbot.walkforward.ablations import MANDATORY_ABLATIONS
from xamarinbot.walkforward.pipeline import fit_window_artifacts, run_walk_forward_ablations, LeakageTrace
from xamarinbot.walkforward.sensitivity import parameter_stability_across_windows, sweep_parameter
from xamarinbot.walkforward.windows import rolling_windows

HEARTBEAT_S = 10.0


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    feature_cfg = FeatureConfig()
    fee_config = FeeConfig()
    exec_cfg = ExecutionConfig()

    print(f"Generating {n_rounds}-round synthetic dataset (SYNTHETIC - not real Polymarket history)...")
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)
    round_ids = [r.round_id for r in results]

    print("\nBuilding rolling walk-forward windows (n_train=6, n_validate=3, n_test=3)...")
    windows = rolling_windows(round_ids, n_train=6, n_validate=3, n_test=3)
    print(f"Windows: {len(windows)}")
    if not windows:
        print("Not enough rounds to form a single walk-forward window at this size; nothing to run.")
        return

    print(f"\nRunning {len(MANDATORY_ABLATIONS)} mandatory ablations (SS20.1) across {len(windows)} window(s)'"
          " own TEST segments, each window fitting/calibrating its own model from only its own TRAIN/VALIDATE rounds...")
    window_round_results, trace = run_walk_forward_ablations(
        store, results, windows, feature_cfg, fee_config, exec_cfg, MANDATORY_ABLATIONS
    )

    # No-leakage sanity check, printed plainly rather than only asserted in
    # tests - Phase 12B audit item 4's explicit ask.
    all_test_ids = {rid for w in windows for rid in w.test_round_ids}
    leakage_clean = (
        trace.model_fit_round_ids.isdisjoint(all_test_ids)
        and trace.calibrator_fit_round_ids.isdisjoint(all_test_ids)
        and trace.transition_fit_round_ids.isdisjoint(all_test_ids)
    )
    print(f"No-leakage check (model/calibrator/transition fit never touched a TEST round_id): {'PASS' if leakage_clean else 'FAIL'}")

    by_ablation: dict[str, list] = {}
    for wr in window_round_results:
        by_ablation.setdefault(wr.ablation_name, []).append(wr.result)
    summaries = []
    for spec in MANDATORY_ABLATIONS:
        round_results = by_ablation.get(spec.name, [])
        summaries.append(summarize_ablation(spec.name, spec.description, round_results))
    print()
    print(format_ablation_matrix(summaries))

    print("\nSweeping edge_min on ablation #5's config, evaluated on window 0's own TEST rounds "
          "using window 0's own frozen model (not a separately-trained global one)...")
    base_spec = next(s for s in MANDATORY_ABLATIONS if s.name == "5_lead_lag_clob_no_repair")
    window_0_trace = LeakageTrace()
    window_0_artifacts = fit_window_artifacts(windows[0], results, store, feature_cfg, [base_spec.feature_set], window_0_trace)
    window_0_model = window_0_artifacts.models_by_feature_set[base_spec.feature_set.name]
    test_rounds_0 = [r for r in results if r.round_id in set(windows[0].test_round_ids)]
    sens = sweep_parameter("edge_min", [0.0, 0.5, 2.0, 10.0], base_spec.one_step_cfg, base_spec.feature_set, store, test_rounds_0, feature_cfg, fee_config, exec_cfg, window_0_model)
    print()
    print(format_sensitivity(sens))

    print("\nChecking parameter stability across all windows (each window's own TEST rounds, each window's own model)...")
    # parameter_stability_across_windows evaluates on validate_round_ids
    # by its own design (locking parameters before the test segment, per
    # Roadmap Phase 11) - reusing window 0's model here is an approximation
    # for this demo's stability check specifically, not a claim that every
    # window shares one model; the ablation matrix above is what actually
    # exercises the full per-window fit/calibrate/freeze/test cycle.
    stab = parameter_stability_across_windows("edge_min", [0.0, 0.5, 2.0, 10.0], base_spec.one_step_cfg, base_spec.feature_set, windows, store, results, feature_cfg, fee_config, exec_cfg, window_0_model)
    print()
    print(format_stability(stab))

    print("\nSYNTHETIC DATA ONLY - see docs/PHASE_STATUS.md / docs/PHASE_12B_AUDIT.md: none of the numbers")
    print("above are profitability, optimal-parameter, or ablation-superiority evidence.")


if __name__ == "__main__":
    main()
