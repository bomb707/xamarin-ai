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
    window_round_results, traces = run_walk_forward_ablations(
        store, results, windows, feature_cfg, fee_config, exec_cfg, MANDATORY_ABLATIONS
    )

    # No-leakage sanity check, printed plainly rather than only asserted in
    # tests - Phase 12B audit item 4's explicit ask. Checked PER WINDOW
    # (Phase 12B Tranche 1.1 item 4): a round that was one window's TEST
    # round is explicitly allowed to become a LATER window's TRAIN/VALIDATE
    # round (windows.py's own documented reuse), so pooling every window's
    # fit/calibrator/transition round_ids together and checking disjointness
    # against every window's test round_ids pooled together would flag that
    # legitimate reuse as a false leak. Each window's own trace is checked
    # only against that same window's own train/validate/test round_ids.
    leakage_clean = True
    for window in windows:
        trace = traces.get(window.window_index)
        if trace is None:
            continue
        train_ids = set(window.train_round_ids)
        validate_ids = set(window.validate_round_ids)
        test_ids = set(window.test_round_ids)
        window_clean = (
            train_ids.isdisjoint(validate_ids)
            and train_ids.isdisjoint(test_ids)
            and validate_ids.isdisjoint(test_ids)
            and trace.model_fit_round_ids.isdisjoint(test_ids)
            and trace.calibrator_fit_round_ids.isdisjoint(test_ids)
            and trace.transition_fit_round_ids.isdisjoint(test_ids)
        )
        leakage_clean = leakage_clean and window_clean
    print(f"No-leakage check (per window: TRAIN/VALIDATE/TEST disjoint, model/calibrator/transition fit never touched "
          f"that window's own TEST round_id): {'PASS' if leakage_clean else 'FAIL'}")

    by_ablation: dict[str, list] = {}
    for wr in window_round_results:
        by_ablation.setdefault(wr.ablation_name, []).append(wr.result)
    summaries = []
    for spec in MANDATORY_ABLATIONS:
        round_results = by_ablation.get(spec.name, [])
        summaries.append(summarize_ablation(spec.name, spec.description, round_results))
    print()
    print(format_ablation_matrix(summaries))

    print("\nFitting each window's own frozen calibrated model for the edge_min sensitivity/stability analysis below "
          "(Phase 12B Tranche 1.1 item 3 - never one externally-shared model reused across windows)...")
    base_spec = next(s for s in MANDATORY_ABLATIONS if s.name == "5_lead_lag_clob_no_repair")
    window_artifacts = {}
    for window in windows:
        window_artifacts[window.window_index] = fit_window_artifacts(
            window, results, store, feature_cfg, [base_spec.feature_set], LeakageTrace()
        )

    print("\nSweeping edge_min on ablation #5's config, evaluated on window 0's own VALIDATE rounds "
          "using window 0's own frozen model (Phase 12B Tranche 1.1 item 2 - never window 0's TEST rounds; "
          "parameters must be locked before the test segment is ever touched)...")
    window_0_model = window_artifacts[0].models_by_feature_set[base_spec.feature_set.name]
    validate_rounds_0 = [r for r in results if r.round_id in set(windows[0].validate_round_ids)]
    sens = sweep_parameter("edge_min", [0.0, 0.5, 2.0, 10.0], base_spec.one_step_cfg, base_spec.feature_set, store, validate_rounds_0, feature_cfg, fee_config, exec_cfg, window_0_model)
    print()
    print(format_sensitivity(sens))

    print("\nChecking parameter stability across all windows (each window's own VALIDATE rounds, each window's own "
          "frozen calibrated model - Phase 12B Tranche 1.1 item 3)...")
    stab = parameter_stability_across_windows(
        "edge_min", [0.0, 0.5, 2.0, 10.0], base_spec.one_step_cfg, base_spec.feature_set, windows,
        window_artifacts, store, results, feature_cfg, fee_config, exec_cfg,
    )
    print()
    print(format_stability(stab))

    print("\nSYNTHETIC DATA ONLY - see docs/PHASE_STATUS.md / docs/PHASE_12B_AUDIT.md: none of the numbers")
    print("above are profitability, optimal-parameter, or ablation-superiority evidence.")


if __name__ == "__main__":
    main()
