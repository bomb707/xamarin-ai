#!/usr/bin/env python3
"""Phase 6 end-to-end demo: classify the seed regime at every decision
point across the synthetic dataset, journal every transition, and print
the transition-statistics report.

Usage: python scripts/run_regime_classifier_demo.py [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.journal.writer import JournalWriter
from xamarinbot.regime.classifier import RegimeClassifier, to_journal_record
from xamarinbot.reports.regime_report import build_transition_report, format_transition_report
from xamarinbot.synthetic.rounds import generate_synthetic_dataset

HEARTBEAT_S = 5.0


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)
    feature_cfg = FeatureConfig()
    journal = JournalWriter(":memory:")

    snapshot_action_counts: dict[str, int] = {}
    n_snapshots = 0

    for result in results:
        events = store.all_events(result.round_id)
        clock = ReplayClock(store, result.round_id)
        classifier = RegimeClassifier(round_id=result.round_id)

        for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
            fv = compute(events, result.round_id, decision_ts, result.p0, feature_cfg)
            if not isinstance(fv, FeatureVector):
                continue
            snapshot = classifier.observe(fv)
            n_snapshots += 1
            snapshot_action_counts[snapshot.seed_action.value] = snapshot_action_counts.get(snapshot.seed_action.value, 0) + 1

        for transition in classifier.transitions:
            journal.write(to_journal_record(transition))

    print(f"Rounds: {n_rounds}  |  snapshots classified: {n_snapshots}")
    print(f"Seed action distribution across all snapshots: {snapshot_action_counts}")
    print()
    report = build_transition_report(journal)
    print(format_transition_report(report))


if __name__ == "__main__":
    main()
