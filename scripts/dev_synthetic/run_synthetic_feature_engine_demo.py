#!/usr/bin/env python3
"""Phase 4 end-to-end demo: compute the causal feature vector at a
sequence of decision points across the synthetic dataset, journal every
valid FeatureStateRecord (invalid-state points are skipped, not defaulted),
and print the SS22.1 lead-lag empirical table.

Uses a 5s heartbeat rather than every synthetic tick (~1s) - the feature
engine re-derives everything from the full causal event list on every call
(no incremental state, by design - see features/engine.py), so a coarser
cadence keeps this demo fast while still producing a reasonably sized
empirical table. Real-time/replay usage would call compute() at whatever
event-driven-plus-heartbeat cadence Phase 2's ReplayRunner provides.

Usage: python scripts/run_feature_engine_demo.py [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
# `devtools` (the synthetic data fabricator) lives at the repo root,
# deliberately outside the shipped package - see Phase 12C.1 item 4.
sys.path.insert(0, str(_REPO_ROOT))

from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute, to_journal_record
from xamarinbot.features.types import FeatureVector, InvalidFeatureState
from xamarinbot.journal.writer import JournalWriter
from xamarinbot.reports.leadlag_report import build_leadlag_report, format_leadlag_report
from devtools.synthetic.rounds import generate_synthetic_dataset

HEARTBEAT_S = 5.0


def run_round(store: EventStore, round_id: str, cfg: FeatureConfig, journal: JournalWriter, p0: float) -> tuple[int, int]:
    clock = ReplayClock(store, round_id)
    events = store.all_events(round_id)

    n_valid = 0
    n_invalid = 0
    for decision_time in clock.decision_points(heartbeat=HEARTBEAT_S):
        result = compute(events, round_id, decision_time, p0, cfg)
        if isinstance(result, FeatureVector):
            journal.write(to_journal_record(result))
            n_valid += 1
        else:
            assert isinstance(result, InvalidFeatureState)
            n_invalid += 1
    return n_valid, n_invalid


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)

    cfg = FeatureConfig()
    journal = JournalWriter(":memory:")
    outcomes = {r.round_id: r.outcome.value for r in results}

    total_valid = total_invalid = 0
    for r in results:
        n_valid, n_invalid = run_round(store, r.round_id, cfg, journal, r.p0)
        total_valid += n_valid
        total_invalid += n_invalid

    print(f"Rounds: {n_rounds}  |  valid feature points: {total_valid}  |  invalid (skipped): {total_invalid}")
    print()
    report = build_leadlag_report(journal, outcomes)
    print(format_leadlag_report(report))


if __name__ == "__main__":
    main()
