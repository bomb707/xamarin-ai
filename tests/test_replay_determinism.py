"""Roadmap Phase 0 verification: "Replay the same dataset twice and compare
identical decisions." Phase 2 exit gate: "Replayed portfolio ledger matches
recorded fills exactly when fills are known."
"""
from __future__ import annotations

from dataclasses import asdict

import run_synthetic_baseline_replay as rbr

from xamarinbot.baseline.config import BaselineConfig
from xamarinbot.events.store import EventStore
from xamarinbot.journal.schema import AuditRecord, FillRecord, PortfolioStateRecord, SettlementRecord
from xamarinbot.journal.writer import JournalWriter
from devtools.synthetic.rounds import generate_synthetic_dataset


def _run_once(n_rounds: int = 6):
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)
    cfg = BaselineConfig()
    journal = JournalWriter(":memory:")
    for result in results:
        rbr.run_round(store, result.round_id, result.outcome, cfg, journal)
    return journal


def _dump(journal: JournalWriter, record_cls) -> list[dict]:
    return [asdict(r) for r in journal.read(record_cls)]


def test_replaying_same_synthetic_dataset_twice_yields_identical_journal():
    j1 = _run_once()
    j2 = _run_once()

    for cls in (AuditRecord, FillRecord, PortfolioStateRecord, SettlementRecord):
        assert _dump(j1, cls) == _dump(j2, cls), f"{cls.__name__} rows differ between replays"


def test_synthetic_dataset_generation_itself_is_deterministic():
    s1 = EventStore(":memory:")
    r1 = generate_synthetic_dataset(s1, n_rounds=3)
    s2 = EventStore(":memory:")
    r2 = generate_synthetic_dataset(s2, n_rounds=3)

    assert [x.outcome for x in r1] == [x.outcome for x in r2]
    assert [x.final_reference for x in r1] == [x.final_reference for x in r2]
    for rid in [x.round_id for x in r1]:
        payloads1 = [(e.event_type, e.payload) for e in s1.all_events(rid)]
        payloads2 = [(e.event_type, e.payload) for e in s2.all_events(rid)]
        assert payloads1 == payloads2


def test_replayed_portfolio_ledger_matches_settlement_reconciliation():
    """Settlement realized_pnl must equal Pi_U or Pi_D of the final
    journaled PortfolioState for that round, per the outcome."""
    journal = _run_once(n_rounds=10)
    for settlement in journal.read(SettlementRecord):
        states = journal.read(PortfolioStateRecord, settlement.round_id)
        assert states, f"no PortfolioState journaled for {settlement.round_id} despite a settlement"
        final = states[-1]
        expected = final.Pi_U if settlement.outcome == "UP" else final.Pi_D
        assert abs(settlement.realized_pnl - expected) < 1e-9
        expected_payout = final.U if settlement.outcome == "UP" else final.D
        assert abs(settlement.payout - expected_payout) < 1e-9
