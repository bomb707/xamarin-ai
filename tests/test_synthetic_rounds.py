"""Phase 12B Tranche 1 regression test (Addendum A): `generate_synthetic_dataset`
must never silently generate identical "training" and "evaluation" data
under different labels. Every prior call site restarted round numbering
at index 0 on every call, so two separate stores' round_id="synthetic-round-0000"
were literally the same simulated market path - this pins down the fix
(`id_offset`) so that regression can't reappear silently."""
from __future__ import annotations

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.synthetic.rounds import generate_synthetic_dataset


def test_default_id_offset_reproduces_prior_behavior():
    """id_offset=0 (the default) must be unchanged from before this fix -
    every existing call site that doesn't pass id_offset must keep working
    identically."""
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=3)
    assert [r.round_id for r in results] == ["synthetic-round-0000", "synthetic-round-0001", "synthetic-round-0002"]


def test_disjoint_id_offset_produces_disjoint_round_ids():
    train_store = EventStore(":memory:")
    train_results = generate_synthetic_dataset(train_store, n_rounds=5, id_offset=0)

    eval_store = EventStore(":memory:")
    eval_results = generate_synthetic_dataset(eval_store, n_rounds=3, id_offset=5)

    train_ids = {r.round_id for r in train_results}
    eval_ids = {r.round_id for r in eval_results}
    assert train_ids.isdisjoint(eval_ids)


def test_disjoint_id_offset_produces_genuinely_different_market_content():
    """The regression this test guards against: two separate calls with
    overlapping index ranges (the old default behavior) produce
    byte-identical underlying market paths, not just different labels.
    A correct fix must produce genuinely different content, not just
    different round_id strings for the same simulated path."""
    train_store = EventStore(":memory:")
    generate_synthetic_dataset(train_store, n_rounds=1, id_offset=0)
    train_spot = [e for e in train_store.all_events("synthetic-round-0000") if e.event_type is EventType.SPOT]

    eval_store = EventStore(":memory:")
    generate_synthetic_dataset(eval_store, n_rounds=1, id_offset=1)
    eval_spot = [e for e in eval_store.all_events("synthetic-round-0001") if e.event_type is EventType.SPOT]

    assert train_spot[0].payload["value"] != eval_spot[0].payload["value"]


def test_overlapping_id_offset_ranges_do_reproduce_identical_content():
    """Documents the exact leakage mechanism the fix addresses: with the
    *same* id_offset (the old, buggy default for every call site), two
    separate stores generate byte-identical content for the same index -
    this is not hypothetical, it's what every pre-fix call site did."""
    store_a = EventStore(":memory:")
    generate_synthetic_dataset(store_a, n_rounds=1, id_offset=0)
    spot_a = [e for e in store_a.all_events("synthetic-round-0000") if e.event_type is EventType.SPOT]

    store_b = EventStore(":memory:")
    generate_synthetic_dataset(store_b, n_rounds=1, id_offset=0)
    spot_b = [e for e in store_b.all_events("synthetic-round-0000") if e.event_type is EventType.SPOT]

    assert [e.payload["value"] for e in spot_a] == [e.payload["value"] for e in spot_b]


def test_walk_forward_demo_pattern_train_and_eval_are_disjoint():
    """Reproduces the exact pattern from scripts/run_walk_forward_ablation_demo.py
    and friends (train pool then eval pool) and asserts the fix holds
    end-to-end, not just at the unit level."""
    n_train = 15
    train_store = EventStore(":memory:")
    train_results = generate_synthetic_dataset(train_store, n_rounds=n_train, id_offset=0)

    eval_store = EventStore(":memory:")
    eval_results = generate_synthetic_dataset(eval_store, n_rounds=6, id_offset=n_train)

    assert {r.round_id for r in train_results}.isdisjoint({r.round_id for r in eval_results})
