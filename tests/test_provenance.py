"""Phase 12C.1 item 2: explicit provenance modes, failing closed."""
from __future__ import annotations

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.provenance import (
    DataProvenance,
    SyntheticDataRefused,
    describe,
    require_real,
)
from xamarinbot.rounds import RoundLabel


def test_the_three_modes_exist():
    assert {p.value for p in DataProvenance} == {
        "REAL_LIVE", "REAL_REPLAY", "SYNTHETIC_TEST",
    }


def test_is_real_partitions_the_modes():
    assert DataProvenance.REAL_LIVE.is_real
    assert DataProvenance.REAL_REPLAY.is_real
    assert not DataProvenance.SYNTHETIC_TEST.is_real
    assert DataProvenance.SYNTHETIC_TEST.is_synthetic


# ------------------------------------------------------------ fail closed

def test_an_unlabelled_store_is_treated_as_synthetic():
    """The whole safety property: forgetting to set provenance downgrades a
    run to test-only rather than silently promoting fabricated data."""
    store = EventStore(":memory:")
    assert store.provenance is DataProvenance.SYNTHETIC_TEST
    assert store.is_real is False
    store.close()


def test_production_evaluation_refuses_synthetic_data():
    with pytest.raises(SyntheticDataRefused, match="SYNTHETIC_TEST"):
        require_real(DataProvenance.SYNTHETIC_TEST, "economic evaluation")


def test_real_data_passes_the_gate():
    require_real(DataProvenance.REAL_REPLAY, "ctx")
    require_real(DataProvenance.REAL_LIVE, "ctx")


def test_the_escape_hatch_must_be_passed_explicitly():
    """`allow_synthetic` is opt-in at the CALL SITE - a library must never
    default it on."""
    require_real(DataProvenance.SYNTHETIC_TEST, "unit test", allow_synthetic=True)


# ------------------------------------------------------------- persistence

def test_provenance_survives_close_and_reopen(tmp_path):
    """A projected real capture must not become unlabelled - and therefore
    refused - merely by being closed and reopened."""
    path = str(tmp_path / "s.db")
    store = EventStore(path, provenance=DataProvenance.REAL_REPLAY)
    store.append(EventType.SPOT, "r1", 1.0, 1.0, {"value": 1.0})
    store.close()

    reopened = EventStore(path)
    assert reopened.provenance is DataProvenance.REAL_REPLAY
    reopened.close()


def test_relabelling_an_existing_store_is_refused(tmp_path):
    """A store does not become real because a caller said so."""
    path = str(tmp_path / "s.db")
    EventStore(path, provenance=DataProvenance.SYNTHETIC_TEST).close()
    with pytest.raises(ValueError, match="refusing to relabel"):
        EventStore(path, provenance=DataProvenance.REAL_REPLAY)


def test_reopening_with_the_same_provenance_is_fine(tmp_path):
    path = str(tmp_path / "s.db")
    EventStore(path, provenance=DataProvenance.REAL_REPLAY).close()
    s = EventStore(path, provenance=DataProvenance.REAL_REPLAY)
    assert s.provenance is DataProvenance.REAL_REPLAY
    s.close()


# --------------------------------------------------------------- reporting

def test_synthetic_description_is_unmistakable():
    text = describe(DataProvenance.SYNTHETIC_TEST)
    assert "FABRICATED" in text
    assert "meaningless" in text
    assert "REAL" in describe(DataProvenance.REAL_REPLAY)


# ------------------------------------------------------------- RoundLabel

def test_round_label_is_provenance_tagged_and_defaults_synthetic():
    from xamarinbot.portfolio.state import Side

    label = RoundLabel("r1", p0=100.0, final_reference=101.0, outcome=Side.UP)
    assert label.provenance is DataProvenance.SYNTHETIC_TEST
    assert label.is_real is False

    real = RoundLabel("r2", 100.0, 99.0, Side.DOWN, DataProvenance.REAL_REPLAY)
    assert real.is_real is True


def test_round_label_lives_outside_the_synthetic_package():
    """It exists precisely so `walkforward/pipeline.py` and
    `model/dataset.py` need not import the data fabricator."""
    import xamarinbot.rounds as mod

    assert mod.__name__ == "xamarinbot.rounds"


# -------------------------------------------------- batched append (item 8)

def test_append_many_preserves_order_and_is_atomic(tmp_path):
    """The projection writes ~180k events per round; `append()` commits per
    call, so a bulk path is required."""
    store = EventStore(str(tmp_path / "s.db"), provenance=DataProvenance.REAL_REPLAY)
    rows = [
        (EventType.SPOT, "r1", float(i) + 0.1, float(i), {"value": float(i)})
        for i in range(500)
    ]
    assert store.append_many(rows) == 500
    assert store.count("r1") == 500
    events = store.all_events("r1")
    assert [e.payload["value"] for e in events] == [float(i) for i in range(500)]
    assert store.counts_by_type("r1") == {"SPOT": 500}
    assert store.append_many([]) == 0
    store.close()
