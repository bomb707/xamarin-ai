"""Data provenance (Phase 12C.1 item 2).

One canonical answer to "where did this data actually come from?", carried by
every recorder, replay, shadow and model dataset in the system.

The invariant this exists to make structurally true:

    Production/live code can never accidentally consume synthetic data.

Before this module, that was a matter of reading import statements carefully:
`replay/feeds.py` (a genuine deterministic replay adapter) and
`synthetic/rounds.py` (a genuine data fabricator) were both reachable from the
same `EventStore`, and nothing downstream could tell which had populated it. A
`ShadowRunner` handed a synthetic store produced exactly the same shaped
result object as one handed a real capture.

Fail-closed by design
---------------------
`DataProvenance.SYNTHETIC_TEST` is the DEFAULT for a bare `EventStore`. That is
deliberate and is the whole safety property: an unlabelled store is treated as
fabricated until something proves otherwise, so forgetting to set provenance
downgrades a run to "test only" rather than silently promoting fabricated data
to production evidence.
"""
from __future__ import annotations

from enum import Enum


class DataProvenance(str, Enum):
    """Where a dataset's observations actually came from."""

    #: Observed from the live venue in real time by the recorder.
    REAL_LIVE = "REAL_LIVE"
    #: Replayed from a real capture. Every observation is a real one that was
    #: genuinely received; only the clock is replayed.
    REAL_REPLAY = "REAL_REPLAY"
    #: Fabricated by a generator. Useful for unit/property tests and for
    #: proving a pipeline runs end to end. Says nothing whatsoever about
    #: market behaviour, edge, or profitability.
    SYNTHETIC_TEST = "SYNTHETIC_TEST"

    @property
    def is_real(self) -> bool:
        return self in (DataProvenance.REAL_LIVE, DataProvenance.REAL_REPLAY)

    @property
    def is_synthetic(self) -> bool:
        return self is DataProvenance.SYNTHETIC_TEST


class SyntheticDataRefused(RuntimeError):
    """Raised when synthetic data reaches a path that must only ever see real
    observations - production execution, or any economic evaluation whose
    numbers a human might act on."""


def require_real(
    provenance: DataProvenance,
    context: str,
    *,
    allow_synthetic: bool = False,
) -> None:
    """Gate a production / economic-evaluation path on real data.

    `allow_synthetic=True` is the explicit test-only escape hatch item 2
    permits. It must be passed at the CALL SITE by something that has
    deliberately opted in - a unit test, or a `scripts/dev_synthetic/` demo -
    never defaulted on by a library.
    """
    if provenance.is_real or allow_synthetic:
        return
    raise SyntheticDataRefused(
        f"{context}: refusing to run on {provenance.value} data. This path "
        "reports results a human may act on, so it accepts REAL_LIVE or "
        "REAL_REPLAY only. Pass allow_synthetic=True explicitly if this is a "
        "test or a dev_synthetic demo."
    )


def describe(provenance: DataProvenance) -> str:
    """One-line banner for reports. Item 2: reports must print provenance
    prominently, so a reader can never mistake a synthetic run for evidence."""
    if provenance is DataProvenance.REAL_LIVE:
        return "REAL_LIVE - observed live from the venue"
    if provenance is DataProvenance.REAL_REPLAY:
        return "REAL_REPLAY - replayed from a real capture"
    return (
        "SYNTHETIC_TEST - FABRICATED DATA. Structural/pipeline evidence only; "
        "any PnL, edge or win-rate figure below is meaningless."
    )
