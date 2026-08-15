"""Provenance-tagged round label (Phase 12C.1 items 1, 2).

`walkforward/pipeline.py` and `model/dataset.py` used to import a
`SyntheticRoundResult` dataclass from the synthetic generator purely to have
a typed carrier for "this round's id, its opening reference price, and how it
settled". That single import was the only thing preventing a clean
`src/xamarinbot/** may not import fabricated-data machinery` boundary - two
genuinely production modules had a hard dependency on the data fabricator.

`RoundLabel` is that carrier, made neutral and provenance-tagged. A real
projected capture and a synthetic generator both produce one; consumers can
tell them apart, which is exactly what they could not do before.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.portfolio.state import Side
from xamarinbot.provenance import DataProvenance


@dataclass(frozen=True)
class RoundLabel:
    """One round's identity, opening reference and settled outcome."""

    round_id: str
    #: Settlement reference price at the round's open. For a real BTC
    #: 5-minute market this is the declared settlement basis (Chainlink
    #: TWAP-60) observed at or before `start_ts` - never a default.
    p0: float
    #: Settlement reference price at the round's close, on the same basis.
    final_reference: float
    outcome: Side
    provenance: DataProvenance = DataProvenance.SYNTHETIC_TEST

    @property
    def is_real(self) -> bool:
        return self.provenance.is_real
