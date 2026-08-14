"""Lead-lag empirical table (Roadmap SS22.1 "Recommended immediate next
sprint": "Produce the first empirical table: P(UP) and realized edge by gap
bucket x spot direction x CLOB direction x time bucket.")

Each journaled FeatureStateRecord is one observation of (state, eventual
round outcome); P(UP | bucket) is the fraction of observations landing in
that bucket whose round settled UP, and realized_edge is its deviation from
a naive 0.5 prior. "Realized edge" isn't given an exact formula in the
source docs beyond this description - this is the natural reading given
Phase 5 (the q-model, the real fair-value reference) doesn't exist yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.journal.schema import FeatureStateRecord
from xamarinbot.journal.writer import JournalWriter


def _bucket(value: float, size: float) -> float:
    return round(value / size) * size


@dataclass(frozen=True)
class LeadLagBucket:
    gap_bucket_bp: float
    spot_direction: int
    clob_direction: int
    time_regime: str
    n: int
    p_up: float
    realized_edge: float


@dataclass
class LeadLagReport:
    canonical_horizon_s: float
    gap_bucket_size_bp: float
    buckets: list[LeadLagBucket] = field(default_factory=list)


def build_leadlag_report(
    journal: JournalWriter,
    outcomes: dict[str, str],
    canonical_horizon_s: float = 1.0,
    gap_bucket_size_bp: float = 50.0,
) -> LeadLagReport:
    records = journal.read(FeatureStateRecord)
    horizon_key = str(canonical_horizon_s)
    grouped: dict[tuple[float, int, int, str], list[int]] = {}

    for rec in records:
        outcome = outcomes.get(rec.round_id)
        if outcome is None:
            continue
        features = rec.features
        spot_ret = features.get("spot_returns_bp", {}).get(horizon_key)
        clob_dir = features.get("clob_direction", {}).get(horizon_key)
        if spot_ret is None or clob_dir is None:
            continue

        spot_dir = 1 if spot_ret > 0 else (-1 if spot_ret < 0 else 0)
        gap_bucket = _bucket(features["gap_twap_bp"], gap_bucket_size_bp)
        time_regime = features["time_regime"]

        key = (gap_bucket, spot_dir, int(clob_dir), time_regime)
        grouped.setdefault(key, []).append(1 if outcome == "UP" else 0)

    buckets = [
        LeadLagBucket(
            gap_bucket_bp=gap_bucket,
            spot_direction=spot_dir,
            clob_direction=clob_dir,
            time_regime=time_regime,
            n=len(hits),
            p_up=sum(hits) / len(hits),
            realized_edge=(sum(hits) / len(hits)) - 0.5,
        )
        for (gap_bucket, spot_dir, clob_dir, time_regime), hits in sorted(grouped.items())
    ]
    return LeadLagReport(canonical_horizon_s=canonical_horizon_s, gap_bucket_size_bp=gap_bucket_size_bp, buckets=buckets)


def format_leadlag_report(report: LeadLagReport) -> str:
    header = (
        f"=== Lead-Lag Empirical Table "
        f"(canonical horizon={report.canonical_horizon_s}s, gap bucket={report.gap_bucket_size_bp}bp) ==="
    )
    col = f"{'gap_bp':>8} {'spot_dir':>8} {'clob_dir':>8} {'time_regime':>22} {'n':>5} {'P(UP)':>8} {'edge':>8}"
    lines = [header, col]
    for b in report.buckets:
        lines.append(
            f"{b.gap_bucket_bp:>8.0f} {b.spot_direction:>8} {b.clob_direction:>8} {b.time_regime:>22} "
            f"{b.n:>5} {b.p_up:>8.1%} {b.realized_edge:>+8.1%}"
        )
    return "\n".join(lines)
