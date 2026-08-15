"""Real freshness from real timestamps (Phase 12C item 10).

`ShadowRunner` previously passed a literal `is_fresh=True` into both the
controller and the order supervisor. That constant made SS21's freshness
gate and SS16's FEED_STALE cancel trigger unreachable: no matter what the
feeds did, every decision claimed a live view. It was survivable while every
input came from a synthetic replay where staleness could not occur; it is
not survivable for a service reading four independent real streams.

Freshness here is computed from the SOURCE timestamp of each required
input - the exchange's stamp for the book, the Chainlink observation time
for the reference and TWAP values, the Binance trade time - never from
local receive time (item 4). Using receive time would report a dead feed as
fresh for as long as our own socket kept delivering anything at all.

Staleness budgets are per-feed because the feeds genuinely differ:

  market book       2.0s   High-rate stream (~130 msg/s measured); two
                           seconds of silence on a live BTC book is already
                           anomalous.
  chainlink ref     5.0s   Published at ~1 Hz; 5s allows a few missed
                           observations before the settlement reference is
                           considered unusable.
  chainlink twap    5.0s   Same cadence, same reasoning.
  binance           5.0s   Same cadence. It is a leading signal, not a
                           settlement source, so this is not made stricter.

A missing input is NOT stale - it is UNKNOWN, and it is treated as at least
as bad as stale. Collapsing the two would let "we never received a single
Chainlink observation" and "the last one was 6 seconds ago" look identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FeedKind(str, Enum):
    BOOK = "book"
    CHAINLINK_REFERENCE = "chainlink_reference"
    CHAINLINK_TWAP_30 = "chainlink_twap_30"
    CHAINLINK_TWAP_60 = "chainlink_twap_60"
    BINANCE = "binance"


class FeedStatus(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    MISSING = "MISSING"   # never observed
    UNUSABLE = "UNUSABLE"  # observed, but the value cannot be used (e.g. gapped book)


@dataclass(frozen=True)
class FreshnessPolicy:
    max_age_s: dict[FeedKind, float] = field(
        default_factory=lambda: {
            FeedKind.BOOK: 2.0,
            FeedKind.CHAINLINK_REFERENCE: 5.0,
            FeedKind.CHAINLINK_TWAP_30: 5.0,
            FeedKind.CHAINLINK_TWAP_60: 5.0,
            FeedKind.BINANCE: 5.0,
        }
    )
    #: Inputs that must be FRESH for a new ALPHA decision to be allowed.
    #: The 30s TWAP is deliberately NOT required: markets configured for a
    #: 60s window do not need it, and requiring both would block decisions
    #: on a feed the round does not use.
    required: frozenset[FeedKind] = frozenset(
        {
            FeedKind.BOOK,
            FeedKind.CHAINLINK_REFERENCE,
            FeedKind.CHAINLINK_TWAP_60,
            FeedKind.BINANCE,
        }
    )
    #: A source timestamp this far ahead of local time means the clocks
    #: disagree; the observation is not trusted as "very fresh".
    max_clock_skew_s: float = 2.0

    def limit_for(self, kind: FeedKind) -> float:
        return self.max_age_s.get(kind, 5.0)


@dataclass(frozen=True)
class FeedFreshness:
    kind: FeedKind
    status: FeedStatus
    age_s: float | None
    source_ts: float | None
    limit_s: float
    detail: str = ""

    @property
    def is_fresh(self) -> bool:
        return self.status is FeedStatus.FRESH


@dataclass(frozen=True)
class FreshnessReport:
    """The freshness of every input at one decision instant."""

    decision_ts: float
    feeds: dict[FeedKind, FeedFreshness]
    required: frozenset[FeedKind]

    @property
    def is_fresh(self) -> bool:
        """True only when EVERY required input is fresh. This is the value
        that replaces the old hardcoded `True`."""
        return all(self.feeds[k].is_fresh for k in self.required if k in self.feeds) and all(
            k in self.feeds for k in self.required
        )

    @property
    def failures(self) -> list[FeedFreshness]:
        out = []
        for k in self.required:
            f = self.feeds.get(k)
            if f is None:
                out.append(FeedFreshness(k, FeedStatus.MISSING, None, None, 0.0, "feed not reported"))
            elif not f.is_fresh:
                out.append(f)
        return out

    @property
    def reason(self) -> str | None:
        """A single explicit reason string for the journal / decision record,
        or None when everything required is fresh."""
        fails = self.failures
        if not fails:
            return None
        return "; ".join(
            f"{f.kind.value}:{f.status.value}"
            + (f" age={f.age_s:.2f}s>{f.limit_s:.2f}s" if f.age_s is not None else "")
            for f in fails
        )

    def as_dict(self) -> dict:
        return {
            "decision_ts": self.decision_ts,
            "is_fresh": self.is_fresh,
            "reason": self.reason,
            "feeds": {
                k.value: {
                    "status": v.status.value,
                    "age_s": v.age_s,
                    "source_ts": v.source_ts,
                    "limit_s": v.limit_s,
                    "detail": v.detail,
                }
                for k, v in self.feeds.items()
            },
        }


def evaluate_feed(
    kind: FeedKind,
    source_ts: float | None,
    now: float,
    policy: FreshnessPolicy,
    *,
    usable: bool = True,
    detail: str = "",
) -> FeedFreshness:
    limit = policy.limit_for(kind)
    if source_ts is None:
        return FeedFreshness(kind, FeedStatus.MISSING, None, None, limit, detail or "never observed")
    age = now - source_ts
    if not usable:
        return FeedFreshness(kind, FeedStatus.UNUSABLE, age, source_ts, limit, detail or "observed but unusable")
    if age < -policy.max_clock_skew_s:
        return FeedFreshness(
            kind, FeedStatus.UNUSABLE, age, source_ts, limit,
            f"source timestamp {-age:.2f}s ahead of local clock; skew exceeds "
            f"{policy.max_clock_skew_s}s",
        )
    status = FeedStatus.FRESH if age <= limit else FeedStatus.STALE
    return FeedFreshness(kind, status, age, source_ts, limit, detail)


def evaluate_freshness(
    now: float,
    source_timestamps: dict[FeedKind, float | None],
    policy: FreshnessPolicy | None = None,
    unusable: frozenset[FeedKind] = frozenset(),
    details: dict[FeedKind, str] | None = None,
) -> FreshnessReport:
    """Build the full report from each feed's latest SOURCE timestamp."""
    policy = policy or FreshnessPolicy()
    details = details or {}
    feeds = {
        kind: evaluate_feed(
            kind, ts, now, policy,
            usable=kind not in unusable,
            detail=details.get(kind, ""),
        )
        for kind, ts in source_timestamps.items()
    }
    for kind in policy.required:
        feeds.setdefault(
            kind,
            FeedFreshness(kind, FeedStatus.MISSING, None, None, policy.limit_for(kind), "never observed"),
        )
    return FreshnessReport(decision_ts=now, feeds=feeds, required=policy.required)
