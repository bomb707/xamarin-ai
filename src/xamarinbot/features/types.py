"""Feature vector and explicit invalid-state types (Roadmap Phase 4
verification: "Missing/stale inputs produce explicit invalid state.")"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TimeRegime(str, Enum):
    """Strategy doc SS6.1 "Soft time regimes" - initial engineering
    regimes, not immutable alpha rules."""

    DISCOVERY = "DISCOVERY"  # 0-60s
    CORE_TRADING = "CORE_TRADING"  # 60-180s
    COMPRESSION = "COMPRESSION"  # 180-240s
    LATE_CONTROLLED = "LATE_CONTROLLED"  # 240-270s
    SETTLEMENT_PROTECTION = "SETTLEMENT_PROTECTION"  # 270-300s


def time_regime_for(t: float) -> TimeRegime:
    if t < 60.0:
        return TimeRegime.DISCOVERY
    if t < 180.0:
        return TimeRegime.CORE_TRADING
    if t < 240.0:
        return TimeRegime.COMPRESSION
    if t < 270.0:
        return TimeRegime.LATE_CONTROLLED
    return TimeRegime.SETTLEMENT_PROTECTION


class InvalidReason(str, Enum):
    MISSING_MARKET_CONFIG = "MISSING_MARKET_CONFIG"
    MISSING_TWAP = "MISSING_TWAP"
    MISSING_SPOT = "MISSING_SPOT"
    MISSING_BOOK = "MISSING_BOOK"
    STALE_TWAP = "STALE_TWAP"
    STALE_SPOT = "STALE_SPOT"
    STALE_BOOK = "STALE_BOOK"
    INSUFFICIENT_VOLATILITY_HISTORY = "INSUFFICIENT_VOLATILITY_HISTORY"


@dataclass(frozen=True)
class InvalidFeatureState:
    """Explicit "we cannot compute features right now" marker - callers
    must not silently substitute a default/zero value for a missing input."""

    round_id: str
    decision_ts: float
    reason: InvalidReason


@dataclass(frozen=True)
class FeatureVector:
    """The causal state vector (Roadmap Phase 4 objective). All fields are
    computable from data with event_time <= decision_ts only."""

    feature_version: str
    round_id: str
    decision_ts: float
    t: float
    tau: float
    time_regime: TimeRegime

    p0: float
    twap: float
    spot: float
    clob_mid: float  # M_t, Polymarket UP midpoint

    # SS5.1 gaps (bp)
    gap_twap_bp: float  # G_T
    gap_spot_bp: float  # G_S
    lead_gap_bp: float  # L

    # SS5.2 modeled TWAP pressure
    twap_pressure_model: float

    # SS6 time/volatility-normalized settlement distance
    realized_vol: float  # sigma_t
    z_gap: float

    # SS9 probability model input: Z_spot, vol-normalized spot momentum at
    # the canonical horizon. SS9 names Z_spot in the logit formula but
    # doesn't define it; this build uses the same vol*sqrt(time)
    # normalization convention as Z_gap for consistency (see
    # features/engine.py). None if the canonical horizon's return isn't
    # available yet (early in the round).
    z_spot: float | None

    # SS7 CLOB features
    clob_log_odds: float  # L_clob(t)
    z_clob: float
    ofi: float
    spread: float

    # SS4/SS7/SS21: fixed-horizon returns, keyed by horizon in seconds
    spot_returns_bp: dict[float, float] = field(default_factory=dict)
    clob_log_odds_returns: dict[float, float] = field(default_factory=dict)
    clob_direction: dict[float, int] = field(default_factory=dict)  # sign(M_t - M_{t-h})


FeatureResult = FeatureVector | InvalidFeatureState
