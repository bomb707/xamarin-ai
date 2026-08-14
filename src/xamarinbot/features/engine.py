"""FeatureEngine v1 (Roadmap Phase 4).

`compute()` is a pure, stateless function of a causal event list: it always
re-derives everything from `events` filtered to `event_time <= decision_ts`,
so live and replay computation are the same code path (Roadmap Phase 4
verification: "Feature output invariant between live and replay") and
future events can never leak in even if the caller's `events` list contains
some (verification: "No feature may consume a future timestamp") - the
filter is applied here, not trusted from the caller.
"""
from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import asdict, dataclass

from xamarinbot.events.types import Event, EventType
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.types import (
    FeatureResult,
    FeatureVector,
    InvalidFeatureState,
    InvalidReason,
    time_regime_for,
)
from xamarinbot.journal.schema import FeatureStateRecord

_LOG_ODDS_EPS = 1e-4


def _series(events: list[Event], event_type: EventType, value_key: str = "value") -> list[tuple[float, float]]:
    matches = sorted((e for e in events if e.event_type is event_type), key=lambda e: e.event_time)
    return [(e.event_time, e.payload[value_key]) for e in matches]


def _value_at_or_before(series: list[tuple[float, float]], cutoff: float) -> tuple[float, float] | None:
    times = [t for t, _ in series]
    idx = bisect.bisect_right(times, cutoff)
    return series[idx - 1] if idx else None


@dataclass(frozen=True)
class _BookPoint:
    ts: float
    mid: float
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


def _book_timeseries(events: list[Event], side_value: str) -> list[_BookPoint]:
    """Replays BOOK_SNAPSHOT/BOOK_DELTA for one side into a time series of
    top-of-book points, recorded only once both a bid and an ask level are
    known. Returns points sorted by event_time."""
    relevant = sorted(
        (e for e in events if e.event_type in (EventType.BOOK_SNAPSHOT, EventType.BOOK_DELTA) and e.payload.get("side") == side_value),
        key=lambda e: e.sort_key,
    )
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    out: list[_BookPoint] = []
    for e in relevant:
        if e.event_type is EventType.BOOK_SNAPSHOT:
            bids = {p: s for p, s in e.payload["bids"]}
            asks = {p: s for p, s in e.payload["asks"]}
        else:
            book = bids if e.payload["book"] == "bids" else asks
            price, size = e.payload["price"], e.payload["size"]
            if size == 0:
                book.pop(price, None)
            else:
                book[price] = size
        if bids and asks:
            best_bid, best_ask = max(bids), min(asks)
            out.append(
                _BookPoint(
                    ts=e.event_time,
                    mid=(best_bid + best_ask) / 2.0,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_size=bids[best_bid],
                    ask_size=asks[best_ask],
                )
            )
    return out


def _log_odds(p: float) -> float:
    p = min(1.0 - _LOG_ODDS_EPS, max(_LOG_ODDS_EPS, p))
    return math.log(p / (1.0 - p))


def compute(
    events: list[Event],
    round_id: str,
    decision_ts: float,
    p0: float,
    cfg: FeatureConfig,
) -> FeatureResult:
    causal = [e for e in events if e.event_time <= decision_ts]  # the causal invariant, enforced here

    config_events = sorted(
        (e for e in causal if e.event_type is EventType.MARKET_CONFIG), key=lambda e: e.event_time
    )
    if not config_events:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.MISSING_MARKET_CONFIG)
    market_config = config_events[-1].payload
    round_start_ts = market_config["start_ts"]
    round_end_ts = market_config["end_ts"]
    twap_window_s = float(market_config["twap_window_seconds"])

    t = decision_ts - round_start_ts
    tau = round_end_ts - decision_ts

    spot_series = _series(causal, EventType.SPOT)
    twap_series = _series(causal, EventType.TWAP)

    spot_point = _value_at_or_before(spot_series, decision_ts)
    if spot_point is None:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.MISSING_SPOT)
    spot_ts, spot = spot_point
    if decision_ts - spot_ts > cfg.freshness_s:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.STALE_SPOT)

    twap_point = _value_at_or_before(twap_series, decision_ts)
    if twap_point is None:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.MISSING_TWAP)
    twap_ts, twap = twap_point
    if decision_ts - twap_ts > cfg.freshness_s:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.STALE_TWAP)

    book_series = _book_timeseries(causal, "UP")
    book_mid_series = [(p.ts, p.mid) for p in book_series]
    book_idx = bisect.bisect_right([p.ts for p in book_series], decision_ts) - 1
    if book_idx < 0:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.MISSING_BOOK)
    book_point = book_series[book_idx]
    if decision_ts - book_point.ts > cfg.freshness_s:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.STALE_BOOK)
    clob_mid = book_point.mid

    # --- SS5.1 gaps ---
    gap_twap_bp = 10_000.0 * (twap - p0) / p0
    gap_spot_bp = 10_000.0 * (spot - p0) / p0
    lead_gap_bp = 10_000.0 * (spot - twap) / twap

    # --- SS5.2 modeled TWAP pressure: V_T,model(t) = (S_t - S_(t-W)) / W ---
    spot_w_ago = _value_at_or_before(spot_series, decision_ts - twap_window_s)
    s_t_minus_w = spot_w_ago[1] if spot_w_ago is not None else p0  # early-round fallback: P0 approximates pre-round price
    twap_pressure_model = (spot - s_t_minus_w) / twap_window_s

    # --- SS6 realized volatility + Z_gap ---
    window_points = [(ts, v) for ts, v in spot_series if decision_ts - cfg.vol_window_s <= ts <= decision_ts]
    if len(window_points) < cfg.vol_min_samples + 1:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.INSUFFICIENT_VOLATILITY_HISTORY)
    scaled_returns = []
    for (t_prev, s_prev), (t_now, s_now) in zip(window_points, window_points[1:]):
        dt = t_now - t_prev
        if dt <= 0 or s_prev <= 0:
            continue
        r = math.log(s_now / s_prev)
        scaled_returns.append(r / math.sqrt(dt))
    if len(scaled_returns) < cfg.vol_min_samples:
        return InvalidFeatureState(round_id, decision_ts, InvalidReason.INSUFFICIENT_VOLATILITY_HISTORY)
    realized_vol = statistics.stdev(scaled_returns)
    sigma_floor = 1e-6
    z_gap = math.log(twap / p0) / (max(realized_vol, sigma_floor) * math.sqrt(max(tau, 1e-6)))

    # --- SS7 CLOB features ---
    clob_log_odds = _log_odds(clob_mid)
    ofi = (
        0.0
        if (book_point.bid_size + book_point.ask_size) == 0
        else (book_point.bid_size - book_point.ask_size) / (book_point.bid_size + book_point.ask_size)
    )
    spread = book_point.spread

    z_window = [_log_odds(p.mid) for p in book_series if decision_ts - cfg.clob_z_window_s <= p.ts <= decision_ts]
    if len(z_window) >= 2:
        median_lo = statistics.median(z_window)
        mad = statistics.median([abs(x - median_lo) for x in z_window])
        robust_scale = max(mad * 1.4826, cfg.clob_min_robust_scale)
        z_clob = (clob_log_odds - median_lo) / robust_scale
    else:
        z_clob = 0.0  # degenerate early-round case: not enough book history yet

    spot_returns_bp: dict[float, float] = {}
    for h in cfg.spot_horizons_s:
        prior = _value_at_or_before(spot_series, decision_ts - h)
        if prior is not None and prior[1] > 0:
            spot_returns_bp[h] = 10_000.0 * (spot - prior[1]) / prior[1]

    # --- SS9 Z_spot: vol-normalized spot momentum at the canonical horizon,
    # same vol*sqrt(time) convention as Z_gap. None (not 0.0) when the
    # canonical horizon isn't available yet - a missing feature must read as
    # missing, not as "no momentum".
    z_spot: float | None = None
    canonical_prior = _value_at_or_before(spot_series, decision_ts - cfg.canonical_horizon_s)
    if canonical_prior is not None and canonical_prior[1] > 0:
        spot_log_return = math.log(spot / canonical_prior[1])
        z_spot = spot_log_return / (max(realized_vol, sigma_floor) * math.sqrt(cfg.canonical_horizon_s))

    clob_log_odds_returns: dict[float, float] = {}
    clob_direction: dict[float, int] = {}
    for h in cfg.clob_horizons_s:
        prior = _value_at_or_before(book_mid_series, decision_ts - h)
        if prior is not None:
            prior_lo = _log_odds(prior[1])
            clob_log_odds_returns[h] = clob_log_odds - prior_lo
            delta = clob_mid - prior[1]
            clob_direction[h] = 1 if delta > 0 else (-1 if delta < 0 else 0)

    return FeatureVector(
        feature_version=cfg.feature_version,
        round_id=round_id,
        decision_ts=decision_ts,
        t=t,
        tau=tau,
        time_regime=time_regime_for(t),
        p0=p0,
        twap=twap,
        spot=spot,
        clob_mid=clob_mid,
        gap_twap_bp=gap_twap_bp,
        gap_spot_bp=gap_spot_bp,
        lead_gap_bp=lead_gap_bp,
        twap_pressure_model=twap_pressure_model,
        realized_vol=realized_vol,
        z_gap=z_gap,
        z_spot=z_spot,
        clob_log_odds=clob_log_odds,
        z_clob=z_clob,
        ofi=ofi,
        spread=spread,
        spot_returns_bp=spot_returns_bp,
        clob_log_odds_returns=clob_log_odds_returns,
        clob_direction=clob_direction,
    )


def to_journal_record(fv: FeatureVector) -> FeatureStateRecord:
    """Flattens a FeatureVector into the SS20 feature_state journal entity."""
    payload = asdict(fv)
    payload["time_regime"] = fv.time_regime.value
    return FeatureStateRecord(
        round_id=fv.round_id,
        decision_ts=fv.decision_ts,
        feature_version=fv.feature_version,
        features=payload,
    )
