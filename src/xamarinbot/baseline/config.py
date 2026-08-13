"""Baseline (V1) parameters (Roadmap Phase 0: "Record all existing
parameters: decision window, lookbacks, minimum gap, clip, ask range, fee
fallback, freshness, limit delta, retries, residual/spend caps.")

Every field name here matches a term named explicitly in that Phase 0 step.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    strategy_version: str = "baseline-v1"

    # decision window: evaluate only in [15, 270] seconds into the round
    # (Strategy doc SS1: "evaluates from 15 to 270 seconds")
    decision_window_start_s: float = 15.0
    decision_window_end_s: float = 270.0

    # lookbacks: momentum horizons for CLOB midpoint and BTC spot
    clob_lookback_s: float = 3.0
    spot_lookback_s: float = 3.0

    # minimum gap: TWAP-vs-price-to-beat gap must exceed this (bp) to count
    # as a directional TWAP signal rather than "flat"
    minimum_gap_bp: float = 1.0

    # clip: fixed minimum order size (shares) for a taker entry
    clip: float = 5.0

    # directional-lead sizing: extra size added when the spot-vs-TWAP lead
    # exceeds this threshold (bp), on top of `clip`, capped by residual_cap.
    # NOTE: the source docs describe "directional-lead sizing" narratively
    # (Strategy doc SS1) but do not give a V1 formula; lead_size_bonus /
    # lead_bonus_threshold_bp are this reconstruction's parameterization -
    # see baseline/strategy.py module docstring and docs/PHASE_STATUS.md.
    lead_bonus_threshold_bp: float = 5.0
    lead_size_bonus: float = 5.0

    # ask range: maximum executable bid-ask-spread-equivalent distance (bp)
    # between the best ask and the current CLOB fair-value reference; a
    # fixed one-tick-ish spread is already a few hundred bp at moderate
    # prices (e.g. 1c on a $0.20 mid = 500bp), so this is a wide-quote/
    # illiquidity cap, not a tight range check.
    ask_range_bp: float = 800.0

    # fee fallback: crypto feeRate used when live market fee config is
    # unavailable [P1]
    fee_fallback_rate: float = 0.07

    # freshness: max staleness (s) for any input feed before skipping
    freshness_s: float = 5.0

    # limit delta: marketable-limit price offset above best ask, so a FAK
    # order reliably crosses the spread
    limit_delta: float = 0.02

    # retries: max resubmission attempts for a rejected/unmatched order
    retries: int = 2

    # average-price guard: skip a taker entry whose limit price is at or
    # above this level (already-expensive outcome, Strategy doc SS3)
    avg_price_guard: float = 0.95

    # risk caps
    residual_cap: float = 50.0  # max |R| = |U-D| this round
    spend_cap: float = 100.0  # max total C this round
