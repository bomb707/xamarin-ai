"""Feature engineering parameters (Roadmap Phase 4; horizons match the
Strategy doc SS21 Parameter Registry: "spot horizons: 0.25/0.5/1/3/5/10 s
candidates", "CLOB horizons: 0.25/0.5/1/3/5 s candidates")."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    feature_version: str = "features-v1"

    spot_horizons_s: tuple[float, ...] = (0.25, 0.5, 1.0, 3.0, 5.0, 10.0)
    clob_horizons_s: tuple[float, ...] = (0.25, 0.5, 1.0, 3.0, 5.0)

    # Z_clob window: lookback used for the median(L_clob window) and
    # robust_scale in Z_clob = (L_clob(t) - median(window)) / robust_scale
    # (Strategy doc SS7). Window length isn't specified in the source docs;
    # chosen to comfortably contain the largest clob_horizons_s entry.
    clob_z_window_s: float = 30.0
    # Floor for the robust scale (MAD * 1.4826, the normal-consistent MAD
    # scale factor) so a quiet/flat window doesn't produce a divide-by-zero
    # or an exploding Z-score.
    clob_min_robust_scale: float = 0.05

    # Realized volatility window for sigma_t (Strategy doc SS6: "sigma_t
    # must be consistently scaled to the time unit used in tau. ... use
    # empirical realized volatility from the same BTC reference stream").
    # Convention used here: sigma_t is the per-sqrt-second stdev of spot
    # log returns (stdev of per-tick log returns, scaled by the average
    # tick interval), so sigma_t * sqrt(tau) is dimensionally consistent
    # with tau in seconds, as SS6 requires.
    vol_window_s: float = 60.0
    vol_min_samples: int = 5

    # canonical horizon used for Z_spot and (in reports/model training) as
    # the "the" spot/CLOB direction feature when a single horizon is needed
    # rather than the full per-horizon dict.
    canonical_horizon_s: float = 1.0

    # required freshness for TWAP/spot/book inputs, seconds
    freshness_s: float = 5.0
