"""Design-vector construction for the q model (Roadmap Phase 5, Strategy
doc SS9): logit(q_t) = beta0 + beta1*Z_gap + beta2*L + beta3*Z_spot +
beta4*Z_clob + beta5*OFI + beta6*tau + interactions.

Suggested interactions (SS9): Z_gap*tau, L*tau, Z_spot*Z_clob,
volatility*tau. `FeatureSet` lets the same machinery build the TWAP-only,
current-BTC-only (spot-only), and combined lead-lag models the Roadmap
Phase 5 step asks to compare - the TWAP-only/spot-only ablations each get
their own tau interaction for a fair comparison, which is this build's
extension beyond SS9's formula (written specifically for the combined
model), not a literal spec formula.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.features.types import FeatureVector

BASE_FEATURE_NAMES = ("z_gap", "lead_gap_bp", "z_spot", "z_clob", "ofi", "tau", "realized_vol")


@dataclass(frozen=True)
class FeatureSet:
    name: str
    base: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return self.base + tuple(f"{a}*{b}" for a, b in self.interactions)


def base_value_map(fv: FeatureVector) -> dict[str, float] | None:
    """None if a required-but-not-yet-available input (currently only
    Z_spot) is missing - callers must treat that example as unusable, not
    silently zero-fill it."""
    if fv.z_spot is None:
        return None
    return {
        "z_gap": fv.z_gap,
        "lead_gap_bp": fv.lead_gap_bp,
        "z_spot": fv.z_spot,
        "z_clob": fv.z_clob,
        "ofi": fv.ofi,
        "tau": fv.tau,
        "realized_vol": fv.realized_vol,
    }


def design_vector(fv: FeatureVector, feature_set: FeatureSet) -> list[float] | None:
    values = base_value_map(fv)
    if values is None:
        return None
    vec = [values[name] for name in feature_set.base]
    vec.extend(values[a] * values[b] for a, b in feature_set.interactions)
    return vec


TWAP_ONLY = FeatureSet("twap_only", base=("z_gap", "tau"), interactions=(("z_gap", "tau"),))
SPOT_ONLY = FeatureSet("spot_only", base=("z_spot", "tau"), interactions=(("z_spot", "tau"),))
# Roadmap Phase 11 / SS20.1 mandatory ablation #4: "V2 TWAP + current-BTC
# lead-lag model" - Z_gap and L (the lead gap) only, no CLOB/OFI. Distinct
# from COMBINED_LEAD_LAG below, which is really ablation #5's "Lead-lag +
# CLOB" - the two are deliberately different feature sets, not the same
# one under two names.
LEAD_LAG_ONLY = FeatureSet(
    "lead_lag_only", base=("z_gap", "lead_gap_bp", "tau"), interactions=(("z_gap", "tau"), ("lead_gap_bp", "tau"))
)
COMBINED_LEAD_LAG = FeatureSet(
    "combined_lead_lag",
    base=("z_gap", "lead_gap_bp", "z_spot", "z_clob", "ofi", "tau"),
    interactions=(("z_gap", "tau"), ("lead_gap_bp", "tau"), ("z_spot", "z_clob"), ("realized_vol", "tau")),
)
