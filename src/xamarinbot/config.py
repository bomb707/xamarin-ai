"""Versioned config and feature flags (Roadmap §2 Architecture and Version
Strategy, Phase 0 audit requirements).

"Use feature flags and versioned configuration so the baseline, one-step
optimizer, and MPC can run side by side on identical event streams."
"Add a strategy_version and config_hash to every journaled
decision/fill/settlement."
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass


def config_hash(config: object) -> str:
    """Stable content hash of a (dataclass) config, stamped onto every
    journaled decision/fill/settlement so any promoted config is
    reproducible and auditable."""
    payload = asdict(config) if is_dataclass(config) else config
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FeatureFlags:
    """Lets the baseline, one-step optimizer, and MPC run side by side on
    identical event streams; each later controller is off until its own
    phase is implemented and its exit gate passes."""

    use_baseline_v1: bool = True
    use_one_step_controller: bool = False  # Phase 8 - not yet implemented
    use_mpc_controller: bool = False  # Phase 10 - not yet implemented
