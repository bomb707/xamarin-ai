"""STRATEGY_V0_MANIFEST - the frozen strategy under shadow evaluation.

Why this exists
---------------
"The current strategy" was, until now, whatever the defaults of
`FeatureConfig`, `OneStepConfig`, `RegimeConfig` and `ExecutionConfig`
happened to be at the moment a run started. That is not a strategy anyone
can evaluate: a shadow run measures a specific configuration, and if the
configuration can drift between runs - or between a run and the analysis of
it - the measurement means nothing.

This module pins one immutable manifest and stamps its hash onto every
decision record. If a config changes, the hash changes, and the records
from before and after are visibly not comparable rather than silently
pooled.

The manifest deliberately does NOT choose the parameters. It records the
ones already in the codebase. Changing a value here is a strategy change and
must be a separate, explicit act - never a side effect of a shadow run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.features import FeatureSet
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.regime.config import RegimeConfig

STRATEGY_VERSION = "STRATEGY_V0"

#: The decision cadence, in seconds after the round opens. Identical to
#: `model/real_dataset.decision_grid` - the training grid and the live
#: strategy clock MUST be the same schedule, or the model is fitted at
#: decision points the bot never actually makes.
DECISION_GRID_START_S = 15.0
DECISION_GRID_END_S = 270.0
DECISION_GRID_STEP_S = 3.0


def _stable_hash(payload: dict) -> str:
    """A hash that depends on the VALUES, not on dict ordering or float
    repr drift, so the same config always yields the same id."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _config_payload(obj) -> dict:
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)
                if not f.name.startswith("_")}
    return dict(vars(obj))


@dataclass(frozen=True)
class StrategyManifest:
    """Everything that determines what the bot does, and its identity."""

    strategy_version: str
    controller: str
    model_version: str
    feature_set_name: str
    feature_config_hash: str
    one_step_config_hash: str
    regime_config_hash: str
    execution_config_hash: str
    maker_enabled: bool
    hedge_enabled: bool
    buffer_enabled: bool
    mpc_enabled: bool
    decision_grid: tuple[float, ...] = field(default=())

    @property
    def config_hash(self) -> str:
        """One id for the whole strategy. This is what goes on every record."""
        return _stable_hash({
            "strategy_version": self.strategy_version,
            "controller": self.controller,
            "model_version": self.model_version,
            "feature_set": self.feature_set_name,
            "feature_config": self.feature_config_hash,
            "one_step_config": self.one_step_config_hash,
            "regime_config": self.regime_config_hash,
            "execution_config": self.execution_config_hash,
            "maker": self.maker_enabled, "hedge": self.hedge_enabled,
            "buffer": self.buffer_enabled, "mpc": self.mpc_enabled,
            "grid": list(self.decision_grid),
        })

    def as_dict(self) -> dict:
        return {
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "controller": self.controller,
            "model_version": self.model_version,
            "feature_set": self.feature_set_name,
            "feature_config_hash": self.feature_config_hash,
            "one_step_config_hash": self.one_step_config_hash,
            "regime_config_hash": self.regime_config_hash,
            "execution_config_hash": self.execution_config_hash,
            "maker_enabled": self.maker_enabled,
            "hedge_enabled": self.hedge_enabled,
            "buffer_enabled": self.buffer_enabled,
            "mpc_enabled": self.mpc_enabled,
            "decision_grid_s": list(self.decision_grid),
        }


def decision_grid(
    start_s: float = DECISION_GRID_START_S,
    end_s: float = DECISION_GRID_END_S,
    step_s: float = DECISION_GRID_STEP_S,
) -> tuple[float, ...]:
    """`t = 15, 18, ..., 270` seconds after the open.

    The FROZEN strategy clock. Deliberately independent of market activity:
    the CLOB publishes ~130 messages/second, and letting message rate set
    decision count would mean a busier market got more chances to trade for
    no reason connected to the opportunity.
    """
    out, t = [], start_s
    while t <= end_s + 1e-9:
        out.append(round(t, 6))
        t += step_s
    return tuple(out)


def build_manifest(
    feature_cfg: FeatureConfig,
    one_step_cfg: OneStepConfig,
    regime_cfg: RegimeConfig,
    exec_cfg: ExecutionConfig,
    feature_set: FeatureSet | None,
    model_version: str,
    *,
    controller: str = "OneStepController",
    mpc_enabled: bool = False,
) -> StrategyManifest:
    return StrategyManifest(
        strategy_version=STRATEGY_VERSION,
        controller=controller,
        model_version=model_version,
        feature_set_name=feature_set.name if feature_set else "NONE",
        feature_config_hash=_stable_hash(_config_payload(feature_cfg)),
        one_step_config_hash=_stable_hash(_config_payload(one_step_cfg)),
        regime_config_hash=_stable_hash(_config_payload(regime_cfg)),
        execution_config_hash=_stable_hash(_config_payload(exec_cfg)),
        maker_enabled=bool(getattr(one_step_cfg, "maker_enabled", False)),
        hedge_enabled=bool(getattr(one_step_cfg, "hedge_enabled", True)),
        buffer_enabled=bool(getattr(one_step_cfg, "buffer_enabled", True)),
        mpc_enabled=mpc_enabled,
        decision_grid=decision_grid(),
    )


def strategy_v0_configs():
    """The concrete parameters STRATEGY_V0 runs with.

    AUDIT FINDING, recorded here rather than hidden: before this manifest
    there was no canonical strategy configuration anywhere in the codebase.
    `OneStepConfig` has no default `g_min` at all, and every caller invented
    its own values - the synthetic demos alone use three different sets
    (`g_min` of -100, -100 and -200, `edge_min` of 0.0 and 0.5). "The
    current strategy" was therefore not a well-defined object.

    The values below are taken verbatim from
    `scripts/dev_synthetic/run_synthetic_shadow_demo.py`, which is the only
    existing precedent for a SHADOW configuration. They are adopted so that
    STRATEGY_V0 is at least pinned and reproducible - NOT because they have
    been validated against a real market. They were chosen to make a
    synthetic demo produce visible activity.

    Consequently these are a PLACEHOLDER pending explicit approval. Any
    profitability claim made under this config would be a claim about
    parameters selected for a demo, and changing them is a strategy change
    that must be deliberate - the hash will show it.
    """
    return OneStepConfig(
        g_min=-100.0,
        spend_cap=200.0,
        position_limit=200.0,
        edge_min=0.0,
    )


#: The model that would be used if real shadow started right now.
#:
#: There is no Gate-A-frozen REAL model. A synthetic-trained model exists in
#: the dev tooling and is deliberately NOT wired here: using it would make
#: the pipeline produce confident-looking ALPHA sized against a probability
#: fitted to fabricated rounds. `MODEL_UNAVAILABLE` is the correct and
#: intended state until Gate A produces a frozen real model.
NO_REAL_MODEL = "NONE:no_gate_a_model_frozen"
