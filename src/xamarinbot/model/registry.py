"""Model registry (Roadmap Phase 5 deliverable: "Freeze model artifact,
feature version, training window and metrics into ModelRegistry." Exit
gate: "No production use until calibration is acceptable.")

A minimal in-memory registry: enough for champion/challenger comparison and
rollback (concepts named again in Phase 14) without building the full
persistence/promotion-pipeline machinery that phase actually calls for.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace

from xamarinbot.model.calibration import IsotonicCalibrator, PlattCalibrator
from xamarinbot.model.logistic import LogisticModel

Calibrator = IsotonicCalibrator | PlattCalibrator | None


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    feature_set_name: str
    feature_version: str
    training_window: tuple[float, float]
    model: LogisticModel
    calibrator: Calibrator
    metrics: dict[str, float]
    created_at: float
    promoted: bool = False


def _content_hash(model: LogisticModel, feature_version: str, training_window: tuple[float, float]) -> str:
    payload = dict(
        feature_set_name=model.feature_set_name,
        column_names=model.column_names,
        weights=model.weights,
        bias=model.bias,
        feature_version=feature_version,
        training_window=training_window,
    )
    canonical = str(sorted(payload.items()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def make_artifact(
    model: LogisticModel,
    feature_version: str,
    training_window: tuple[float, float],
    metrics: dict[str, float],
    calibrator: Calibrator = None,
) -> ModelArtifact:
    model_id = f"{model.feature_set_name}-{_content_hash(model, feature_version, training_window)}"
    return ModelArtifact(
        model_id=model_id,
        feature_set_name=model.feature_set_name,
        feature_version=feature_version,
        training_window=training_window,
        model=model,
        calibrator=calibrator,
        metrics=dict(metrics),
        created_at=time.time(),
    )


class PromotionGateError(RuntimeError):
    """Raised instead of silently promoting a poorly-calibrated model -
    matches the Phase 5 exit gate "No production use until calibration is
    acceptable.\""""


@dataclass
class ModelRegistry:
    _artifacts: dict[str, ModelArtifact] = field(default_factory=dict)
    _champion_id: str | None = None

    def register(self, artifact: ModelArtifact) -> None:
        self._artifacts[artifact.model_id] = artifact

    def get(self, model_id: str) -> ModelArtifact:
        return self._artifacts[model_id]

    def promote(self, model_id: str, max_brier: float = 0.25) -> ModelArtifact:
        """0.25 is the Brier score of a constant-0.5 predictor - the
        minimum bar an out-of-sample-evaluated model must clear before
        promotion, per the Phase 5 exit gate ("must beat or justify
        complexity relative to simpler baselines")."""
        artifact = self._artifacts[model_id]
        brier = artifact.metrics.get("brier")
        if brier is None or brier > max_brier:
            raise PromotionGateError(
                f"model {model_id} fails promotion gate: brier={brier} not <= {max_brier}"
            )
        promoted = replace(artifact, promoted=True)
        self._artifacts[model_id] = promoted
        self._champion_id = model_id
        return promoted

    def champion(self) -> ModelArtifact | None:
        return self._artifacts.get(self._champion_id) if self._champion_id else None

    def rollback(self, model_id: str) -> ModelArtifact:
        """Roadmap Phase 10/14: "Rollback to previous model version" - a
        one-command switch of which artifact is champion, no retraining."""
        if model_id not in self._artifacts:
            raise KeyError(model_id)
        self._champion_id = model_id
        return self._artifacts[model_id]
