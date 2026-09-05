"""The product-facing inference contract.

This is what the Express API consumes, so it is versioned and stable in the same
way the training artifacts are. Three rules shape it:

**A score is not a probability.** ``raw_score`` is the model's ranking scalar --
unbounded, and meaningful only as an ordering within one episode.
``normalized_score`` projects the episode's raw scores onto 0-100 and says so in
``score_scale``. Neither is calibrated against human labels, and
``is_calibrated_probability`` is therefore ``False`` and is carried in the
payload rather than assumed by the reader.

**Every result names the model that produced it.** ``model_variant``,
``model_run_id``, ``checkpoint_git_commit`` and the schema versions travel with
the scores, so a ranking shown in a demo can be traced to the exact weights that
produced it.

**Absence is reported, not filled in.** A candidate whose features could not be
built is listed in ``excluded`` with a reason. It is never scored from a
zero-filled vector, and the ranked list is exactly as long as the set of
candidates that really were scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from slotify_rank.config.versions import PACKAGE_VERSION

__all__ = [
    "INFERENCE_SCHEMA_VERSION",
    "ScoreScale",
    "ScoredCandidate",
    "ExcludedCandidate",
    "ModelIdentity",
    "InferenceResult",
]

INFERENCE_SCHEMA_VERSION = "inference-v1.0.0"

#: How ``normalized_score`` was derived. ``episode_relative_min_max`` is the
#: only scale a learned ranker can honestly offer today: its raw score is
#: unbounded, so any absolute presentation would be invented.
ScoreScale = Literal["episode_relative_min_max"]


@dataclass(frozen=True)
class ScoredCandidate:
    candidate_id: str
    insertion_time_seconds: float
    insertion_ms: int
    raw_score: float
    normalized_score: float
    rank: int
    #: Pre-sigmoid auxiliary output when the model has that head. Present for
    #: inspection only; it is not calibrated and must not be shown as a percent.
    acceptability_logit: float | None
    #: Which learned modalities this candidate actually had. A candidate with no
    #: transcript is scored with its text block masked, and this says so.
    audio_available: bool
    text_available: bool
    feature_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "insertion_time_seconds": self.insertion_time_seconds,
            "insertion_ms": self.insertion_ms,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "rank": self.rank,
            "acceptability_logit": self.acceptability_logit,
            "audio_available": self.audio_available,
            "text_available": self.text_available,
            "feature_status": self.feature_status,
            "provenance": "learned_ranker",
        }


@dataclass(frozen=True)
class ExcludedCandidate:
    candidate_id: str
    reason: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ModelIdentity:
    """Enough to reproduce a ranking from the repository."""

    model_variant: str
    model_run_id: str
    checkpoint_path: str
    checkpoint_git_commit: str
    parameter_count: int
    handcrafted_dimension: int
    audio_dimension: int
    text_dimension: int
    feature_spec_version: str
    feature_pipeline_version: str
    input_schema_version: str
    normalizer_version: str
    #: What the weights were fitted on. `synthetic_fixture` and `weak_heuristic`
    #: are NOT evidence of ranking quality and are stated here so no consumer can
    #: present them as such.
    training_label_source: str
    training_data_provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_variant": self.model_variant,
            "model_run_id": self.model_run_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_git_commit": self.checkpoint_git_commit,
            "parameter_count": self.parameter_count,
            "handcrafted_dimension": self.handcrafted_dimension,
            "audio_dimension": self.audio_dimension,
            "text_dimension": self.text_dimension,
            "feature_spec_version": self.feature_spec_version,
            "feature_pipeline_version": self.feature_pipeline_version,
            "input_schema_version": self.input_schema_version,
            "normalizer_version": self.normalizer_version,
            "training_label_source": self.training_label_source,
            "training_data_provenance": self.training_data_provenance,
        }


@dataclass(frozen=True)
class InferenceResult:
    episode_id: str
    duration_seconds: float | None
    ranked: tuple[ScoredCandidate, ...]
    excluded: tuple[ExcludedCandidate, ...]
    model: ModelIdentity
    score_scale: ScoreScale = "episode_relative_min_max"
    is_calibrated_probability: bool = False
    candidate_count: int = 0
    warnings: tuple[str, ...] = ()
    timings_seconds: dict[str, float] = field(default_factory=dict)
    schema_version: str = INFERENCE_SCHEMA_VERSION
    package_version: str = PACKAGE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "episode_id": self.episode_id,
            "duration_seconds": self.duration_seconds,
            "provenance": "learned_ranker",
            "score_scale": self.score_scale,
            "is_calibrated_probability": self.is_calibrated_probability,
            "candidate_count": self.candidate_count,
            "scored_count": len(self.ranked),
            "excluded_count": len(self.excluded),
            "model": self.model.to_dict(),
            "ranked": [candidate.to_dict() for candidate in self.ranked],
            "excluded": [candidate.to_dict() for candidate in self.excluded],
            "warnings": list(self.warnings),
            "timings_seconds": dict(self.timings_seconds),
        }


def normalize_scores(raw_scores: Sequence[float]) -> list[float]:
    """Project raw scores onto 0-100 relative to this episode.

    Identical scores map to 50 rather than to an arbitrary spread: a ranker that
    could not separate its candidates should not appear to have done so.
    """
    values = [float(score) for score in raw_scores]
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low < 1e-9:
        return [50.0 for _ in values]
    return [round(100.0 * (value - low) / (high - low), 2) for value in values]
