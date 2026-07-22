"""The candidate feature record and the handcrafted feature vocabulary.

**Feature ordering is data, not convention.** The canonical order is stored in
the manifest header and every record is written against it. Relying on dict
insertion order instead would work perfectly until someone reordered two lines
in an extractor, at which point every stored vector would be silently
mis-columned -- a bug that produces plausible numbers and no error.

**Vectors do not live in JSON.** Handcrafted features are a few dozen scalars
per candidate, so they go inline. The 384-dimensional embeddings do not: they
are written once per episode as a ``.npy`` matrix and the record stores a
*reference* -- file plus row id. A corpus of 100k candidates with four
1536-float JSON arrays each would be tens of gigabytes of text that no reader
could mmap.

**Missing modalities are recorded, never faked.** ``audio_embedding_available``
and ``text_embedding_available`` are explicit. A future PyTorch dataset can zero
the vector and set a mask; it must never be handed a random or silently-zero
embedding it cannot distinguish from a real one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from slotify_rank.config.versions import (
    CANDIDATE_GENERATION_VERSION,
    FEATURE_MANIFEST_SCHEMA_VERSION,
    FEATURE_PIPELINE_VERSION,
    FEATURE_SPEC_VERSION,
    SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS,
)

__all__ = [
    "FeatureStatus",
    "EmbeddingReference",
    "FeatureSpec",
    "CandidateFeatureRecord",
]

#: ``complete`` requires *both* learned modalities. The partial states are not a
#: failure -- a candidate in an untranscribed episode legitimately has audio
#: only -- but they are counted separately so "complete multimodal records"
#: never quietly includes them.
FeatureStatus = Literal[
    "complete",
    "audio_only",
    "text_only",
    "handcrafted_only",
    "failed",
]


@dataclass(frozen=True)
class EmbeddingReference:
    """Where one candidate's vector lives inside an episode-level array."""

    #: Repository-relative path to the ``.npy`` matrix.
    path: str
    #: Row identity within that matrix. Resolved through the sidecar's row_ids,
    #: never by position, so a row inserted upstream cannot shift every lookup.
    row_id: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("EmbeddingReference.path must not be empty")
        if "\\" in self.path:
            raise ValueError(
                f"EmbeddingReference.path must use forward slashes, got {self.path!r}"
            )
        if self.dimension <= 0:
            raise ValueError("EmbeddingReference.dimension must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EmbeddingReference":
        return cls(
            path=str(raw["path"]),
            row_id=str(raw["row_id"]),
            dimension=int(raw["dimension"]),
        )


@dataclass(frozen=True)
class FeatureSpec:
    """The ordered handcrafted feature vocabulary for one pipeline version.

    Built once from the configured window scales and reused for every candidate.
    Names are sorted so the order depends only on the *set* of features, not on
    the order the extractors happened to emit them.
    """

    names: tuple[str, ...]
    spec_version: str = FEATURE_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("A feature spec must contain at least one feature")
        duplicates = sorted({n for n in self.names if list(self.names).count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate feature name(s) in spec: {duplicates}")
        if list(self.names) != sorted(self.names):
            raise ValueError(
                "Feature names must be stored in sorted order so the column "
                "layout cannot depend on extractor emission order"
            )

    @property
    def dimension(self) -> int:
        return len(self.names)

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "FeatureSpec":
        return cls(names=tuple(sorted(set(names))))

    def vectorize(
        self, values: Mapping[str, float], missing: Mapping[str, bool]
    ) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        """Project a name-keyed mapping onto the canonical column order.

        Unknown names are an error, not something to drop: a feature the spec
        does not know about means the extractors and the spec have diverged, and
        continuing would silently discard a column from every record.
        """
        unknown = sorted(set(values) - set(self.names))
        if unknown:
            raise ValueError(
                f"Feature(s) {unknown} are not in the spec (version "
                f"{self.spec_version}). The extractors and the spec have "
                "diverged; bump FEATURE_SPEC_VERSION rather than dropping columns."
            )
        absent = sorted(set(self.names) - set(values))
        if absent:
            raise ValueError(
                f"Feature(s) {absent} are in the spec but were not extracted. A "
                "record with a short vector would be mis-columned on load."
            )
        vector = tuple(float(values[name]) for name in self.names)
        mask = tuple(bool(missing[name]) for name in self.names)
        return vector, mask


@dataclass(frozen=True)
class CandidateFeatureRecord:
    """Everything Phase 3 knows about one candidate.

    ``handcrafted_feature_values`` is positional against the manifest's
    ``handcrafted_feature_names``; the two are written together and validated
    together.
    """

    episode_id: str
    candidate_id: str
    timestamp_ms: int
    dataset_split: str
    handcrafted_feature_values: tuple[float, ...]
    handcrafted_missing_mask: tuple[bool, ...]
    feature_status: FeatureStatus
    audio_sha256: str = ""
    transcript_version: str = ""
    candidate_generation_version: str = CANDIDATE_GENERATION_VERSION
    feature_pipeline_version: str = FEATURE_PIPELINE_VERSION
    feature_spec_version: str = FEATURE_SPEC_VERSION
    audio_embedding_reference: dict[str, EmbeddingReference] = field(
        default_factory=dict
    )
    text_embedding_reference: dict[str, EmbeddingReference] = field(
        default_factory=dict
    )
    audio_embedding_dimension: int | None = None
    text_embedding_dimension: int | None = None
    audio_embedding_available: bool = False
    text_embedding_available: bool = False
    transcript_available: bool = False
    #: Which transcript segments produced the text context, for auditability.
    context_segment_ids_before: tuple[str, ...] = ()
    context_segment_ids_after: tuple[str, ...] = ()
    failure_reason: str | None = None
    schema_version: str = FEATURE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.handcrafted_feature_values) != len(self.handcrafted_missing_mask):
            raise ValueError(
                f"{self.candidate_id}: {len(self.handcrafted_feature_values)} feature "
                f"values but {len(self.handcrafted_missing_mask)} mask entries. A "
                "value without a mask cannot be interpreted."
            )
        if self.timestamp_ms < 0:
            raise ValueError(f"{self.candidate_id}: timestamp_ms must be non-negative")
        if self.feature_status not in (
            "complete",
            "audio_only",
            "text_only",
            "handcrafted_only",
            "failed",
        ):
            raise ValueError(
                f"{self.candidate_id}: unknown feature_status {self.feature_status!r}"
            )
        if self.feature_status == "failed" and not self.failure_reason:
            raise ValueError(
                f"{self.candidate_id}: feature_status 'failed' requires a "
                "failure_reason"
            )
        # The status must agree with the availability flags, or every count
        # derived from either one is untrustworthy.
        expected = _status_for(
            self.audio_embedding_available, self.text_embedding_available
        )
        if self.feature_status != "failed" and self.feature_status != expected:
            raise ValueError(
                f"{self.candidate_id}: feature_status {self.feature_status!r} "
                f"contradicts availability flags (audio="
                f"{self.audio_embedding_available}, text="
                f"{self.text_embedding_available}), which imply {expected!r}"
            )
        if self.audio_embedding_available and not self.audio_embedding_reference:
            raise ValueError(
                f"{self.candidate_id}: marked audio-available but carries no "
                "embedding reference"
            )
        if self.text_embedding_available and not self.text_embedding_reference:
            raise ValueError(
                f"{self.candidate_id}: marked text-available but carries no "
                "embedding reference"
            )

    @property
    def eligible_for_training(self) -> bool:
        """Usable as a training example.

        ``failed`` records are excluded. Partial-modality records are *not*:
        the whole point of carrying explicit masks is that a model can train on
        them.
        """
        return self.feature_status != "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "candidate_id": self.candidate_id,
            "timestamp_ms": self.timestamp_ms,
            "dataset_split": self.dataset_split,
            "audio_sha256": self.audio_sha256,
            "transcript_version": self.transcript_version,
            "candidate_generation_version": self.candidate_generation_version,
            "feature_pipeline_version": self.feature_pipeline_version,
            "feature_spec_version": self.feature_spec_version,
            "handcrafted_feature_values": list(self.handcrafted_feature_values),
            "handcrafted_missing_mask": list(self.handcrafted_missing_mask),
            "audio_embedding_reference": {
                key: reference.to_dict()
                for key, reference in self.audio_embedding_reference.items()
            },
            "text_embedding_reference": {
                key: reference.to_dict()
                for key, reference in self.text_embedding_reference.items()
            },
            "audio_embedding_dimension": self.audio_embedding_dimension,
            "text_embedding_dimension": self.text_embedding_dimension,
            "audio_embedding_available": self.audio_embedding_available,
            "text_embedding_available": self.text_embedding_available,
            "transcript_available": self.transcript_available,
            "context_segment_ids_before": list(self.context_segment_ids_before),
            "context_segment_ids_after": list(self.context_segment_ids_after),
            "feature_status": self.feature_status,
            "failure_reason": self.failure_reason,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CandidateFeatureRecord":
        version = str(raw.get("schema_version", ""))
        if version not in SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported feature record schema_version {version!r}; this build "
                f"reads {sorted(SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS)}"
            )
        return cls(
            episode_id=str(raw["episode_id"]),
            candidate_id=str(raw["candidate_id"]),
            timestamp_ms=int(raw["timestamp_ms"]),
            dataset_split=str(raw["dataset_split"]),
            handcrafted_feature_values=tuple(
                float(value) for value in raw["handcrafted_feature_values"]
            ),
            handcrafted_missing_mask=tuple(
                bool(value) for value in raw["handcrafted_missing_mask"]
            ),
            feature_status=str(raw["feature_status"]),  # type: ignore[arg-type]
            audio_sha256=str(raw.get("audio_sha256", "")),
            transcript_version=str(raw.get("transcript_version", "")),
            candidate_generation_version=str(
                raw.get("candidate_generation_version", "")
            ),
            feature_pipeline_version=str(raw.get("feature_pipeline_version", "")),
            feature_spec_version=str(raw.get("feature_spec_version", "")),
            audio_embedding_reference={
                key: EmbeddingReference.from_mapping(value)
                for key, value in (raw.get("audio_embedding_reference") or {}).items()
            },
            text_embedding_reference={
                key: EmbeddingReference.from_mapping(value)
                for key, value in (raw.get("text_embedding_reference") or {}).items()
            },
            audio_embedding_dimension=raw.get("audio_embedding_dimension"),
            text_embedding_dimension=raw.get("text_embedding_dimension"),
            audio_embedding_available=bool(raw.get("audio_embedding_available", False)),
            text_embedding_available=bool(raw.get("text_embedding_available", False)),
            transcript_available=bool(raw.get("transcript_available", False)),
            context_segment_ids_before=tuple(
                raw.get("context_segment_ids_before", ())
            ),
            context_segment_ids_after=tuple(raw.get("context_segment_ids_after", ())),
            failure_reason=raw.get("failure_reason"),
            schema_version=version,
        )


def _status_for(audio_available: bool, text_available: bool) -> FeatureStatus:
    if audio_available and text_available:
        return "complete"
    if audio_available:
        return "audio_only"
    if text_available:
        return "text_only"
    return "handcrafted_only"


def status_for(audio_available: bool, text_available: bool) -> FeatureStatus:
    """Public spelling of the availability-to-status mapping."""
    return _status_for(audio_available, text_available)
