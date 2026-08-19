"""Typed, versioned dataset records: episodes and dataset candidates.

Two rules govern everything here.

**Determinism.** An episode's identity is derived from the SHA-256 of its
original audio, so importing the same file twice -- from a different directory,
on a different machine, in a different order -- produces the same
``episode_id`` and therefore the same candidate IDs, labels and split
assignment. Nothing about identity depends on wall-clock time, insertion order
or absolute paths.

**Separation of label provenance.** ``label_status`` distinguishes ``human``,
``weak_heuristic`` and ``unlabelled``. These are never summed into one
"labelled" figure: the evidence matrix tracks them as separate quantities
precisely because collapsing them would overstate the human effort.

The third rule is enforced rather than merely documented: candidates the
*product* invents when it cannot find three real breakpoints carry
``is_synthetic=True`` and are pinned to ``eligible_for_labelling=False`` and
``eligible_for_evaluation=False`` in ``__post_init__``. They exist in the
manifest for auditability and can never enter a dataset count.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from slotify_rank.config.versions import (
    CANDIDATE_GENERATION_VERSION,
    DATASET_CANDIDATE_SCHEMA_VERSION,
    EPISODE_SCHEMA_VERSION,
    PREPROCESSING_VERSION,
    SOURCE_MANIFEST_VERSION,
    SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS,
    SUPPORTED_EPISODE_SCHEMA_VERSIONS,
)

__all__ = [
    "SourceType",
    "ContentType",
    "EpisodeStatus",
    "LabelStatus",
    "DatasetSplit",
    "DatasetCandidateSource",
    "TARGET_DOMAIN_CONTENT_TYPES",
    "EpisodeRecord",
    "DatasetCandidate",
    "EnergySummary",
    "slugify",
    "make_episode_id",
]

#: How an episode's audio got here. Deliberately closed: broad web scraping is
#: out of scope, so there is no "crawled" member.
SourceType = Literal["local_file", "direct_download", "existing_repository_fixture"]

#: What the audio *is*. Drives the target-domain checks: the final test split
#: must be podcast-like, and music must never enter headline statistics.
ContentType = Literal[
    "podcast",
    "interview",
    "conversational",
    "narrated",
    "meeting",
    "music",
    "other",
]

#: Content types that match the product use case. ``meeting`` (AMI and similar)
#: is supplemental only -- see docs/dataset-card.md -- and ``music`` is
#: explicitly out of domain even though the product supports a song mode.
TARGET_DOMAIN_CONTENT_TYPES: frozenset[str] = frozenset(
    {"podcast", "interview", "conversational", "narrated"}
)

EpisodeStatus = Literal["registered", "fetched", "probed", "normalized", "failed"]

LabelStatus = Literal["unlabelled", "human", "weak_heuristic"]

DatasetSplit = Literal["train", "validation", "test", "development", "unassigned"]

#: Generator that proposed a timestamp. ``product_padding`` is not a generator:
#: it marks the product's invented fallbacks, which are excluded by construction.
DatasetCandidateSource = Literal[
    "silence",
    "pause",
    "rms_minimum",
    "transcript_segment_end",
    "fixed_interval",
    "product_padding",
]

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def slugify(text: str, max_length: int = 48) -> str:
    """ASCII, lowercase, hyphen-separated. Deterministic and filename-safe."""
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP_RE.sub("-", ascii_text).strip("-")
    return slug[:max_length].strip("-")


def make_episode_id(title: str, sha256: str) -> str:
    """Deterministic episode identity: ``<title-slug>_<first 12 of sha256>``.

    Keying on the *content* hash rather than the path means a re-import is
    idempotent and two copies of the same recording collapse to one episode,
    which is what stops the same audio being counted twice in the corpus totals.
    The slug is for humans; the hash carries the identity.
    """
    digest = str(sha256).strip().lower()
    if len(digest) != 64:
        raise ValueError(f"sha256 must be 64 hex characters, got {sha256!r}")
    slug = slugify(title) or "episode"
    return f"{slug}_{digest[:12]}"


def _require_relative(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string or null")
    if "\\" in text:
        raise ValueError(
            f"{field_name} must use forward slashes (got {text!r}); use "
            "slotify_rank.data.paths.to_repo_relative to build it"
        )
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        raise ValueError(
            f"{field_name} must be repository-relative, not absolute (got {text!r})"
        )
    if ".." in text.split("/"):
        raise ValueError(f"{field_name} must not traverse upward (got {text!r})")
    return text


@dataclass(frozen=True)
class EpisodeRecord:
    """One audio item in the corpus.

    Nullable-field rules:

    * ``series_id`` is **never** null -- an episode with no series is its own
      series (``make_series_id`` falls back to the episode id), because the
      splitter groups on this field and a null would silently disable leakage
      protection.
    * ``license_name`` / ``license_url`` / ``attribution`` are null only for
      ``local_file`` and ``existing_repository_fixture``, which are private and
      not redistributed. They are **required** for ``direct_download``.
    * ``duration_ms`` / ``sample_rate_hz`` / ``channels`` / ``file_format`` are
      null until ``probe`` has run, and are non-null for status ``probed`` and
      ``normalized``.
    * ``normalized_path`` is null until ``normalize`` has run.
    * ``transcript_path`` is null unless the user supplied a timestamped
      transcript; Phase 2 never generates one.
    """

    episode_id: str
    series_id: str
    title: str
    source_type: SourceType
    source_uri: str
    source_name: str
    license_name: str | None
    license_url: str | None
    attribution: str | None
    language: str
    content_type: ContentType
    original_path: str
    sha256: str
    status: EpisodeStatus = "registered"
    normalized_path: str | None = None
    normalized_sha256: str | None = None
    #: Duration of the 16 kHz render. Recorded separately from ``duration_ms``
    #: (the original's) so a lossy decode that drops or pads audio is visible
    #: rather than being folded into the corpus total.
    normalized_duration_ms: int | None = None
    duration_ms: int | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    file_format: str | None = None
    transcript_path: str | None = None
    is_target_domain: bool = True
    notes: str | None = None
    source_manifest_version: str = SOURCE_MANIFEST_VERSION
    preprocessing_version: str | None = None
    schema_version: str = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.episode_id):
            raise ValueError(
                f"episode_id must match {_ID_RE.pattern} (got {self.episode_id!r})"
            )
        if not _ID_RE.match(self.series_id):
            raise ValueError(
                f"series_id must match {_ID_RE.pattern} (got {self.series_id!r})"
            )
        if self.source_type not in (
            "local_file",
            "direct_download",
            "existing_repository_fixture",
        ):
            raise ValueError(f"Unknown source_type {self.source_type!r}")
        if self.content_type not in (
            "podcast",
            "interview",
            "conversational",
            "narrated",
            "meeting",
            "music",
            "other",
        ):
            raise ValueError(f"Unknown content_type {self.content_type!r}")
        if len(self.sha256) != 64:
            raise ValueError(f"sha256 must be 64 hex characters, got {self.sha256!r}")
        if not self.source_name:
            raise ValueError(f"{self.episode_id}: source_name is required")
        if self.source_type == "direct_download" and not (
            self.license_name and self.license_url
        ):
            raise ValueError(
                f"{self.episode_id}: license_name and license_url are required for "
                "direct_download sources -- a redistributable remote file must "
                "state its licence"
            )
        _require_relative(self.original_path, "original_path")
        _require_relative(self.normalized_path, "normalized_path")
        _require_relative(self.transcript_path, "transcript_path")
        if self.duration_ms is not None and self.duration_ms <= 0:
            raise ValueError(
                f"{self.episode_id}: duration_ms must be positive, got {self.duration_ms}"
            )
        if self.channels is not None and self.channels < 1:
            raise ValueError(f"{self.episode_id}: channels must be >= 1")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError(f"{self.episode_id}: sample_rate_hz must be positive")
        if self.status == "normalized" and not self.normalized_path:
            raise ValueError(
                f"{self.episode_id}: status 'normalized' requires normalized_path"
            )

    @property
    def duration_seconds(self) -> float | None:
        return None if self.duration_ms is None else self.duration_ms / 1000.0

    @property
    def duration_hours(self) -> float:
        return 0.0 if self.duration_ms is None else self.duration_ms / 3_600_000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> "EpisodeRecord":
        payload = self.to_dict()
        payload.update(changes)
        return EpisodeRecord(**payload)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EpisodeRecord":
        if not isinstance(raw, Mapping):
            raise TypeError(f"Episode record must be a mapping, got {type(raw).__name__}")
        version = str(raw.get("schema_version", ""))
        if version not in SUPPORTED_EPISODE_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported episode schema_version {version!r}; this build reads "
                f"{sorted(SUPPORTED_EPISODE_SCHEMA_VERSIONS)}"
            )
        known = {f for f in EpisodeRecord.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"Episode record has unknown field(s): {sorted(unknown)}. Refusing to "
                "drop data silently."
            )
        return cls(**{key: raw[key] for key in raw})


@dataclass(frozen=True)
class EnergySummary:
    """Local loudness around a candidate, in dBFS.

    Cheap, deterministic, and computable from the normalized WAV with no model.
    ``window_ms`` records the analysis window so a value is interpretable later.
    """

    window_ms: int
    mean_dbfs: float
    min_dbfs: float
    max_dbfs: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EnergySummary":
        return cls(
            window_ms=int(raw["window_ms"]),
            mean_dbfs=float(raw["mean_dbfs"]),
            min_dbfs=float(raw["min_dbfs"]),
            max_dbfs=float(raw["max_dbfs"]),
        )


@dataclass(frozen=True)
class DatasetCandidate:
    """A candidate breakpoint in the dataset (as opposed to a product slot).

    Shares ``candidate_id`` construction with the Phase 1 ranking record
    (:func:`slotify_rank.candidates.schema.make_candidate_id`) so labels, model
    predictions and baseline rankings all join on the same key.

    ``raw_component_scores`` and ``heuristic_score`` are produced by the Phase 1
    scorer, unchanged, so a dataset row always carries the baseline's opinion of
    it. That is what makes "the model beat the heuristic" checkable per-row.
    """

    episode_id: str
    candidate_id: str
    timestamp_ms: int
    candidate_sources: tuple[str, ...]
    is_synthetic: bool = False
    eligible_for_labelling: bool = True
    eligible_for_evaluation: bool = True
    pause_duration_ms: int = 0
    silence_duration_ms: int = 0
    local_energy_summary: EnergySummary | None = None
    sentence_end: bool | None = None
    transcript_segment_id: str | None = None
    transcript_before: str | None = None
    transcript_after: str | None = None
    normalized_episode_position: float | None = None
    raw_component_scores: dict[str, Any] = field(default_factory=dict)
    heuristic_score: float | None = None
    baseline_version: str = ""
    config_version: str = ""
    candidate_generation_version: str = CANDIDATE_GENERATION_VERSION
    preprocessing_version: str = PREPROCESSING_VERSION
    label_status: LabelStatus = "unlabelled"
    dataset_split: DatasetSplit = "unassigned"
    merged_from_ms: tuple[int, ...] = ()
    schema_version: str = DATASET_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_ms, int) or isinstance(self.timestamp_ms, bool):
            raise TypeError(
                f"timestamp_ms must be an int (canonical identity is integer "
                f"milliseconds), got {self.timestamp_ms!r}"
            )
        if self.timestamp_ms < 0:
            raise ValueError(f"timestamp_ms must be non-negative, got {self.timestamp_ms}")
        if not self.candidate_sources:
            raise ValueError(
                f"{self.candidate_id}: candidate_sources must not be empty -- a "
                "candidate with no provenance cannot be audited"
            )
        if self.pause_duration_ms < 0 or self.silence_duration_ms < 0:
            raise ValueError(f"{self.candidate_id}: durations must be non-negative")
        # The synthetic-exclusion invariant, enforced rather than trusted.
        if self.is_synthetic and (
            self.eligible_for_labelling or self.eligible_for_evaluation
        ):
            raise ValueError(
                f"{self.candidate_id}: synthetic product padding must have "
                "eligible_for_labelling=False and eligible_for_evaluation=False"
            )
        if self.label_status not in ("unlabelled", "human", "weak_heuristic"):
            raise ValueError(f"Unknown label_status {self.label_status!r}")
        if self.dataset_split not in (
            "train",
            "validation",
            "test",
            "development",
            "unassigned",
        ):
            raise ValueError(f"Unknown dataset_split {self.dataset_split!r}")

    @property
    def timestamp_seconds(self) -> float:
        return self.timestamp_ms / 1000.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_sources"] = list(self.candidate_sources)
        payload["merged_from_ms"] = list(self.merged_from_ms)
        return payload

    def replace(self, **changes: Any) -> "DatasetCandidate":
        payload = self.to_dict()
        payload.update(changes)
        return DatasetCandidate.from_mapping(payload)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetCandidate":
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Candidate record must be a mapping, got {type(raw).__name__}"
            )
        version = str(raw.get("schema_version", ""))
        if version not in SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported candidate schema_version {version!r}; this build reads "
                f"{sorted(SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS)}"
            )
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"Candidate record has unknown field(s): {sorted(unknown)}. Refusing "
                "to drop data silently."
            )
        payload = dict(raw)
        payload["candidate_sources"] = tuple(payload.get("candidate_sources", ()))
        payload["merged_from_ms"] = tuple(
            int(value) for value in payload.get("merged_from_ms", ())
        )
        energy = payload.get("local_energy_summary")
        if isinstance(energy, Mapping):
            payload["local_energy_summary"] = EnergySummary.from_mapping(energy)
        return cls(**payload)


def sort_candidates(
    candidates: Sequence[DatasetCandidate],
) -> list[DatasetCandidate]:
    """Canonical ordering: episode, then timestamp, then candidate id.

    Manifests are written in this order so a regenerated manifest diffs cleanly
    against the previous one.
    """
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.episode_id,
            candidate.timestamp_ms,
            candidate.candidate_id,
        ),
    )
