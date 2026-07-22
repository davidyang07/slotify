"""Turning Phase 3 feature records into eligible training examples.

Every candidate that does *not* become a training example is counted under a
named reason and, for the first few of each kind, kept with its detail message.
That accounting is the point of this module: a training run that reports "412
examples" against a corpus of 900 candidates is only trustworthy if the missing
488 can be attributed. Silent dropping is how a dataset loses its held-out split
and nobody finds out until the numbers are already in a report.

No feature is computed here and no model is loaded. Embeddings are read from the
per-episode ``.npy`` arrays the Phase 3 pipeline already wrote; a run of this
loader never imports Whisper or MiniLM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate
from slotify_rank.datasets.labels import LabelSet
from slotify_rank.datasets.schema import (
    AUDIO_EMBEDDING_PARTS,
    NATIVE_EMBEDDING_DIMENSION,
    TEXT_EMBEDDING_PARTS,
    TEXT_STORED_SIDES,
    DatasetSchema,
    TrainingExample,
)
from slotify_rank.embeddings.store import StaleEmbeddingCache, read_embeddings, row_lookup
from slotify_rank.features.schema import CandidateFeatureRecord

__all__ = [
    "EligibilityConfig",
    "Exclusion",
    "LoadedDataset",
    "build_examples",
]

#: Splits a training example may legitimately belong to. ``unassigned`` is
#: excluded: a candidate with no split cannot be placed on either side of the
#: train/validation boundary, and guessing is how leakage starts.
_ASSIGNED_SPLITS = frozenset({"train", "validation", "test", "development"})

#: How many detail messages to retain per exclusion reason. Enough to diagnose,
#: bounded so a systematically broken corpus does not produce a 900-line report.
_MAX_DETAILS_PER_REASON = 5


@dataclass(frozen=True)
class EligibilityConfig:
    """What this training run is willing to train on."""

    #: Feature pipeline version the run requires. Empty means "whatever the
    #: manifest says", used by tests and by the first run against a new corpus.
    required_feature_pipeline_version: str = ""
    required_feature_spec_version: str = ""
    #: When True, only ``complete`` records (both learned modalities present)
    #: become examples. The Phase 4 smoke run prefers complete records; a real
    #: run over a partly-untranscribed corpus would set this False and let the
    #: availability masks do their job.
    require_complete_multimodal: bool = False
    #: When False, unlabelled candidates still become examples (used for
    #: inference-only scoring). Supervised training always requires labels.
    require_labels: bool = True
    #: Auxiliary head enabled -> an acceptability target is mandatory.
    require_acceptability_label: bool = True
    #: Splits to keep. Empty keeps every assigned split.
    splits: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_feature_pipeline_version": self.required_feature_pipeline_version,
            "required_feature_spec_version": self.required_feature_spec_version,
            "require_complete_multimodal": self.require_complete_multimodal,
            "require_labels": self.require_labels,
            "require_acceptability_label": self.require_acceptability_label,
            "splits": list(self.splits),
        }


@dataclass(frozen=True)
class Exclusion:
    candidate_id: str
    reason: str
    detail: str


@dataclass
class LoadedDataset:
    """Eligible examples plus a full account of everything that was dropped."""

    examples: list[TrainingExample] = field(default_factory=list)
    schema: DatasetSchema | None = None
    exclusions: list[Exclusion] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    label_set: LabelSet | None = None

    @property
    def episode_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for example in self.examples:
            seen.setdefault(example.episode_id, None)
        return list(seen)

    def by_split(self, split: str) -> list[TrainingExample]:
        return [example for example in self.examples if example.split == split]

    def summary(self) -> dict[str, Any]:
        splits = Counter(example.split for example in self.examples)
        episodes_per_split: dict[str, int] = {}
        for split in splits:
            episodes_per_split[split] = len(
                {e.episode_id for e in self.examples if e.split == split}
            )
        modality = Counter(
            (
                "complete"
                if example.audio_available and example.text_available
                else "audio_only"
                if example.audio_available
                else "text_only"
                if example.text_available
                else "handcrafted_only"
            )
            for example in self.examples
        )
        details: dict[str, list[str]] = {}
        for exclusion in self.exclusions:
            bucket = details.setdefault(exclusion.reason, [])
            if len(bucket) < _MAX_DETAILS_PER_REASON:
                bucket.append(f"{exclusion.candidate_id}: {exclusion.detail}")
        return {
            "candidate_count": len(self.examples),
            "episode_count": len(self.episode_ids),
            "candidates_by_split": dict(sorted(splits.items())),
            "episodes_by_split": dict(sorted(episodes_per_split.items())),
            "modality_coverage": dict(sorted(modality.items())),
            "exclusion_counts": dict(sorted(self.counts.items())),
            "exclusion_details": details,
            "schema": self.schema.to_dict() if self.schema else None,
            "labels": self.label_set.to_dict() if self.label_set else None,
        }


class _EmbeddingCache:
    """One read per ``.npy`` per load, resolved by row id."""

    def __init__(self, paths: DataPaths):
        self._paths = paths
        self._cache: dict[str, tuple[np.ndarray, dict[str, int], int]] = {}

    def rows(self, relative_path: str) -> tuple[np.ndarray, dict[str, int], int]:
        cached = self._cache.get(relative_path)
        if cached is not None:
            return cached
        absolute = self._paths.absolute(relative_path)
        matrix, metadata = read_embeddings(absolute)
        entry = (matrix, row_lookup(metadata), metadata.dimension)
        self._cache[relative_path] = entry
        return entry


def _gather(
    cache: _EmbeddingCache,
    references: Mapping[str, Any],
    keys: Sequence[str],
    native_dimension: int,
) -> dict[str, np.ndarray]:
    """Fetch one vector per key, by row id. Raises when anything is missing."""
    vectors: dict[str, np.ndarray] = {}
    for key in keys:
        reference = references.get(key)
        if reference is None:
            raise KeyError(f"no {key!r} embedding reference")
        matrix, lookup, dimension = cache.rows(reference.path)
        if dimension != native_dimension:
            raise StaleEmbeddingCache(
                f"{reference.path} stores {dimension}-dimensional rows but the "
                f"training schema expects {native_dimension}"
            )
        index = lookup.get(reference.row_id)
        if index is None:
            raise KeyError(f"row {reference.row_id!r} is not in {reference.path}")
        vectors[key] = np.asarray(matrix[index], dtype=np.float32)
    return vectors


def _build_audio_block(vectors: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([vectors[part] for part in AUDIO_EMBEDDING_PARTS])


def _build_text_block(vectors: Mapping[str, np.ndarray]) -> np.ndarray:
    """Concatenate the stored sides plus their difference and product.

    Pure arithmetic over cached MiniLM output -- no model is loaded. See
    :mod:`slotify_rank.datasets.schema` for why the derived parts exist.
    """
    before = vectors["before"]
    after = vectors["after"]
    parts = {
        "before": before,
        "after": after,
        "difference": after - before,
        "product": before * after,
    }
    return np.concatenate([parts[name] for name in TEXT_EMBEDDING_PARTS])


def build_examples(
    paths: DataPaths,
    header: Mapping[str, Any],
    records: Sequence[CandidateFeatureRecord],
    labels: LabelSet | None,
    config: EligibilityConfig | None = None,
    candidates: Sequence[DatasetCandidate] | None = None,
    split_lookup: Mapping[str, str] | None = None,
    native_embedding_dimension: int = NATIVE_EMBEDDING_DIMENSION,
) -> LoadedDataset:
    """Project feature records onto training examples, accounting for every drop."""
    config = config or EligibilityConfig()
    result = LoadedDataset(label_set=labels)

    feature_names = tuple(str(name) for name in header.get("handcrafted_feature_names", ()))
    if not feature_names:
        raise ValueError(
            "The feature manifest header declares no handcrafted_feature_names. "
            "Without the column layout the stored vectors cannot be interpreted."
        )
    schema = DatasetSchema(
        handcrafted_feature_names=feature_names,
        handcrafted_dimension=len(feature_names),
        audio_dimension=native_embedding_dimension * len(AUDIO_EMBEDDING_PARTS),
        text_dimension=native_embedding_dimension * len(TEXT_EMBEDDING_PARTS),
        native_embedding_dimension=native_embedding_dimension,
        feature_spec_version=str(header.get("feature_spec_version", "")),
        feature_pipeline_version=str(header.get("feature_pipeline_version", "")),
    )
    result.schema = schema

    by_candidate_id = {c.candidate_id: c for c in (candidates or ())}
    cache = _EmbeddingCache(paths)

    seen: set[str] = set()
    for record in sorted(records, key=lambda r: (r.episode_id, r.timestamp_ms, r.candidate_id)):
        candidate_id = record.candidate_id
        if candidate_id in seen:
            raise ValueError(
                f"Duplicate candidate_id {candidate_id!r} in the feature manifest. "
                "A duplicated example would be trained on twice and would appear "
                "on both sides of a pair with itself."
            )
        seen.add(candidate_id)

        reason, detail, example = _classify(
            record=record,
            schema=schema,
            config=config,
            labels=labels,
            dataset_candidate=by_candidate_id.get(candidate_id),
            split_lookup=split_lookup or {},
            cache=cache,
            native_embedding_dimension=native_embedding_dimension,
        )
        result.counts[reason] += 1
        if reason == "eligible" and example is not None:
            result.examples.append(example)
        else:
            result.exclusions.append(Exclusion(candidate_id, reason, detail))

    return result


def _classify(
    record: CandidateFeatureRecord,
    schema: DatasetSchema,
    config: EligibilityConfig,
    labels: LabelSet | None,
    dataset_candidate: DatasetCandidate | None,
    split_lookup: Mapping[str, str],
    cache: _EmbeddingCache,
    native_embedding_dimension: int,
) -> tuple[str, str, TrainingExample | None]:
    """Decide one candidate's fate. Returns ``(reason, detail, example)``."""
    candidate_id = record.candidate_id

    # -- versions ---------------------------------------------------------
    if config.required_feature_pipeline_version and (
        record.feature_pipeline_version != config.required_feature_pipeline_version
    ):
        return (
            "unsupported_version",
            f"feature_pipeline_version {record.feature_pipeline_version!r} != "
            f"required {config.required_feature_pipeline_version!r}",
            None,
        )
    if config.required_feature_spec_version and (
        record.feature_spec_version != config.required_feature_spec_version
    ):
        return (
            "unsupported_version",
            f"feature_spec_version {record.feature_spec_version!r} != required "
            f"{config.required_feature_spec_version!r}",
            None,
        )
    if record.feature_spec_version and schema.feature_spec_version and (
        record.feature_spec_version != schema.feature_spec_version
    ):
        return (
            "stale_features",
            f"record spec {record.feature_spec_version!r} disagrees with the "
            f"manifest header {schema.feature_spec_version!r}",
            None,
        )

    # -- structural integrity ---------------------------------------------
    if not record.episode_id:
        return "invalid_features", "record carries no episode_id", None
    if record.feature_status == "failed":
        return (
            "invalid_features",
            f"Phase 3 marked this record failed: {record.failure_reason}",
            None,
        )
    if len(record.handcrafted_feature_values) != schema.handcrafted_dimension:
        return (
            "stale_features",
            f"{len(record.handcrafted_feature_values)} handcrafted values but the "
            f"manifest header declares {schema.handcrafted_dimension} columns",
            None,
        )

    # -- candidate provenance ---------------------------------------------
    if dataset_candidate is not None:
        if dataset_candidate.is_synthetic:
            return "synthetic_excluded", "synthetic product padding", None
        if not dataset_candidate.eligible_for_labelling:
            return "ineligible_candidate", "eligible_for_labelling=False", None
        if not dataset_candidate.eligible_for_evaluation:
            return "ineligible_candidate", "eligible_for_evaluation=False", None

    # -- split -------------------------------------------------------------
    split = record.dataset_split
    manifest_split = split_lookup.get(record.episode_id)
    if manifest_split is not None and manifest_split != split:
        return (
            "split_mismatch",
            f"feature record says {split!r} but the split manifest says "
            f"{manifest_split!r}",
            None,
        )
    if split not in _ASSIGNED_SPLITS:
        return "split_mismatch", f"split {split!r} is not an assigned split", None
    if config.splits and split not in config.splits:
        return (
            "split_mismatch",
            f"split {split!r} is not among the requested splits {list(config.splits)}",
            None,
        )

    # -- handcrafted values ------------------------------------------------
    handcrafted = np.asarray(record.handcrafted_feature_values, dtype=np.float32)
    if not np.isfinite(handcrafted).all():
        bad = int(np.count_nonzero(~np.isfinite(handcrafted)))
        return "invalid_features", f"{bad} non-finite handcrafted value(s)", None
    mask = np.asarray(record.handcrafted_missing_mask, dtype=bool)

    # -- learned modalities -------------------------------------------------
    audio = np.zeros(schema.audio_dimension, dtype=np.float32)
    text = np.zeros(schema.text_dimension, dtype=np.float32)
    audio_available = False
    text_available = False

    if record.audio_embedding_available:
        try:
            vectors = _gather(
                cache,
                record.audio_embedding_reference,
                AUDIO_EMBEDDING_PARTS,
                native_embedding_dimension,
            )
        except (KeyError, FileNotFoundError) as error:
            return "missing_features", f"audio embedding unreadable: {error}", None
        except StaleEmbeddingCache as error:
            return "stale_features", f"audio embedding stale: {error}", None
        audio = _build_audio_block(vectors)
        audio_available = True

    if record.text_embedding_available:
        try:
            vectors = _gather(
                cache,
                record.text_embedding_reference,
                TEXT_STORED_SIDES,
                native_embedding_dimension,
            )
        except (KeyError, FileNotFoundError) as error:
            return "missing_features", f"text embedding unreadable: {error}", None
        except StaleEmbeddingCache as error:
            return "stale_features", f"text embedding stale: {error}", None
        text = _build_text_block(vectors)
        text_available = True

    if config.require_complete_multimodal and not (audio_available and text_available):
        return (
            "missing_features",
            f"require_complete_multimodal is set but status is "
            f"{record.feature_status!r}",
            None,
        )
    if not np.isfinite(audio).all() or not np.isfinite(text).all():
        return "invalid_features", "a learned embedding contains NaN or infinity", None
    # A stored vector that is exactly zero everywhere is indistinguishable from
    # the sentinel used for an unavailable modality, so it cannot be trained on
    # as if it were real.
    if audio_available and not audio.any():
        return "invalid_features", "audio embedding is entirely zero", None
    if text_available and not text.any():
        return "invalid_features", "text embedding is entirely zero", None

    # -- labels ------------------------------------------------------------
    quality_score = float("nan")
    is_acceptable = False
    if config.require_labels:
        label = labels.get(candidate_id) if labels else None
        if label is None:
            return "missing_label", "no human label for this candidate", None
        quality_score = label.quality_score
        if not np.isfinite(quality_score):
            return "missing_label", "human quality score is not finite", None
        is_acceptable = label.is_acceptable
    else:
        label = labels.get(candidate_id) if labels else None
        if label is not None:
            quality_score = label.quality_score
            is_acceptable = label.is_acceptable
        else:
            quality_score = 0.0

    return (
        "eligible",
        "",
        TrainingExample(
            candidate_id=candidate_id,
            episode_id=record.episode_id,
            split=split,
            handcrafted=handcrafted,
            handcrafted_missing_mask=mask,
            audio=audio,
            audio_available=audio_available,
            text=text,
            text_available=text_available,
            quality_score=quality_score,
            is_acceptable=is_acceptable,
            feature_status=record.feature_status,
        ),
    )
