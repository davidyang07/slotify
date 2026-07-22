"""Joining handcrafted features and embeddings into candidate-level records.

Assembly is where the pieces meet, so it is also where they can silently
disagree. Three joins happen here and each is made by *identity*, never by
position or order:

* candidate -> handcrafted features, by ``candidate_id``;
* candidate -> pooled audio embedding rows, by ``candidate_id#kind``;
* candidate -> MiniLM rows, by ``candidate_id#side``.

The feature *spec* is finalized here rather than in the extractors, because two
features -- the cosine similarity between the before/after context embeddings
and its complement -- cannot exist until the text vectors do. Every record is
then vectorized against that one spec, so the manifest's column layout is
uniform by construction.

A candidate present in the manifest but absent from the extracted features is a
failure, recorded as one. It is never dropped: a silently shorter feature set is
how a corpus loses a tenth of its data without anyone noticing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from slotify_rank.config.versions import (
    FEATURE_MANIFEST_SCHEMA_VERSION,
    FEATURE_PIPELINE_VERSION,
    FEATURE_SPEC_VERSION,
    TRANSCRIPTION_VERSION,
)
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate
from slotify_rank.embeddings.minilm_text import cosine_similarity
from slotify_rank.embeddings.store import load_embeddings, row_lookup
from slotify_rank.embeddings.whisper_audio import CANDIDATE_EMBEDDING_KINDS
from slotify_rank.features.schema import (
    CandidateFeatureRecord,
    EmbeddingReference,
    FeatureSpec,
    status_for,
)

__all__ = [
    "AssemblyResult",
    "assemble_episode",
    "write_feature_manifest",
    "read_feature_manifest",
]

#: Features that can only be computed once both text vectors exist.
_EMBEDDING_DERIVED_FEATURES = ("context_cosine_similarity", "semantic_change_score")


@dataclass
class AssemblyResult:
    """Records plus the exclusions that produced them."""

    records: list[CandidateFeatureRecord] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    excluded_synthetic: int = 0

    def extend(self, other: "AssemblyResult") -> None:
        self.records.extend(other.records)
        self.failures.extend(other.failures)
        self.excluded_synthetic += other.excluded_synthetic


def _reference(path: Path, paths: DataPaths, row_id: str, dimension: int):
    return EmbeddingReference(
        path=paths.relative(path), row_id=row_id, dimension=dimension
    )


def assemble_episode(
    paths: DataPaths,
    episode_id: str,
    candidates: Sequence[DatasetCandidate],
    handcrafted: Mapping[str, Any] | None,
    split_lookup: Mapping[str, str],
    audio_array_path: Path,
    text_array_path: Path,
    spec: FeatureSpec,
) -> AssemblyResult:
    """Build the feature records for one episode.

    Missing embeddings are not an error: an episode with no transcript
    legitimately produces ``audio_only`` records. Missing *handcrafted* features
    are an error, because every candidate must have them.
    """
    from slotify_rank.pipeline.stages import audio_row_id, text_row_id

    result = AssemblyResult()
    eligible = []
    for candidate in candidates:
        if candidate.is_synthetic:
            result.excluded_synthetic += 1
            continue
        eligible.append(candidate)

    if not eligible:
        return result

    if handcrafted is None:
        for candidate in eligible:
            result.failures.append(
                (candidate.candidate_id, "handcrafted features were never extracted")
            )
        return result

    extracted = handcrafted.get("candidates", {})
    transcript_available = bool(handcrafted.get("transcript_available", False))
    audio_sha256 = str(handcrafted.get("audio_sha256", ""))

    audio_loaded = load_embeddings(audio_array_path)
    text_loaded = load_embeddings(text_array_path)
    audio_matrix, audio_metadata = audio_loaded if audio_loaded else (None, None)
    text_matrix, text_metadata = text_loaded if text_loaded else (None, None)
    audio_rows = row_lookup(audio_metadata) if audio_metadata else {}
    text_rows = row_lookup(text_metadata) if text_metadata else {}

    for candidate in eligible:
        record = extracted.get(candidate.candidate_id)
        if record is None:
            result.failures.append(
                (candidate.candidate_id, "no handcrafted features for this candidate")
            )
            continue

        values = dict(record["values"])
        missing = dict(record["missing"])

        # -- text embeddings ------------------------------------------------
        before_vector = None
        after_vector = None
        text_references: dict[str, EmbeddingReference] = {}
        if text_matrix is not None and text_metadata is not None:
            for side in ("before", "after"):
                row_id = text_row_id(candidate.candidate_id, side)
                index = text_rows.get(row_id)
                if index is None:
                    continue
                vector = text_matrix[index]
                if side == "before":
                    before_vector = vector
                else:
                    after_vector = vector
                text_references[side] = _reference(
                    text_array_path, paths, row_id, text_metadata.dimension
                )

        similarity = cosine_similarity(before_vector, after_vector)
        values["context_cosine_similarity"] = 0.0 if similarity is None else similarity
        missing["context_cosine_similarity"] = similarity is None
        values["semantic_change_score"] = (
            0.0 if similarity is None else 1.0 - similarity
        )
        missing["semantic_change_score"] = similarity is None

        text_available = before_vector is not None and after_vector is not None

        # -- audio embeddings -----------------------------------------------
        audio_references: dict[str, EmbeddingReference] = {}
        if audio_matrix is not None and audio_metadata is not None:
            for kind in CANDIDATE_EMBEDDING_KINDS:
                row_id = audio_row_id(candidate.candidate_id, kind)
                if row_id not in audio_rows:
                    continue
                audio_references[kind] = _reference(
                    audio_array_path, paths, row_id, audio_metadata.dimension
                )
        # "Available" means the representations a model would actually consume:
        # both sides of the break plus the surrounding context.
        audio_available = all(
            kind in audio_references for kind in ("before", "after", "context")
        )
        if not audio_available:
            audio_references = dict(audio_references)

        # Vectorize against the one canonical spec, which raises rather than
        # silently truncating if this candidate's names diverged.
        try:
            vector, mask = spec.vectorize(values, missing)
        except ValueError as error:
            result.failures.append((candidate.candidate_id, str(error)))
            continue

        result.records.append(
            CandidateFeatureRecord(
                episode_id=episode_id,
                candidate_id=candidate.candidate_id,
                timestamp_ms=candidate.timestamp_ms,
                dataset_split=split_lookup.get(episode_id, candidate.dataset_split),
                handcrafted_feature_values=vector,
                handcrafted_missing_mask=mask,
                feature_status=status_for(audio_available, text_available),
                audio_sha256=audio_sha256,
                transcript_version=TRANSCRIPTION_VERSION if transcript_available else "",
                audio_embedding_reference=audio_references if audio_available else {},
                text_embedding_reference=text_references if text_available else {},
                audio_embedding_dimension=(
                    audio_metadata.dimension if audio_metadata else None
                ),
                text_embedding_dimension=(
                    text_metadata.dimension if text_metadata else None
                ),
                audio_embedding_available=audio_available,
                text_embedding_available=text_available,
                transcript_available=transcript_available,
                context_segment_ids_before=tuple(
                    record.get("context_segment_ids_before", ())
                ),
                context_segment_ids_after=tuple(
                    record.get("context_segment_ids_after", ())
                ),
            )
        )

    return result


def build_spec(records: Iterable[Mapping[str, Any]]) -> FeatureSpec:
    """Canonical feature spec from the extracted name sets.

    Every episode must contribute exactly the same names; a divergence means the
    extractors ran under different configurations, and vectorizing across it
    would mis-column half the corpus.
    """
    name_sets: list[frozenset[str]] = []
    for payload in records:
        for record in payload.get("candidates", {}).values():
            name_sets.append(frozenset(record["values"]))
    if not name_sets:
        raise ValueError("No extracted features found; run the acoustic stage first.")

    reference = name_sets[0]
    for other in name_sets[1:]:
        if other != reference:
            difference = sorted(reference.symmetric_difference(other))
            raise ValueError(
                f"Episodes produced different feature names ({difference[:5]}). "
                "The extractors ran under different configurations; re-run the "
                "acoustic stage with --force."
            )
    return FeatureSpec.from_names(
        sorted(reference | set(_EMBEDDING_DERIVED_FEATURES))
    )


def write_feature_manifest(
    path: Path,
    spec: FeatureSpec,
    records: Sequence[CandidateFeatureRecord],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the JSONL manifest: one header line, then one record per line.

    JSONL rather than a single JSON document so the manifest streams: a training
    dataset reads it line by line without holding the whole corpus in memory,
    and a partial write is detectable as a truncated final line rather than as
    an unparseable file.
    """
    header = {
        "record_type": "header",
        "schema_version": FEATURE_MANIFEST_SCHEMA_VERSION,
        "feature_spec_version": spec.spec_version,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "handcrafted_feature_names": list(spec.names),
        "handcrafted_feature_count": spec.dimension,
        "record_count": len(records),
        **(dict(metadata) if metadata else {}),
    }
    lines = [json.dumps(header, ensure_ascii=False, sort_keys=True)]
    ordered = sorted(records, key=lambda r: (r.episode_id, r.timestamp_ms, r.candidate_id))
    for record in ordered:
        lines.append(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
    atomic_write_bytes(Path(path), ("\n".join(lines) + "\n").encode("utf-8"))


def read_feature_manifest(
    path: Path,
) -> tuple[dict[str, Any], list[CandidateFeatureRecord]]:
    """Read the manifest header and records."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Feature manifest not found: {file_path}")

    header: dict[str, Any] | None = None
    records: list[CandidateFeatureRecord] = []
    for number, line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{file_path}:{number} is not valid JSON: {error}") from error
        if payload.get("record_type") == "header":
            header = payload
            continue
        records.append(CandidateFeatureRecord.from_mapping(payload))

    if header is None:
        raise ValueError(
            f"{file_path} has no header line; the feature column layout is "
            "unknown and the records cannot be interpreted."
        )
    return header, records
