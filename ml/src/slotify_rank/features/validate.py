"""Integrity checks over the assembled feature manifest.

The failure mode this guards against is not a crash -- it is a corpus that looks
fine and is quietly wrong: a vector one column short, an embedding that belongs
to the neighbouring candidate, a test-split episode whose features were built
from a train-split transcript. All of those train a model successfully and
produce a number nobody can reproduce.

Every check returns an error rather than raising, so one run reports everything
wrong at once instead of one problem per invocation. The CLI exits non-zero when
any error is present. Warnings are reported and do not fail the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from slotify_rank.config.versions import (
    FEATURE_MANIFEST_SCHEMA_VERSION,
    FEATURE_SPEC_VERSION,
    SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS,
)
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.embeddings.store import read_embeddings
from slotify_rank.features.schema import CandidateFeatureRecord

__all__ = ["ValidationIssue", "FeatureValidationReport", "validate_features"]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    subject: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "subject": self.subject}


@dataclass
class FeatureValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    checked: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str, subject: str = "") -> None:
        self.errors.append(ValidationIssue(code, message, subject))

    def warn(self, code: str, message: str, subject: str = "") -> None:
        self.warnings.append(ValidationIssue(code, message, subject))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "checked": self.checked,
        }


def validate_features(
    paths: DataPaths,
    header: Mapping[str, Any],
    records: Sequence[CandidateFeatureRecord],
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    split_lookup: Mapping[str, str] | None = None,
    deep: bool = False,
) -> FeatureValidationReport:
    """Validate an assembled feature manifest against its inputs."""
    report = FeatureValidationReport()
    split_lookup = dict(split_lookup or {})

    # -- header and spec ---------------------------------------------------
    schema_version = str(header.get("schema_version", ""))
    if schema_version not in SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS:
        report.error(
            "unsupported_schema_version",
            f"Feature manifest schema_version {schema_version!r} is not readable by "
            f"this build ({sorted(SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS)})",
        )
        # Nothing below can be trusted if the layout is unknown.
        return report

    names: list[str] = list(header.get("handcrafted_feature_names", []))
    if not names:
        report.error(
            "missing_feature_names",
            "The manifest header lists no handcrafted feature names, so the "
            "column layout of every record is unknown",
        )
        return report
    if names != sorted(names):
        report.error(
            "unsorted_feature_names",
            "Handcrafted feature names are not in sorted order; the canonical "
            "column layout must be deterministic",
        )
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        report.error(
            "duplicate_feature_names", f"Duplicate feature name(s): {duplicates}"
        )
    if str(header.get("feature_spec_version", "")) != FEATURE_SPEC_VERSION:
        report.error(
            "stale_feature_spec",
            f"Manifest was built under feature spec "
            f"{header.get('feature_spec_version')!r} but this build is "
            f"{FEATURE_SPEC_VERSION!r}. Re-run the pipeline.",
        )

    expected_dimension = len(names)
    known_candidates = {candidate.candidate_id: candidate for candidate in candidates}
    known_episodes = {episode.episode_id: episode for episode in episodes}

    # -- per-record checks -------------------------------------------------
    seen: dict[str, int] = {}
    for record in records:
        subject = record.candidate_id

        if record.candidate_id in seen:
            report.error(
                "duplicate_record",
                f"{record.candidate_id} appears more than once in the manifest",
                subject,
            )
        seen[record.candidate_id] = seen.get(record.candidate_id, 0) + 1

        if len(record.handcrafted_feature_values) != expected_dimension:
            report.error(
                "wrong_vector_dimension",
                f"{record.candidate_id} has "
                f"{len(record.handcrafted_feature_values)} feature values but the "
                f"header declares {expected_dimension}",
                subject,
            )
        if len(record.handcrafted_missing_mask) != len(
            record.handcrafted_feature_values
        ):
            report.error(
                "mask_length_mismatch",
                f"{record.candidate_id} has "
                f"{len(record.handcrafted_missing_mask)} mask entries for "
                f"{len(record.handcrafted_feature_values)} values",
                subject,
            )

        values = np.asarray(record.handcrafted_feature_values, dtype=np.float64)
        if values.size and not np.isfinite(values).all():
            bad = [
                names[index]
                for index in np.flatnonzero(~np.isfinite(values))
                if index < len(names)
            ]
            report.error(
                "non_finite_feature",
                f"{record.candidate_id} has NaN or infinite value(s) for {bad[:5]}",
                subject,
            )

        source = known_candidates.get(record.candidate_id)
        if source is None:
            report.error(
                "unknown_candidate",
                f"{record.candidate_id} is not present in the candidate manifest",
                subject,
            )
        else:
            if source.is_synthetic:
                report.error(
                    "synthetic_candidate_included",
                    f"{record.candidate_id} is synthetic product padding and must "
                    "never receive features or enter an eligible count",
                    subject,
                )
            if source.timestamp_ms != record.timestamp_ms:
                report.error(
                    "timestamp_mismatch",
                    f"{record.candidate_id} is at {record.timestamp_ms} ms in the "
                    f"feature manifest but {source.timestamp_ms} ms in the "
                    "candidate manifest",
                    subject,
                )

        episode = known_episodes.get(record.episode_id)
        if episode is None:
            report.error(
                "unknown_episode",
                f"{record.candidate_id} references unknown episode "
                f"{record.episode_id}",
                subject,
            )
        else:
            duration = episode.normalized_duration_ms or episode.duration_ms
            if duration and not 0 <= record.timestamp_ms <= duration:
                report.error(
                    "timestamp_out_of_bounds",
                    f"{record.candidate_id} at {record.timestamp_ms} ms lies "
                    f"outside episode [0, {duration}] ms",
                    subject,
                )
            if episode.normalized_sha256 and record.audio_sha256:
                if episode.normalized_sha256 != record.audio_sha256:
                    report.error(
                        "audio_checksum_mismatch",
                        f"{record.candidate_id} was built from audio "
                        f"{record.audio_sha256[:12]} but the episode now has "
                        f"{episode.normalized_sha256[:12]}. Re-run the pipeline.",
                        subject,
                    )

        # Split agreement: features built under one split assignment and read
        # under another is how leakage enters without anyone editing a split.
        if split_lookup:
            expected_split = split_lookup.get(record.episode_id)
            if expected_split and expected_split != record.dataset_split:
                report.error(
                    "split_mismatch",
                    f"{record.candidate_id} is recorded in split "
                    f"{record.dataset_split!r} but the split manifest assigns its "
                    f"episode to {expected_split!r}",
                    subject,
                )

        # Availability flags must be backed by real references.
        if record.audio_embedding_available and not record.audio_embedding_reference:
            report.error(
                "missing_audio_reference",
                f"{record.candidate_id} is marked audio-available with no reference",
                subject,
            )
        if record.text_embedding_available and not record.text_embedding_reference:
            report.error(
                "missing_text_reference",
                f"{record.candidate_id} is marked text-available with no reference",
                subject,
            )
        if record.feature_status == "complete" and not (
            record.audio_embedding_available and record.text_embedding_available
        ):
            report.error(
                "incomplete_marked_complete",
                f"{record.candidate_id} is marked 'complete' without both learned "
                "modalities",
                subject,
            )

    # -- candidate coverage ------------------------------------------------
    covered = set(seen)
    expected_candidates = {
        candidate.candidate_id
        for candidate in candidates
        if not candidate.is_synthetic
        and candidate.episode_id in {record.episode_id for record in records}
    }
    uncovered = sorted(expected_candidates - covered)
    if uncovered:
        report.warn(
            "uncovered_candidates",
            f"{len(uncovered)} eligible candidate(s) in processed episodes have no "
            f"feature record, e.g. {uncovered[:3]}",
        )

    # -- embedding arrays --------------------------------------------------
    if deep:
        _validate_arrays(paths, records, report)

    report.checked = {
        "records": len(records),
        "episodes": len({record.episode_id for record in records}),
        "handcrafted_feature_count": expected_dimension,
        "deep": deep,
    }
    return report


def _validate_arrays(
    paths: DataPaths,
    records: Sequence[CandidateFeatureRecord],
    report: FeatureValidationReport,
) -> None:
    """Open every referenced array and confirm the rows resolve.

    Slow, so it is opt-in. It is the only check that proves a reference is not
    merely well-formed but actually points at a vector.
    """
    cache: dict[str, Any] = {}
    for record in records:
        references = list(record.audio_embedding_reference.items()) + list(
            record.text_embedding_reference.items()
        )
        for kind, reference in references:
            if reference.path not in cache:
                try:
                    matrix, metadata = read_embeddings(paths.absolute(reference.path))
                    cache[reference.path] = (
                        matrix,
                        metadata,
                        {row: i for i, row in enumerate(metadata.row_ids)},
                    )
                except (FileNotFoundError, ValueError) as error:
                    cache[reference.path] = None
                    report.error(
                        "unreadable_embedding_array",
                        f"{reference.path}: {error}",
                        record.candidate_id,
                    )
            entry = cache[reference.path]
            if entry is None:
                continue
            matrix, metadata, lookup = entry
            if reference.row_id not in lookup:
                report.error(
                    "missing_embedding_row",
                    f"{record.candidate_id}: row {reference.row_id!r} is absent "
                    f"from {reference.path}",
                    record.candidate_id,
                )
                continue
            if metadata.dimension != reference.dimension:
                report.error(
                    "embedding_dimension_mismatch",
                    f"{record.candidate_id}: reference declares dimension "
                    f"{reference.dimension} but the array has {metadata.dimension}",
                    record.candidate_id,
                )
                continue
            vector = matrix[lookup[reference.row_id]]
            if not np.isfinite(vector).all():
                report.error(
                    "non_finite_embedding",
                    f"{record.candidate_id}: {kind} embedding contains NaN or "
                    "infinity",
                    record.candidate_id,
                )
