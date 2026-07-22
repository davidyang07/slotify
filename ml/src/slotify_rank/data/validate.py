"""Comprehensive dataset validation.

Two severities, and the distinction is the whole design:

**ERROR** — an integrity violation. The dataset is describing something that is
not true: a duplicate ID, a checksum that no longer matches, a candidate outside
its episode, a label pointing at nothing, an episode in two splits, synthetic
padding marked eligible. Any error means the command exits non-zero and no
downstream number should be trusted.

**WARNING** — a state that is incomplete but legitimate during development: a
smoke dataset too small to split meaningfully, episodes not yet normalized, no
labels yet. These do not fail the command, because failing them would make the
early workflow unusable, but they are printed and included in the report so a
"clean" run is never confused with a complete one.

Checksum verification is opt-in (``--deep``) because it re-hashes every file in
the corpus; everything else runs in milliseconds and is always on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from slotify_rank.config.versions import (
    PACKAGE_VERSION,
    SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS,
    SUPPORTED_EPISODE_SCHEMA_VERSIONS,
)
from slotify_rank.data.checksum import sha256_file
from slotify_rank.data.paths import DataPaths, is_within
from slotify_rank.data.schema import (
    TARGET_DOMAIN_CONTENT_TYPES,
    DatasetCandidate,
    EpisodeRecord,
)
from slotify_rank.data.splits import SplitManifest
from slotify_rank.labelling.database import (
    MAX_QUALITY_SCORE,
    MIN_QUALITY_SCORE,
    LabelRecord,
)

__all__ = ["Finding", "ValidationReport", "validate_dataset"]

_ABSOLUTE_PATH_RE = re.compile(r"^(/|[A-Za-z]:[\\/]|\\\\)")


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warning"
    check: str
    message: str
    subject: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
            "subject": self.subject,
        }


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def error(self, check: str, message: str, subject: str | None = None) -> None:
        self.findings.append(Finding("error", check, message, subject))

    def warn(self, check: str, message: str, subject: str | None = None) -> None:
        self.findings.append(Finding("warning", check, message, subject))

    def ran(self, check: str) -> None:
        self.checks_run.append(check)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "package_version": PACKAGE_VERSION,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "checks_run": sorted(set(self.checks_run)),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def write(self, path: Path) -> None:
        from slotify_rank.data.checksum import atomic_write_bytes

        atomic_write_bytes(
            Path(path),
            (json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n").encode(
                "utf-8"
            ),
        )


def _check_duplicates(
    report: ValidationReport,
    items: Iterable[str],
    check: str,
    label: str,
) -> None:
    report.ran(check)
    seen: set[str] = set()
    duplicated: set[str] = set()
    for value in items:
        if value in seen:
            duplicated.add(value)
        seen.add(value)
    for value in sorted(duplicated):
        report.error(check, f"Duplicate {label}: {value}", value)


def _validate_episodes(
    report: ValidationReport,
    episodes: Sequence[EpisodeRecord],
    paths: DataPaths,
    deep: bool,
) -> None:
    _check_duplicates(
        report, (e.episode_id for e in episodes), "duplicate_episode_ids", "episode_id"
    )

    report.ran("episode_schema_version")
    report.ran("episode_source_metadata")
    report.ran("episode_license_metadata")
    report.ran("episode_paths_within_data_roots")
    report.ran("episode_duration_valid")
    report.ran("episode_audio_readable")
    report.ran("episode_preprocessing_version")
    report.ran("episode_absolute_paths")
    if deep:
        report.ran("episode_checksums")

    for episode in episodes:
        subject = episode.episode_id
        if episode.schema_version not in SUPPORTED_EPISODE_SCHEMA_VERSIONS:
            report.error(
                "episode_schema_version",
                f"Unsupported schema_version {episode.schema_version!r}",
                subject,
            )
        if not episode.source_name or not episode.source_uri:
            report.error(
                "episode_source_metadata",
                "source_name and source_uri are both required",
                subject,
            )
        if episode.source_type == "direct_download" and not (
            episode.license_name and episode.license_url
        ):
            report.error(
                "episode_license_metadata",
                "a publicly downloaded episode must declare license_name and license_url",
                subject,
            )
        if episode.source_type in ("local_file", "existing_repository_fixture") and not (
            episode.license_name
        ):
            report.warn(
                "episode_license_metadata",
                "no licence declared; treated as private and non-redistributable",
                subject,
            )

        for field_name in ("original_path", "normalized_path", "transcript_path"):
            value = getattr(episode, field_name)
            if value is None:
                continue
            if _ABSOLUTE_PATH_RE.match(value):
                report.error(
                    "episode_absolute_paths",
                    f"{field_name} is an absolute, machine-specific path: {value!r}",
                    subject,
                )
                continue
            resolved = paths.repo_root / value
            allowed = (
                paths.data_root,
                paths.repo_root / "backend",
                paths.repo_root / "ml",
            )
            if not any(is_within(resolved, root) for root in allowed):
                report.error(
                    "episode_paths_within_data_roots",
                    f"{field_name} points outside the configured data roots: {value!r}",
                    subject,
                )

        if episode.status == "normalized":
            if episode.duration_ms is None or episode.duration_ms <= 0:
                report.error(
                    "episode_duration_valid",
                    "a normalized episode must have a positive duration_ms",
                    subject,
                )
            if not episode.preprocessing_version:
                report.error(
                    "episode_preprocessing_version",
                    "a normalized episode must record its preprocessing_version",
                    subject,
                )
            normalized = (
                None
                if episode.normalized_path is None
                else paths.absolute(episode.normalized_path)
            )
            if normalized is None or not normalized.is_file():
                report.error(
                    "episode_audio_readable",
                    f"normalized audio is missing at {episode.normalized_path!r}",
                    subject,
                )
            elif normalized.stat().st_size == 0:
                report.error(
                    "episode_audio_readable",
                    "normalized audio is zero-length",
                    subject,
                )
        else:
            report.warn(
                "episode_audio_readable",
                f"status is {episode.status!r}, not 'normalized'; it contributes "
                "nothing to processed-hours statistics",
                subject,
            )

        if deep:
            original = paths.repo_root / episode.original_path
            if original.is_file():
                actual = sha256_file(original)
                if actual != episode.sha256:
                    report.error(
                        "episode_checksums",
                        f"original audio SHA-256 mismatch (manifest {episode.sha256}, "
                        f"disk {actual})",
                        subject,
                    )
            else:
                report.error(
                    "episode_checksums",
                    f"original audio missing at {episode.original_path!r}",
                    subject,
                )
            if episode.normalized_path and episode.normalized_sha256:
                normalized = paths.absolute(episode.normalized_path)
                if normalized.is_file():
                    actual = sha256_file(normalized)
                    if actual != episode.normalized_sha256:
                        report.error(
                            "episode_checksums",
                            "normalized audio SHA-256 mismatch; re-run "
                            "`dataset normalize --force`",
                            subject,
                        )


def _validate_candidates(
    report: ValidationReport,
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
) -> None:
    _check_duplicates(
        report,
        (c.candidate_id for c in candidates),
        "duplicate_candidate_ids",
        "candidate_id",
    )
    for check in (
        "candidate_schema_version",
        "candidate_episode_exists",
        "candidate_within_episode_bounds",
        "candidate_source_consistency",
        "synthetic_candidate_eligibility",
        "candidate_preprocessing_version",
    ):
        report.ran(check)

    episode_by_id = {episode.episode_id: episode for episode in episodes}
    for candidate in candidates:
        subject = candidate.candidate_id
        if candidate.schema_version not in SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS:
            report.error(
                "candidate_schema_version",
                f"Unsupported schema_version {candidate.schema_version!r}",
                subject,
            )
        episode = episode_by_id.get(candidate.episode_id)
        if episode is None:
            report.error(
                "candidate_episode_exists",
                f"references unknown episode {candidate.episode_id!r}",
                subject,
            )
            continue
        if episode.duration_ms is not None and candidate.timestamp_ms > episode.duration_ms:
            report.error(
                "candidate_within_episode_bounds",
                f"timestamp {candidate.timestamp_ms} ms is past the end of the "
                f"episode ({episode.duration_ms} ms)",
                subject,
            )
        if not candidate.candidate_sources:
            report.error(
                "candidate_source_consistency", "has no candidate_sources", subject
            )
        if candidate.is_synthetic:
            if "product_padding" not in candidate.candidate_sources:
                report.error(
                    "candidate_source_consistency",
                    "is_synthetic but is not flagged product_padding",
                    subject,
                )
            if candidate.eligible_for_labelling or candidate.eligible_for_evaluation:
                report.error(
                    "synthetic_candidate_eligibility",
                    "synthetic product padding is marked eligible; it must never "
                    "enter labelling, training or evaluation",
                    subject,
                )
        elif "product_padding" in candidate.candidate_sources:
            report.error(
                "candidate_source_consistency",
                "flagged product_padding but is_synthetic is false",
                subject,
            )
        if not candidate.preprocessing_version:
            report.error(
                "candidate_preprocessing_version",
                "missing preprocessing_version",
                subject,
            )


def _validate_labels(
    report: ValidationReport,
    labels: Sequence[LabelRecord],
    candidates: Sequence[DatasetCandidate],
) -> None:
    for check in (
        "label_candidate_exists",
        "label_rating_range",
        "duplicate_labels",
        "label_not_on_synthetic",
    ):
        report.ran(check)

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    seen: set[tuple[str, str]] = set()
    for label in labels:
        subject = f"{label.annotator_id}/{label.candidate_id}"
        candidate = by_id.get(label.candidate_id)
        if candidate is None:
            report.error(
                "label_candidate_exists",
                f"label references candidate {label.candidate_id!r}, which is not in "
                "the candidate manifest",
                subject,
            )
        elif candidate.is_synthetic:
            report.error(
                "label_not_on_synthetic",
                "a human label exists on synthetic product padding",
                subject,
            )
        if not MIN_QUALITY_SCORE <= label.quality_score <= MAX_QUALITY_SCORE:
            report.error(
                "label_rating_range",
                f"quality_score {label.quality_score} is outside "
                f"{MIN_QUALITY_SCORE}-{MAX_QUALITY_SCORE}",
                subject,
            )
        key = (label.annotator_id, label.candidate_id)
        if key in seen:
            report.error(
                "duplicate_labels",
                "more than one active label for this annotator and candidate",
                subject,
            )
        seen.add(key)

    if not labels:
        report.warn(
            "label_candidate_exists",
            "no human labels yet; held-out evaluation is not possible until there are",
        )


def _validate_splits(
    report: ValidationReport,
    manifest: SplitManifest | None,
    episodes: Sequence[EpisodeRecord],
    candidates: Sequence[DatasetCandidate],
) -> None:
    for check in (
        "split_leakage_by_episode",
        "split_leakage_by_series",
        "split_candidate_consistency",
        "test_partition_target_domain",
    ):
        report.ran(check)

    if manifest is None:
        report.warn(
            "split_leakage_by_episode",
            "no split manifest; run `dataset split` before reporting any held-out "
            "metric",
        )
        return

    if manifest.degraded:
        report.warn(
            "split_leakage_by_episode",
            f"split is degraded: {manifest.reason}",
        )

    episode_splits: dict[str, set[str]] = {}
    for assignment in manifest.assignments:
        for episode_id in assignment.episode_ids:
            episode_splits.setdefault(episode_id, set()).add(assignment.split)
    for episode_id, splits in sorted(episode_splits.items()):
        if len(splits) > 1:
            report.error(
                "split_leakage_by_episode",
                f"episode appears in multiple partitions: {sorted(splits)}",
                episode_id,
            )

    episode_by_id = {episode.episode_id: episode for episode in episodes}
    series_splits: dict[str, set[str]] = {}
    for episode_id, splits in episode_splits.items():
        episode = episode_by_id.get(episode_id)
        if episode is None:
            report.error(
                "split_candidate_consistency",
                "split manifest references an episode not in the episode manifest",
                episode_id,
            )
            continue
        series_splits.setdefault(episode.series_id, set()).update(splits)
    for series_id, splits in sorted(series_splits.items()):
        if len(splits) > 1:
            report.error(
                "split_leakage_by_series",
                f"series spans multiple partitions: {sorted(splits)}. Episodes of one "
                "series share hosts, room and vocabulary; splitting them leaks.",
                series_id,
            )

    by_episode = manifest.by_episode
    for candidate in candidates:
        if candidate.dataset_split == "unassigned":
            continue
        expected = by_episode.get(candidate.episode_id)
        if expected is not None and candidate.dataset_split != expected:
            report.error(
                "split_candidate_consistency",
                f"candidate is marked {candidate.dataset_split!r} but its episode is "
                f"in {expected!r}",
                candidate.candidate_id,
            )

    test_episode_ids = [
        episode_id for episode_id, split in by_episode.items() if split == "test"
    ]
    if test_episode_ids:
        target = [
            episode_id
            for episode_id in test_episode_ids
            if (episode := episode_by_id.get(episode_id)) is not None
            and episode.is_target_domain
            and episode.content_type in TARGET_DOMAIN_CONTENT_TYPES
        ]
        if not target:
            report.error(
                "test_partition_target_domain",
                "the test partition contains no podcast-like target-domain audio. "
                "Headline results must be measured on the product's actual domain; "
                "music and meeting corpora cannot define them.",
            )
        elif len(target) < len(test_episode_ids):
            report.warn(
                "test_partition_target_domain",
                f"{len(test_episode_ids) - len(target)} of {len(test_episode_ids)} "
                "test episodes are out of domain; report their metrics separately",
            )
    elif not manifest.degraded:
        report.warn(
            "test_partition_target_domain", "the test partition is empty"
        )


def validate_dataset(
    episodes: Sequence[EpisodeRecord],
    candidates: Sequence[DatasetCandidate],
    paths: DataPaths,
    labels: Sequence[LabelRecord] = (),
    split_manifest: SplitManifest | None = None,
    deep: bool = False,
) -> ValidationReport:
    """Run every check. Errors mean the dataset is not trustworthy."""
    report = ValidationReport()
    if not episodes:
        report.warn("dataset_not_empty", "the episode manifest is empty")
    report.ran("dataset_not_empty")

    _validate_episodes(report, episodes, paths, deep)
    _validate_candidates(report, candidates, episodes)
    _validate_labels(report, labels, candidates)
    _validate_splits(report, split_manifest, episodes, candidates)
    return report
