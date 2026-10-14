"""Frozen label snapshot.

Before the real experiment runs, exactly what it is reproduced against is pinned
here: the hashes of the label export, the candidate manifest, the feature
manifest and the split, plus the label distribution and the reason every
excluded label was excluded. A snapshot is immutable once written -- a
correction is a new version, never an overwrite -- because a result reported
against ``label-snapshot-v1`` must always mean the same bytes.

The snapshot references private data (candidate ids, pseudonymous annotator
ids), so it is written under the git-ignored data root by default. The committed
experiment manifest carries only its hash, not its contents.

A snapshot nobody checks pins nothing. :func:`verify_against_snapshot` is what
``evaluation compare`` runs before it reads a single label: the export it was
handed must be byte-identical to the one frozen, and the candidate, feature and
split manifests must be the ones the snapshot recorded. Any difference is a
refusal -- an evaluation against labels or candidates that moved after the
freeze is not the frozen experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import (
    LABEL_RUBRIC_VERSION,
    LABEL_SNAPSHOT_SCHEMA_VERSION,
    PACKAGE_VERSION,
)
from slotify_rank.data.checksum import atomic_write_bytes, sha256_file, sha256_text
from slotify_rank.datasets.labels import AggregatedLabel

__all__ = [
    "FrozenLabelSnapshot",
    "build_snapshot",
    "write_snapshot",
    "read_snapshot",
    "verify_against_snapshot",
]


@dataclass(frozen=True)
class FrozenLabelSnapshot:
    snapshot_version: str
    schema_version: str
    label_manifest_hash: str
    candidate_manifest_hash: str
    feature_manifest_hash: str | None
    split_manifest_hash: str | None
    queue_hash: str | None
    rubric_version: str
    annotator_ids: tuple[str, ...]
    exported_at: str
    unique_candidate_count: int
    label_distribution: Mapping[str, Any]
    excluded_labels: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "schema_version": self.schema_version,
            "package_version": PACKAGE_VERSION,
            "label_manifest_hash": self.label_manifest_hash,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "feature_manifest_hash": self.feature_manifest_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "queue_hash": self.queue_hash,
            "rubric_version": self.rubric_version,
            "annotator_ids": list(self.annotator_ids),
            "exported_at": self.exported_at,
            "unique_candidate_count": self.unique_candidate_count,
            "label_distribution": dict(self.label_distribution),
            "excluded_labels": [dict(entry) for entry in self.excluded_labels],
        }

    def content_hash(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenLabelSnapshot":
        return cls(
            snapshot_version=str(raw["snapshot_version"]),
            schema_version=str(raw["schema_version"]),
            label_manifest_hash=str(raw["label_manifest_hash"]),
            candidate_manifest_hash=str(raw["candidate_manifest_hash"]),
            feature_manifest_hash=raw.get("feature_manifest_hash"),
            split_manifest_hash=raw.get("split_manifest_hash"),
            queue_hash=raw.get("queue_hash"),
            rubric_version=str(raw["rubric_version"]),
            annotator_ids=tuple(raw.get("annotator_ids", ())),
            exported_at=str(raw["exported_at"]),
            unique_candidate_count=int(raw["unique_candidate_count"]),
            label_distribution=dict(raw.get("label_distribution", {})),
            excluded_labels=tuple(raw.get("excluded_labels", ())),
        )


def build_snapshot(
    aggregated: Mapping[str, AggregatedLabel],
    candidate_split: Mapping[str, str],
    label_export_path: Path,
    candidate_manifest_path: Path,
    feature_manifest_path: Path | None,
    split_manifest_path: Path | None,
    excluded_labels: Sequence[Mapping[str, Any]],
    snapshot_version: str = "v1",
    queue_hash: str | None = None,
) -> FrozenLabelSnapshot:
    """Assemble a snapshot from the export, the manifests and the exclusions."""
    score_histogram: dict[str, int] = {}
    by_split: dict[str, int] = {}
    acceptable = 0
    annotators: set[str] = set()
    for cid, label in aggregated.items():
        bucket = str(int(round(label.quality_score)))
        score_histogram[bucket] = score_histogram.get(bucket, 0) + 1
        split = candidate_split.get(cid, "unassigned")
        by_split[split] = by_split.get(split, 0) + 1
        if label.is_acceptable:
            acceptable += 1
        annotators.update(label.annotator_ids)

    distribution = {
        "by_quality_score": dict(sorted(score_histogram.items())),
        "by_split": dict(sorted(by_split.items())),
        "acceptable_count": acceptable,
        "unacceptable_count": len(aggregated) - acceptable,
    }

    return FrozenLabelSnapshot(
        snapshot_version=snapshot_version,
        schema_version=LABEL_SNAPSHOT_SCHEMA_VERSION,
        label_manifest_hash=(
            sha256_file(label_export_path) if label_export_path.is_file() else ""
        ),
        candidate_manifest_hash=(
            sha256_file(candidate_manifest_path)
            if candidate_manifest_path.is_file()
            else ""
        ),
        feature_manifest_hash=(
            sha256_file(feature_manifest_path)
            if feature_manifest_path and feature_manifest_path.is_file()
            else None
        ),
        split_manifest_hash=(
            sha256_file(split_manifest_path)
            if split_manifest_path and split_manifest_path.is_file()
            else None
        ),
        queue_hash=queue_hash,
        rubric_version=LABEL_RUBRIC_VERSION,
        annotator_ids=tuple(sorted(annotators)),
        exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        unique_candidate_count=len(aggregated),
        label_distribution=distribution,
        excluded_labels=tuple(excluded_labels),
    )


def write_snapshot(
    path: Path, snapshot: FrozenLabelSnapshot, force: bool = False
) -> None:
    """Write a snapshot, refusing to overwrite a different one of the same version."""
    destination = Path(path)
    payload = json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if destination.exists() and not force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        # Compare on identity-defining content only; the export timestamp is
        # allowed to differ for a byte-identical label set.
        existing_body = {k: v for k, v in existing.items() if k != "exported_at"}
        new_body = {k: v for k, v in snapshot.to_dict().items() if k != "exported_at"}
        if existing_body == new_body:
            return
        raise FileExistsError(
            f"{destination} already exists with different content. A frozen label "
            "snapshot is immutable: a correction is a new version, never an "
            "overwrite. Bump snapshot_version."
        )
    atomic_write_bytes(destination, payload.encode("utf-8"))


def read_snapshot(path: Path) -> FrozenLabelSnapshot:
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if str(raw.get("schema_version")) != LABEL_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"{file_path} is snapshot schema {raw.get('schema_version')!r}, but "
            f"this build reads {LABEL_SNAPSHOT_SCHEMA_VERSION!r}."
        )
    return FrozenLabelSnapshot.from_mapping(raw)


def verify_against_snapshot(
    snapshot: FrozenLabelSnapshot,
    label_export: Path,
    candidate_manifest: Path | None = None,
    feature_manifest: Path | None = None,
    split_manifest: Path | None = None,
) -> list[str]:
    """Every way the artifacts on disk differ from what ``snapshot`` froze.

    Empty means verified. The label export is always checked -- it is what the
    snapshot exists to pin -- and a snapshot that recorded no export hash cannot
    verify one. The manifests are checked whenever the snapshot recorded them.
    """
    mismatches: list[str] = []

    def current(path: Path | None) -> str | None:
        return sha256_file(Path(path)) if path and Path(path).is_file() else None

    label_hash = current(label_export)
    if not snapshot.label_manifest_hash:
        mismatches.append(
            f"snapshot {snapshot.snapshot_version} recorded no label export hash (it "
            "was frozen before an export existed), so no export can be verified "
            "against it; re-export and freeze a new snapshot version"
        )
    elif label_hash is None:
        mismatches.append(f"label export {Path(label_export).name} does not exist")
    elif label_hash != snapshot.label_manifest_hash:
        mismatches.append(
            f"label export {Path(label_export).name} hashes to {label_hash[:16]}..., "
            f"but snapshot {snapshot.snapshot_version} froze "
            f"{snapshot.label_manifest_hash[:16]}..."
        )

    for name, recorded, path in (
        ("candidate manifest", snapshot.candidate_manifest_hash or None, candidate_manifest),
        ("feature manifest", snapshot.feature_manifest_hash, feature_manifest),
        ("split manifest", snapshot.split_manifest_hash, split_manifest),
    ):
        if not recorded:
            continue
        observed = current(path)
        if observed != recorded:
            mismatches.append(
                f"the {name} "
                + ("is missing" if observed is None else f"hashes to {observed[:16]}...")
                + f", but snapshot {snapshot.snapshot_version} froze {recorded[:16]}..."
            )
    return mismatches
