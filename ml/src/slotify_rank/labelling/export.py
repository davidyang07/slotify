"""Export labels from SQLite to versioned JSONL.

The database is the working store; the export is the artifact. Exports carry the
rubric version, the acceptability threshold in force, and the schema versions of
the manifests they were joined against, so a label file is interpretable years
after the session that produced it.

Parquet is deliberately not the default: JSONL diffs, streams, and needs no
extra dependency. A ``--format parquet`` path is offered only when ``pyarrow``
happens to be installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import (
    DATASET_CANDIDATE_SCHEMA_VERSION,
    LABEL_RUBRIC_VERSION,
    LABEL_SCHEMA_VERSION,
    PACKAGE_VERSION,
)
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.schema import DatasetCandidate
from slotify_rank.labelling.database import LabelDatabase, LabelRecord

__all__ = ["ExportResult", "export_labels", "build_export_rows"]


@dataclass(frozen=True)
class ExportResult:
    path: Path
    metadata_path: Path
    row_count: int
    annotator_count: int
    orphan_count: int
    repeat_count: int = 0


def build_export_rows(
    labels: Sequence[LabelRecord],
    candidates: Mapping[str, DatasetCandidate],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Join labels onto candidates. Returns ``(rows, orphan_candidate_ids)``.

    Orphans -- labels whose candidate no longer exists in the manifest -- are
    reported, not dropped silently and not exported. A label with no candidate
    cannot be trained on and its existence usually means a manifest was
    regenerated with different parameters.

    **Consistency repeats are excluded.** A repeat is the same annotator judging
    the same candidate a second time to measure whether they agree with
    themselves. Exporting it would make the pooling step average the two
    judgements and quietly turn a quality control into extra supervision, so the
    repeats stay in the database and are read by
    :func:`slotify_rank.labelling.quality.check_label_quality` instead.
    """
    rows: list[dict[str, Any]] = []
    orphans: list[str] = []
    for label in labels:
        if label.is_repeat:
            continue
        candidate = candidates.get(label.candidate_id)
        if candidate is None:
            orphans.append(label.candidate_id)
            continue
        rows.append(
            {
                **label.to_dict(),
                "timestamp_ms": candidate.timestamp_ms,
                "candidate_sources": list(candidate.candidate_sources),
                "dataset_split": candidate.dataset_split,
                "heuristic_score": candidate.heuristic_score,
                "baseline_version": candidate.baseline_version,
                "candidate_generation_version": candidate.candidate_generation_version,
                "preprocessing_version": candidate.preprocessing_version,
                "candidate_schema_version": candidate.schema_version,
                "label_source": "human",
            }
        )
    rows.sort(key=lambda row: (row["episode_id"], row["candidate_id"], row["annotator_id"]))
    return rows, sorted(set(orphans))


def export_labels(
    database: LabelDatabase,
    candidates: Sequence[DatasetCandidate],
    destination: Path,
    dataset_version: str = "v1",
) -> ExportResult:
    """Write ``labels_<version>.jsonl`` plus a sidecar metadata file."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    labels = database.all_labels()
    rows, orphans = build_export_rows(labels, by_id)
    repeats = sum(1 for label in labels if label.is_repeat)

    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    destination = Path(destination)
    atomic_write_bytes(destination, body.encode("utf-8"))

    metadata = {
        "dataset_version": dataset_version,
        "package_version": PACKAGE_VERSION,
        "rubric_version": LABEL_RUBRIC_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "graded_relevance_rule": "graded_relevance = quality_score - 1 (0-4)",
        "candidate_schema_version": DATASET_CANDIDATE_SCHEMA_VERSION,
        "acceptable_threshold": database.acceptable_threshold,
        "acceptable_rule": f"quality_score >= {database.acceptable_threshold}",
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": len(rows),
        "annotators": sorted({row["annotator_id"] for row in rows}),
        "orphan_label_candidate_ids": orphans,
        "excluded_repeat_judgement_count": repeats,
        "label_source": "human",
        "note": (
            "Every row here is a human judgement. Weak or heuristic labels are "
            "never written to this file, and blind consistency repeats are "
            "excluded so a quality control cannot become extra supervision."
        ),
    }
    metadata_path = destination.with_suffix(destination.suffix + ".meta.json")
    atomic_write_bytes(
        metadata_path,
        (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    return ExportResult(
        path=destination,
        metadata_path=metadata_path,
        row_count=len(rows),
        annotator_count=len(metadata["annotators"]),
        orphan_count=len(orphans),
        repeat_count=repeats,
    )
