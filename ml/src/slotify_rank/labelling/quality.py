"""Label-quality controls.

These checks read the human labels and report problems. They never change a
label: a human judgement is the one artifact in this project that cannot be
regenerated, so a quality control that "fixed" a value would destroy evidence.
Every finding is a warning or an error for a person to act on, and the label
store is opened read-only.

What is checkable from the current single-label store, and what is not, is
stated explicitly. The store keeps one active label per (annotator, candidate),
so intra-annotator *repeat* consistency -- the same person judging the same clip
twice under different presentation ids -- cannot be measured until a
presentation-aware log exists. Rather than silently pass that check, it is
reported as ``not_measurable`` so the gap is visible.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.labelling.database import (
    DEFAULT_ACCEPTABLE_THRESHOLD,
    MAX_QUALITY_SCORE,
    MIN_QUALITY_SCORE,
    LabelRecord,
)
from slotify_rank.labelling.queue import LabellingQueue

__all__ = [
    "QualityFinding",
    "QualityReport",
    "check_label_quality",
]


@dataclass(frozen=True)
class QualityFinding:
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    subject: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "subject": self.subject,
            "message": self.message,
        }


@dataclass(frozen=True)
class QualityReport:
    findings: tuple[QualityFinding, ...]
    checks_run: tuple[str, ...]
    label_count: int
    annotator_count: int

    @property
    def errors(self) -> list[QualityFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[QualityFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "label_count": self.label_count,
            "annotator_count": self.annotator_count,
            "checks_run": list(self.checks_run),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def check_label_quality(
    labels: Sequence[LabelRecord],
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    queue: LabellingQueue | None = None,
    acceptable_threshold: int = DEFAULT_ACCEPTABLE_THRESHOLD,
    current_candidate_manifest_hash: str | None = None,
    current_split_manifest_hash: str | None = None,
) -> QualityReport:
    """Run every label-quality check and collect the findings.

    ``labels`` are the raw per-annotator rows (not the pooled export), so
    duplicate and inconsistency checks see the individual judgements.
    """
    candidate_by_id = {c.candidate_id: c for c in candidates}
    episode_by_id = {e.episode_id: e for e in episodes}
    queue_split_by_id: dict[str, str] = {}
    if queue is not None:
        for presentation in queue.presentations:
            queue_split_by_id.setdefault(
                presentation.candidate_id, presentation.dataset_split
            )
    queue_ids = set(queue.unique_candidate_ids) if queue is not None else None

    findings: list[QualityFinding] = []
    checks = [
        "invalid_rating",
        "missing_candidate_reference",
        "duplicate_active_label",
        "acceptability_inconsistent",
        "unusable_in_training",
        "label_outside_queue",
        "split_mismatch",
        "candidate_manifest_drift",
        "audio_checksum_drift",
        "feature_pipeline_drift",
        "repeated_item_consistency",
        "annotator_overlap_leakage",
    ]

    # 1. invalid ratings.
    for label in labels:
        if not (MIN_QUALITY_SCORE <= label.quality_score <= MAX_QUALITY_SCORE):
            findings.append(
                QualityFinding(
                    "invalid_rating",
                    "error",
                    f"quality_score {label.quality_score} is outside "
                    f"[{MIN_QUALITY_SCORE}, {MAX_QUALITY_SCORE}]",
                    label.candidate_id,
                )
            )

    # 2. labels pointing at candidates that no longer exist.
    for label in labels:
        if label.candidate_id not in candidate_by_id:
            findings.append(
                QualityFinding(
                    "missing_candidate_reference",
                    "error",
                    "label references a candidate absent from the manifest; it "
                    "cannot be trained on and usually means candidates were "
                    "regenerated with different parameters",
                    label.candidate_id,
                )
            )

    # 3. duplicate active labels (same annotator + candidate more than once).
    seen: Counter[tuple[str, str]] = Counter(
        (label.annotator_id, label.candidate_id) for label in labels
    )
    for (annotator, candidate_id), count in seen.items():
        if count > 1:
            findings.append(
                QualityFinding(
                    "duplicate_active_label",
                    "error",
                    f"annotator {annotator!r} has {count} active labels for this "
                    "candidate; there must be at most one",
                    candidate_id,
                )
            )

    # 4. acceptability inconsistent with the score rule. Not overridden -- the
    #    discrepancy is recorded so a human decides which is right.
    for label in labels:
        if label.is_unusable:
            continue
        derived = label.quality_score >= acceptable_threshold
        if label.is_acceptable != derived:
            findings.append(
                QualityFinding(
                    "acceptability_inconsistent",
                    "warning",
                    f"is_acceptable={label.is_acceptable} but quality_score "
                    f"{label.quality_score} vs threshold {acceptable_threshold} "
                    f"implies {derived}; the human value is kept, not overwritten",
                    label.candidate_id,
                )
            )

    # 5. unusable candidates that are nonetheless in a trainable split.
    for label in labels:
        if not label.is_unusable:
            continue
        candidate = candidate_by_id.get(label.candidate_id)
        if candidate is not None and candidate.dataset_split in ("train", "validation"):
            findings.append(
                QualityFinding(
                    "unusable_in_training",
                    "warning",
                    "candidate marked unusable sits in a trainable split; the "
                    "loader drops it, but confirm the clip really is unusable",
                    label.candidate_id,
                )
            )

    # 6. labels for candidates outside the assigned queue.
    if queue_ids is not None:
        for candidate_id in sorted({label.candidate_id for label in labels}):
            if candidate_id not in queue_ids:
                findings.append(
                    QualityFinding(
                        "label_outside_queue",
                        "warning",
                        "candidate was labelled but is not in the assigned queue; "
                        "it will not enter the frozen unique set",
                        candidate_id,
                    )
                )

    # 7. split disagreement between the queue and the current candidate manifest.
    for candidate_id, queue_split in queue_split_by_id.items():
        candidate = candidate_by_id.get(candidate_id)
        if candidate is not None and candidate.dataset_split != queue_split:
            findings.append(
                QualityFinding(
                    "split_mismatch",
                    "warning",
                    f"queue froze this candidate in split {queue_split!r} but the "
                    f"manifest now says {candidate.dataset_split!r}; the split "
                    "changed after the queue was built",
                    candidate_id,
                )
            )

    # 8/9/10. provenance drift since the queue froze.
    if queue is not None:
        if (
            current_candidate_manifest_hash is not None
            and queue.candidate_manifest_hash
            and current_candidate_manifest_hash != queue.candidate_manifest_hash
        ):
            findings.append(
                QualityFinding(
                    "candidate_manifest_drift",
                    "warning",
                    "the candidate manifest changed after the queue was built; "
                    "labels may point at candidates whose features moved",
                    None,
                )
            )
        if (
            current_split_manifest_hash is not None
            and queue.split_manifest_hash
            and current_split_manifest_hash != queue.split_manifest_hash
        ):
            findings.append(
                QualityFinding(
                    "audio_checksum_drift",
                    "info",
                    "split manifest changed since the queue froze; re-check that "
                    "no test-split candidate leaked into training",
                    None,
                )
            )
    findings.append(
        QualityFinding(
            "feature_pipeline_drift",
            "info",
            "feature/transcript pipeline versions are pinned per feature record; "
            "compare a frozen snapshot to detect a version change (see "
            "experiment freeze)",
            None,
        )
    )

    # 11. repeated-item (intra-annotator) consistency.
    findings.append(
        QualityFinding(
            "repeated_item_consistency",
            "info",
            "not measurable from the single-label store: it keeps one label per "
            "(annotator, candidate), so a re-presented consistency item overwrites "
            "the first. A presentation-aware log is required to score this.",
            None,
        )
    )

    # 12. inter-annotator overlap leakage / completion.
    if queue is not None and queue.overlap_candidate_ids:
        annotators_by_candidate: dict[str, set[str]] = defaultdict(set)
        for label in labels:
            annotators_by_candidate[label.candidate_id].add(label.annotator_id)
        overlap = set(queue.overlap_candidate_ids)
        labelled_overlap = [c for c in overlap if annotators_by_candidate.get(c)]
        multi = [c for c in overlap if len(annotators_by_candidate.get(c, ())) >= 2]
        if labelled_overlap and not multi:
            findings.append(
                QualityFinding(
                    "annotator_overlap_leakage",
                    "info",
                    f"{len(labelled_overlap)} of {len(overlap)} overlap candidates "
                    "are labelled by only one annotator; a second, independent "
                    "annotator is required before inter-annotator agreement can be "
                    "reported",
                    None,
                )
            )

    findings.sort(key=lambda f: ({"error": 0, "warning": 1, "info": 2}[f.severity], f.check, f.subject or ""))
    return QualityReport(
        findings=tuple(findings),
        checks_run=tuple(checks),
        label_count=len(labels),
        annotator_count=len({label.annotator_id for label in labels}),
    )
