"""Reading human labels for supervised training.

Human labels are the **only** default supervised source. Weak or heuristic
labels are never loaded here: the whole value of the resume claim rests on the
model having learned from human judgement, and a loader that silently accepts a
heuristic-derived score would make that claim unverifiable from the code. A
future configuration may name weak labels explicitly; there is deliberately no
flag that does it by accident.

Multiple annotators per candidate are aggregated deterministically:

* ``quality_score`` -- arithmetic mean of the annotators' 1-5 scores, so a
  candidate two people scored 4 and 5 sits between two candidates they agreed
  on. Pair generation then thresholds on the *difference* of these means.
* ``is_acceptable`` -- majority vote, with an exact tie broken by the aggregated
  quality score against the exported acceptability threshold. Averaging booleans
  and rounding would silently make 0.5 depend on floating-point luck.
* ``is_unusable`` -- any annotator marking a candidate unusable removes it. One
  person noticing the clip is corrupt outweighs another not noticing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "AggregatedLabel",
    "LabelSet",
    "aggregate_labels",
    "read_label_export",
]


@dataclass(frozen=True)
class AggregatedLabel:
    """One candidate's supervised target, pooled over its annotators."""

    candidate_id: str
    episode_id: str
    quality_score: float
    is_acceptable: bool
    annotator_count: int
    annotator_ids: tuple[str, ...]
    rubric_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "quality_score": self.quality_score,
            "is_acceptable": self.is_acceptable,
            "annotator_count": self.annotator_count,
            "annotator_ids": list(self.annotator_ids),
            "rubric_version": self.rubric_version,
        }


@dataclass(frozen=True)
class LabelSet:
    """Aggregated labels plus the provenance needed to reject a mismatched set."""

    labels: Mapping[str, AggregatedLabel]
    rubric_versions: tuple[str, ...]
    acceptable_threshold: float
    source: str
    unusable_candidate_ids: tuple[str, ...] = ()
    label_source: str = "human"

    def __len__(self) -> int:
        return len(self.labels)

    def get(self, candidate_id: str) -> AggregatedLabel | None:
        return self.labels.get(candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label_source": self.label_source,
            "labelled_candidate_count": len(self.labels),
            "rubric_versions": list(self.rubric_versions),
            "acceptable_threshold": self.acceptable_threshold,
            "unusable_candidate_count": len(self.unusable_candidate_ids),
        }


def aggregate_labels(
    rows: Iterable[Mapping[str, Any]],
    acceptable_threshold: float,
    source: str = "",
) -> LabelSet:
    """Pool per-annotator rows into one target per candidate."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label_source = str(row.get("label_source", "human"))
        if label_source != "human":
            raise ValueError(
                f"{source or 'label rows'}: candidate "
                f"{row.get('candidate_id')!r} carries label_source={label_source!r}. "
                "Only human labels are a supervised source; weak labels must be "
                "enabled by an explicitly named configuration."
            )
        grouped.setdefault(str(row["candidate_id"]), []).append(row)

    labels: dict[str, AggregatedLabel] = {}
    unusable: list[str] = []
    rubric_versions: set[str] = set()

    for candidate_id in sorted(grouped):
        entries = sorted(grouped[candidate_id], key=lambda r: str(r["annotator_id"]))
        if any(bool(entry.get("is_unusable", False)) for entry in entries):
            unusable.append(candidate_id)
            continue
        scores = [float(entry["quality_score"]) for entry in entries]
        mean_score = sum(scores) / len(scores)
        votes = sum(1 for entry in entries if bool(entry.get("is_acceptable", False)))
        if votes * 2 == len(entries):
            acceptable = mean_score >= acceptable_threshold
        else:
            acceptable = votes * 2 > len(entries)
        for entry in entries:
            rubric_versions.add(str(entry.get("rubric_version", "")))
        labels[candidate_id] = AggregatedLabel(
            candidate_id=candidate_id,
            episode_id=str(entries[0]["episode_id"]),
            quality_score=mean_score,
            is_acceptable=acceptable,
            annotator_count=len(entries),
            annotator_ids=tuple(str(entry["annotator_id"]) for entry in entries),
            rubric_version=sorted(
                {str(entry.get("rubric_version", "")) for entry in entries}
            )[0],
        )

    return LabelSet(
        labels=labels,
        rubric_versions=tuple(sorted(v for v in rubric_versions if v)),
        acceptable_threshold=acceptable_threshold,
        source=source,
        unusable_candidate_ids=tuple(unusable),
    )


def read_label_export(path: Path) -> LabelSet:
    """Read ``labels_<version>.jsonl`` and its ``.meta.json`` sidecar.

    The sidecar carries the acceptability threshold that was in force when the
    labels were exported. Re-deriving it from a default here would silently
    re-bin every acceptability target if the threshold ever changed, so a
    missing sidecar is a hard failure rather than a fallback.
    """
    export_path = Path(path)
    if not export_path.is_file():
        raise FileNotFoundError(f"Label export not found: {export_path}")

    metadata_path = export_path.with_suffix(export_path.suffix + ".meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{export_path} has no metadata sidecar ({metadata_path.name}). The "
            "acceptability threshold that produced these labels is unknown, so "
            "the auxiliary targets cannot be interpreted."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "acceptable_threshold" not in metadata:
        raise ValueError(f"{metadata_path} does not record 'acceptable_threshold'")

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        export_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{export_path}:{number} is not valid JSON: {error}"
            ) from error

    return aggregate_labels(
        rows,
        acceptable_threshold=float(metadata["acceptable_threshold"]),
        source=str(export_path),
    )
