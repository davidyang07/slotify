"""Weak, heuristic-derived labels for the bootstrap distillation run.

**Read this before quoting any number produced from these labels.**

There are no human labels in this repository yet. That blocks the Phase 5B
experiment gate, and it should: a ranking-quality claim needs human judgement.
It also blocks something much smaller and entirely separate -- proving that the
*inference path* works end to end, that a real PyTorch checkpoint really does
consume real multimodal features of real audio and return a real ordering.

This module unblocks only the second thing. It grades each candidate by binning
``heuristic_offline_v1``'s own score into the 1-5 rubric, within its episode. A
model trained on that is a **distillation of the baseline onto multimodal
features**. It is:

* a genuine multimodal PyTorch ranker, genuinely trained, genuinely served;
* evidence that the feature pipeline, checkpoint format, normalizer and
  inference adapter agree with each other;
* **not** evidence that the model ranks ad breaks well, because its teacher is
  the very baseline any improvement would have to be measured against.

Every artifact it touches is stamped ``label_source: weak_heuristic``. The
loader refuses these rows unless the caller names the source, the training
summary records it, the checkpoint records it, the inference response reports
it, and ``evaluation compare`` refuses to publish a headline improvement
computed from them.

Within-episode quantile binning, rather than a global threshold on the raw
score, for one reason: ranking metrics are computed within an episode, so a
target that is constant inside an episode carries no ranking information at
all. A quiet, evenly-paced episode whose candidates all score 0.6 must still
produce an ordering.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.schema import DatasetCandidate

__all__ = [
    "WEAK_LABEL_SOURCE",
    "WEAK_RUBRIC_VERSION",
    "WEAK_ANNOTATOR_ID",
    "WeakLabelSummary",
    "grade_within_episode",
    "build_weak_label_rows",
    "write_weak_label_export",
]

WEAK_LABEL_SOURCE = "weak_heuristic"
WEAK_RUBRIC_VERSION = "weak-heuristic-rubric-v1"
#: Not a person. Named so it can never be mistaken for one in an annotator
#: count, and so `label_statistics.json` keeps reporting zero human annotators.
WEAK_ANNOTATOR_ID = "weak-heuristic-v1"

#: Acceptability threshold on the derived 1-5 grade. Matches the rubric's own
#: "acceptable" boundary so the auxiliary head sees the same cut a human
#: labelling round would apply.
WEAK_ACCEPTABLE_THRESHOLD = 4.0


@dataclass(frozen=True)
class WeakLabelSummary:
    episode_count: int
    candidate_count: int
    grades: dict[int, int]
    episodes_without_spread: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_source": WEAK_LABEL_SOURCE,
            "rubric_version": WEAK_RUBRIC_VERSION,
            "episode_count": self.episode_count,
            "candidate_count": self.candidate_count,
            "grades": {str(grade): count for grade, count in sorted(self.grades.items())},
            "episodes_without_spread": list(self.episodes_without_spread),
            "warning": (
                "WEAK LABELS. Derived from heuristic_offline_v1's own score, not "
                "from human judgement. A model trained on these distills the "
                "baseline and cannot be compared against it."
            ),
        }


def grade_within_episode(scores: Sequence[float]) -> list[int]:
    """Bin one episode's heuristic scores onto the 1-5 rubric.

    Rank-based rather than value-based: the candidates are ordered and cut into
    five equal-count bands. Ties receive the same grade, so two candidates the
    baseline could not separate are not separated here either.
    """
    count = len(scores)
    if count == 0:
        return []
    if count == 1:
        # A single candidate has no within-episode ordering to express. The
        # middle grade says exactly that: no preference was derived.
        return [3]

    order = sorted(range(count), key=lambda index: (scores[index], index))
    # Rank each candidate, giving tied scores the same (minimum) rank so an
    # arbitrary index order cannot manufacture a preference.
    ranks = [0] * count
    position = 0
    while position < count:
        end = position
        while end + 1 < count and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        for index in order[position : end + 1]:
            ranks[index] = position
        position = end + 1

    grades: list[int] = []
    for rank in ranks:
        band = int(5 * rank / count) + 1
        grades.append(min(5, max(1, band)))
    return grades


def build_weak_label_rows(
    candidates: Iterable[DatasetCandidate],
) -> tuple[list[dict[str, Any]], WeakLabelSummary]:
    """One weak label row per eligible candidate, grouped by episode."""
    by_episode: dict[str, list[DatasetCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.is_synthetic:
            continue
        if not candidate.eligible_for_labelling:
            continue
        if candidate.heuristic_score is None:
            continue
        by_episode[candidate.episode_id].append(candidate)

    rows: list[dict[str, Any]] = []
    grade_counts: dict[int, int] = defaultdict(int)
    flat: list[str] = []

    for episode_id in sorted(by_episode):
        episode_candidates = sorted(
            by_episode[episode_id], key=lambda c: (c.timestamp_ms, c.candidate_id)
        )
        scores = [float(c.heuristic_score) for c in episode_candidates]
        if len(set(scores)) == 1 and len(scores) > 1:
            flat.append(episode_id)
        grades = grade_within_episode(scores)
        for candidate, grade in zip(episode_candidates, grades):
            grade_counts[grade] += 1
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "episode_id": candidate.episode_id,
                    "annotator_id": WEAK_ANNOTATOR_ID,
                    "quality_score": float(grade),
                    "is_acceptable": grade >= WEAK_ACCEPTABLE_THRESHOLD,
                    "is_unusable": False,
                    "rubric_version": WEAK_RUBRIC_VERSION,
                    "label_source": WEAK_LABEL_SOURCE,
                    "heuristic_score": round(float(candidate.heuristic_score), 6),
                }
            )

    summary = WeakLabelSummary(
        episode_count=len(by_episode),
        candidate_count=len(rows),
        grades=dict(grade_counts),
        episodes_without_spread=tuple(flat),
    )
    return rows, summary


def write_weak_label_export(
    path: Path, rows: Sequence[Mapping[str, Any]], summary: WeakLabelSummary
) -> None:
    """Write the JSONL export and the sidecar the loader requires."""
    destination = Path(path)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_bytes(destination, payload.encode("utf-8"))

    metadata = {
        "acceptable_threshold": WEAK_ACCEPTABLE_THRESHOLD,
        "label_source": WEAK_LABEL_SOURCE,
        "rubric_version": WEAK_RUBRIC_VERSION,
        "human_annotator_count": 0,
        "human_labelled_candidate_count": 0,
        **summary.to_dict(),
    }
    atomic_write_bytes(
        destination.with_suffix(destination.suffix + ".meta.json"),
        (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
