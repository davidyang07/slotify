"""The Phase 5B readiness gate.

This module inspects the current human labels and reports, condition by
condition, whether the genuine-label experiment may begin. It never trains and
never fabricates: every number comes from the label store, the manifests and the
frozen split. When the gate is not met it says exactly which conditions failed,
in terms an operator can act on.

The gate is intentionally strict, and the thresholds live in one place
(:class:`ReadinessGate`) so they cannot be quietly weakened in a call site. The
CLI's ``--require-ready`` turns a failing gate into a non-zero exit, so a
training script can depend on it.

The conditions, and why each exists:

* **>= 200 human-labelled unique candidates**, **>= 8 episodes**, **>= 6 series.**
  A ranking metric over a handful of episodes is noise; series breadth is what
  stops the model learning one show's editing rhythm.
* **>= 4 training / >= 1 validation / >= 1 test series.** A held-out number needs
  a held-out series, and validation must be a different series from both.
* **Usable within-episode pairs in train and validation.** The pairwise
  objective learns nothing from an episode whose labels are all equal.
* **Graded relevant candidates in test.** NDCG is undefined on a test episode
  with no relevant candidate, so at least one must exist.
* **All labels pass integrity checks**, and **all labelled candidates have a
  complete multimodal feature record** -- a model cannot consume a candidate
  whose features failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import EXPERIMENT_MANIFEST_VERSION, PACKAGE_VERSION
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.datasets.labels import AggregatedLabel
from slotify_rank.labelling.database import LabelRecord
from slotify_rank.labelling.quality import QualityReport
from slotify_rank.ranking.pairs import PairConfig

__all__ = [
    "ReadinessGate",
    "ReadinessReport",
    "compute_readiness",
    "count_usable_pairs",
]


@dataclass(frozen=True)
class ReadinessGate:
    """The Phase 5B entry conditions. Do not weaken to advance the phase."""

    min_unique_candidates: int = 200
    min_episodes: int = 8
    min_series: int = 6
    min_train_series: int = 4
    min_validation_series: int = 1
    min_test_series: int = 1
    #: Human 1-5 score at or above which a test candidate counts as "relevant"
    #: for NDCG. Matches the evaluation default.
    relevance_threshold: float = 4.0
    #: Minimum score gap for a within-episode preference pair.
    min_pair_score_difference: float = field(
        default_factory=lambda: PairConfig().minimum_score_difference
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_unique_candidates": self.min_unique_candidates,
            "min_episodes": self.min_episodes,
            "min_series": self.min_series,
            "min_train_series": self.min_train_series,
            "min_validation_series": self.min_validation_series,
            "min_test_series": self.min_test_series,
            "relevance_threshold": self.relevance_threshold,
            "min_pair_score_difference": self.min_pair_score_difference,
        }


def count_usable_pairs(
    labels_by_episode: Mapping[str, Sequence[AggregatedLabel]],
    min_difference: float,
) -> tuple[int, list[str]]:
    """Count within-episode preference pairs and list episodes with none.

    A pair is usable when two candidates in the same episode differ in pooled
    quality score by at least ``min_difference``. This mirrors the eligibility
    rule in :mod:`slotify_rank.ranking.pairs` without constructing full training
    examples, so readiness can be computed before any feature tensor is built.
    """
    total = 0
    without: list[str] = []
    for episode_id in sorted(labels_by_episode):
        members = list(labels_by_episode[episode_id])
        episode_pairs = 0
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                if abs(left.quality_score - right.quality_score) >= min_difference:
                    episode_pairs += 1
        total += episode_pairs
        if episode_pairs == 0:
            without.append(episode_id)
    return total, without


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    gate: Mapping[str, Any]
    blocking_reasons: tuple[str, ...]
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERIMENT_MANIFEST_VERSION,
            "package_version": PACKAGE_VERSION,
            "phase_5b_ready": self.ready,
            "gate": dict(self.gate),
            "blocking_reasons": list(self.blocking_reasons),
            "metrics": dict(self.metrics),
        }

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def compute_readiness(
    aggregated: Mapping[str, AggregatedLabel],
    raw_labels: Sequence[LabelRecord],
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    quality: QualityReport,
    feature_status_by_id: Mapping[str, str] | None = None,
    gate: ReadinessGate | None = None,
) -> ReadinessReport:
    """Assemble the readiness report from labels, manifests and a quality run."""
    gate = gate or ReadinessGate()
    candidate_by_id = {c.candidate_id: c for c in candidates}
    series_by_episode = {e.episode_id: e.series_id for e in episodes}
    feature_status = feature_status_by_id or {}

    # Only labelled candidates that still exist in the manifest are counted; an
    # orphan label cannot be trained on and is reported by the quality check.
    labelled = {
        cid: label
        for cid, label in aggregated.items()
        if cid in candidate_by_id
    }

    split_of = {cid: candidate_by_id[cid].dataset_split for cid in labelled}
    episodes_labelled = {label.episode_id for label in labelled.values()}
    series_labelled = {
        series_by_episode.get(label.episode_id, label.episode_id)
        for label in labelled.values()
    }

    def series_in_split(split: str) -> set[str]:
        return {
            series_by_episode.get(labelled[cid].episode_id, labelled[cid].episode_id)
            for cid in labelled
            if split_of[cid] == split
        }

    train_series = series_in_split("train")
    val_series = series_in_split("validation")
    test_series = series_in_split("test")

    # Usable pairs per split.
    def by_episode(split: str) -> dict[str, list[AggregatedLabel]]:
        grouped: dict[str, list[AggregatedLabel]] = {}
        for cid, label in labelled.items():
            if split_of[cid] == split:
                grouped.setdefault(label.episode_id, []).append(label)
        return grouped

    train_pairs, train_no_pair = count_usable_pairs(
        by_episode("train"), gate.min_pair_score_difference
    )
    val_pairs, val_no_pair = count_usable_pairs(
        by_episode("validation"), gate.min_pair_score_difference
    )

    # Graded relevant candidates in the test split.
    test_relevant = [
        cid
        for cid, label in labelled.items()
        if split_of[cid] == "test" and label.quality_score >= gate.relevance_threshold
    ]

    # Quality-score histogram and acceptability counts (pooled).
    score_histogram: dict[str, int] = {}
    acceptable = 0
    for label in labelled.values():
        bucket = str(int(round(label.quality_score)))
        score_histogram[bucket] = score_histogram.get(bucket, 0) + 1
        if label.is_acceptable:
            acceptable += 1

    labels_by_split = {
        split: sum(1 for cid in labelled if split_of[cid] == split)
        for split in sorted(set(split_of.values()))
    }

    # Feature completeness over the labelled set.
    complete = sum(
        1 for cid in labelled if feature_status.get(cid) == "complete"
    )
    missing_or_failed = sorted(
        cid
        for cid in labelled
        if feature_status.get(cid) in (None, "failed")
    )
    features_all_complete = bool(labelled) and complete == len(labelled)

    metrics = {
        "human_labelled_unique_candidates": len(labelled),
        "human_labelled_episodes": len(episodes_labelled),
        "human_labelled_series": len(series_labelled),
        "labels_by_split": labels_by_split,
        "labels_by_quality_score": dict(sorted(score_histogram.items())),
        "acceptable_count": acceptable,
        "unacceptable_count": len(labelled) - acceptable,
        "train_series": sorted(train_series),
        "validation_series": sorted(val_series),
        "test_series": sorted(test_series),
        "usable_pairs_train": train_pairs,
        "usable_pairs_validation": val_pairs,
        "episodes_without_usable_pairs_train": train_no_pair,
        "episodes_without_usable_pairs_validation": val_no_pair,
        "test_relevant_candidate_count": len(test_relevant),
        "feature_complete_labelled": complete,
        "feature_incomplete_labelled": sorted(missing_or_failed),
        "labels_pass_integrity": quality.ok,
        "integrity_error_count": len(quality.errors),
        "integrity_warning_count": len(quality.warnings),
    }

    reasons: list[str] = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            reasons.append(reason)

    require(
        len(labelled) >= gate.min_unique_candidates,
        f"human-labelled unique candidates {len(labelled)} < {gate.min_unique_candidates}",
    )
    require(
        len(episodes_labelled) >= gate.min_episodes,
        f"human-labelled episodes {len(episodes_labelled)} < {gate.min_episodes}",
    )
    require(
        len(series_labelled) >= gate.min_series,
        f"represented series {len(series_labelled)} < {gate.min_series}",
    )
    require(
        len(train_series) >= gate.min_train_series,
        f"training series {len(train_series)} < {gate.min_train_series}",
    )
    require(
        len(val_series) >= gate.min_validation_series,
        f"validation series {len(val_series)} < {gate.min_validation_series}",
    )
    require(
        len(test_series) >= gate.min_test_series,
        f"test series {len(test_series)} < {gate.min_test_series}",
    )
    require(train_pairs > 0, "training split contains no usable ranking pairs")
    require(val_pairs > 0, "validation split contains no usable ranking pairs")
    require(
        len(test_relevant) > 0,
        "test split contains no graded relevant candidates "
        f"(quality_score >= {gate.relevance_threshold})",
    )
    require(quality.ok, f"labels fail integrity validation ({len(quality.errors)} error(s))")
    require(
        features_all_complete,
        "not all labelled candidates have a complete multimodal feature record"
        + (
            f" ({len(missing_or_failed)} missing/failed)"
            if missing_or_failed
            else ""
        ),
    )

    return ReadinessReport(
        ready=not reasons,
        gate=gate.to_dict(),
        blocking_reasons=tuple(reasons),
        metrics=metrics,
    )
