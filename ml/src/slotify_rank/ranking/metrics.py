"""Validation metrics, computed **within** each episode.

The one rule: candidates from different episodes are never placed in the same
ranked list. Pooling them produces a number that looks like NDCG and is not one
-- an episode with many high-scoring candidates would supply the top-k for
episodes it has nothing to do with, and the metric would mostly measure how
episodes differ in average label rather than how well the model ranks.

The metric implementations themselves are Phase 1's
(:mod:`slotify_rank.evaluation.metrics`), reused rather than reimplemented. This
module only groups scores by episode, converts them into that module's
prediction/judgement records, and reports which episodes had to be excluded and
why. A second NDCG implementation would eventually disagree with the first, and
the headline comparison in Phase 5 has to be against the same metric the
baseline was measured with.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from slotify_rank.datasets.ranking_dataset import EpisodeGroups
from slotify_rank.evaluation.metrics import (
    EpisodeJudgements,
    EpisodePrediction,
    MetricConfig,
    binary_f1,
    evaluate_rankings,
)

__all__ = ["ValidationMetrics", "evaluate_validation"]


@dataclass(frozen=True)
class ValidationMetrics:
    """Episode-grouped ranking metrics plus the auxiliary classification ones."""

    ndcg_at_k: float | None
    pairwise_accuracy: float | None
    precision_at_k: float | None
    recall_at_k: float | None
    mrr: float | None
    episode_count: int
    candidate_count: int
    episodes_scored: int
    episodes_excluded: int
    exclusion_reasons: Mapping[str, int]
    classification_f1: float | None = None
    classification_accuracy: float | None = None
    per_episode: tuple[Mapping[str, Any], ...] = ()
    k: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            f"ndcg_at_{self.k}": self.ndcg_at_k,
            "pairwise_accuracy": self.pairwise_accuracy,
            f"precision_at_{self.k}": self.precision_at_k,
            f"recall_at_{self.k}": self.recall_at_k,
            "mrr": self.mrr,
            "episode_count": self.episode_count,
            "candidate_count": self.candidate_count,
            "episodes_scored": self.episodes_scored,
            "episodes_excluded_from_ndcg": self.episodes_excluded,
            "ndcg_exclusion_reasons": dict(self.exclusion_reasons),
            "classification_f1": self.classification_f1,
            "classification_accuracy": self.classification_accuracy,
            "k": self.k,
        }


def evaluate_validation(
    groups: EpisodeGroups,
    scores: torch.Tensor,
    acceptability_logits: torch.Tensor | None = None,
    config: MetricConfig | None = None,
) -> ValidationMetrics:
    """Rank each episode independently and macro-average across episodes.

    ``scores`` is indexed by *materialized row*, matching what the model
    produced for the whole split; each episode selects its own rows from it.
    """
    config = config or MetricConfig()
    features = groups.features

    predictions: list[EpisodePrediction] = []
    judgements: list[EpisodeJudgements] = []
    exclusions: Counter = Counter()

    for group in groups:
        relevance = groups.relevance(group)
        if not relevance:
            exclusions["episode_has_no_eligible_candidate"] += 1
            continue
        episode_scores = {
            features.candidate_ids[row]: float(scores[row]) for row in group.rows
        }
        ranked = sorted(
            episode_scores,
            # Ties broken by candidate id so a ranking is reproducible rather
            # than dependent on sort stability across platforms.
            key=lambda candidate_id: (-episode_scores[candidate_id], candidate_id),
        )
        if max(relevance.values()) <= config.gain_offset:
            # Every candidate has zero gain, so the ideal DCG is zero and NDCG is
            # genuinely undefined. Counted, never scored as 0.0.
            exclusions["no_candidate_above_gain_offset"] += 1
        predictions.append(
            EpisodePrediction(
                episode_id=group.episode_id,
                ranked_candidate_ids=ranked,
                scores=episode_scores,
            )
        )
        judgements.append(
            EpisodeJudgements(episode_id=group.episode_id, relevance=relevance)
        )

    if not predictions:
        return ValidationMetrics(
            ndcg_at_k=None,
            pairwise_accuracy=None,
            precision_at_k=None,
            recall_at_k=None,
            mrr=None,
            episode_count=len(groups),
            candidate_count=groups.candidate_count,
            episodes_scored=0,
            episodes_excluded=len(groups),
            exclusion_reasons=dict(exclusions),
            k=config.k,
        )

    report = evaluate_rankings(predictions, judgements, config)
    aggregate = report.aggregate
    excluded = report.coverage["ndcg_at_k_n_skipped"]

    classification_f1 = None
    classification_accuracy = None
    if acceptability_logits is not None:
        rows = [row for group in groups for row in group.rows]
        truth = [bool(features.is_acceptable[row] >= 0.5) for row in rows]
        predicted = [bool(acceptability_logits[row] > 0.0) for row in rows]
        classification_f1 = binary_f1(truth, predicted)
        classification_accuracy = (
            sum(1 for a, b in zip(truth, predicted) if a == b) / len(truth)
            if truth
            else None
        )

    return ValidationMetrics(
        ndcg_at_k=aggregate["ndcg_at_k"],
        pairwise_accuracy=aggregate["pairwise_accuracy"],
        precision_at_k=aggregate["precision_at_k"],
        recall_at_k=aggregate["recall_at_k"],
        mrr=aggregate["mrr"],
        episode_count=len(groups),
        candidate_count=groups.candidate_count,
        episodes_scored=report.n_episodes,
        episodes_excluded=excluded,
        exclusion_reasons=dict(exclusions),
        classification_f1=classification_f1,
        classification_accuracy=classification_accuracy,
        per_episode=tuple(episode.to_dict() for episode in report.per_episode),
        k=config.k,
    )
