"""Episode-level ranking metrics for the Slotify breakpoint ranker.

Every metric is computed **per episode** and then macro-averaged over episodes,
because episodes differ enormously in candidate count and a micro-average would
let a handful of long episodes dominate.

Undefined cases return :data:`None` rather than ``0.0``, and are excluded from
the aggregate with a reported ``n_skipped``. This is deliberate: scoring an
undefined episode as zero deflates a model, and scoring it as one inflates it.
Both would make the headline comparison meaningless.

Definitions used here
---------------------

``relevance``
    A graded human label per candidate. The project's rubric is 1-5
    (see ``docs/multimodal-ranking-mvp-plan.md`` §12).

``gain``
    ``2 ** max(0, relevance - gain_offset) - 1`` with ``gain_offset = 1.0``, so
    a rubric score of 1 ("disruptive") contributes zero gain and 5 contributes
    15. NDCG uses graded gain.

``binary relevance``
    ``relevance >= relevance_threshold`` (default 4.0, i.e. "strong natural
    break" or better). Precision, recall, F1 and MRR use binary relevance.

Phase 1 scope: these operate on saved rankings. No trained model exists yet, so
no relative-improvement figure is computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import fmean
from typing import Mapping, Sequence

__all__ = [
    "MetricConfig",
    "EpisodeJudgements",
    "EpisodePrediction",
    "EpisodeMetrics",
    "EvaluationReport",
    "dcg",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "binary_f1",
    "mean_reciprocal_rank",
    "pairwise_ranking_accuracy",
    "evaluate_episode",
    "evaluate_rankings",
]

DEFAULT_K = 3
DEFAULT_RELEVANCE_THRESHOLD = 4.0
DEFAULT_GAIN_OFFSET = 1.0
DEFAULT_MIN_PAIR_GAP = 1.0


@dataclass(frozen=True)
class MetricConfig:
    """Knobs shared by every metric, so a report can state exactly what it used."""

    k: int = DEFAULT_K
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    gain_offset: float = DEFAULT_GAIN_OFFSET
    min_pair_gap: float = DEFAULT_MIN_PAIR_GAP
    #: When True, Recall@k divides by ``min(n_relevant, k)`` so an episode with
    #: more than k relevant candidates is not penalised for a k-length list.
    #: When False, it divides by ``n_relevant`` (textbook recall).
    normalize_recall: bool = True

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.min_pair_gap <= 0:
            raise ValueError(f"min_pair_gap must be positive, got {self.min_pair_gap}")

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "relevance_threshold": self.relevance_threshold,
            "gain_offset": self.gain_offset,
            "min_pair_gap": self.min_pair_gap,
            "normalize_recall": self.normalize_recall,
            "recall_denominator": "min(n_relevant, k)"
            if self.normalize_recall
            else "n_relevant",
        }


@dataclass(frozen=True)
class EpisodeJudgements:
    """Human relevance labels for one episode."""

    episode_id: str
    relevance: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        for candidate_id, value in self.relevance.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Relevance for {candidate_id!r} must be numeric, got {value!r}"
                )
            if math.isnan(value):
                raise ValueError(f"Relevance for {candidate_id!r} is NaN")


@dataclass(frozen=True)
class EpisodePrediction:
    """A ranked list of candidate IDs for one episode, best first."""

    episode_id: str
    ranked_candidate_ids: Sequence[str]
    #: Optional continuous scores, required only by pairwise ranking accuracy.
    scores: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        seen: set[str] = set()
        for candidate_id in self.ranked_candidate_ids:
            if candidate_id in seen:
                raise ValueError(
                    f"Duplicate candidate_id {candidate_id!r} in the ranking for "
                    f"episode {self.episode_id!r}"
                )
            seen.add(candidate_id)


def _gain(relevance: float, gain_offset: float) -> float:
    return 2 ** max(0.0, relevance - gain_offset) - 1


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the standard ``log2(i + 1)`` discount."""
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def _validate(prediction: EpisodePrediction, judgements: EpisodeJudgements) -> None:
    if prediction.episode_id != judgements.episode_id:
        raise ValueError(
            f"Episode mismatch: prediction {prediction.episode_id!r} vs "
            f"judgements {judgements.episode_id!r}"
        )
    unknown = [
        candidate_id
        for candidate_id in prediction.ranked_candidate_ids
        if candidate_id not in judgements.relevance
    ]
    if unknown:
        # A predicted ID with no label means the prediction and the label set
        # were built from different candidate generations, which silently
        # corrupts every metric. Fail loudly instead.
        raise ValueError(
            f"Episode {prediction.episode_id!r}: {len(unknown)} predicted candidate(s) "
            f"have no relevance label (e.g. {unknown[:3]}). Predictions and labels must "
            "come from the same candidate generation."
        )


def ndcg_at_k(
    ranked_candidate_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int = DEFAULT_K,
    gain_offset: float = DEFAULT_GAIN_OFFSET,
) -> float | None:
    """Graded NDCG@k. ``None`` when the ideal DCG is zero (no gain available)."""
    ideal_gains = sorted(
        (_gain(value, gain_offset) for value in relevance.values()), reverse=True
    )[:k]
    idcg = dcg(ideal_gains)
    if idcg == 0:
        return None
    actual = [
        _gain(relevance[candidate_id], gain_offset)
        for candidate_id in ranked_candidate_ids[:k]
    ]
    return dcg(actual) / idcg


def _relevant_ids(relevance: Mapping[str, float], threshold: float) -> set[str]:
    return {cid for cid, value in relevance.items() if value >= threshold}


def precision_at_k(
    ranked_candidate_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float | None:
    """Fraction of the top-k that is relevant.

    The denominator is ``min(k, len(ranked))`` so an episode that simply has
    fewer than k candidates is not penalised for candidates it never had.
    ``None`` when the ranking is empty.
    """
    top = list(ranked_candidate_ids[:k])
    if not top:
        return None
    relevant = _relevant_ids(relevance, threshold)
    return sum(1 for cid in top if cid in relevant) / len(top)


def recall_at_k(
    ranked_candidate_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    normalize: bool = True,
) -> float | None:
    """Share of relevant candidates retrieved in the top-k.

    ``None`` when the episode has no relevant candidate at all -- recall is
    genuinely undefined there, and returning 0.0 would punish a model for an
    episode where nothing was findable.
    """
    relevant = _relevant_ids(relevance, threshold)
    if not relevant:
        return None
    hits = sum(1 for cid in ranked_candidate_ids[:k] if cid in relevant)
    denominator = min(len(relevant), k) if normalize else len(relevant)
    return hits / denominator


def binary_f1(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float | None:
    """Binary F1 over aligned label/prediction sequences.

    ``None`` when neither a true positive, a false positive nor a false negative
    exists (i.e. precision and recall are both undefined).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length, got {len(y_true)} and {len(y_pred)}"
        )
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth and pred)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if pred and not truth)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth and not pred)
    if tp + fp + fn == 0:
        return None
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def f1_at_k(
    ranked_candidate_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float | None:
    """Binary F1 treating the top-k as the positive prediction set."""
    if not relevance:
        return None
    top = set(ranked_candidate_ids[:k])
    candidate_ids = sorted(relevance)
    y_true = [relevance[cid] >= threshold for cid in candidate_ids]
    y_pred = [cid in top for cid in candidate_ids]
    return binary_f1(y_true, y_pred)


def mean_reciprocal_rank(
    ranked_candidate_ids: Sequence[str],
    relevance: Mapping[str, float],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float | None:
    """Reciprocal rank of the first relevant candidate.

    ``None`` when the episode has no relevant candidate. ``0.0`` when relevant
    candidates exist but none appear in the ranking -- that is a real miss, not
    an undefined case.
    """
    relevant = _relevant_ids(relevance, threshold)
    if not relevant:
        return None
    for index, candidate_id in enumerate(ranked_candidate_ids):
        if candidate_id in relevant:
            return 1 / (index + 1)
    return 0.0


def pairwise_ranking_accuracy(
    scores: Mapping[str, float],
    relevance: Mapping[str, float],
    min_gap: float = DEFAULT_MIN_PAIR_GAP,
) -> float | None:
    """Share of clearly-ordered label pairs that the scores order correctly.

    Only pairs whose relevance differs by at least ``min_gap`` are counted, so
    the metric is not dominated by near-ties the annotator could not have
    distinguished. A score tie counts as half credit, the standard convention
    (it is neither correct nor a confident error). ``None`` when no eligible
    pair exists.
    """
    ids = sorted(set(scores) & set(relevance))
    correct = 0.0
    total = 0
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            gap = relevance[left] - relevance[right]
            if abs(gap) < min_gap:
                continue
            total += 1
            score_gap = scores[left] - scores[right]
            if score_gap == 0:
                correct += 0.5
            elif (score_gap > 0) == (gap > 0):
                correct += 1.0
    if total == 0:
        return None
    return correct / total


@dataclass(frozen=True)
class EpisodeMetrics:
    """All metrics for one episode. ``None`` marks an undefined metric."""

    episode_id: str
    n_candidates_ranked: int
    n_labelled: int
    n_relevant: int
    ndcg_at_k: float | None
    precision_at_k: float | None
    recall_at_k: float | None
    f1_at_k: float | None
    mrr: float | None
    pairwise_accuracy: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "n_candidates_ranked": self.n_candidates_ranked,
            "n_labelled": self.n_labelled,
            "n_relevant": self.n_relevant,
            "ndcg_at_k": self.ndcg_at_k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "mrr": self.mrr,
            "pairwise_accuracy": self.pairwise_accuracy,
        }


def evaluate_episode(
    prediction: EpisodePrediction,
    judgements: EpisodeJudgements,
    config: MetricConfig | None = None,
) -> EpisodeMetrics:
    """Compute every metric for a single episode."""
    config = config or MetricConfig()
    _validate(prediction, judgements)
    relevance = judgements.relevance
    ranked = list(prediction.ranked_candidate_ids)

    return EpisodeMetrics(
        episode_id=prediction.episode_id,
        n_candidates_ranked=len(ranked),
        n_labelled=len(relevance),
        n_relevant=len(_relevant_ids(relevance, config.relevance_threshold)),
        ndcg_at_k=ndcg_at_k(ranked, relevance, config.k, config.gain_offset),
        precision_at_k=precision_at_k(
            ranked, relevance, config.k, config.relevance_threshold
        ),
        recall_at_k=recall_at_k(
            ranked,
            relevance,
            config.k,
            config.relevance_threshold,
            config.normalize_recall,
        ),
        f1_at_k=f1_at_k(ranked, relevance, config.k, config.relevance_threshold),
        mrr=mean_reciprocal_rank(ranked, relevance, config.relevance_threshold),
        pairwise_accuracy=(
            pairwise_ranking_accuracy(prediction.scores, relevance, config.min_pair_gap)
            if prediction.scores is not None
            else None
        ),
    )


@dataclass(frozen=True)
class EvaluationReport:
    """Macro-averaged metrics plus the per-episode detail behind them."""

    config: MetricConfig
    per_episode: tuple[EpisodeMetrics, ...]
    aggregate: Mapping[str, float | None]
    coverage: Mapping[str, int]
    n_episodes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_config": self.config.to_dict(),
            "n_episodes": self.n_episodes,
            "aggregate": dict(self.aggregate),
            "coverage": dict(self.coverage),
            "per_episode": [episode.to_dict() for episode in self.per_episode],
        }


_METRIC_FIELDS = (
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "mrr",
    "pairwise_accuracy",
)


def evaluate_rankings(
    predictions: Sequence[EpisodePrediction],
    judgements: Sequence[EpisodeJudgements],
    config: MetricConfig | None = None,
) -> EvaluationReport:
    """Evaluate a set of episodes and macro-average the result.

    Raises on duplicate episode IDs on either side, and on a prediction with no
    matching judgements. Episodes for which a metric is undefined are excluded
    from that metric's average and counted in ``coverage``.
    """
    config = config or MetricConfig()

    prediction_ids = [p.episode_id for p in predictions]
    duplicates = {eid for eid in prediction_ids if prediction_ids.count(eid) > 1}
    if duplicates:
        raise ValueError(f"Duplicate episode_id(s) in predictions: {sorted(duplicates)}")

    judgement_ids = [j.episode_id for j in judgements]
    judgement_duplicates = {eid for eid in judgement_ids if judgement_ids.count(eid) > 1}
    if judgement_duplicates:
        raise ValueError(
            f"Duplicate episode_id(s) in judgements: {sorted(judgement_duplicates)}"
        )

    by_id = {j.episode_id: j for j in judgements}
    missing = [eid for eid in prediction_ids if eid not in by_id]
    if missing:
        raise ValueError(f"No judgements for predicted episode(s): {missing}")

    per_episode = tuple(
        evaluate_episode(prediction, by_id[prediction.episode_id], config)
        for prediction in predictions
    )

    aggregate: dict[str, float | None] = {}
    coverage: dict[str, int] = {"n_episodes": len(per_episode)}
    for name in _METRIC_FIELDS:
        values = [
            value
            for value in (getattr(episode, name) for episode in per_episode)
            if value is not None
        ]
        aggregate[name] = fmean(values) if values else None
        coverage[f"{name}_n_evaluated"] = len(values)
        coverage[f"{name}_n_skipped"] = len(per_episode) - len(values)

    return EvaluationReport(
        config=config,
        per_episode=per_episode,
        aggregate=aggregate,
        coverage=coverage,
        n_episodes=len(per_episode),
    )
