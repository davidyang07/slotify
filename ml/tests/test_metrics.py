"""Tests for the episode-level ranking metrics.

Expected values are hand-computed in the test bodies rather than taken from the
implementation, so a change in the implementation cannot quietly redefine a
metric. No network, no audio, no models.
"""

from __future__ import annotations

import math

import pytest

from slotify_rank.evaluation.metrics import (
    EpisodeJudgements,
    EpisodePrediction,
    MetricConfig,
    binary_f1,
    dcg,
    evaluate_episode,
    evaluate_rankings,
    f1_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    pairwise_ranking_accuracy,
    precision_at_k,
    recall_at_k,
)

# Rubric scores 1-5. gain = 2 ** max(0, rel - 1) - 1  ->  1:0, 2:1, 3:3, 4:7, 5:15
RELEVANCE = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 1.0, "e": 2.0}


# --------------------------------------------------------------------------
# DCG / NDCG
# --------------------------------------------------------------------------


def test_dcg_matches_hand_computation():
    # gains 3, 1 -> 3/log2(2) + 1/log2(3) = 3 + 0.63092975...
    assert dcg([3.0, 1.0]) == pytest.approx(3 + 1 / math.log2(3))


def test_dcg_of_empty_list_is_zero():
    assert dcg([]) == 0.0


def test_ndcg_of_the_ideal_ordering_is_one():
    assert ndcg_at_k(["a", "b", "c"], RELEVANCE, k=3) == pytest.approx(1.0)


def test_ndcg_of_the_worst_ordering_is_low():
    value = ndcg_at_k(["d", "e", "c"], RELEVANCE, k=3)
    assert value is not None
    assert 0.0 < value < 0.4


def test_ndcg_is_hand_verifiable():
    # ranking b, a: gains 7, 15 -> 7/1 + 15/log2(3) = 16.4649...
    # ideal a, b:   gains 15, 7 -> 15/1 + 7/log2(3) = 19.4165...
    actual = 7 + 15 / math.log2(3)
    ideal = 15 + 7 / math.log2(3)
    assert ndcg_at_k(["b", "a"], RELEVANCE, k=2) == pytest.approx(actual / ideal)


def test_ndcg_is_bounded_by_zero_and_one():
    for ranking in (["a", "b", "c"], ["d", "e", "a"], ["c"], ["e", "d"]):
        value = ndcg_at_k(ranking, RELEVANCE, k=3)
        assert value is not None
        assert 0.0 <= value <= 1.0


def test_ndcg_is_undefined_when_no_candidate_carries_gain():
    """All labels at the floor -> ideal DCG is zero, so NDCG is undefined."""
    assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, k=3) is None


def test_ndcg_handles_fewer_candidates_than_k():
    value = ndcg_at_k(["a"], RELEVANCE, k=3)
    assert value is not None
    # Best possible with one slot is a; ideal@3 uses three slots, so < 1.
    assert 0.0 < value < 1.0


def test_ndcg_of_empty_ranking_is_zero_not_undefined():
    """Relevant candidates existed and none were returned: that is a real miss."""
    assert ndcg_at_k([], RELEVANCE, k=3) == 0.0


# --------------------------------------------------------------------------
# Precision / Recall / F1
# --------------------------------------------------------------------------


def test_precision_at_k_counts_only_candidates_above_threshold():
    # threshold 4.0 -> relevant = {a, b}; top-3 = a, c, d -> 1/3
    assert precision_at_k(["a", "c", "d"], RELEVANCE, k=3) == pytest.approx(1 / 3)


def test_precision_denominator_shrinks_for_short_rankings():
    """An episode with two candidates is not penalised for lacking a third."""
    assert precision_at_k(["a", "b"], RELEVANCE, k=3) == pytest.approx(1.0)


def test_precision_of_empty_ranking_is_undefined():
    assert precision_at_k([], RELEVANCE, k=3) is None


def test_recall_normalized_by_min_relevant_and_k():
    # relevant = {a, b}; min(2, 3) = 2; one hit -> 0.5
    assert recall_at_k(["a", "c", "d"], RELEVANCE, k=3) == pytest.approx(0.5)


def test_recall_unnormalized_uses_all_relevant():
    relevance = {f"c{i}": 5.0 for i in range(6)}
    ranked = ["c0", "c1", "c2"]
    assert recall_at_k(ranked, relevance, k=3, normalize=False) == pytest.approx(0.5)
    assert recall_at_k(ranked, relevance, k=3, normalize=True) == pytest.approx(1.0)


def test_recall_is_undefined_without_relevant_candidates():
    assert recall_at_k(["d", "e"], {"d": 1.0, "e": 2.0}, k=3) is None


def test_binary_f1_hand_computed():
    # tp=2, fp=1, fn=1 -> p=2/3, r=2/3, f1=2/3
    y_true = [True, True, False, True]
    y_pred = [True, True, True, False]
    assert binary_f1(y_true, y_pred) == pytest.approx(2 / 3)


def test_binary_f1_is_zero_without_true_positives():
    assert binary_f1([True, False], [False, True]) == 0.0


def test_binary_f1_is_undefined_when_nothing_is_positive():
    assert binary_f1([False, False], [False, False]) is None


def test_binary_f1_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        binary_f1([True], [True, False])


def test_f1_at_k_treats_the_top_k_as_the_positive_set():
    # relevant = {a, b}; predicted = {a, c, d} -> tp=1, fp=2, fn=1
    # p = 1/3, r = 1/2 -> f1 = 2 * (1/6) / (5/6) = 0.4
    assert f1_at_k(["a", "c", "d"], RELEVANCE, k=3) == pytest.approx(0.4)


# --------------------------------------------------------------------------
# MRR
# --------------------------------------------------------------------------


def test_mrr_uses_the_first_relevant_position():
    assert mean_reciprocal_rank(["c", "a", "b"], RELEVANCE) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["a", "c"], RELEVANCE) == pytest.approx(1.0)
    assert mean_reciprocal_rank(["c", "d", "b"], RELEVANCE) == pytest.approx(1 / 3)


def test_mrr_is_zero_when_relevant_candidates_are_missed():
    assert mean_reciprocal_rank(["c", "d", "e"], RELEVANCE) == 0.0


def test_mrr_is_undefined_without_relevant_candidates():
    assert mean_reciprocal_rank(["d", "e"], {"d": 1.0, "e": 2.0}) is None


# --------------------------------------------------------------------------
# Pairwise ranking accuracy
# --------------------------------------------------------------------------


def test_pairwise_accuracy_is_one_for_a_perfectly_ordered_scorer():
    scores = {"a": 0.9, "b": 0.8, "c": 0.5, "d": 0.1, "e": 0.2}
    assert pairwise_ranking_accuracy(scores, RELEVANCE) == pytest.approx(1.0)


def test_pairwise_accuracy_is_zero_for_a_perfectly_inverted_scorer():
    scores = {"a": 0.1, "b": 0.2, "c": 0.5, "d": 0.9, "e": 0.8}
    assert pairwise_ranking_accuracy(scores, RELEVANCE) == pytest.approx(0.0)


def test_pairwise_accuracy_gives_half_credit_for_score_ties():
    scores = {"a": 0.5, "b": 0.5}
    assert pairwise_ranking_accuracy(scores, {"a": 5.0, "b": 3.0}) == pytest.approx(0.5)


def test_pairwise_accuracy_skips_pairs_below_the_gap():
    scores = {"a": 0.1, "b": 0.9}
    # gap is 0.5 < min_pair_gap of 1.0, so no eligible pair remains
    assert pairwise_ranking_accuracy(scores, {"a": 4.0, "b": 3.5}) is None


def test_pairwise_accuracy_is_undefined_without_eligible_pairs():
    assert pairwise_ranking_accuracy({"a": 1.0}, {"a": 5.0}) is None


# --------------------------------------------------------------------------
# Episode and aggregate evaluation
# --------------------------------------------------------------------------


def _prediction(episode_id="ep-1", ranked=("a", "b", "c"), scores=None):
    return EpisodePrediction(
        episode_id=episode_id, ranked_candidate_ids=list(ranked), scores=scores
    )


def _judgements(episode_id="ep-1", relevance=None):
    return EpisodeJudgements(
        episode_id=episode_id, relevance=dict(relevance or RELEVANCE)
    )


def test_evaluate_episode_populates_every_metric():
    metrics = evaluate_episode(
        _prediction(scores={"a": 0.9, "b": 0.8, "c": 0.5}), _judgements()
    )
    assert metrics.episode_id == "ep-1"
    assert metrics.n_candidates_ranked == 3
    assert metrics.n_labelled == 5
    assert metrics.n_relevant == 2
    assert metrics.ndcg_at_k == pytest.approx(1.0)
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.mrr == pytest.approx(1.0)
    assert metrics.pairwise_accuracy is not None


def test_pairwise_accuracy_is_none_without_scores():
    assert evaluate_episode(_prediction(), _judgements()).pairwise_accuracy is None


def test_duplicate_candidate_ids_in_a_ranking_are_rejected():
    with pytest.raises(ValueError, match="Duplicate candidate_id"):
        EpisodePrediction(episode_id="ep-1", ranked_candidate_ids=["a", "b", "a"])


def test_unlabelled_predicted_candidate_is_rejected():
    """Guards against evaluating predictions from a different candidate generation."""
    with pytest.raises(ValueError, match="no relevance label"):
        evaluate_episode(_prediction(ranked=("a", "zzz")), _judgements())


def test_episode_id_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Episode mismatch"):
        evaluate_episode(_prediction("ep-1"), _judgements("ep-2"))


def test_nan_relevance_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        EpisodeJudgements(episode_id="ep", relevance={"a": float("nan")})


def test_boolean_relevance_is_rejected():
    with pytest.raises(TypeError, match="numeric"):
        EpisodeJudgements(episode_id="ep", relevance={"a": True})


def test_empty_episode_id_is_rejected():
    with pytest.raises(ValueError, match="episode_id"):
        EpisodePrediction(episode_id="", ranked_candidate_ids=["a"])


def test_empty_ranking_yields_defined_but_missing_metrics():
    metrics = evaluate_episode(_prediction(ranked=()), _judgements())
    assert metrics.n_candidates_ranked == 0
    assert metrics.precision_at_k is None
    assert metrics.ndcg_at_k == 0.0
    assert metrics.mrr == 0.0


def test_aggregate_macro_averages_and_reports_coverage():
    predictions = [
        _prediction("ep-1", ("a", "b", "c")),
        _prediction("ep-2", ("x", "y")),
    ]
    judgements = [
        _judgements("ep-1"),
        # ep-2 has no candidate above the floor -> NDCG undefined, skipped
        _judgements("ep-2", {"x": 1.0, "y": 1.0}),
    ]
    report = evaluate_rankings(predictions, judgements)
    assert report.n_episodes == 2
    assert report.coverage["ndcg_at_k_n_evaluated"] == 1
    assert report.coverage["ndcg_at_k_n_skipped"] == 1
    # The average must come from the one defined episode only, not from 0.0.
    assert report.aggregate["ndcg_at_k"] == pytest.approx(1.0)


def test_aggregate_is_none_when_every_episode_is_undefined():
    report = evaluate_rankings(
        [_prediction("ep-1", ("x",))], [_judgements("ep-1", {"x": 1.0})]
    )
    assert report.aggregate["ndcg_at_k"] is None
    assert report.coverage["ndcg_at_k_n_evaluated"] == 0


def test_duplicate_episode_ids_in_predictions_are_rejected():
    with pytest.raises(ValueError, match="Duplicate episode_id"):
        evaluate_rankings(
            [_prediction("ep-1"), _prediction("ep-1")],
            [_judgements("ep-1")],
        )


def test_duplicate_episode_ids_in_judgements_are_rejected():
    with pytest.raises(ValueError, match="Duplicate episode_id"):
        evaluate_rankings([_prediction("ep-1")], [_judgements("ep-1"), _judgements("ep-1")])


def test_missing_judgements_are_rejected():
    with pytest.raises(ValueError, match="No judgements"):
        evaluate_rankings([_prediction("ep-1")], [_judgements("ep-2")])


def test_metric_config_records_its_own_definitions():
    config = MetricConfig(k=5, relevance_threshold=3.0)
    payload = config.to_dict()
    assert payload["k"] == 5
    assert payload["relevance_threshold"] == 3.0
    assert payload["recall_denominator"] == "min(n_relevant, k)"


def test_metric_config_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be positive"):
        MetricConfig(k=0)


def test_report_serializes_to_json_compatible_types():
    report = evaluate_rankings([_prediction()], [_judgements()])
    payload = report.to_dict()
    assert payload["n_episodes"] == 1
    assert "metric_config" in payload
    assert isinstance(payload["per_episode"], list)
