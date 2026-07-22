"""Ranking loss, auxiliary loss, and episode-grouped validation metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slotify_rank.datasets.normalizer import fit_normalizer
from slotify_rank.datasets.ranking_dataset import EpisodeGroups, materialize
from slotify_rank.evaluation.metrics import MetricConfig
from slotify_rank.ranking.losses import (
    LossConfig,
    acceptability_loss,
    class_weight_from_training,
    combined_loss,
    margin_ranking_loss,
    pairwise_accuracy,
)
from slotify_rank.ranking.metrics import evaluate_validation
from tests.test_ranking_pairs import make_example
from tests.training_fixtures import load_corpus


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------


def test_correct_ordering_has_zero_loss_beyond_the_margin():
    left = torch.tensor([2.0])
    right = torch.tensor([0.0])
    target = torch.tensor([1.0])
    assert float(margin_ranking_loss(left, right, target, margin=0.2)) == 0.0


def test_reversed_ordering_has_a_higher_loss():
    target = torch.tensor([1.0])
    correct = margin_ranking_loss(
        torch.tensor([2.0]), torch.tensor([0.0]), target, margin=0.2
    )
    reversed_order = margin_ranking_loss(
        torch.tensor([0.0]), torch.tensor([2.0]), target, margin=0.2
    )
    assert float(reversed_order) > float(correct)
    assert float(reversed_order) == pytest.approx(2.2)


def test_target_minus_one_reverses_the_expectation():
    """target=-1 means the RIGHT candidate should score higher."""
    left = torch.tensor([0.0])
    right = torch.tensor([2.0])
    assert float(
        margin_ranking_loss(left, right, torch.tensor([-1.0]), margin=0.2)
    ) == 0.0
    assert float(
        margin_ranking_loss(left, right, torch.tensor([1.0]), margin=0.2)
    ) == pytest.approx(2.2)


def test_margin_demands_separation():
    left = torch.tensor([0.05])
    right = torch.tensor([0.0])
    target = torch.tensor([1.0])
    assert float(margin_ranking_loss(left, right, target, margin=0.0)) == 0.0
    assert float(margin_ranking_loss(left, right, target, margin=0.2)) > 0.0


def test_a_larger_margin_never_lowers_the_loss():
    left = torch.tensor([1.0, 0.4])
    right = torch.tensor([0.0, 0.3])
    target = torch.tensor([1.0, 1.0])
    losses = [
        float(margin_ranking_loss(left, right, target, margin=margin))
        for margin in (0.0, 0.2, 0.5, 1.0)
    ]
    assert losses == sorted(losses)


def test_pair_weights_shift_the_loss():
    left = torch.tensor([2.0, 0.0])
    right = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 1.0])
    unweighted = float(margin_ranking_loss(left, right, target, margin=0.2))
    # Down-weighting the wrong pair must lower the loss.
    weighted = float(
        margin_ranking_loss(
            left, right, target, margin=0.2, weights=torch.tensor([1.0, 0.1])
        )
    )
    assert weighted < unweighted


def test_zero_total_weight_is_a_hard_failure():
    with pytest.raises(ValueError, match="Total pair weight is zero"):
        margin_ranking_loss(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            weights=torch.tensor([0.0]),
        )


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Shape mismatch"):
        margin_ranking_loss(
            torch.tensor([1.0, 2.0]), torch.tensor([0.0]), torch.tensor([1.0])
        )


# ---------------------------------------------------------------------------
# Pairwise accuracy
# ---------------------------------------------------------------------------


def test_pairwise_accuracy_counts_ties_as_half():
    left = torch.tensor([2.0, 0.0, 1.0])
    right = torch.tensor([0.0, 2.0, 1.0])
    target = torch.tensor([1.0, 1.0, 1.0])
    assert pairwise_accuracy(left, right, target) == pytest.approx((1 + 0 + 0.5) / 3)


def test_pairwise_accuracy_respects_the_target_sign():
    left = torch.tensor([0.0])
    right = torch.tensor([2.0])
    assert pairwise_accuracy(left, right, torch.tensor([-1.0])) == 1.0
    assert pairwise_accuracy(left, right, torch.tensor([1.0])) == 0.0


# ---------------------------------------------------------------------------
# Auxiliary head
# ---------------------------------------------------------------------------


def test_auxiliary_loss_falls_as_predictions_improve():
    targets = torch.tensor([1.0, 0.0])
    good = acceptability_loss(torch.tensor([4.0, -4.0]), targets)
    bad = acceptability_loss(torch.tensor([-4.0, 4.0]), targets)
    assert float(good) < float(bad)


def test_combined_loss_adds_the_weighted_auxiliary_term():
    left = torch.tensor([0.0])
    right = torch.tensor([1.0])
    target = torch.tensor([1.0])
    logits = torch.tensor([0.0, 0.0])
    truth = torch.tensor([1.0, 0.0])

    without = combined_loss(
        left, right, target, config=LossConfig(auxiliary_enabled=False)
    )
    with_aux = combined_loss(
        left,
        right,
        target,
        unique_logits=logits,
        unique_targets=truth,
        config=LossConfig(auxiliary_weight=0.25),
    )
    assert without.auxiliary is None
    assert with_aux.auxiliary is not None
    expected = float(with_aux.ranking) + 0.25 * float(with_aux.auxiliary)
    assert float(with_aux.total) == pytest.approx(expected)
    assert float(without.total) == pytest.approx(float(without.ranking))


def test_auxiliary_weight_of_zero_leaves_the_ranking_loss_alone():
    result = combined_loss(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        unique_logits=torch.tensor([3.0]),
        unique_targets=torch.tensor([0.0]),
        config=LossConfig(auxiliary_weight=0.0),
    )
    assert float(result.total) == pytest.approx(float(result.ranking))


def test_auxiliary_loss_uses_unique_candidates_not_pair_slots():
    """A candidate repeated across pairs must not be counted repeatedly."""
    # Three pairs, but only two distinct candidates on the unique side. The two
    # logits are deliberately asymmetric: with equal per-element losses the
    # repeated mean would coincide with the unique mean and the test would pass
    # even for a genuinely double-counting implementation.
    logits_unique = torch.tensor([2.0, -0.5])
    truth_unique = torch.tensor([1.0, 0.0])
    # The same candidates, as they would appear if the pair slots were used.
    logits_repeated = torch.tensor([2.0, 2.0, 2.0, -0.5])
    truth_repeated = torch.tensor([1.0, 1.0, 1.0, 0.0])

    unique_value = float(acceptability_loss(logits_unique, truth_unique))
    repeated_value = float(acceptability_loss(logits_repeated, truth_repeated))
    assert unique_value != pytest.approx(repeated_value)

    left = torch.tensor([1.0, 1.0, 1.0])
    right = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([1.0, 1.0, 1.0])
    result = combined_loss(
        left,
        right,
        target,
        unique_logits=logits_unique,
        unique_targets=truth_unique,
        config=LossConfig(),
    )
    assert result.unique_candidate_count == 2
    assert result.pair_count == 3
    assert float(result.auxiliary) == pytest.approx(unique_value)


def test_class_weights_come_from_training_only():
    train = [
        make_example("a", "ep1", 5.0),
        make_example("b", "ep1", 5.0),
        make_example("c", "ep1", 1.0),
    ]
    weights = class_weight_from_training(train)
    assert weights["positive_count"] == 2
    assert weights["negative_count"] == 1
    assert weights["positive_weight"] == pytest.approx(0.5)
    assert weights["fitted_on"] == "train"


def test_class_weights_refuse_validation_examples():
    mixed = [
        make_example("a", "ep1", 5.0, split="train"),
        make_example("b", "ep1", 1.0, split="validation"),
    ]
    with pytest.raises(ValueError, match="Refusing to derive class weights"):
        class_weight_from_training(mixed)


def test_a_single_class_split_is_reported_as_degenerate():
    train = [make_example("a", "ep1", 5.0), make_example("b", "ep1", 4.0)]
    weights = class_weight_from_training(train)
    assert weights["degenerate"]
    assert weights["positive_weight"] == 1.0


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="margin must be non-negative"):
        LossConfig(margin=-0.1)
    with pytest.raises(ValueError, match="auxiliary_weight must be non-negative"):
        LossConfig(auxiliary_weight=-1.0)


# ---------------------------------------------------------------------------
# Episode-grouped validation
# ---------------------------------------------------------------------------


def _validation_setup(tmp_path, episode_count: int = 12):
    _, _, loaded = load_corpus(tmp_path, episode_count=episode_count)
    normalizer = fit_normalizer(
        loaded.by_split("train"), loaded.schema.handcrafted_feature_names
    )
    features = materialize(loaded.examples, normalizer)
    validation_rows = [
        index for index, split in enumerate(features.splits) if split == "validation"
    ]
    return features, EpisodeGroups(features, validation_rows)


def test_validation_is_grouped_by_episode(tmp_path):
    features, groups = _validation_setup(tmp_path)
    # A perfect ranker: score exactly the human label.
    metrics = evaluate_validation(groups, features.quality_score)
    assert metrics.episode_count == len(groups)
    assert metrics.ndcg_at_k == pytest.approx(1.0)
    assert len(metrics.per_episode) == metrics.episodes_scored


def test_a_perfect_ranker_beats_a_reversed_one(tmp_path):
    features, groups = _validation_setup(tmp_path)
    good = evaluate_validation(groups, features.quality_score)
    bad = evaluate_validation(groups, -features.quality_score)
    assert good.ndcg_at_k > bad.ndcg_at_k
    assert good.pairwise_accuracy > bad.pairwise_accuracy


def test_pooling_episodes_would_give_a_different_answer():
    """Guards the grouping itself: one global list is not the same metric.

    Two episodes, each ranked *perfectly* within itself, but episode B's scores
    all sit below episode A's. Grouped by episode that is NDCG 1.0. Pooled into
    one list it is not, because B's best candidate lands beneath A's worst.
    """
    from slotify_rank.datasets.schema import TrainingExample

    def example(candidate_id: str, episode_id: str, score: float) -> TrainingExample:
        return TrainingExample(
            candidate_id=candidate_id,
            episode_id=episode_id,
            split="validation",
            handcrafted=np.zeros(2, dtype=np.float32),
            handcrafted_missing_mask=np.zeros(2, dtype=bool),
            audio=np.ones(4, dtype=np.float32),
            audio_available=True,
            text=np.ones(4, dtype=np.float32),
            text_available=True,
            quality_score=score,
            is_acceptable=score >= 3.0,
        )

    examples = [
        example("a1", "epA", 5.0),
        example("a2", "epA", 3.0),
        example("a3", "epA", 1.0),
        example("b1", "epB", 5.0),
        example("b2", "epB", 3.0),
        example("b3", "epB", 1.0),
    ]
    normalizer = fit_normalizer(
        [TrainingExample(**{**e.__dict__, "split": "train"}) for e in examples],
        ("a", "b"),
    )
    features = materialize(examples, normalizer)
    # Perfect within each episode; episode B is globally shifted down.
    scores = torch.tensor([10.0, 9.0, 8.0, 3.0, 2.0, 1.0])

    grouped = evaluate_validation(EpisodeGroups(features), scores)
    assert grouped.episodes_scored == 2
    assert grouped.ndcg_at_k == pytest.approx(1.0)

    # The same scores pooled into a single list: B's candidates are pushed out
    # of the top 3 entirely, so the number is strictly worse.
    pooled_examples = [
        TrainingExample(**{**e.__dict__, "episode_id": "pooled"}) for e in examples
    ]
    pooled_features = materialize(pooled_examples, normalizer)
    pooled = evaluate_validation(EpisodeGroups(pooled_features), scores)
    assert pooled.episodes_scored == 1
    assert pooled.ndcg_at_k < grouped.ndcg_at_k


def test_all_zero_gain_episode_is_excluded_not_scored():
    from slotify_rank.datasets.schema import TrainingExample

    examples = [
        TrainingExample(
            candidate_id=f"c{index}",
            episode_id="ep1",
            split="validation",
            handcrafted=np.zeros(2, dtype=np.float32),
            handcrafted_missing_mask=np.zeros(2, dtype=bool),
            audio=np.ones(4, dtype=np.float32),
            audio_available=True,
            text=np.ones(4, dtype=np.float32),
            text_available=True,
            # Rubric score 1 contributes zero gain, so IDCG is zero.
            quality_score=1.0,
            is_acceptable=False,
        )
        for index in range(3)
    ]
    normalizer = fit_normalizer(
        [
            TrainingExample(**{**e.__dict__, "split": "train"})
            for e in examples
        ],
        ("a", "b"),
    )
    features = materialize(examples, normalizer)
    groups = EpisodeGroups(features)
    metrics = evaluate_validation(groups, torch.tensor([1.0, 2.0, 3.0]))
    assert metrics.ndcg_at_k is None
    assert metrics.episodes_excluded == 1
    assert metrics.exclusion_reasons["no_candidate_above_gain_offset"] == 1


def test_classification_metrics_are_reported_when_logits_are_given(tmp_path):
    features, groups = _validation_setup(tmp_path)
    perfect = torch.where(
        features.is_acceptable > 0.5,
        torch.full_like(features.is_acceptable, 5.0),
        torch.full_like(features.is_acceptable, -5.0),
    )
    metrics = evaluate_validation(groups, features.quality_score, perfect)
    assert metrics.classification_accuracy == pytest.approx(1.0)
    assert metrics.classification_f1 in (None, pytest.approx(1.0))


def test_metric_config_k_is_honoured(tmp_path):
    features, groups = _validation_setup(tmp_path)
    metrics = evaluate_validation(
        groups, features.quality_score, config=MetricConfig(k=1)
    )
    assert metrics.k == 1
    assert "ndcg_at_1" in metrics.to_dict()


def test_ties_are_broken_deterministically(tmp_path):
    features, groups = _validation_setup(tmp_path)
    flat = torch.zeros(len(features))
    first = evaluate_validation(groups, flat)
    second = evaluate_validation(groups, flat)
    assert first.to_dict() == second.to_dict()
