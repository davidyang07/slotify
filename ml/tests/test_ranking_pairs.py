"""Within-episode pair generation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from slotify_rank.datasets.schema import TrainingExample
from slotify_rank.ranking.pairs import PairConfig, generate_pairs, index_pairs
from tests.training_fixtures import load_corpus


def make_example(
    candidate_id: str, episode_id: str, score: float, split: str = "train"
) -> TrainingExample:
    return TrainingExample(
        candidate_id=candidate_id,
        episode_id=episode_id,
        split=split,
        handcrafted=np.zeros(4, dtype=np.float32),
        handcrafted_missing_mask=np.zeros(4, dtype=bool),
        audio=np.ones(8, dtype=np.float32),
        audio_available=True,
        text=np.ones(8, dtype=np.float32),
        text_available=True,
        quality_score=score,
        is_acceptable=score >= 3.0,
    )


def test_pairs_never_cross_episodes():
    examples = [
        make_example("a", "ep1", 5.0),
        make_example("b", "ep1", 1.0),
        make_example("c", "ep2", 5.0),
        make_example("d", "ep2", 1.0),
    ]
    pair_set = generate_pairs(examples, PairConfig(max_pairs_per_episode=None))
    assert len(pair_set) == 2
    for pair in pair_set.pairs:
        preferred_episode = next(
            e.episode_id for e in examples if e.candidate_id == pair.preferred_candidate_id
        )
        other_episode = next(
            e.episode_id
            for e in examples
            if e.candidate_id == pair.nonpreferred_candidate_id
        )
        assert preferred_episode == other_episode == pair.episode_id


def test_mixed_splits_are_refused():
    examples = [
        make_example("a", "ep1", 5.0, split="train"),
        make_example("b", "ep1", 1.0, split="validation"),
    ]
    with pytest.raises(ValueError, match="across splits"):
        generate_pairs(examples)


def test_equal_scores_produce_no_pair():
    examples = [make_example("a", "ep1", 3.0), make_example("b", "ep1", 3.0)]
    pair_set = generate_pairs(examples)
    assert len(pair_set) == 0
    assert pair_set.episodes_without_pairs == ["ep1"]


def test_score_difference_threshold_is_respected():
    examples = [make_example("a", "ep1", 4.0), make_example("b", "ep1", 3.4)]
    assert len(generate_pairs(examples, PairConfig(minimum_score_difference=1.0))) == 0
    assert len(generate_pairs(examples, PairConfig(minimum_score_difference=0.5))) == 1


def test_the_higher_scored_candidate_is_preferred():
    examples = [make_example("low", "ep1", 2.0), make_example("high", "ep1", 5.0)]
    pair = generate_pairs(examples).pairs[0]
    assert pair.preferred_candidate_id == "high"
    assert pair.nonpreferred_candidate_id == "low"
    assert pair.score_difference == pytest.approx(3.0)


def test_direction_shuffling_keeps_the_semantic_fields_fixed():
    examples = [
        make_example(f"c{index}", "ep1", float(1 + (index % 5)))
        for index in range(12)
    ]
    shuffled = generate_pairs(
        examples, PairConfig(shuffle_direction=True, max_pairs_per_episode=None)
    )
    targets = {pair.target for pair in shuffled.pairs}
    assert targets == {1, -1}, "both slot orders should occur"
    for pair in shuffled.pairs:
        if pair.target == 1:
            assert pair.left_candidate_id == pair.preferred_candidate_id
        else:
            assert pair.right_candidate_id == pair.preferred_candidate_id

    unshuffled = generate_pairs(
        examples, PairConfig(shuffle_direction=False, max_pairs_per_episode=None)
    )
    assert {pair.target for pair in unshuffled.pairs} == {1}


def test_pair_ids_are_deterministic_and_content_addressed():
    examples = [make_example("a", "ep1", 5.0), make_example("b", "ep1", 2.0)]
    first = generate_pairs(examples).pairs[0]
    second = generate_pairs(list(reversed(examples))).pairs[0]
    assert first.pair_id == second.pair_id


def test_generation_is_reproducible_for_a_fixed_seed(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    train = loaded.by_split("train")
    config = PairConfig(seed=7, max_pairs_per_episode=5)
    first = generate_pairs(train, config)
    second = generate_pairs(train, config)
    assert [p.pair_id for p in first.pairs] == [p.pair_id for p in second.pairs]
    assert [p.pair_weight for p in first.pairs] == [p.pair_weight for p in second.pairs]


def test_a_different_seed_changes_the_sample(tmp_path):
    _, _, loaded = load_corpus(tmp_path, candidates_per_episode=12)
    train = loaded.by_split("train")
    first = generate_pairs(train, PairConfig(seed=1, max_pairs_per_episode=4))
    second = generate_pairs(train, PairConfig(seed=2, max_pairs_per_episode=4))
    assert [p.pair_id for p in first.pairs] != [p.pair_id for p in second.pairs]


def test_pairs_are_deduplicated():
    examples = [make_example("a", "ep1", 5.0), make_example("b", "ep1", 2.0)]
    pair_set = generate_pairs(examples, PairConfig(max_pairs_per_episode=None))
    ids = [pair.pair_id for pair in pair_set.pairs]
    assert len(ids) == len(set(ids)) == 1


def test_max_pairs_per_episode_is_enforced():
    examples = [
        make_example(f"c{index}", "ep1", float(1 + index % 5)) for index in range(20)
    ]
    pair_set = generate_pairs(examples, PairConfig(max_pairs_per_episode=7))
    assert len(pair_set) == 7
    assert pair_set.pairs_per_episode["ep1"] == 7
    assert pair_set.eligible_before_capping > 7


def test_large_episodes_do_not_dominate():
    small = [make_example(f"s{i}", "small", float(1 + i % 5)) for i in range(4)]
    large = [make_example(f"l{i}", "large", float(1 + i % 5)) for i in range(40)]
    pair_set = generate_pairs(small + large, PairConfig(max_pairs_per_episode=10))
    counts = pair_set.pairs_per_episode
    assert counts["large"] == 10
    assert counts["small"] <= 10
    # Uncapped, "large" would contribute roughly 60x what "small" does.
    assert counts["large"] <= 3 * max(counts["small"], 1)


def test_balanced_sampling_spans_score_differences():
    examples = [
        make_example(f"c{index}", "ep1", float(1 + index % 5)) for index in range(25)
    ]
    balanced = generate_pairs(
        examples, PairConfig(sampling_strategy="balanced", max_pairs_per_episode=8)
    )
    histogram = balanced.score_difference_histogram()
    assert len(histogram) >= 3, f"expected several gap sizes, got {histogram}"


def test_all_strategy_ignores_the_cap():
    examples = [
        make_example(f"c{index}", "ep1", float(1 + index % 5)) for index in range(10)
    ]
    pair_set = generate_pairs(
        examples, PairConfig(sampling_strategy="all", max_pairs_per_episode=3)
    )
    assert len(pair_set) == pair_set.eligible_before_capping > 3


def test_uniform_weights_are_one_by_default():
    examples = [make_example("a", "ep1", 5.0), make_example("b", "ep1", 1.0)]
    pair = generate_pairs(examples).pairs[0]
    assert pair.pair_weight == pytest.approx(1.0)


def test_score_difference_weighting():
    examples = [make_example("a", "ep1", 5.0), make_example("b", "ep1", 1.0)]
    pair = generate_pairs(
        examples, PairConfig(weight_scheme="score_difference")
    ).pairs[0]
    assert pair.pair_weight == pytest.approx(4.0)


def test_hard_pair_emphasis_up_weights_near_ties():
    examples = [
        make_example("a", "ep1", 5.0),
        make_example("b", "ep1", 4.0),
        make_example("c", "ep1", 1.0),
    ]
    pair_set = generate_pairs(
        examples,
        PairConfig(hard_pair_emphasis=1.0, max_pairs_per_episode=None),
    )
    by_gap = {pair.score_difference: pair.pair_weight for pair in pair_set.pairs}
    assert by_gap[1.0] > by_gap[4.0]


def test_episodes_without_pairs_are_reported():
    examples = [
        make_example("a", "flat", 3.0),
        make_example("b", "flat", 3.0),
        make_example("c", "varied", 5.0),
        make_example("d", "varied", 1.0),
    ]
    pair_set = generate_pairs(examples)
    assert pair_set.episodes_without_pairs == ["flat"]
    assert pair_set.contributing_episode_count == 1


def test_summary_reports_the_distribution(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    pair_set = generate_pairs(loaded.by_split("train"), PairConfig(seed=3))
    summary = pair_set.summary()
    assert summary["pair_count"] == len(pair_set)
    assert summary["split"] == "train"
    assert set(summary["pairs_per_episode"]) <= set(
        e.episode_id for e in loaded.by_split("train")
    )
    assert summary["score_difference_histogram"]
    assert summary["config"]["minimum_score_difference"] == 1.0


def test_index_pairs_renders_the_dataset_contract():
    examples = [make_example("a", "ep1", 5.0), make_example("b", "ep1", 1.0)]
    rows = index_pairs(generate_pairs(examples))
    assert set(rows[0]) == {
        "left_candidate_id",
        "right_candidate_id",
        "target",
        "pair_weight",
        "episode_id",
    }


def test_zero_threshold_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        PairConfig(minimum_score_difference=0.0)


def test_no_split_leakage_when_generating_per_split(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    train_ids = {e.candidate_id for e in loaded.by_split("train")}
    validation_ids = {e.candidate_id for e in loaded.by_split("validation")}
    pair_set = generate_pairs(loaded.by_split("train"))
    for pair in pair_set.pairs:
        assert pair.preferred_candidate_id in train_ids
        assert pair.nonpreferred_candidate_id in train_ids
        assert pair.preferred_candidate_id not in validation_ids
