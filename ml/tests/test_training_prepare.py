"""Training configuration, dataset preparation and run identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from slotify_rank.config.settings import find_repo_root
from slotify_rank.models.schema import ModelConfig
from slotify_rank.training.config import (
    TrainingConfig,
    compute_run_id,
    load_training_config,
)
from slotify_rank.training.prepare import dataset_fingerprint, prepare_dataset
from tests.training_fixtures import make_corpus


def prepared(tmp_path, config: TrainingConfig | None = None, **corpus_overrides):
    paths, corpus = make_corpus(
        tmp_path, **{"episode_count": 10, "candidates_per_episode": 8, **corpus_overrides}
    )
    config = config or TrainingConfig()
    return paths, corpus, prepare_dataset(paths, config, corpus.label_export)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_shipped_training_config_loads():
    path = Path(find_repo_root()) / "ml" / "configs" / "training_v1.yaml"
    config = load_training_config(path)
    assert config.margin == 0.2
    assert config.auxiliary_weight == 0.25
    assert config.minimum_score_difference == 1.0
    assert config.device == "auto"
    assert config.metric_k == 3


def test_unknown_training_setting_is_rejected():
    with pytest.raises(ValueError, match="unknown setting"):
        TrainingConfig.from_mapping({"learning_rate": 0.1, "lr_schedule": "cosine"})


def test_validating_on_the_training_split_is_refused():
    with pytest.raises(ValueError, match="validating on the training split"):
        TrainingConfig(train_split="train", validation_split="train")


def test_zero_gradient_clip_is_refused():
    with pytest.raises(ValueError, match="gradient_clip_norm must be positive"):
        TrainingConfig(gradient_clip_norm=0.0)


def test_overrides_are_recorded_not_silently_applied():
    config = TrainingConfig()
    updated, applied = config.with_overrides(epochs=2, max_episodes=3, seed=None)
    assert updated.epochs == 2
    assert updated.max_episodes == 3
    assert updated.seed == config.seed
    assert applied["epochs"] == {"from": config.epochs, "to": 2}
    assert applied["max_episodes"] == {"from": None, "to": 3}


def test_an_unchanged_override_is_not_reported_as_a_change():
    config = TrainingConfig()
    _, applied = config.with_overrides(epochs=config.epochs)
    assert "epochs" not in applied


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="Unknown training override"):
        TrainingConfig().with_overrides(learn_rate=0.1)


def test_sub_configs_are_derived_consistently():
    config = TrainingConfig(margin=0.5, auxiliary_head=False, seed=11)
    assert config.loss_config().margin == 0.5
    assert not config.loss_config().auxiliary_enabled
    assert config.pair_config().seed == 11


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------


def test_run_id_is_stable_for_identical_inputs():
    training = TrainingConfig()
    model = ModelConfig(variant="gated")
    first = compute_run_id(training, model, "fingerprint", "abc123")
    second = compute_run_id(training, model, "fingerprint", "abc123")
    assert first == second
    assert first.startswith("gated-")


def test_run_id_changes_with_any_input():
    training = TrainingConfig()
    model = ModelConfig(variant="gated")
    base = compute_run_id(training, model, "fingerprint", "abc123")
    assert compute_run_id(replace(training, seed=7), model, "fingerprint", "abc123") != base
    assert compute_run_id(training, ModelConfig(variant="concat"), "fingerprint", "abc123") != base
    assert compute_run_id(training, model, "other", "abc123") != base
    assert compute_run_id(training, model, "fingerprint", "def456") != base


def test_dataset_fingerprint_follows_the_bytes(tmp_path):
    features = tmp_path / "features.jsonl"
    labels = tmp_path / "labels.jsonl"
    features.write_text("a\n", encoding="utf-8")
    labels.write_text("b\n", encoding="utf-8")
    first = dataset_fingerprint(features, labels)
    assert dataset_fingerprint(features, labels) == first
    features.write_text("a changed\n", encoding="utf-8")
    assert dataset_fingerprint(features, labels) != first


def test_a_missing_artifact_hashes_as_absent(tmp_path):
    """An absent label export and a nonexistent path are the same input."""
    features = tmp_path / "features.jsonl"
    features.write_text("a\n", encoding="utf-8")
    assert dataset_fingerprint(features, None) == dataset_fingerprint(
        features, tmp_path / "nope.jsonl"
    )
    # ...and both differ from the fingerprint with labels actually present.
    labels = tmp_path / "labels.jsonl"
    labels.write_text("b\n", encoding="utf-8")
    assert dataset_fingerprint(features, labels) != dataset_fingerprint(features, None)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def test_prepare_produces_disjoint_splits(tmp_path):
    _, _, dataset = prepared(tmp_path)
    summary = dataset.summary()
    assert summary["episode_overlap_between_splits"] == []
    assert summary["training_candidate_count"] > 0
    assert summary["validation_candidate_count"] > 0
    train_ids = {dataset.features.candidate_ids[row] for row in dataset.train_rows}
    validation_ids = {
        dataset.features.candidate_ids[row] for row in dataset.validation_rows
    }
    assert not (train_ids & validation_ids)


def test_prepare_fits_the_normalizer_on_train_only(tmp_path):
    _, _, dataset = prepared(tmp_path)
    assert dataset.normalizer.fit_candidate_count == len(dataset.train_rows)
    assert dataset.normalizer.training_split_hash


def test_pairs_never_span_the_split_boundary(tmp_path):
    _, _, dataset = prepared(tmp_path)
    train_ids = {dataset.features.candidate_ids[row] for row in dataset.train_rows}
    for pair in dataset.train_pairs.pairs:
        assert pair.preferred_candidate_id in train_ids
        assert pair.nonpreferred_candidate_id in train_ids
    validation_ids = {
        dataset.features.candidate_ids[row] for row in dataset.validation_rows
    }
    for pair in dataset.validation_pairs.pairs:
        assert pair.preferred_candidate_id in validation_ids


def test_preparation_is_deterministic(tmp_path):
    _, _, first = prepared(tmp_path / "a")
    _, _, second = prepared(tmp_path / "b")
    assert first.fingerprint == second.fingerprint
    assert [p.pair_id for p in first.train_pairs.pairs] == [
        p.pair_id for p in second.train_pairs.pairs
    ]
    assert first.features.candidate_ids == second.features.candidate_ids


def test_limits_are_applied_and_recorded(tmp_path):
    config = TrainingConfig(max_episodes=2, max_candidates=6)
    _, _, dataset = prepared(tmp_path, config)
    limits = dataset.summary()["limits_applied"]
    assert limits["max_episodes"]["limit"] == 2
    assert limits["max_candidates"]["limit"] == 6
    assert dataset.summary()["training_candidate_count"] <= 6
    assert dataset.summary()["validation_candidate_count"] <= 6


def test_max_pairs_limit_is_recorded(tmp_path):
    config = TrainingConfig(max_pairs=5)
    _, _, dataset = prepared(tmp_path, config)
    assert len(dataset.train_pairs) == 5
    assert dataset.summary()["limits_applied"]["max_pairs"]["limit"] == 5


def test_an_empty_validation_split_is_refused(tmp_path):
    config = TrainingConfig(validation_split="test")
    paths, corpus = make_corpus(tmp_path, episode_count=3, candidates_per_episode=6)
    # Episodes 0,1,2 map to train,train,validation -- so "test" is empty.
    with pytest.raises(RuntimeError, match="split is empty"):
        prepare_dataset(paths, config, corpus.label_export)


def test_native_embedding_dimension_is_read_from_the_artifacts(tmp_path):
    paths, corpus = make_corpus(
        tmp_path, episode_count=10, candidates_per_episode=6, native_dimension=96
    )
    dataset = prepare_dataset(paths, TrainingConfig(), corpus.label_export)
    assert dataset.schema.native_embedding_dimension == 96
    assert dataset.schema.audio_dimension == 96 * 4
    assert dataset.schema.text_dimension == 96 * 4


def test_missing_modalities_survive_preparation(tmp_path):
    _, _, dataset = prepared(tmp_path, text_missing_episode_stride=3)
    coverage = dataset.loaded.summary()["modality_coverage"]
    assert coverage.get("audio_only", 0) > 0
    assert float(dataset.features.text_available.min()) == 0.0


def test_summary_reports_the_pair_distribution(tmp_path):
    _, _, dataset = prepared(tmp_path)
    summary = dataset.summary()
    assert summary["training_pair_count"] == len(dataset.train_pairs)
    assert summary["train_pairs"]["score_difference_histogram"]
    assert summary["train_pairs"]["pairs_per_episode"]
