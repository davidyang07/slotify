"""The training loop, checkpoints, early stopping and resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from slotify_rank.datasets.schema import DatasetSchema
from slotify_rank.models.registry import build_model
from slotify_rank.models.schema import ModelConfig
from slotify_rank.training.checkpoint import (
    CheckpointError,
    IncompatibleCheckpoint,
    assert_compatible,
    atomic_torch_save,
    inspect_checkpoint,
    load_checkpoint,
)
from slotify_rank.training.config import TrainingConfig
from slotify_rank.training.early_stopping import EarlyStopping
from slotify_rank.training.prepare import prepare_dataset
from slotify_rank.training.trainer import Trainer, build_seeded_model
from tests.training_fixtures import shared_corpus


def build_trainer(
    tmp_path: Path,
    config: TrainingConfig | None = None,
    variant: str = "handcrafted",
    run_dir: str = "run",
    **corpus_overrides,
):
    paths, corpus = shared_corpus(
        **{"episode_count": 10, "candidates_per_episode": 8, **corpus_overrides}
    )
    config = config or TrainingConfig(epochs=3, early_stopping_patience=0)
    dataset = prepare_dataset(paths, config, corpus.label_export)
    model_config = ModelConfig(variant=variant, auxiliary_head=config.auxiliary_head)
    # Seeded before construction, so two runs with the same seed start from the
    # same weights -- see build_seeded_model.
    model = build_seeded_model(model_config, dataset.schema, config.seed)
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        run_directory=tmp_path / run_dir,
        run_id=f"test-{variant}",
        model_config=model_config.to_dict(),
        label_manifest_hash="labelhash",
    )
    return trainer, dataset, model_config


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_training_runs_on_cpu_and_produces_checkpoints(tmp_path):
    trainer, _, _ = build_trainer(tmp_path)
    result = trainer.train()
    assert result.device == "cpu"
    assert result.epochs_run == 3
    assert Path(result.last_checkpoint_path).is_file()
    assert Path(result.best_checkpoint_path).is_file()
    assert result.parameter_count > 0


def test_the_loss_decreases_on_a_learnable_signal(tmp_path):
    trainer, _, _ = build_trainer(
        tmp_path, TrainingConfig(epochs=8, early_stopping_patience=0)
    )
    result = trainer.train()
    losses = [metrics.train_loss for metrics in result.epoch_metrics]
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"
    accuracies = [m.train_pairwise_accuracy for m in result.epoch_metrics]
    assert accuracies[-1] > accuracies[0]


def test_every_epoch_reports_the_required_statistics(tmp_path):
    trainer, _, _ = build_trainer(tmp_path)
    result = trainer.train()
    for metrics in result.epoch_metrics:
        payload = metrics.to_dict()
        assert payload["train_pair_count"] > 0
        assert payload["train_contributing_episode_count"] > 0
        assert payload["train_auxiliary_loss"] is not None
        assert "train_score_margin_mean" in payload
        assert "train_score_margin_std" in payload
        assert payload["pairs_per_second"] > 0
        assert payload["validation"]["episode_count"] >= 1


def test_validation_is_grouped_by_episode_during_training(tmp_path):
    trainer, dataset, _ = build_trainer(tmp_path)
    result = trainer.train()
    validation = result.epoch_metrics[0].validation
    expected_episodes = len(
        {dataset.features.episode_ids[row] for row in dataset.validation_rows}
    )
    assert validation["episode_count"] == expected_episodes
    assert validation["candidate_count"] == len(dataset.validation_rows)


def test_the_auxiliary_head_can_be_disabled(tmp_path):
    config = TrainingConfig(epochs=2, auxiliary_head=False, early_stopping_patience=0)
    trainer, _, _ = build_trainer(tmp_path, config)
    result = trainer.train()
    assert all(m.train_auxiliary_loss is None for m in result.epoch_metrics)


def test_gradient_clipping_is_applied(tmp_path):
    """A tiny clip norm must bound the parameter movement of one step.

    Deliberately SGD: Adam and AdamW rescale each step by the running second
    moment, so their step size is roughly the learning rate regardless of the
    gradient's magnitude. Under those optimizers clipping barely moves the
    weights at all, and this test would pass whether or not clipping happened.
    """
    config = TrainingConfig(
        epochs=1,
        optimizer="sgd",
        gradient_clip_norm=1e-6,
        learning_rate=1.0,
        early_stopping_patience=0,
    )
    trainer, _, _ = build_trainer(tmp_path, config)
    before = torch.cat([p.detach().flatten().clone() for p in trainer.model.parameters()])
    trainer.train()
    after = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
    movement = float((after - before).norm())
    assert movement < 0.05, f"clipping did not bound the step: {movement}"


def test_mixed_precision_is_refused_on_cpu(tmp_path):
    config = TrainingConfig(epochs=1, mixed_precision=True, device="cpu")
    with pytest.raises(ValueError, match="mixed_precision was requested"):
        build_trainer(tmp_path, config)


def test_class_weights_are_recorded_when_enabled(tmp_path):
    config = TrainingConfig(epochs=1, use_class_weights=True, early_stopping_patience=0)
    trainer, _, _ = build_trainer(tmp_path, config)
    assert trainer.class_weight_metadata["fitted_on"] == "train"
    assert "positive_weight" in trainer.class_weight_metadata


def test_environment_records_device_and_dtype(tmp_path):
    trainer, _, _ = build_trainer(tmp_path)
    environment = trainer.environment()
    assert environment["device"] == "cpu"
    assert environment["resolved_dtype"] == "float32"
    assert environment["mixed_precision"] is False
    assert environment["threads"] >= 1
    assert "torch" in environment["dependency_versions"]


@pytest.mark.parametrize(
    "variant", ["handcrafted", "text_only", "audio_only", "concat", "gated"]
)
def test_every_variant_trains(tmp_path, variant):
    trainer, _, _ = build_trainer(
        tmp_path,
        TrainingConfig(epochs=2, early_stopping_patience=0),
        variant=variant,
        run_dir=f"run-{variant}",
    )
    result = trainer.train()
    assert result.epochs_run == 2
    assert Path(result.last_checkpoint_path).is_file()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_identical_runs_agree(tmp_path):
    config = TrainingConfig(epochs=3, early_stopping_patience=0, seed=17)
    first, _, _ = build_trainer(tmp_path / "a", config, run_dir="r1")
    second, _, _ = build_trainer(tmp_path / "b", config, run_dir="r2")
    left = first.train()
    right = second.train()
    assert [m.train_loss for m in left.epoch_metrics] == pytest.approx(
        [m.train_loss for m in right.epoch_metrics], rel=1e-6
    )
    assert left.best_validation_metric == pytest.approx(right.best_validation_metric)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


def test_early_stopping_fires_on_a_plateau():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.update(0.5, 1)
    assert not stopper.update(0.5001, 2)
    assert not stopper.stopped
    assert not stopper.update(0.5002, 3)
    assert stopper.stopped
    assert stopper.best_epoch == 1
    assert "did not improve" in stopper.stop_reason


def test_min_delta_prevents_noise_from_resetting_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    stopper.update(0.5, 1)
    stopper.update(0.5005, 2)
    stopper.update(0.5009, 3)
    assert stopper.stopped, "sub-threshold wobble should not have counted as progress"


def test_an_undefined_metric_is_neither_progress_nor_a_stall():
    stopper = EarlyStopping(patience=2, min_delta=0.0)
    stopper.update(0.5, 1)
    stopper.update(None, 2)
    assert stopper.undefined_epochs == 1
    assert stopper.epochs_without_improvement == 0
    assert not stopper.stopped


def test_patience_zero_disables_stopping():
    stopper = EarlyStopping(patience=0)
    stopper.update(0.5, 1)
    for epoch in range(2, 20):
        stopper.update(0.1, epoch)
    assert not stopper.stopped


def test_early_stopping_state_round_trips():
    stopper = EarlyStopping(patience=3, min_delta=0.01)
    stopper.update(0.7, 4)
    stopper.update(0.6, 5)
    restored = EarlyStopping()
    restored.load_state_dict(stopper.state_dict())
    assert restored.best_metric == stopper.best_metric
    assert restored.best_epoch == stopper.best_epoch
    assert restored.epochs_without_improvement == stopper.epochs_without_improvement


def test_early_stopping_selects_the_best_checkpoint(tmp_path):
    config = TrainingConfig(epochs=20, early_stopping_patience=2, seed=5)
    trainer, _, _ = build_trainer(tmp_path, config)
    result = trainer.train()
    assert result.epochs_run < 20, "early stopping should have cut the run short"
    best = [m for m in result.epoch_metrics if m.is_best][-1]
    assert best.epoch == result.best_epoch
    payload = load_checkpoint(Path(result.best_checkpoint_path))
    assert payload["epoch"] == result.best_epoch


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_contains_every_required_field(tmp_path):
    trainer, dataset, model_config = build_trainer(tmp_path)
    result = trainer.train()
    payload = load_checkpoint(Path(result.last_checkpoint_path))
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "global_step",
        "best_validation_metric",
        "training_config",
        "training_config_hash",
        "model_config",
        "model_variant",
        "input_schema",
        "feature_names",
        "feature_pipeline_version",
        "normalizer_reference",
        "dataset_split_hash",
        "label_manifest_hash",
        "random_seed",
        "resolved_device",
        "resolved_dependency_versions",
        "git_commit",
    ):
        assert field in payload, f"missing {field}"
    assert payload["feature_names"] == list(dataset.schema.handcrafted_feature_names)
    assert payload["model_variant"] == model_config.variant


def test_checkpoints_load_with_weights_only(tmp_path):
    """No arbitrary Python object may be needed to read a checkpoint."""
    trainer, _, _ = build_trainer(tmp_path)
    result = trainer.train()
    raw = torch.load(
        Path(result.last_checkpoint_path), map_location="cpu", weights_only=True
    )
    assert isinstance(raw, dict)
    assert "model_state_dict" in raw


def test_a_corrupted_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a torch file at all")
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_a_truncated_checkpoint_is_rejected(tmp_path):
    trainer, _, _ = build_trainer(tmp_path)
    result = trainer.train()
    source = Path(result.last_checkpoint_path).read_bytes()
    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(source[: len(source) // 2])
    with pytest.raises(CheckpointError):
        load_checkpoint(truncated)


def test_a_missing_checkpoint_is_rejected(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")


def test_an_incompatible_variant_is_rejected(tmp_path):
    trainer, dataset, _ = build_trainer(tmp_path)
    result = trainer.train()
    payload = load_checkpoint(Path(result.last_checkpoint_path))
    with pytest.raises(IncompatibleCheckpoint, match="variant"):
        assert_compatible(payload, model_variant="gated", schema=dataset.schema)


def test_an_incompatible_input_dimension_is_rejected(tmp_path):
    trainer, dataset, model_config = build_trainer(tmp_path)
    result = trainer.train()
    payload = load_checkpoint(Path(result.last_checkpoint_path))
    other = DatasetSchema(
        handcrafted_feature_names=tuple(f"g{i}" for i in range(5)),
        handcrafted_dimension=5,
        audio_dimension=dataset.schema.audio_dimension,
        text_dimension=dataset.schema.text_dimension,
        native_embedding_dimension=dataset.schema.native_embedding_dimension,
    )
    with pytest.raises(IncompatibleCheckpoint, match="handcrafted"):
        assert_compatible(payload, model_variant=model_config.variant, schema=other)


def test_reordered_features_are_rejected(tmp_path):
    trainer, dataset, model_config = build_trainer(tmp_path)
    result = trainer.train()
    payload = load_checkpoint(Path(result.last_checkpoint_path))
    names = list(dataset.schema.handcrafted_feature_names)
    reordered = DatasetSchema(
        handcrafted_feature_names=tuple([names[1], names[0], *names[2:]]),
        handcrafted_dimension=len(names),
        audio_dimension=dataset.schema.audio_dimension,
        text_dimension=dataset.schema.text_dimension,
        native_embedding_dimension=dataset.schema.native_embedding_dimension,
        feature_spec_version=dataset.schema.feature_spec_version,
        feature_pipeline_version=dataset.schema.feature_pipeline_version,
    )
    with pytest.raises(IncompatibleCheckpoint, match="ordering"):
        assert_compatible(payload, model_variant=model_config.variant, schema=reordered)


def test_a_foreign_split_hash_is_rejected_unless_overridden(tmp_path):
    trainer, dataset, model_config = build_trainer(tmp_path)
    result = trainer.train()
    payload = load_checkpoint(Path(result.last_checkpoint_path))
    with pytest.raises(IncompatibleCheckpoint, match="dataset_split_hash"):
        assert_compatible(
            payload,
            model_variant=model_config.variant,
            schema=dataset.schema,
            dataset_split_hash="a-different-corpus",
        )
    # Inference-only use on a new corpus is legitimate and explicitly allowed.
    assert_compatible(
        payload,
        model_variant=model_config.variant,
        schema=dataset.schema,
        dataset_split_hash="a-different-corpus",
        allow_split_mismatch=True,
    )


def test_an_unsupported_schema_version_is_rejected(tmp_path):
    path = tmp_path / "old.pt"
    atomic_torch_save(
        path,
        {
            "checkpoint_schema_version": "checkpoint-schema-v0.1.0",
            "model_state_dict": {},
            "model_variant": "gated",
            "input_schema": "{}",
        },
    )
    with pytest.raises(IncompatibleCheckpoint, match="Unsupported checkpoint"):
        load_checkpoint(path)


def test_inspect_reports_without_building_a_model(tmp_path):
    trainer, _, _ = build_trainer(tmp_path)
    result = trainer.train()
    report = inspect_checkpoint(Path(result.best_checkpoint_path))
    assert report["model_variant"] == "handcrafted"
    assert report["parameter_count"] == trainer.model.parameter_count()
    assert report["resolved_device"] == "cpu"
    json.dumps(report)


def test_checkpoint_writes_are_atomic(tmp_path):
    """No partial file is ever visible at the destination path."""
    destination = tmp_path / "nested" / "ckpt.pt"
    atomic_torch_save(destination, {"a": torch.zeros(3)})
    assert destination.is_file()
    # The temp files the writer used must not have been left behind.
    leftovers = [p for p in destination.parent.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers


def test_a_failed_checkpoint_write_leaves_no_debris(tmp_path):
    destination = tmp_path / "ckpt.pt"

    class Unserializable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_torch_save(destination, {"bad": Unserializable()})
    assert not destination.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_restores_the_full_training_state(tmp_path):
    config = TrainingConfig(epochs=2, early_stopping_patience=0, seed=3)
    trainer, _, _ = build_trainer(tmp_path, config, run_dir="first")
    result = trainer.train()

    resumed, _, _ = build_trainer(
        tmp_path / "again", TrainingConfig(epochs=5, early_stopping_patience=0, seed=3),
        run_dir="second",
    )
    payload = resumed.resume_from(Path(result.last_checkpoint_path))
    assert resumed.start_epoch == 3
    assert resumed.global_step == payload["global_step"]
    assert resumed.early_stopping.best_epoch == trainer.early_stopping.best_epoch
    assert resumed.early_stopping.best_metric == pytest.approx(
        trainer.early_stopping.best_metric
    )


def test_interrupted_then_resumed_matches_an_uninterrupted_run(tmp_path):
    """The headline resume guarantee, within CPU floating-point tolerance.

    Bit-identity is not claimed: reduction order in BLAS kernels is not seed
    controlled, so two mathematically equal sequences can differ in the last
    few bits and compound over epochs. 1e-4 is well below any difference that
    would change a ranking decision.
    """
    seed = 11
    total_epochs = 6
    stop_after = 2

    uninterrupted, _, _ = build_trainer(
        tmp_path / "whole",
        TrainingConfig(epochs=total_epochs, early_stopping_patience=0, seed=seed),
        run_dir="whole",
    )
    whole = uninterrupted.train()

    partial, _, _ = build_trainer(
        tmp_path / "part",
        TrainingConfig(epochs=stop_after, early_stopping_patience=0, seed=seed),
        run_dir="part",
    )
    first_half = partial.train()

    resumed, _, _ = build_trainer(
        tmp_path / "part2",
        TrainingConfig(epochs=total_epochs, early_stopping_patience=0, seed=seed),
        run_dir="part2",
    )
    resumed.resume_from(Path(first_half.last_checkpoint_path))
    second_half = resumed.train()

    assert first_half.epochs_run + second_half.epochs_run == total_epochs

    whole_losses = [m.train_loss for m in whole.epoch_metrics]
    stitched = [m.train_loss for m in first_half.epoch_metrics] + [
        m.train_loss for m in second_half.epoch_metrics
    ]
    assert stitched == pytest.approx(whole_losses, abs=1e-4)

    whole_ndcg = whole.best_validation_metric
    resumed_best = max(
        value
        for value in (first_half.best_validation_metric, second_half.best_validation_metric)
        if value is not None
    )
    assert resumed_best == pytest.approx(whole_ndcg, abs=1e-4)


def test_resuming_a_finished_run_has_nothing_to_do(tmp_path):
    config = TrainingConfig(epochs=2, early_stopping_patience=0)
    trainer, _, _ = build_trainer(tmp_path, config, run_dir="done")
    result = trainer.train()
    resumed, _, _ = build_trainer(tmp_path / "b", config, run_dir="done2")
    resumed.resume_from(Path(result.last_checkpoint_path))
    assert resumed.start_epoch > config.epochs
