"""The product-facing inference path.

These tests never touch audio or a real model download. They train a tiny
ranker on the synthetic fixture corpus, save a checkpoint, and then exercise
the predictor exactly as the Express API does: load, verify, score, rank.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from slotify_rank.datasets.schema import DatasetSchema, TrainingExample
from slotify_rank.inference.predictor import (
    PredictorError,
    RankerPredictor,
    score_examples,
)
from slotify_rank.inference.schema import (
    INFERENCE_SCHEMA_VERSION,
    ExcludedCandidate,
    InferenceResult,
    normalize_scores,
)
from slotify_rank.models.schema import ModelConfig
from slotify_rank.training.checkpoint import IncompatibleCheckpoint, load_checkpoint
from slotify_rank.training.config import TrainingConfig
from slotify_rank.training.prepare import prepare_dataset
from slotify_rank.training.trainer import Trainer, build_seeded_model
from tests.training_fixtures import shared_corpus


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory) -> Path:
    """A real checkpoint from a real (tiny, synthetic) training run."""
    run_dir = tmp_path_factory.mktemp("inference-run")
    paths, corpus = shared_corpus(episode_count=10, candidates_per_episode=8)
    config = TrainingConfig(epochs=2, early_stopping_patience=0)
    dataset = prepare_dataset(paths, config, corpus.label_export)
    model_config = ModelConfig(variant="gated", auxiliary_head=config.auxiliary_head)
    model = build_seeded_model(model_config, dataset.schema, config.seed)
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        run_directory=run_dir,
        run_id="inference-test",
        model_config=model_config.to_dict(),
        label_manifest_hash="labelhash",
    )
    trainer.train()

    from slotify_rank.datasets.normalizer import write_normalizer

    write_normalizer(run_dir / "normalizer.json", dataset.normalizer)
    checkpoint = run_dir / "best_checkpoint.pt"
    assert checkpoint.is_file(), "the trainer did not write a best checkpoint"
    return checkpoint


@pytest.fixture(scope="module")
def predictor(trained_checkpoint: Path) -> RankerPredictor:
    return RankerPredictor.load(trained_checkpoint)


def make_example(
    schema: DatasetSchema,
    candidate_id: str,
    seed: int,
    audio_available: bool = True,
    text_available: bool = True,
) -> TrainingExample:
    rng = np.random.default_rng(seed)
    return TrainingExample(
        candidate_id=candidate_id,
        episode_id="upload",
        split="development",
        handcrafted=rng.normal(size=schema.handcrafted_dimension).astype(np.float32),
        handcrafted_missing_mask=np.zeros(schema.handcrafted_dimension, dtype=bool),
        audio=(
            rng.normal(size=schema.audio_dimension).astype(np.float32)
            if audio_available
            else np.zeros(schema.audio_dimension, dtype=np.float32)
        ),
        audio_available=audio_available,
        text=(
            rng.normal(size=schema.text_dimension).astype(np.float32)
            if text_available
            else np.zeros(schema.text_dimension, dtype=np.float32)
        ),
        text_available=text_available,
        quality_score=0.0,
        is_acceptable=False,
        feature_status="complete" if text_available else "audio_only",
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_reports_the_model_identity(predictor: RankerPredictor) -> None:
    identity = predictor.identity
    assert identity.model_variant == "gated"
    assert identity.parameter_count > 0
    assert identity.handcrafted_dimension == predictor.schema.handcrafted_dimension
    assert identity.normalizer_version == predictor.normalizer.normalizer_version


def test_loaded_model_is_in_eval_mode(predictor: RankerPredictor) -> None:
    assert predictor.model.training is False
    for module in predictor.model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.training is False


def test_a_missing_normalizer_is_a_hard_failure(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    lonely = tmp_path / "best_checkpoint.pt"
    lonely.write_bytes(trained_checkpoint.read_bytes())
    with pytest.raises(PredictorError, match="No normalizer"):
        RankerPredictor.load(lonely)


def test_a_checkpoint_for_another_schema_is_rejected(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    payload = load_checkpoint(trained_checkpoint)
    schema = dict(payload["input_schema"])
    schema["handcrafted_feature_names"] = list(schema["handcrafted_feature_names"])[:-1]
    schema["handcrafted_dimension"] = len(schema["handcrafted_feature_names"])
    payload["input_schema"] = json.dumps(schema, sort_keys=True)
    payload["feature_names"] = schema["handcrafted_feature_names"]

    from slotify_rank.training.checkpoint import atomic_torch_save

    broken = tmp_path / "best_checkpoint.pt"
    atomic_torch_save(broken, payload)
    (tmp_path / "normalizer.json").write_text(
        (trained_checkpoint.parent / "normalizer.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(IncompatibleCheckpoint):
        RankerPredictor.load(broken)


def test_a_corrupt_checkpoint_is_rejected(tmp_path: Path) -> None:
    from slotify_rank.training.checkpoint import CheckpointError

    broken = tmp_path / "best_checkpoint.pt"
    broken.write_bytes(b"not a checkpoint")
    (tmp_path / "normalizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointError):
        RankerPredictor.load(broken)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_scoring_is_deterministic(predictor: RankerPredictor) -> None:
    examples = [make_example(predictor.schema, f"upload:{i:09d}", seed=i) for i in range(5)]
    first, _ = predictor.score(examples)
    second, _ = predictor.score(examples)
    assert first == second


def test_an_empty_candidate_set_scores_nothing(predictor: RankerPredictor) -> None:
    assert predictor.score([]) == ([], [])
    assert score_examples(predictor, [], {}, {}) == []


def test_a_candidate_with_no_transcript_is_scored_with_text_masked(
    predictor: RankerPredictor,
) -> None:
    examples = [
        make_example(predictor.schema, "upload:000001000", seed=1, text_available=False)
    ]
    scored = score_examples(
        predictor, examples, {"upload:000001000": 1000}, {"upload:000001000": "audio_only"}
    )
    assert len(scored) == 1
    assert scored[0].text_available is False
    assert scored[0].feature_status == "audio_only"


def test_a_wrong_width_example_is_rejected_rather_than_reshaped(
    predictor: RankerPredictor,
) -> None:
    example = make_example(predictor.schema, "upload:000001000", seed=3)
    truncated = TrainingExample(
        candidate_id=example.candidate_id,
        episode_id=example.episode_id,
        split=example.split,
        handcrafted=example.handcrafted[:-1],
        handcrafted_missing_mask=example.handcrafted_missing_mask[:-1],
        audio=example.audio,
        audio_available=example.audio_available,
        text=example.text,
        text_available=example.text_available,
        quality_score=0.0,
        is_acceptable=False,
    )
    with pytest.raises(IncompatibleCheckpoint, match="handcrafted features"):
        predictor.score([truncated])


def test_rank_order_matches_the_raw_scores(predictor: RankerPredictor) -> None:
    examples = [make_example(predictor.schema, f"upload:{i:09d}", seed=i) for i in range(6)]
    timestamps = {example.candidate_id: index * 1000 for index, example in enumerate(examples)}
    scored = score_examples(predictor, examples, timestamps, {})

    assert [entry.rank for entry in scored] == list(range(1, len(examples) + 1))
    raw = [entry.raw_score for entry in scored]
    assert raw == sorted(raw, reverse=True)


def test_ties_break_deterministically_on_timestamp(predictor: RankerPredictor) -> None:
    example = make_example(predictor.schema, "upload:000002000", seed=7)
    twin = TrainingExample(
        candidate_id="upload:000001000",
        episode_id=example.episode_id,
        split=example.split,
        handcrafted=example.handcrafted,
        handcrafted_missing_mask=example.handcrafted_missing_mask,
        audio=example.audio,
        audio_available=example.audio_available,
        text=example.text,
        text_available=example.text_available,
        quality_score=0.0,
        is_acceptable=False,
    )
    timestamps = {"upload:000002000": 2000, "upload:000001000": 1000}
    scored = score_examples(predictor, [example, twin], timestamps, {})
    assert scored[0].candidate_id == "upload:000001000"


def test_the_normalizer_is_applied_not_bypassed(predictor: RankerPredictor) -> None:
    """Feeding raw values whose statistics the normalizer cannot represent must
    change the score, proving the transform is on the path."""
    base = make_example(predictor.schema, "upload:000001000", seed=11)
    shifted = TrainingExample(
        candidate_id=base.candidate_id,
        episode_id=base.episode_id,
        split=base.split,
        handcrafted=base.handcrafted + 10.0,
        handcrafted_missing_mask=base.handcrafted_missing_mask,
        audio=base.audio,
        audio_available=base.audio_available,
        text=base.text,
        text_available=base.text_available,
        quality_score=0.0,
        is_acceptable=False,
    )
    first, _ = predictor.score([base])
    second, _ = predictor.score([shifted])
    assert first[0] != second[0]


def test_non_finite_features_are_refused(predictor: RankerPredictor) -> None:
    base = make_example(predictor.schema, "upload:000001000", seed=13)
    with pytest.raises(ValueError):
        TrainingExample(
            candidate_id=base.candidate_id,
            episode_id=base.episode_id,
            split=base.split,
            handcrafted=np.full_like(base.handcrafted, np.nan),
            handcrafted_missing_mask=base.handcrafted_missing_mask,
            audio=base.audio,
            audio_available=base.audio_available,
            text=base.text,
            text_available=base.text_available,
            quality_score=0.0,
            is_acceptable=False,
        )


# ---------------------------------------------------------------------------
# The response document
# ---------------------------------------------------------------------------


def test_normalized_scores_do_not_invent_a_spread() -> None:
    assert normalize_scores([]) == []
    assert normalize_scores([2.5]) == [50.0]
    assert normalize_scores([1.0, 1.0, 1.0]) == [50.0, 50.0, 50.0]
    assert normalize_scores([-1.0, 0.0, 1.0]) == [0.0, 50.0, 100.0]


def test_the_result_never_claims_calibration(predictor: RankerPredictor) -> None:
    examples = [make_example(predictor.schema, "upload:000001000", seed=17)]
    scored = score_examples(predictor, examples, {"upload:000001000": 1000}, {})
    result = InferenceResult(
        episode_id="upload",
        duration_seconds=60.0,
        ranked=tuple(scored),
        excluded=(ExcludedCandidate("upload:000002000", "missing_features", "no audio"),),
        model=predictor.identity,
        candidate_count=2,
    )
    payload = result.to_dict()
    assert payload["is_calibrated_probability"] is False
    assert payload["score_scale"] == "episode_relative_min_max"
    assert payload["schema_version"] == INFERENCE_SCHEMA_VERSION
    assert payload["provenance"] == "learned_ranker"
    assert payload["excluded_count"] == 1
    assert payload["scored_count"] == 1
    assert payload["ranked"][0]["provenance"] == "learned_ranker"


def test_the_result_reports_the_training_label_source(
    predictor: RankerPredictor,
) -> None:
    """A model trained on anything other than human labels must say so, because
    that is what stops its numbers being quoted as a quality result."""
    assert predictor.identity.training_label_source in {
        "human",
        "weak_heuristic",
        "synthetic",
        "unknown",
    }
