"""The product-facing inference path.

These tests never touch audio or a real model download. They train a tiny
ranker on the synthetic fixture corpus, save a checkpoint, and then exercise
the predictor exactly as the Express API does: load, verify, score, rank.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from slotify_rank.config.settings import find_repo_root
from slotify_rank.datasets.normalizer import read_normalizer, write_normalizer
from slotify_rank.datasets.schema import DatasetSchema, TrainingExample
from slotify_rank.inference.predictor import (
    PredictorError,
    RankerPredictor,
    rank_examples,
    score_examples,
)
from slotify_rank.inference.schema import (
    INFERENCE_SCHEMA_VERSION,
    WEAK_LABEL_DISCLOSURE,
    ExcludedCandidate,
    InferenceResult,
    normalize_scores,
)
from slotify_rank.models.schema import ModelConfig
from slotify_rank.training.checkpoint import (
    IncompatibleCheckpoint,
    atomic_torch_save,
    load_checkpoint,
)
from slotify_rank.training.config import TrainingConfig
from slotify_rank.training.prepare import prepare_dataset
from slotify_rank.training.trainer import Trainer, build_seeded_model
from tests.training_fixtures import SYNTHETIC_SOURCES, shared_corpus


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory) -> Path:
    """A real checkpoint from a real (tiny, synthetic) training run."""
    run_dir = tmp_path_factory.mktemp("inference-run")
    paths, corpus = shared_corpus(episode_count=10, candidates_per_episode=8)
    config = TrainingConfig(epochs=2, early_stopping_patience=0)
    dataset = prepare_dataset(
        paths, config, corpus.label_export, allowed_label_sources=SYNTHETIC_SOURCES
    )
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


def _twins(predictor: RankerPredictor) -> list[TrainingExample]:
    example = make_example(predictor.schema, "upload:000002000", seed=7)
    return [example, replace(example, candidate_id="upload:000001000")]


def test_ties_break_deterministically_on_timestamp(predictor: RankerPredictor) -> None:
    timestamps = {"upload:000002000": 2000, "upload:000001000": 1000}
    scored = score_examples(predictor, _twins(predictor), timestamps, {})
    assert scored[0].candidate_id == "upload:000001000"


def test_identical_candidates_are_not_stretched_into_a_spread(
    predictor: RankerPredictor,
) -> None:
    """Two copies of one vector in one float32 batch can score a few 1e-9 apart.

    Min-max normalizing that noise would report 100 and 0 to the user as
    `placement_score`. After quantization the two are one score, normalized to
    the no-spread value, and the raw scores the response reports agree.
    """
    timestamps = {"upload:000002000": 2000, "upload:000001000": 1000}
    scored = score_examples(predictor, _twins(predictor), timestamps, {})
    assert [entry.normalized_score for entry in scored] == [50.0, 50.0]
    assert scored[0].raw_score == scored[1].raw_score


def test_noise_below_a_quantum_never_reaches_the_normalized_score() -> None:
    assert normalize_scores([0.4, 0.4 + 7.45e-9]) == [50.0, 50.0]
    assert normalize_scores([0.4, 0.400001]) == [0.0, 100.0]


def test_ordering_normalization_and_raw_score_agree(predictor: RankerPredictor) -> None:
    examples = [make_example(predictor.schema, f"upload:{i:09d}", seed=i) for i in range(6)]
    timestamps = {example.candidate_id: i * 1000 for i, example in enumerate(examples)}
    scored = score_examples(predictor, examples, timestamps, {})
    raw = [entry.raw_score for entry in scored]
    normalized = [entry.normalized_score for entry in scored]
    assert raw == sorted(raw, reverse=True)
    assert normalized == sorted(normalized, reverse=True)
    assert all(value == round(value, 6) for value in raw)


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


# ---------------------------------------------------------------------------
# The served schema is compared with the checkpoint's, by name
# ---------------------------------------------------------------------------


def _served(schema: DatasetSchema, names) -> DatasetSchema:
    return replace(
        schema, handcrafted_feature_names=tuple(names), handcrafted_dimension=len(names)
    )


def _reordered(example: TrainingExample, order) -> TrainingExample:
    index = np.asarray(order)
    return replace(
        example,
        handcrafted=example.handcrafted[index],
        handcrafted_missing_mask=example.handcrafted_missing_mask[index],
    )


def test_same_width_with_different_names_is_refused(predictor: RankerPredictor) -> None:
    """Equal width is exactly the case a width check cannot see."""
    names = list(predictor.schema.handcrafted_feature_names)
    renamed = _served(predictor.schema, ["something_else", *names[1:]])
    example = make_example(predictor.schema, "upload:000001000", seed=1)
    with pytest.raises(IncompatibleCheckpoint, match="absent from the served features"):
        rank_examples(predictor, [example], renamed, {}, {})


def test_a_reordered_layout_is_projected_back_by_name(predictor: RankerPredictor) -> None:
    names = list(predictor.schema.handcrafted_feature_names)
    order = list(reversed(range(len(names))))
    served = _served(predictor.schema, [names[i] for i in order])
    examples = [make_example(predictor.schema, f"upload:{i:09d}", seed=i) for i in range(4)]
    timestamps = {e.candidate_id: i * 1000 for i, e in enumerate(examples)}

    expected = score_examples(predictor, examples, timestamps, {})
    ranking = rank_examples(
        predictor, [_reordered(e, order) for e in examples], served, timestamps, {}
    )
    assert [c.to_dict() for c in ranking.ranked] == [c.to_dict() for c in expected]


def test_a_superset_layout_is_projected_and_the_extra_column_withheld(
    predictor: RankerPredictor,
) -> None:
    names = list(predictor.schema.handcrafted_feature_names)
    served = _served(predictor.schema, ["heuristic_extra_column", *names])
    examples = [make_example(predictor.schema, f"upload:{i:09d}", seed=i) for i in range(3)]
    widened = [
        replace(
            e,
            handcrafted=np.concatenate([[123.0], e.handcrafted]).astype(np.float32),
            handcrafted_missing_mask=np.concatenate([[False], e.handcrafted_missing_mask]),
        )
        for e in examples
    ]
    timestamps = {e.candidate_id: i * 1000 for i, e in enumerate(examples)}
    ranking = rank_examples(predictor, widened, served, timestamps, {})
    expected = score_examples(predictor, examples, timestamps, {})
    assert [c.raw_score for c in ranking.ranked] == [c.raw_score for c in expected]
    assert ranking.withheld_columns == ("heuristic_extra_column",)


def test_a_learned_block_of_another_construction_is_refused(
    predictor: RankerPredictor,
) -> None:
    served = replace(predictor.schema, text_parts=("before", "after", "product", "difference"))
    example = make_example(predictor.schema, "upload:000001000", seed=1)
    with pytest.raises(IncompatibleCheckpoint, match="text_parts"):
        rank_examples(predictor, [example], served, {}, {})


def test_a_feature_spec_difference_is_disclosed_not_refused(
    predictor: RankerPredictor,
) -> None:
    """A version bump elsewhere must not take the served model down, and must
    not pass silently either."""
    served = replace(predictor.schema, feature_spec_version="featurespec-v9.9.9")
    example = make_example(predictor.schema, "upload:000001000", seed=1)
    ranking = rank_examples(predictor, [example], served, {"upload:000001000": 1000}, {})
    assert len(ranking.ranked) == 1
    assert any("featurespec-v9.9.9" in warning for warning in ranking.warnings)


def test_scoring_by_width_alone_is_refused(predictor: RankerPredictor) -> None:
    example = make_example(predictor.schema, "upload:000001000", seed=1)
    with pytest.raises(IncompatibleCheckpoint, match="no schema"):
        rank_examples(predictor, [example], None, {}, {})


# ---------------------------------------------------------------------------
# The normalizer must be the checkpoint's own
# ---------------------------------------------------------------------------


def _copy_run(trained_checkpoint: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("best_checkpoint.pt", "normalizer.json"):
        (destination / name).write_bytes((trained_checkpoint.parent / name).read_bytes())
    return destination / "best_checkpoint.pt"


def test_a_normalizer_with_shifted_column_names_is_refused(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint = _copy_run(trained_checkpoint, tmp_path)
    normalizer = read_normalizer(tmp_path / "normalizer.json")
    names = list(normalizer.feature_names)
    write_normalizer(
        tmp_path / "normalizer.json",
        replace(normalizer, feature_names=tuple([names[1], names[0], *names[2:]])),
    )
    with pytest.raises(IncompatibleCheckpoint, match="normalizer columns differ"):
        RankerPredictor.load(checkpoint)


def test_a_normalizer_fitted_on_another_split_is_refused(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint = _copy_run(trained_checkpoint, tmp_path)
    normalizer = read_normalizer(tmp_path / "normalizer.json")
    foreign = tmp_path / "elsewhere" / "normalizer.json"
    write_normalizer(foreign, replace(normalizer, training_split_hash="0" * 32))
    # Reachable through --normalizer / SLOTIFY_RANKER_NORMALIZER, not only beside it.
    with pytest.raises(IncompatibleCheckpoint, match="different training split"):
        RankerPredictor.load(checkpoint, foreign)


# ---------------------------------------------------------------------------
# What the model was trained on
# ---------------------------------------------------------------------------


def test_new_checkpoints_store_their_label_source(trained_checkpoint: Path) -> None:
    payload = load_checkpoint(trained_checkpoint)
    assert payload["label_source"] == "synthetic"
    assert payload["dropped_handcrafted_features"] == []


def test_the_stored_label_source_wins_over_a_sidecar(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint = _copy_run(trained_checkpoint, tmp_path)
    payload = load_checkpoint(checkpoint)
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"run_id": payload["run_id"], "label_source": "human"}),
        encoding="utf-8",
    )
    assert RankerPredictor.load(checkpoint).identity.training_label_source == "synthetic"


def _legacy(trained_checkpoint: Path, destination: Path) -> tuple[Path, str]:
    """A copy with the provenance keys removed, as the shipped bootstrap has."""
    checkpoint = _copy_run(trained_checkpoint, destination)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("label_source", "data_provenance", "dropped_handcrafted_features"):
        raw.pop(key, None)
    atomic_torch_save(checkpoint, raw)
    return checkpoint, str(raw["run_id"])


def test_a_legacy_checkpoint_falls_back_to_its_own_runs_summary(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint, run_id = _legacy(trained_checkpoint, tmp_path)
    (tmp_path / "training_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "label_source": "weak_heuristic",
                "data_provenance": "weak_supervision",
            }
        ),
        encoding="utf-8",
    )
    identity = RankerPredictor.load(checkpoint).identity
    assert identity.training_label_source == "weak_heuristic"
    assert identity.training_data_provenance == "weak_supervision"


def test_a_legacy_checkpoint_beside_another_runs_summary_is_unknown(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint, _ = _legacy(trained_checkpoint, tmp_path)
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"run_id": "some-other-run", "label_source": "human"}),
        encoding="utf-8",
    )
    assert RankerPredictor.load(checkpoint).identity.training_label_source == "unknown"


def test_a_legacy_checkpoint_with_no_summary_is_unknown(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint, _ = _legacy(trained_checkpoint, tmp_path)
    assert RankerPredictor.load(checkpoint).identity.training_label_source == "unknown"


# ---------------------------------------------------------------------------
# The disclosure in the response
# ---------------------------------------------------------------------------


def _result_for(predictor: RankerPredictor, label_source: str) -> dict:
    identity = replace(predictor.identity, training_label_source=label_source)
    return InferenceResult(
        episode_id="upload",
        duration_seconds=1.0,
        ranked=(),
        excluded=(),
        model=identity,
        warnings=("Transcription was skipped.",),
    ).to_dict()


def test_a_weak_label_model_says_it_distills_the_baseline(
    predictor: RankerPredictor,
) -> None:
    payload = _result_for(predictor, "weak_heuristic")
    assert WEAK_LABEL_DISCLOSURE in payload["warnings"]
    assert "distillation of the heuristic baseline" in WEAK_LABEL_DISCLOSURE
    assert "without any human labels" in WEAK_LABEL_DISCLOSURE
    assert "Transcription was skipped." in payload["warnings"]
    # A warning is not a shape change.
    assert payload["schema_version"] == INFERENCE_SCHEMA_VERSION == "inference-v1.0.0"


def test_a_human_trained_model_carries_no_training_disclosure(
    predictor: RankerPredictor,
) -> None:
    assert _result_for(predictor, "human")["warnings"] == ["Transcription was skipped."]


def test_synthetic_and_unknown_supervision_are_disclosed_too(
    predictor: RankerPredictor,
) -> None:
    assert any("synthetic" in w for w in _result_for(predictor, "synthetic")["warnings"])
    assert any("unknown" in w for w in _result_for(predictor, "unknown")["warnings"])


# ---------------------------------------------------------------------------
# A weak-label run cannot see its teacher, and still serves
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def weak_run(tmp_path_factory):
    from slotify_rank.data import manifests
    from slotify_rank.datasets.normalizer import write_normalizer as write
    from slotify_rank.labelling.weak import build_weak_label_rows, write_weak_label_export

    paths, corpus = shared_corpus(episode_count=10, candidates_per_episode=8)
    directory = tmp_path_factory.mktemp("weak-run")
    rows, summary = build_weak_label_rows(manifests.read_candidates(paths.candidates_manifest))
    export = directory / "weak_labels.jsonl"
    write_weak_label_export(export, rows, summary)

    config = TrainingConfig(epochs=1, early_stopping_patience=0)
    dataset = prepare_dataset(
        paths, config, export, allowed_label_sources=("weak_heuristic",)
    )
    model_config = ModelConfig(variant="gated", auxiliary_head=config.auxiliary_head)
    trainer = Trainer(
        model=build_seeded_model(model_config, dataset.schema, config.seed),
        dataset=dataset,
        config=config,
        run_directory=directory,
        run_id="weak-test",
        model_config=model_config.to_dict(),
    )
    trainer.train()
    write(directory / "normalizer.json", dataset.normalizer)
    return paths, dataset, directory / "best_checkpoint.pt"


def test_a_weak_label_run_withholds_every_heuristic_column(weak_run) -> None:
    _, dataset, checkpoint = weak_run
    names = dataset.schema.handcrafted_feature_names
    assert not [name for name in names if name.startswith("heuristic_")]
    assert dataset.dropped_handcrafted_features == ("heuristic_total_score",)
    assert dataset.normalizer.feature_names == names
    summary = dataset.summary()["dropped_handcrafted_features"]
    assert summary["names"] == ["heuristic_total_score"]
    assert "weak_heuristic" in summary["reason"]

    payload = load_checkpoint(checkpoint)
    assert payload["label_source"] == "weak_heuristic"
    assert payload["dropped_handcrafted_features"] == ["heuristic_total_score"]
    assert "heuristic_total_score" not in payload["input_schema"]["handcrafted_feature_names"]


def test_a_weak_label_checkpoint_scores_the_full_served_layout(weak_run) -> None:
    """Serving still builds every column; the predictor projects by name."""
    from slotify_rank.datasets.loader import EligibilityConfig, build_examples
    from slotify_rank.features.assemble import read_feature_manifest

    paths, _, checkpoint = weak_run

    header, records = read_feature_manifest(paths.features_manifest)
    served = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=None,
        config=EligibilityConfig(require_labels=False, require_acceptability_label=False),
    )
    assert "heuristic_total_score" in served.schema.handcrafted_feature_names

    predictor = RankerPredictor.load(checkpoint)
    assert predictor.identity.training_label_source == "weak_heuristic"
    examples = served.examples[:8]
    ranking = rank_examples(
        predictor,
        examples,
        served.schema,
        {e.candidate_id: i for i, e in enumerate(examples)},
        {},
    )
    assert len(ranking.ranked) == len(examples)
    assert ranking.withheld_columns == ("heuristic_total_score",)
    result = InferenceResult(
        episode_id="upload",
        duration_seconds=None,
        ranked=tuple(ranking.ranked),
        excluded=(),
        model=predictor.identity,
    )
    assert WEAK_LABEL_DISCLOSURE in result.to_dict()["warnings"]


def test_synthetic_and_human_runs_keep_the_heuristic_columns(trained_checkpoint) -> None:
    payload = load_checkpoint(trained_checkpoint)
    assert "heuristic_total_score" in payload["input_schema"]["handcrafted_feature_names"]


# ---------------------------------------------------------------------------
# The shipped checkpoint keeps working through the product path
# ---------------------------------------------------------------------------

SHIPPED = (
    Path(find_repo_root())
    / "artifacts"
    / "training"
    / "gated-d8ed976101aa4c3b"
    / "best_checkpoint.pt"
)


@pytest.mark.skipif(not SHIPPED.is_file(), reason="the committed checkpoint is absent")
def test_the_shipped_checkpoint_loads_scores_and_discloses() -> None:
    predictor = RankerPredictor.load(SHIPPED)
    identity = predictor.identity
    # A legacy checkpoint: provenance comes from its own run's summary.
    assert "label_source" not in load_checkpoint(SHIPPED)
    assert identity.training_label_source == "weak_heuristic"
    assert identity.handcrafted_dimension == 110

    examples = [make_example(predictor.schema, f"upload:{i:09d}", seed=i) for i in range(5)]
    ranking = rank_examples(
        predictor,
        examples,
        predictor.schema,
        {e.candidate_id: i * 1000 for i, e in enumerate(examples)},
        {},
    )
    assert [c.rank for c in ranking.ranked] == [1, 2, 3, 4, 5]
    payload = InferenceResult(
        episode_id="upload",
        duration_seconds=5.0,
        ranked=tuple(ranking.ranked),
        excluded=(),
        model=identity,
    ).to_dict()
    assert payload["warnings"][0] == WEAK_LABEL_DISCLOSURE


def test_inference_reads_the_committed_pipeline_configs_by_default() -> None:
    from slotify_rank.inference.episode import PipelineConfigs, default_config_dir

    configs = PipelineConfigs.load()
    assert Path(configs.source_dir) == default_config_dir()
    assert default_config_dir() == Path(find_repo_root()) / "ml" / "configs"
    # Same values as loading the directory explicitly -- there is one source.
    explicit = PipelineConfigs.load(default_config_dir())
    assert configs == explicit
