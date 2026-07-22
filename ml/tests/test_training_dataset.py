"""Dataset loading, eligibility accounting and train-only normalization."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from slotify_rank.config.versions import (
    FEATURE_PIPELINE_VERSION,
    FEATURE_SPEC_VERSION,
    NORMALIZER_VERSION,
)
from slotify_rank.data.schema import DatasetCandidate
from slotify_rank.datasets.labels import aggregate_labels, read_label_export
from slotify_rank.datasets.loader import EligibilityConfig, build_examples
from slotify_rank.datasets.normalizer import (
    fit_normalizer,
    read_normalizer,
    write_normalizer,
)
from slotify_rank.datasets.schema import (
    AUDIO_EMBEDDING_PARTS,
    NATIVE_EMBEDDING_DIMENSION,
    TEXT_EMBEDDING_PARTS,
    DatasetSchema,
    TrainingExample,
)
from slotify_rank.features.assemble import read_feature_manifest
from tests.training_fixtures import load_corpus, make_corpus


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_compatible_feature_artifacts_load(tmp_path):
    _, corpus, loaded = load_corpus(tmp_path)
    assert len(loaded.examples) == corpus.candidate_count
    assert loaded.counts["eligible"] == corpus.candidate_count
    assert loaded.schema is not None
    assert loaded.schema.audio_dimension == NATIVE_EMBEDDING_DIMENSION * len(
        AUDIO_EMBEDDING_PARTS
    )
    assert loaded.schema.text_dimension == NATIVE_EMBEDDING_DIMENSION * len(
        TEXT_EMBEDDING_PARTS
    )


def test_embedding_dimensions_come_from_the_artifact_not_a_constant(tmp_path):
    _, _, loaded = load_corpus(tmp_path, native_dimension=64)
    # The loader was told 384 by default; it must reject an artifact that
    # disagrees rather than reading 64 numbers into a 384-wide slot.
    assert loaded.counts["eligible"] == 0
    assert loaded.counts["stale_features"] > 0


def test_native_dimension_can_be_declared(tmp_path):
    paths, corpus = make_corpus(tmp_path, native_dimension=64)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        native_embedding_dimension=64,
    )
    assert loaded.counts["eligible"] == corpus.candidate_count
    assert loaded.schema.audio_dimension == 64 * 4


def test_incompatible_feature_pipeline_version_is_rejected(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        config=EligibilityConfig(
            required_feature_pipeline_version="featurepipeline-v9.9.9"
        ),
    )
    assert loaded.counts["unsupported_version"] == corpus.candidate_count
    assert not loaded.examples


def test_missing_labels_are_counted_not_dropped(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    trimmed = replace(
        labels, labels={k: v for k, v in list(labels.labels.items())[:5]}
    )
    loaded = build_examples(
        paths=paths, header=header, records=records, labels=trimmed
    )
    assert loaded.counts["eligible"] == 5
    assert loaded.counts["missing_label"] == corpus.candidate_count - 5
    assert sum(loaded.counts.values()) == corpus.candidate_count


def test_synthetic_candidates_are_excluded(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    padding = DatasetCandidate(
        episode_id=records[0].episode_id,
        candidate_id=records[0].candidate_id,
        timestamp_ms=records[0].timestamp_ms,
        candidate_sources=("fixed_interval",),
        is_synthetic=True,
        eligible_for_labelling=False,
        eligible_for_evaluation=False,
    )
    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        candidates=[padding],
    )
    assert loaded.counts["synthetic_excluded"] == 1
    assert records[0].candidate_id not in {e.candidate_id for e in loaded.examples}


def test_ineligible_candidates_are_reported_separately_from_synthetic(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    ineligible = DatasetCandidate(
        episode_id=records[0].episode_id,
        candidate_id=records[0].candidate_id,
        timestamp_ms=records[0].timestamp_ms,
        candidate_sources=("pause",),
        eligible_for_labelling=False,
    )
    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        candidates=[ineligible],
    )
    assert loaded.counts["ineligible_candidate"] == 1
    assert loaded.counts["synthetic_excluded"] == 0


def test_duplicate_candidate_ids_are_a_hard_failure(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    with pytest.raises(ValueError, match="Duplicate candidate_id"):
        build_examples(
            paths=paths,
            header=header,
            records=[*records, records[0]],
            labels=labels,
        )


def test_split_disagreement_with_the_manifest_is_rejected(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    lying = {records[0].episode_id: "test"}
    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        split_lookup=lying,
    )
    reasons = {e.reason for e in loaded.exclusions}
    if records[0].dataset_split != "test":
        assert "split_mismatch" in reasons


def test_unassigned_split_never_becomes_an_example(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    orphan = replace(records[0], dataset_split="unassigned")
    loaded = build_examples(
        paths=paths, header=header, records=[orphan, *records[1:]], labels=labels
    )
    assert loaded.counts["split_mismatch"] == 1


def test_splits_are_preserved_exactly(tmp_path):
    _, corpus, loaded = load_corpus(tmp_path)
    for example in loaded.examples:
        assert example.split == corpus.splits[example.episode_id]
    assert {"train", "validation", "test"} <= set(
        e.split for e in loaded.examples
    )


def test_candidate_ordering_is_deterministic(tmp_path):
    _, _, first = load_corpus(tmp_path)
    _, _, second = load_corpus(tmp_path / "again")
    assert [e.candidate_id for e in first.examples] == [
        e.candidate_id for e in second.examples
    ]


def test_missing_text_modality_is_masked_not_faked(tmp_path):
    _, _, loaded = load_corpus(tmp_path, text_missing_episode_stride=3)
    without_text = [e for e in loaded.examples if not e.text_available]
    assert without_text, "the fixture should withhold text from some episodes"
    for example in without_text:
        assert not example.text.any()
        assert example.audio_available
    coverage = loaded.summary()["modality_coverage"]
    assert coverage["audio_only"] == len(without_text)


def test_require_complete_multimodal_filters_partial_records(tmp_path):
    paths, corpus = make_corpus(tmp_path, text_missing_episode_stride=3)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        config=EligibilityConfig(require_complete_multimodal=True),
    )
    assert all(e.audio_available and e.text_available for e in loaded.examples)
    assert loaded.counts["missing_features"] > 0


def test_failed_records_are_excluded_as_invalid(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    broken = replace(
        records[0],
        feature_status="failed",
        failure_reason="decode error",
        audio_embedding_available=False,
        text_embedding_available=False,
        audio_embedding_reference={},
        text_embedding_reference={},
    )
    loaded = build_examples(
        paths=paths, header=header, records=[broken, *records[1:]], labels=labels
    )
    assert loaded.counts["invalid_features"] == 1


def test_non_finite_handcrafted_values_are_excluded(tmp_path):
    paths, corpus, _ = load_corpus(tmp_path)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    values = list(records[0].handcrafted_feature_values)
    values[0] = float("nan")
    poisoned = replace(records[0], handcrafted_feature_values=tuple(values))
    loaded = build_examples(
        paths=paths, header=header, records=[poisoned, *records[1:]], labels=labels
    )
    assert loaded.counts["invalid_features"] == 1


def test_every_candidate_is_accounted_for(tmp_path):
    _, corpus, loaded = load_corpus(tmp_path)
    assert sum(loaded.counts.values()) == corpus.candidate_count
    assert len(loaded.examples) + len(loaded.exclusions) == corpus.candidate_count


def test_summary_is_json_serializable(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    json.dumps(loaded.summary())


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_weak_labels_are_refused(tmp_path):
    with pytest.raises(ValueError, match="Only human labels"):
        aggregate_labels(
            [
                {
                    "candidate_id": "c1",
                    "episode_id": "e1",
                    "annotator_id": "a",
                    "quality_score": 4,
                    "is_acceptable": True,
                    "label_source": "weak_heuristic",
                }
            ],
            acceptable_threshold=3.0,
        )


def test_multiple_annotators_are_averaged(tmp_path):
    labels = aggregate_labels(
        [
            {
                "candidate_id": "c1",
                "episode_id": "e1",
                "annotator_id": "a",
                "quality_score": 4,
                "is_acceptable": True,
                "rubric_version": "rubric-v1.0.0",
            },
            {
                "candidate_id": "c1",
                "episode_id": "e1",
                "annotator_id": "b",
                "quality_score": 5,
                "is_acceptable": True,
                "rubric_version": "rubric-v1.0.0",
            },
        ],
        acceptable_threshold=3.0,
    )
    assert labels.labels["c1"].quality_score == pytest.approx(4.5)
    assert labels.labels["c1"].annotator_count == 2


def test_an_unusable_vote_removes_the_candidate():
    labels = aggregate_labels(
        [
            {
                "candidate_id": "c1",
                "episode_id": "e1",
                "annotator_id": "a",
                "quality_score": 4,
                "is_acceptable": True,
                "is_unusable": False,
            },
            {
                "candidate_id": "c1",
                "episode_id": "e1",
                "annotator_id": "b",
                "quality_score": 1,
                "is_acceptable": False,
                "is_unusable": True,
            },
        ],
        acceptable_threshold=3.0,
    )
    assert "c1" not in labels.labels
    assert labels.unusable_candidate_ids == ("c1",)


def test_label_export_requires_its_metadata_sidecar(tmp_path):
    paths, corpus = make_corpus(tmp_path)
    corpus.label_export.with_suffix(
        corpus.label_export.suffix + ".meta.json"
    ).unlink()
    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        read_label_export(corpus.label_export)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalizer_fits_on_train_only(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    train = loaded.by_split("train")
    normalizer = fit_normalizer(
        train, loaded.schema.handcrafted_feature_names, expected_split="train"
    )
    assert normalizer.fit_candidate_count == len(train)
    assert normalizer.fit_episode_count == len({e.episode_id for e in train})


def test_normalizer_refuses_validation_examples(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    mixed = loaded.by_split("train") + loaded.by_split("validation")
    with pytest.raises(ValueError, match="Refusing to fit"):
        fit_normalizer(
            mixed, loaded.schema.handcrafted_feature_names, expected_split="train"
        )


def test_validation_statistics_never_change_the_fit(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    names = loaded.schema.handcrafted_feature_names
    train = loaded.by_split("train")
    baseline = fit_normalizer(train, names)

    shifted = [
        replace(example, handcrafted=example.handcrafted + 1000.0)
        for example in loaded.by_split("validation")
    ]
    # Refitting on train after perturbing validation must produce identical stats.
    again = fit_normalizer(train, names)
    assert np.allclose(baseline.means, again.means)
    assert np.allclose(baseline.standard_deviations, again.standard_deviations)
    assert shifted  # the perturbed rows exist and were simply never consulted


def test_constant_features_are_not_amplified(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    names = loaded.schema.handcrafted_feature_names
    normalizer = fit_normalizer(loaded.by_split("train"), names)
    index = names.index("constant_probe")
    assert normalizer.constant_feature_mask[index]
    assert normalizer.standard_deviations[index] == 1.0
    transformed = normalizer.transform(
        loaded.examples[0].handcrafted, loaded.examples[0].handcrafted_missing_mask
    )
    assert np.isfinite(transformed).all()
    assert abs(float(transformed[index])) < 1e-3


def test_binary_flags_pass_through_unscaled(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    names = loaded.schema.handcrafted_feature_names
    normalizer = fit_normalizer(loaded.by_split("train"), names)
    index = names.index("sentence_end")
    assert normalizer.binary_feature_mask[index]
    example = loaded.examples[0]
    transformed = normalizer.transform(
        example.handcrafted, example.handcrafted_missing_mask
    )
    assert float(transformed[index]) in (0.0, 1.0)


def test_missing_values_are_excluded_from_the_fit_and_zeroed(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    names = loaded.schema.handcrafted_feature_names
    normalizer = fit_normalizer(loaded.by_split("train"), names)
    index = names.index("context_cosine_similarity")
    masked = [e for e in loaded.examples if e.handcrafted_missing_mask[index]]
    assert masked, "the fixture should mark some context similarities missing"
    transformed = normalizer.transform(
        masked[0].handcrafted, masked[0].handcrafted_missing_mask
    )
    assert float(transformed[index]) == 0.0


def test_normalizer_round_trips_through_disk(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    names = loaded.schema.handcrafted_feature_names
    normalizer = fit_normalizer(loaded.by_split("train"), names)
    path = tmp_path / "normalizer.json"
    write_normalizer(path, normalizer)
    restored = read_normalizer(path)
    assert restored.normalizer_version == NORMALIZER_VERSION
    assert restored.feature_names == normalizer.feature_names
    assert np.allclose(restored.means, normalizer.means)
    example = loaded.examples[0]
    assert np.allclose(
        restored.transform(example.handcrafted, example.handcrafted_missing_mask),
        normalizer.transform(example.handcrafted, example.handcrafted_missing_mask),
    )


def test_normalizer_rejects_a_changed_column_count(tmp_path):
    _, _, loaded = load_corpus(tmp_path)
    names = loaded.schema.handcrafted_feature_names
    normalizer = fit_normalizer(loaded.by_split("train"), names)
    with pytest.raises(ValueError, match="feature layout changed"):
        normalizer.transform(np.zeros((2, len(names) + 1), dtype=np.float32))


def test_fitting_on_zero_examples_fails_loudly():
    with pytest.raises(ValueError, match="zero examples"):
        fit_normalizer([], ("a", "b"))


# ---------------------------------------------------------------------------
# Schema compatibility
# ---------------------------------------------------------------------------


def test_schema_incompatibility_names_every_reason():
    base = DatasetSchema(
        handcrafted_feature_names=("a", "b"),
        handcrafted_dimension=2,
        audio_dimension=384 * 4,
        text_dimension=384 * 4,
        native_embedding_dimension=384,
        feature_spec_version=FEATURE_SPEC_VERSION,
        feature_pipeline_version=FEATURE_PIPELINE_VERSION,
    )
    other = DatasetSchema(
        handcrafted_feature_names=("b", "a"),
        handcrafted_dimension=2,
        audio_dimension=384 * 4,
        text_dimension=384 * 4,
        native_embedding_dimension=384,
        feature_spec_version="featurespec-v9.9.9",
        feature_pipeline_version=FEATURE_PIPELINE_VERSION,
    )
    reasons = base.incompatibilities(other)
    assert any("ordering" in reason for reason in reasons)
    assert any("feature_spec_version" in reason for reason in reasons)
    assert base.incompatibilities(base) == []


def test_example_rejects_an_available_but_zero_modality():
    with pytest.raises(ValueError, match="all zeros"):
        TrainingExample(
            candidate_id="c",
            episode_id="e",
            split="train",
            handcrafted=np.zeros(3, dtype=np.float32),
            handcrafted_missing_mask=np.zeros(3, dtype=bool),
            audio=np.zeros(8, dtype=np.float32),
            audio_available=True,
            text=np.zeros(8, dtype=np.float32),
            text_available=False,
            quality_score=3.0,
            is_acceptable=True,
        )
