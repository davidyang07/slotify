"""Assembly joins, validation and statistics.

Everything is built from synthetic handcrafted payloads and synthetic embedding
arrays, so the join logic is exercised without loading a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slotify_rank.config.versions import (
    FEATURE_MANIFEST_SCHEMA_VERSION,
    FEATURE_SPEC_VERSION,
)
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.embeddings.store import EmbeddingMetadata, write_embeddings
from slotify_rank.features.assemble import (
    assemble_episode,
    build_spec,
    read_feature_manifest,
    write_feature_manifest,
)
from slotify_rank.features.schema import CandidateFeatureRecord, FeatureSpec
from slotify_rank.features.stats import compute_feature_statistics
from slotify_rank.features.validate import validate_features
from slotify_rank.pipeline.stages import audio_row_id, eligible_candidates, text_row_id

EPISODE = "ep_0123456789ab"
AUDIO_DIM = 384
TEXT_DIM = 384
BASE_FEATURES = {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}


def _paths(tmp_path: Path) -> DataPaths:
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    return paths


def _candidate(timestamp_ms: int, synthetic: bool = False) -> DatasetCandidate:
    return DatasetCandidate(
        episode_id=EPISODE,
        candidate_id=f"{EPISODE}:{timestamp_ms:09d}",
        timestamp_ms=timestamp_ms,
        candidate_sources=("product_padding",) if synthetic else ("silence",),
        is_synthetic=synthetic,
        eligible_for_labelling=not synthetic,
        eligible_for_evaluation=not synthetic,
    )


def _episode() -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=EPISODE,
        series_id="series-a",
        title="Fixture",
        source_type="existing_repository_fixture",
        source_uri="backend/audio_tests/convo.mp3",
        source_name="fixture",
        license_name=None,
        license_url=None,
        attribution=None,
        language="en",
        content_type="podcast",
        original_path="backend/audio_tests/convo.mp3",
        sha256="a" * 64,
        status="normalized",
        normalized_path="data/normalized/ep.wav",
        normalized_sha256="b" * 64,
        normalized_duration_ms=60_000,
        duration_ms=60_000,
    )


def _handcrafted(candidates, with_text: bool = True) -> dict:
    records = {}
    for candidate in candidates:
        records[candidate.candidate_id] = {
            "timestamp_ms": candidate.timestamp_ms,
            "values": dict(BASE_FEATURES),
            "missing": {name: False for name in BASE_FEATURES},
            "context_before_text": "Something before." if with_text else "",
            "context_after_text": "Something after." if with_text else "",
            "context_segment_ids_before": ["seg-1"] if with_text else [],
            "context_segment_ids_after": ["seg-2"] if with_text else [],
        }
    return {
        "episode_id": EPISODE,
        "identity_digest": "d" * 64,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "audio_sha256": "b" * 64,
        "transcript_available": with_text,
        "episode_duration_ms": 60_000,
        "candidates": records,
    }


def _write_audio(paths: DataPaths, candidates, kinds=("before", "after", "context", "difference")):
    rows, row_ids = [], []
    for candidate in candidates:
        for kind in kinds:
            rows.append(np.full(AUDIO_DIM, 0.5, dtype=np.float32))
            row_ids.append(audio_row_id(candidate.candidate_id, kind))
    path = paths.audio_embeddings_dir / f"{EPISODE}.audio.npy"
    write_embeddings(
        path,
        np.vstack(rows) if rows else np.zeros((0, AUDIO_DIM), dtype=np.float32),
        EmbeddingMetadata(
            kind="audio_embedding",
            episode_id=EPISODE,
            model_id="openai/whisper-tiny.en",
            model_revision="main",
            dimension=AUDIO_DIM,
            row_ids=tuple(row_ids),
            identity_digest="e" * 64,
        ),
    )
    return path


def _write_text(paths: DataPaths, candidates, sides=("before", "after")):
    rows, row_ids = [], []
    rng = np.random.default_rng(0)
    for candidate in candidates:
        for side in sides:
            vector = rng.normal(size=TEXT_DIM).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
            row_ids.append(text_row_id(candidate.candidate_id, side))
    path = paths.text_embeddings_dir / f"{EPISODE}.text.npy"
    write_embeddings(
        path,
        np.vstack(rows) if rows else np.zeros((0, TEXT_DIM), dtype=np.float32),
        EmbeddingMetadata(
            kind="text_embedding",
            episode_id=EPISODE,
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            model_revision="main",
            dimension=TEXT_DIM,
            row_ids=tuple(row_ids),
            identity_digest="f" * 64,
        ),
    )
    return path


def _assemble(paths, candidates, handcrafted, audio_path, text_path, spec=None):
    spec = spec or build_spec([handcrafted])
    return (
        assemble_episode(
            paths=paths,
            episode_id=EPISODE,
            candidates=candidates,
            handcrafted=handcrafted,
            split_lookup={EPISODE: "train"},
            audio_array_path=audio_path,
            text_array_path=text_path,
            spec=spec,
        ),
        spec,
    )


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def test_a_complete_record_carries_both_modalities(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, _ = _assemble(
        paths, candidates, handcrafted, _write_audio(paths, candidates), _write_text(paths, candidates)
    )
    record = result.records[0]
    assert record.feature_status == "complete"
    assert record.audio_embedding_available and record.text_embedding_available
    assert record.audio_embedding_dimension == 384
    assert record.text_embedding_dimension == 384


def test_embeddings_join_by_row_id_not_by_position(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000), _candidate(20_000)]
    handcrafted = _handcrafted(candidates)
    result, _ = _assemble(
        paths, candidates, handcrafted, _write_audio(paths, candidates), _write_text(paths, candidates)
    )
    for record in result.records:
        for reference in record.audio_embedding_reference.values():
            assert reference.row_id.startswith(record.candidate_id)


def test_split_is_taken_from_the_split_manifest(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, _ = _assemble(
        paths, candidates, handcrafted, _write_audio(paths, candidates), _write_text(paths, candidates)
    )
    assert result.records[0].dataset_split == "train"


def test_assembly_is_deterministic(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000), _candidate(20_000)]
    handcrafted = _handcrafted(candidates)
    audio, text = _write_audio(paths, candidates), _write_text(paths, candidates)
    first, spec = _assemble(paths, candidates, handcrafted, audio, text)
    second, _ = _assemble(paths, candidates, handcrafted, audio, text, spec)
    assert [r.to_dict() for r in first.records] == [r.to_dict() for r in second.records]


# ---------------------------------------------------------------------------
# Missing modalities
# ---------------------------------------------------------------------------


def test_no_text_embeddings_yields_an_audio_only_record(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates, with_text=False)
    result, _ = _assemble(
        paths,
        candidates,
        handcrafted,
        _write_audio(paths, candidates),
        paths.text_embeddings_dir / "absent.npy",
    )
    record = result.records[0]
    assert record.feature_status == "audio_only"
    assert record.text_embedding_available is False
    assert record.text_embedding_reference == {}


def test_no_audio_embeddings_yields_a_text_only_record(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, _ = _assemble(
        paths,
        candidates,
        handcrafted,
        paths.audio_embeddings_dir / "absent.npy",
        _write_text(paths, candidates),
    )
    assert result.records[0].feature_status == "text_only"


def test_neither_modality_yields_a_handcrafted_only_record(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates, with_text=False)
    result, _ = _assemble(
        paths,
        candidates,
        handcrafted,
        paths.audio_embeddings_dir / "absent.npy",
        paths.text_embeddings_dir / "absent.npy",
    )
    assert result.records[0].feature_status == "handcrafted_only"


def test_a_missing_side_masks_the_similarity_features(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, spec = _assemble(
        paths,
        candidates,
        handcrafted,
        _write_audio(paths, candidates),
        _write_text(paths, candidates, sides=("before",)),
    )
    record = result.records[0]
    index = spec.names.index("context_cosine_similarity")
    assert record.handcrafted_missing_mask[index] is True
    assert record.handcrafted_feature_values[index] == 0.0


def test_a_partial_audio_set_is_not_marked_available(tmp_path: Path):
    """'before' alone is not a usable audio representation."""
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, _ = _assemble(
        paths,
        candidates,
        handcrafted,
        _write_audio(paths, candidates, kinds=("before",)),
        _write_text(paths, candidates),
    )
    assert result.records[0].audio_embedding_available is False


# ---------------------------------------------------------------------------
# Exclusions and failures
# ---------------------------------------------------------------------------


def test_synthetic_candidates_are_excluded_and_counted(tmp_path: Path):
    paths = _paths(tmp_path)
    real = _candidate(10_000)
    synthetic = _candidate(20_000, synthetic=True)
    handcrafted = _handcrafted([real])
    result, _ = _assemble(
        paths,
        [real, synthetic],
        handcrafted,
        _write_audio(paths, [real]),
        _write_text(paths, [real]),
    )
    assert len(result.records) == 1
    assert result.excluded_synthetic == 1


def test_eligible_candidates_filters_synthetic_padding():
    candidates = [_candidate(1_000), _candidate(2_000, synthetic=True)]
    eligible = eligible_candidates(candidates, EPISODE)
    assert [c.timestamp_ms for c in eligible] == [1_000]


def test_a_candidate_with_no_extracted_features_is_reported(tmp_path: Path):
    paths = _paths(tmp_path)
    present = _candidate(10_000)
    absent = _candidate(20_000)
    handcrafted = _handcrafted([present])
    result, _ = _assemble(
        paths,
        [present, absent],
        handcrafted,
        _write_audio(paths, [present]),
        _write_text(paths, [present]),
    )
    assert len(result.records) == 1
    assert result.failures and absent.candidate_id in result.failures[0][0]


def test_absent_handcrafted_features_fail_every_candidate(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    result = assemble_episode(
        paths=paths,
        episode_id=EPISODE,
        candidates=candidates,
        handcrafted=None,
        split_lookup={},
        audio_array_path=paths.audio_embeddings_dir / "a.npy",
        text_array_path=paths.text_embeddings_dir / "t.npy",
        spec=FeatureSpec.from_names(["alpha"]),
    )
    assert result.records == []
    assert len(result.failures) == 1


def test_divergent_feature_names_across_episodes_are_rejected():
    first = _handcrafted([_candidate(1_000)])
    second = _handcrafted([_candidate(2_000)])
    second["candidates"][f"{EPISODE}:000002000"]["values"]["extra"] = 1.0
    with pytest.raises(ValueError, match="different feature names"):
        build_spec([first, second])


# ---------------------------------------------------------------------------
# Manifest round trip
# ---------------------------------------------------------------------------


def test_manifest_round_trips(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000), _candidate(20_000)]
    handcrafted = _handcrafted(candidates)
    result, spec = _assemble(
        paths, candidates, handcrafted, _write_audio(paths, candidates), _write_text(paths, candidates)
    )
    write_feature_manifest(paths.features_manifest, spec, result.records)
    header, records = read_feature_manifest(paths.features_manifest)
    assert header["handcrafted_feature_count"] == spec.dimension
    assert len(records) == 2
    assert records[0].to_dict() == sorted(
        result.records, key=lambda r: r.timestamp_ms
    )[0].to_dict()


def test_a_manifest_without_a_header_is_rejected(tmp_path: Path):
    """Records alone are uninterpretable: the column layout lives in the header."""
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, spec = _assemble(
        paths, candidates, handcrafted, _write_audio(paths, candidates), _write_text(paths, candidates)
    )
    write_feature_manifest(paths.features_manifest, spec, result.records)

    lines = paths.features_manifest.read_text(encoding="utf-8").splitlines()
    headerless = tmp_path / "headerless.jsonl"
    headerless.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no header line"):
        read_feature_manifest(headerless)


def test_an_unsupported_record_schema_version_is_rejected(tmp_path: Path):
    path = tmp_path / "features.jsonl"
    path.write_text(
        json.dumps({"record_type": "header", "handcrafted_feature_names": ["a"]})
        + "\n"
        + json.dumps({"schema_version": "feature-record-schema-v99"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported feature record schema_version"):
        read_feature_manifest(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(tmp_path, records, spec, candidates, split_lookup=None, deep=False):
    paths = _paths(tmp_path)
    write_feature_manifest(paths.features_manifest, spec, records)
    header, loaded = read_feature_manifest(paths.features_manifest)
    return validate_features(
        paths=paths,
        header=header,
        records=loaded,
        candidates=candidates,
        episodes=[_episode()],
        split_lookup=split_lookup,
        deep=deep,
    )


def _good(tmp_path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, spec = _assemble(
        paths, candidates, handcrafted, _write_audio(paths, candidates), _write_text(paths, candidates)
    )
    return result.records, spec, candidates


def test_a_well_formed_manifest_validates(tmp_path: Path):
    records, spec, candidates = _good(tmp_path)
    assert _validate(tmp_path, records, spec, candidates).ok


def test_deep_validation_opens_every_referenced_array(tmp_path: Path):
    records, spec, candidates = _good(tmp_path)
    assert _validate(tmp_path, records, spec, candidates, deep=True).ok


def test_a_wrong_vector_dimension_is_an_error(tmp_path: Path):
    records, spec, candidates = _good(tmp_path)
    broken = CandidateFeatureRecord(
        **{
            **records[0].to_dict(),
            "handcrafted_feature_values": (1.0,),
            "handcrafted_missing_mask": (False,),
            "audio_embedding_reference": records[0].audio_embedding_reference,
            "text_embedding_reference": records[0].text_embedding_reference,
        }
    )
    report = _validate(tmp_path, [broken], spec, candidates)
    assert any(issue.code == "wrong_vector_dimension" for issue in report.errors)


def test_a_non_finite_feature_is_an_error(tmp_path: Path):
    records, spec, candidates = _good(tmp_path)
    values = list(records[0].handcrafted_feature_values)
    values[0] = float("nan")
    broken = CandidateFeatureRecord(
        **{
            **records[0].to_dict(),
            "handcrafted_feature_values": tuple(values),
            "handcrafted_missing_mask": records[0].handcrafted_missing_mask,
            "audio_embedding_reference": records[0].audio_embedding_reference,
            "text_embedding_reference": records[0].text_embedding_reference,
        }
    )
    report = _validate(tmp_path, [broken], spec, candidates)
    assert any(issue.code == "non_finite_feature" for issue in report.errors)


def test_a_candidate_absent_from_the_manifest_is_an_error(tmp_path: Path):
    records, spec, _ = _good(tmp_path)
    report = _validate(tmp_path, records, spec, candidates=[])
    assert any(issue.code == "unknown_candidate" for issue in report.errors)


def test_a_duplicate_record_is_an_error(tmp_path: Path):
    records, spec, candidates = _good(tmp_path)
    report = _validate(tmp_path, records + records, spec, candidates)
    assert any(issue.code == "duplicate_record" for issue in report.errors)


def test_a_synthetic_candidate_in_the_manifest_is_an_error(tmp_path: Path):
    records, spec, _ = _good(tmp_path)
    synthetic = _candidate(10_000, synthetic=True)
    report = _validate(tmp_path, records, spec, [synthetic])
    assert any(issue.code == "synthetic_candidate_included" for issue in report.errors)


def test_a_timestamp_outside_the_episode_is_an_error(tmp_path: Path):
    records, spec, candidates = _good(tmp_path)
    broken = CandidateFeatureRecord(
        **{
            **records[0].to_dict(),
            "timestamp_ms": 900_000,
            "handcrafted_feature_values": records[0].handcrafted_feature_values,
            "handcrafted_missing_mask": records[0].handcrafted_missing_mask,
            "audio_embedding_reference": records[0].audio_embedding_reference,
            "text_embedding_reference": records[0].text_embedding_reference,
        }
    )
    report = _validate(tmp_path, [broken], spec, candidates)
    assert any(issue.code == "timestamp_out_of_bounds" for issue in report.errors)


def test_an_audio_checksum_mismatch_is_an_error(tmp_path: Path):
    records, spec, candidates = _good(tmp_path)
    broken = CandidateFeatureRecord(
        **{
            **records[0].to_dict(),
            "audio_sha256": "c" * 64,
            "handcrafted_feature_values": records[0].handcrafted_feature_values,
            "handcrafted_missing_mask": records[0].handcrafted_missing_mask,
            "audio_embedding_reference": records[0].audio_embedding_reference,
            "text_embedding_reference": records[0].text_embedding_reference,
        }
    )
    report = _validate(tmp_path, [broken], spec, candidates)
    assert any(issue.code == "audio_checksum_mismatch" for issue in report.errors)


def test_a_split_disagreement_is_an_error(tmp_path: Path):
    """Leakage enters here without anyone editing a split manifest."""
    records, spec, candidates = _good(tmp_path)
    report = _validate(
        tmp_path, records, spec, candidates, split_lookup={EPISODE: "test"}
    )
    assert any(issue.code == "split_mismatch" for issue in report.errors)


def test_a_stale_feature_spec_version_is_an_error(tmp_path: Path):
    records, _, candidates = _good(tmp_path)
    stale = FeatureSpec(
        names=("alpha", "beta", "context_cosine_similarity", "gamma", "semantic_change_score"),
        spec_version="featurespec-v0.0.1",
    )
    report = _validate(tmp_path, records, stale, candidates)
    assert any(issue.code == "stale_feature_spec" for issue in report.errors)


def test_a_missing_embedding_row_is_caught_by_deep_validation(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    audio = _write_audio(paths, candidates)
    text = _write_text(paths, candidates)
    result, spec = _assemble(paths, candidates, handcrafted, audio, text)
    # Rewrite the audio array with different row ids underneath the references.
    write_embeddings(
        audio,
        np.zeros((1, AUDIO_DIM), dtype=np.float32),
        EmbeddingMetadata(
            kind="audio_embedding",
            episode_id=EPISODE,
            model_id="openai/whisper-tiny.en",
            model_revision="main",
            dimension=AUDIO_DIM,
            row_ids=("unrelated-row",),
            identity_digest="e" * 64,
        ),
    )
    write_feature_manifest(paths.features_manifest, spec, result.records)
    header, loaded = read_feature_manifest(paths.features_manifest)
    report = validate_features(
        paths=paths,
        header=header,
        records=loaded,
        candidates=candidates,
        episodes=[_episode()],
        deep=True,
    )
    assert any(issue.code == "missing_embedding_row" for issue in report.errors)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_statistics_separate_complete_from_partial_records(tmp_path: Path):
    paths = _paths(tmp_path)
    complete = _candidate(10_000)
    audio_only = _candidate(20_000)
    handcrafted = _handcrafted([complete, audio_only])
    handcrafted["candidates"][audio_only.candidate_id]["context_before_text"] = ""
    handcrafted["candidates"][audio_only.candidate_id]["context_after_text"] = ""
    result, spec = _assemble(
        paths,
        [complete, audio_only],
        handcrafted,
        _write_audio(paths, [complete, audio_only]),
        _write_text(paths, [complete]),
    )
    write_feature_manifest(paths.features_manifest, spec, result.records)
    header, records = read_feature_manifest(paths.features_manifest)
    statistics = compute_feature_statistics(paths, header, records, [_episode()])

    assert statistics["candidates"]["complete_multimodal"] == 1
    assert statistics["candidates"]["audio_only"] == 1
    # Never summed into one "has features" figure.
    assert statistics["candidates"]["processed"] == 2


def test_statistics_record_both_native_and_constructed_text_dimensions(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, spec = _assemble(
        paths, candidates, handcrafted, _write_audio(paths, candidates), _write_text(paths, candidates)
    )
    write_feature_manifest(paths.features_manifest, spec, result.records)
    header, records = read_feature_manifest(paths.features_manifest)
    statistics = compute_feature_statistics(paths, header, records, [_episode()])

    embeddings = statistics["embeddings"]
    assert embeddings["text_embedding_native_dimensions"] == [384]
    assert embeddings["constructed_text_vector_dimension"] == 1536
    assert embeddings["audio_embedding_dimensions"] == [384]


def test_statistics_report_the_missing_value_rate_per_feature(tmp_path: Path):
    paths = _paths(tmp_path)
    candidates = [_candidate(10_000)]
    handcrafted = _handcrafted(candidates)
    result, spec = _assemble(
        paths,
        candidates,
        handcrafted,
        _write_audio(paths, candidates),
        _write_text(paths, candidates, sides=("before",)),
    )
    write_feature_manifest(paths.features_manifest, spec, result.records)
    header, records = read_feature_manifest(paths.features_manifest)
    statistics = compute_feature_statistics(paths, header, records, [_episode()])
    assert statistics["features"]["missing_value_rate"]["context_cosine_similarity"] == 1.0
    assert statistics["features"]["missing_value_rate"]["alpha"] == 0.0
