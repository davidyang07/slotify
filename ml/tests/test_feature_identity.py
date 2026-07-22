"""Cache identity, stage state and Phase 3 configuration loading.

Network-free and model-free, like every unit test in this suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slotify_rank.config.feature_settings import (
    AcousticConfig,
    AudioEmbeddingConfig,
    FeatureConfig,
    TranscriptContextConfig,
    TranscriptionConfig,
    WindowConfig,
    load_embedding_config,
    load_feature_config,
    load_pipeline_config,
    load_transcription_config,
)
from slotify_rank.config.settings import find_repo_root
from slotify_rank.pipeline.identity import (
    CacheIdentity,
    canonical_json,
    config_digest,
    library_versions,
)
from slotify_rank.pipeline.state import StageEntry, StageLedger

CONFIG_DIR = find_repo_root() / "ml" / "configs"


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------


def test_identity_is_insensitive_to_key_order():
    first = CacheIdentity("transcript", {"a": 1, "b": 2})
    second = CacheIdentity("transcript", {"b": 2, "a": 1})
    assert first.digest == second.digest


def test_identity_changes_when_any_input_changes():
    base = CacheIdentity("transcript", {"model": "tiny.en", "sha": "abc"})
    changed = CacheIdentity("transcript", {"model": "base.en", "sha": "abc"})
    assert base.digest != changed.digest


def test_identity_kind_participates_in_the_digest():
    """Two stages sharing every input must not share a cache key."""
    inputs = {"episode_id": "ep", "sha": "abc"}
    assert (
        CacheIdentity("audio_embedding", inputs).digest
        != CacheIdentity("text_embedding", inputs).digest
    )


def test_identity_rejects_unserializable_inputs_at_construction():
    with pytest.raises(TypeError, match="cannot take part in a cache identity"):
        CacheIdentity("transcript", {"path": object()})


def test_tuples_and_sets_digest_stably():
    """Set iteration order is not stable between interpreter runs."""
    assert canonical_json({"s": {"b", "a"}}) == canonical_json({"s": {"a", "b"}})
    assert canonical_json({"t": (1, 2)}) == canonical_json({"t": [1, 2]})


def test_missing_metadata_is_never_a_match():
    identity = CacheIdentity("transcript", {"a": 1})
    assert identity.matches(None) is False
    assert identity.matches({}) is False
    assert identity.matches({"digest": identity.digest}) is True


def test_explain_mismatch_names_the_changed_input():
    before = CacheIdentity("transcript", {"model": "tiny.en", "sha": "abc"})
    after = CacheIdentity("transcript", {"model": "base.en", "sha": "abc"})
    reasons = after.explain_mismatch(before.to_dict())
    assert any("model" in reason for reason in reasons)
    assert not any("sha" in reason for reason in reasons)


def test_explain_mismatch_reports_absent_metadata():
    assert CacheIdentity("transcript", {"a": 1}).explain_mismatch(None) == [
        "no cache metadata recorded"
    ]


def test_library_versions_records_absent_packages_distinctly():
    versions = library_versions("torch", "definitely-not-a-real-package")
    assert versions["definitely-not-a-real-package"] == "absent"
    assert versions["torch"] != "absent"


def test_short_digest_is_a_prefix_of_the_full_digest():
    identity = CacheIdentity("transcript", {"a": 1})
    assert identity.digest.startswith(identity.short)
    assert len(identity.short) == 16


# ---------------------------------------------------------------------------
# Stage ledger
# ---------------------------------------------------------------------------


def test_unknown_episode_is_missing():
    ledger = StageLedger("transcribe")
    assert ledger.get("nope").status == "missing"
    assert ledger.needs_work("nope", "digest") is True


def test_complete_entry_under_a_different_identity_reads_as_stale():
    ledger = StageLedger("transcribe")
    ledger.mark_complete("ep", "digest-a")
    assert ledger.status_for("ep", "digest-a") == "complete"
    assert ledger.status_for("ep", "digest-b") == "stale"
    assert ledger.needs_work("ep", "digest-b") is True


def test_failed_entries_are_skipped_unless_retry_is_requested():
    ledger = StageLedger("transcribe")
    ledger.mark_failed("ep", "decode error")
    assert ledger.needs_work("ep", "digest", retry_failed=False) is False
    assert ledger.needs_work("ep", "digest", retry_failed=True) is True


def test_partial_entries_are_reprocessed():
    """A process that died mid-stage left untrusted output."""
    ledger = StageLedger("transcribe")
    ledger.mark_partial("ep")
    assert ledger.status_for("ep", "digest") == "partial"
    assert ledger.needs_work("ep", "digest") is True


def test_complete_entry_requires_an_identity():
    with pytest.raises(ValueError, match="must record the cache identity"):
        StageEntry(episode_id="ep", status="complete")


def test_failed_entry_requires_a_reason():
    with pytest.raises(ValueError, match="must record why"):
        StageEntry(episode_id="ep", status="failed")


def test_ledger_round_trips_through_disk(tmp_path: Path):
    ledger = StageLedger("transcribe")
    ledger.mark_complete("ep-a", "digest-a", segments=12)
    ledger.mark_failed("ep-b", "no audio")
    path = tmp_path / "transcribe.json"
    ledger.write(path)

    reloaded = StageLedger.read(path, "transcribe")
    assert reloaded.status_for("ep-a", "digest-a") == "complete"
    assert reloaded.get("ep-a").details == {"segments": 12}
    assert reloaded.get("ep-b").failure_reason == "no audio"


def test_missing_ledger_reads_as_empty(tmp_path: Path):
    assert StageLedger.read(tmp_path / "absent.json", "transcribe").counts() == {
        "complete": 0,
        "failed": 0,
        "partial": 0,
        "stale": 0,
    }


def test_corrupt_ledger_raises_rather_than_silently_recomputing(tmp_path: Path):
    path = tmp_path / "transcribe.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        StageLedger.read(path, "transcribe")


def test_ledger_rejects_a_file_written_for_another_stage(tmp_path: Path):
    path = tmp_path / "ledger.json"
    StageLedger("transcribe").write(path)
    with pytest.raises(ValueError, match="ledger for stage"):
        StageLedger.read(path, "acoustic")


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown stage"):
        StageLedger("not-a-stage")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_repository_configs_load():
    transcription = load_transcription_config(CONFIG_DIR / "transcription_v1.yaml")
    assert transcription.model_id == "openai/whisper-tiny.en"
    assert transcription.chunk_ms == 30_000

    features = load_feature_config(CONFIG_DIR / "features_v1.yaml")
    assert features.windows.as_pairs() == (
        ("short", 1000),
        ("medium", 3000),
        ("context", 10000),
    )

    embeddings = load_embedding_config(CONFIG_DIR / "embeddings_v1.yaml")
    assert embeddings.text.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert embeddings.audio.chunk_ms == 30_000

    assert load_pipeline_config(CONFIG_DIR / "feature_pipeline_v1.yaml").workers >= 1


def test_repository_configs_match_the_built_in_defaults():
    """The YAML files must document the defaults, not diverge from them."""
    assert (
        load_transcription_config(CONFIG_DIR / "transcription_v1.yaml").digest
        == TranscriptionConfig().digest
    )
    assert (
        load_feature_config(CONFIG_DIR / "features_v1.yaml").digest
        == FeatureConfig().digest
    )


def test_unknown_setting_is_rejected(tmp_path: Path):
    path = tmp_path / "transcription.yaml"
    path.write_text("transcription:\n  nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown setting"):
        load_transcription_config(path)


def test_sampled_decoding_is_rejected():
    with pytest.raises(ValueError, match="not reproducible"):
        TranscriptionConfig(temperature=0.7)


def test_overlap_must_be_smaller_than_the_chunk():
    with pytest.raises(ValueError, match="strictly less than chunk_ms"):
        TranscriptionConfig(chunk_ms=30_000, overlap_ms=30_000)


def test_audio_chunk_length_is_pinned_to_whisper_receptive_field():
    with pytest.raises(ValueError, match="padding frames"):
        AudioEmbeddingConfig(chunk_ms=10_000)


def test_window_scales_must_be_ordered():
    with pytest.raises(ValueError, match="short_ms <= medium_ms <= context_ms"):
        WindowConfig(short_ms=5_000, medium_ms=1_000, context_ms=10_000)


def test_acoustic_n_fft_must_be_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        AcousticConfig(n_fft=500)


def test_transcript_context_gap_must_be_positive():
    with pytest.raises(ValueError, match="max_gap_ms must be positive"):
        TranscriptContextConfig(max_gap_ms=0)


def test_config_digest_changes_with_any_setting():
    assert TranscriptionConfig().digest != TranscriptionConfig(chunk_ms=20_000).digest
    assert FeatureConfig().digest != FeatureConfig(
        windows=WindowConfig(short_ms=500)
    ).digest


def test_pipeline_settings_are_not_part_of_any_cache_identity():
    """Worker counts change runtime, not bytes."""
    from slotify_rank.config.feature_settings import PipelineConfig

    assert PipelineConfig(workers=1).digest != PipelineConfig(workers=4).digest
    # ...but the pipeline config never feeds an identity; the feature and
    # transcription digests are what stages use, and they are unaffected.
    assert load_feature_config(CONFIG_DIR / "features_v1.yaml").digest == config_digest(
        FeatureConfig().to_dict()
    )
