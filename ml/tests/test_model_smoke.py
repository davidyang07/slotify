"""Opt-in verification against the real local models.

Everything else in this suite is network-free and model-free. These tests are
the exception: they download (once, into the Hugging Face cache) and run the
actual ``whisper-tiny.en`` and ``all-MiniLM-L6-v2`` weights on one short
repository fixture.

They are excluded from the default run. To execute them::

    .venv/Scripts/python.exe -m pytest -m model_smoke

Their purpose is narrow and specific: to prove that the dimensions, the
temporal resolution and the cache behaviour asserted by the mocked tests are
the ones the real models actually produce. A mocked 384 is a claim; this is the
evidence.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from slotify_rank.config.feature_settings import (
    AudioEmbeddingConfig,
    TextEmbeddingConfig,
    TranscriptionConfig,
    WindowConfig,
)
from slotify_rank.config.settings import find_repo_root
from slotify_rank.data.audio_io import samples_duration_ms
from slotify_rank.embeddings.minilm_text import (
    build_constructed_vector,
    cosine_similarity,
)
from slotify_rank.embeddings.pooling import pool_across_chunks

pytestmark = [pytest.mark.model_smoke, pytest.mark.slow]

#: The shortest spoken-word fixture in the repository (~19 s).
FIXTURE = (
    find_repo_root()
    / "data"
    / "normalized"
    / "fixture-podcast-style-monologue_eefa0528f230.wav"
)

TINY_EN_DIMENSION = 384
MINILM_DIMENSION = 384


needs_fixture = pytest.mark.skipif(
    not FIXTURE.is_file(),
    reason=(
        "Normalized smoke fixture is absent; run `dataset import-local`, "
        "`dataset probe` and `dataset normalize` first."
    ),
)


@pytest.fixture(scope="module")
def audio():
    from slotify_rank.transcription.local_whisper import load_audio_samples

    return load_audio_samples(FIXTURE)


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------


@needs_fixture
def test_real_whisper_produces_a_timestamped_transcript(audio, capsys):
    from slotify_rank.transcription.local_whisper import WhisperTranscriber

    samples, sample_rate = audio
    transcriber = WhisperTranscriber(config=TranscriptionConfig()).load()
    started = time.perf_counter()
    try:
        transcript = transcriber.transcribe(
            episode_id="smoke_0123456789ab",
            audio=samples,
            audio_sha256="a" * 64,
            identity_digest="d" * 64,
            sample_rate=sample_rate,
        )
    finally:
        transcriber.release()
    elapsed = time.perf_counter() - started

    assert transcript.segment_count > 0
    assert transcript.word_count > 0
    # The shared definition, not a local one: rounding and truncation differ by
    # a millisecond, which is exactly the bug this helper exists to prevent.
    duration_ms = samples_duration_ms(samples.size, sample_rate)
    for segment in transcript.segments:
        assert 0 <= segment.start_ms <= segment.end_ms <= duration_ms
    with capsys.disabled():
        print(
            f"\n  whisper-tiny.en: {transcript.segment_count} segments, "
            f"{transcript.word_count} words, {elapsed:.1f}s for "
            f"{duration_ms / 1000:.1f}s of audio "
            f"({duration_ms / 1000 / elapsed:.2f}x real time)"
        )


@needs_fixture
def test_real_whisper_encoder_is_384_dimensional(audio, capsys):
    """The headline dimension claim, against the real weights."""
    from slotify_rank.embeddings.whisper_audio import WhisperAudioEncoder

    samples, sample_rate = audio
    encoder = WhisperAudioEncoder(config=AudioEmbeddingConfig()).load()
    started = time.perf_counter()
    try:
        assert encoder.dimension == TINY_EN_DIMENSION
        # Explicitly not 768 -- that is whisper-base's width.
        assert encoder.dimension != 768
        states = encoder.encode_episode("smoke_0123456789ab", samples, sample_rate)
    finally:
        encoder.release()
    elapsed = time.perf_counter() - started

    assert states.dimension == TINY_EN_DIMENSION
    for _, array in states.chunks:
        assert array.shape[1] == TINY_EN_DIMENSION
        assert np.isfinite(array).all()
    with capsys.disabled():
        print(
            f"\n  whisper encoder: dimension {states.dimension}, "
            f"{states.frame_duration_ms} ms/frame, {elapsed:.1f}s"
        )


@needs_fixture
def test_real_encoder_temporal_resolution_is_twenty_milliseconds(audio):
    """Derived from the real feature extractor and the real output shape."""
    from slotify_rank.embeddings.whisper_audio import WhisperAudioEncoder

    samples, sample_rate = audio
    encoder = WhisperAudioEncoder(config=AudioEmbeddingConfig()).load()
    try:
        states = encoder.encode_episode("smoke_0123456789ab", samples, sample_rate)
    finally:
        encoder.release()
    assert states.frame_duration_ms == pytest.approx(20.0)
    # 30 s window at 20 ms per frame.
    assert states.chunks[0][1].shape[0] == 1_500


@needs_fixture
def test_real_pooled_candidate_embeddings_are_384_dimensional(audio):
    from slotify_rank.embeddings.whisper_audio import (
        CANDIDATE_EMBEDDING_KINDS,
        WhisperAudioEncoder,
        pool_candidate_embeddings,
    )
    from slotify_rank.features.windows import build_windows

    samples, sample_rate = audio
    duration_ms = int(samples.size * 1000 / sample_rate)
    encoder = WhisperAudioEncoder(config=AudioEmbeddingConfig()).load()
    try:
        states = encoder.encode_episode("smoke_0123456789ab", samples, sample_rate)
    finally:
        encoder.release()

    windows = build_windows(duration_ms // 2, duration_ms, WindowConfig().as_pairs())
    pooled = pool_candidate_embeddings(states, windows, include_difference=True)
    for kind in CANDIDATE_EMBEDDING_KINDS:
        assert pooled[kind] is not None, kind
        assert pooled[kind].shape == (TINY_EN_DIMENSION,)
        assert np.isfinite(pooled[kind]).all()


# ---------------------------------------------------------------------------
# MiniLM
# ---------------------------------------------------------------------------


def test_real_minilm_native_dimension_is_384(capsys):
    from slotify_rank.embeddings.minilm_text import MiniLmTextEncoder

    encoder = MiniLmTextEncoder(config=TextEmbeddingConfig()).load()
    started = time.perf_counter()
    try:
        assert encoder.dimension == MINILM_DIMENSION
        vectors = encoder.encode(
            [
                "So that wraps up the first half of today's discussion.",
                "Now let's move on to something completely different.",
            ]
        )
    finally:
        encoder.release()
    elapsed = time.perf_counter() - started

    assert vectors.shape == (2, MINILM_DIMENSION)
    assert np.isfinite(vectors).all()
    with capsys.disabled():
        print(f"\n  MiniLM: dimension {vectors.shape[1]}, {elapsed:.1f}s for 2 texts")


def test_real_constructed_transcript_vector_is_1536_dimensional(capsys):
    """1536 = 384 x 4 blocks. Arithmetic, not a model output width."""
    from slotify_rank.embeddings.minilm_text import MiniLmTextEncoder

    encoder = MiniLmTextEncoder(config=TextEmbeddingConfig()).load()
    try:
        vectors = encoder.encode(
            [
                "And that concludes our look at the first topic.",
                "Right, so let's talk about something entirely unrelated.",
            ]
        )
    finally:
        encoder.release()

    constructed = build_constructed_vector(vectors[0], vectors[1])
    assert constructed.shape == (1536,)
    assert constructed.shape[0] == vectors.shape[1] * 4
    with capsys.disabled():
        print(
            f"\n  constructed transcript vector: {vectors.shape[1]} x 4 = "
            f"{constructed.shape[0]}"
        )


def test_real_embeddings_are_normalized_and_similarity_behaves(capsys):
    from slotify_rank.embeddings.minilm_text import MiniLmTextEncoder

    encoder = MiniLmTextEncoder(config=TextEmbeddingConfig()).load()
    try:
        vectors = encoder.encode(
            [
                "The cat sat quietly on the warm windowsill.",
                "A cat was resting on the sunny window ledge.",
                "Quarterly revenue exceeded analyst expectations.",
            ]
        )
    finally:
        encoder.release()

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    related = cosine_similarity(vectors[0], vectors[1])
    unrelated = cosine_similarity(vectors[0], vectors[2])
    assert related > unrelated
    with capsys.disabled():
        print(
            f"\n  cosine similarity: related {related:.3f}, unrelated {unrelated:.3f}"
        )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@needs_fixture
def test_the_real_pipeline_assembles_a_complete_multimodal_candidate(capsys):
    """The acceptance criterion, end to end, with real weights."""
    from slotify_rank.data import manifests
    from slotify_rank.data.paths import DataPaths
    from slotify_rank.features.assemble import read_feature_manifest

    paths = DataPaths()
    if not paths.features_manifest.is_file():
        pytest.skip(
            "No feature manifest; run `pipeline features` against the smoke corpus."
        )

    header, records = read_feature_manifest(paths.features_manifest)
    complete = [r for r in records if r.feature_status == "complete"]
    assert complete, "no complete multimodal record was assembled"

    record = complete[0]
    assert record.audio_embedding_dimension == TINY_EN_DIMENSION
    assert record.text_embedding_dimension == MINILM_DIMENSION
    assert len(record.handcrafted_feature_values) == header["handcrafted_feature_count"]
    assert np.isfinite(record.handcrafted_feature_values).all()
    with capsys.disabled():
        print(
            f"\n  assembled {len(records)} record(s), {len(complete)} complete; "
            f"{header['handcrafted_feature_count']} handcrafted features, "
            f"audio {record.audio_embedding_dimension}, "
            f"text {record.text_embedding_dimension} "
            f"(constructed {record.text_embedding_dimension * 4})"
        )
