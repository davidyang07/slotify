"""Whisper encoder frame/time alignment, pooling and the embedding store.

Every test here uses synthetic encoder states. No model is downloaded and no
network is touched -- which is the whole reason the alignment arithmetic lives
in a pure module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slotify_rank.config.feature_settings import WindowConfig
from slotify_rank.embeddings.pooling import (
    FrameGrid,
    derive_frame_duration_ms,
    frame_span_for_window,
    mean_pool,
    pool_across_chunks,
    valid_frame_count,
)
from slotify_rank.embeddings.store import (
    EmbeddingMetadata,
    StaleEmbeddingCache,
    load_embeddings,
    read_embeddings,
    select_rows,
    sidecar_path,
    write_embeddings,
)
from slotify_rank.features.windows import build_windows
from slotify_rank.pipeline.identity import CacheIdentity

#: whisper-tiny.en: 384-dimensional encoder states, 1500 frames per 30 s window.
TINY_EN_DIMENSION = 384
FRAMES_PER_CHUNK = 1_500
CHUNK_MS = 30_000
FRAME_MS = 20.0


def _states(frames: int = FRAMES_PER_CHUNK, dim: int = TINY_EN_DIMENSION):
    """Encoder states where frame i is the constant vector i.

    Makes a pooled mean directly readable as the mean frame index, so a
    misalignment shows up as an arithmetic error rather than as noise.
    """
    return np.tile(
        np.arange(frames, dtype=np.float32).reshape(frames, 1), (1, dim)
    )


def _grid(start_ms: int = 0, end_ms: int = CHUNK_MS, frames: int = FRAMES_PER_CHUNK):
    return FrameGrid(
        chunk_start_ms=start_ms,
        chunk_end_ms=end_ms,
        frame_duration_ms=FRAME_MS,
        total_frames=frames,
    )


# ---------------------------------------------------------------------------
# Temporal resolution
# ---------------------------------------------------------------------------


def test_tiny_en_temporal_resolution_is_twenty_milliseconds():
    """160/16000 s per mel frame, x2 for the encoder's stride-2 conv."""
    assert (
        derive_frame_duration_ms(
            hop_length=160,
            sampling_rate=16_000,
            chunk_ms=CHUNK_MS,
            encoder_frames=FRAMES_PER_CHUNK,
        )
        == 20.0
    )


def test_a_disagreeing_frame_count_is_refused_rather_than_guessed():
    """Preprocessing and observed shape must agree, or alignment is unknown."""
    with pytest.raises(ValueError, match="temporal resolution is ambiguous"):
        derive_frame_duration_ms(
            hop_length=160,
            sampling_rate=16_000,
            chunk_ms=CHUNK_MS,
            encoder_frames=150,  # would imply 200 ms per frame
        )


def test_the_undocumented_two_hundred_millisecond_assumption_is_rejected():
    """A 0.2 s resolution is wrong by 10x and must not silently pass."""
    with pytest.raises(ValueError, match="ambiguous"):
        derive_frame_duration_ms(160, 16_000, CHUNK_MS, encoder_frames=150)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_padding_frames_are_excluded_from_the_valid_region():
    """Whisper emits 1500 frames whether it saw 30 s or 2 s of audio."""
    assert valid_frame_count(2_000, FRAME_MS, FRAMES_PER_CHUNK) == 100
    assert valid_frame_count(CHUNK_MS, FRAME_MS, FRAMES_PER_CHUNK) == FRAMES_PER_CHUNK


def test_the_frame_straddling_the_end_of_audio_is_kept():
    """Dropping it would lose the final moments of every episode."""
    assert valid_frame_count(2_010, FRAME_MS, FRAMES_PER_CHUNK) == 101


def test_valid_frames_never_exceed_the_encoder_output():
    assert valid_frame_count(60_000, FRAME_MS, FRAMES_PER_CHUNK) == FRAMES_PER_CHUNK


def test_pooling_a_short_chunk_ignores_the_padding():
    """A 2 s chunk padded to 30 s must pool only the first 100 frames."""
    grid = _grid(start_ms=0, end_ms=2_000)
    pooled = pool_across_chunks([(grid, _states())], 0, 2_000)
    assert pooled is not None
    # Mean of frame indices 0..99 is 49.5, not the mean over all 1500 frames.
    assert pooled[0] == pytest.approx(49.5)


# ---------------------------------------------------------------------------
# Frame/time mapping
# ---------------------------------------------------------------------------


def test_a_window_maps_to_the_expected_frames():
    grid = _grid()
    assert frame_span_for_window(grid, 0, 1_000) == (0, 50)
    assert frame_span_for_window(grid, 1_000, 2_000) == (50, 100)


def test_chunk_offset_is_applied_to_the_frame_mapping():
    """A chunk starting at 25 s must map episode time, not chunk time."""
    grid = _grid(start_ms=25_000, end_ms=55_000)
    assert frame_span_for_window(grid, 25_000, 26_000) == (0, 50)
    assert frame_span_for_window(grid, 26_000, 27_000) == (50, 100)


def test_a_window_outside_the_chunk_contributes_nothing():
    grid = _grid(start_ms=25_000, end_ms=55_000)
    assert frame_span_for_window(grid, 0, 1_000) == (0, 0)


def test_a_window_partially_overlapping_a_chunk_is_clipped():
    grid = _grid(start_ms=25_000, end_ms=55_000)
    first, last = frame_span_for_window(grid, 24_000, 26_000)
    assert (first, last) == (0, 50)


def test_pooling_across_two_chunks_averages_over_every_frame():
    """Not a mean of per-chunk means -- the chunks contribute unequally."""
    left = (_grid(0, CHUNK_MS), _states())
    right = (_grid(25_000, 55_000), _states())
    pooled = pool_across_chunks([left, right], 24_000, 26_000)
    assert pooled is not None
    # Chunk 0 contributes frames 1200..1299 (values 1200..1299).
    # Chunk 1 contributes frames 0..49 (values 0..49).
    expected = (sum(range(1_200, 1_300)) + sum(range(0, 50))) / 150
    assert pooled[0] == pytest.approx(expected)


def test_an_empty_span_pools_to_none_not_to_zeros():
    """Zeros would be indistinguishable from real silence."""
    assert mean_pool(_states(), 5, 5) is None
    assert pool_across_chunks([(_grid(), _states())], 40_000, 41_000) is None


# ---------------------------------------------------------------------------
# Candidate windows
# ---------------------------------------------------------------------------


def test_candidate_windows_pool_to_the_expected_dimension():
    grid = _grid()
    windows = build_windows(10_000, 30_000, WindowConfig().as_pairs())
    before = windows.before["medium"]
    pooled = pool_across_chunks([(grid, _states())], before.start_ms, before.end_ms)
    assert pooled is not None
    assert pooled.shape == (TINY_EN_DIMENSION,)


def test_a_boundary_candidate_pools_from_the_available_side_only():
    grid = _grid(0, 30_000)
    windows = build_windows(0, 30_000, WindowConfig().as_pairs())
    before = windows.before["medium"]
    after = windows.after["medium"]
    assert pool_across_chunks([(grid, _states())], before.start_ms, before.end_ms) is None
    assert pool_across_chunks([(grid, _states())], after.start_ms, after.end_ms) is not None


def test_pooling_rejects_non_finite_encoder_output():
    states = _states().copy()
    states[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        pool_across_chunks([(_grid(), states)], 0, 1_000)


# ---------------------------------------------------------------------------
# Embedding store
# ---------------------------------------------------------------------------


def _metadata(digest: str = "d" * 64, dimension: int = TINY_EN_DIMENSION, rows=("a", "b")):
    return EmbeddingMetadata(
        kind="audio_embedding",
        episode_id="ep_0123456789ab",
        model_id="openai/whisper-tiny.en",
        model_revision="main",
        dimension=dimension,
        row_ids=tuple(rows),
        identity_digest=digest,
    )


def test_arrays_round_trip_with_their_metadata(tmp_path: Path):
    array = np.random.default_rng(0).normal(size=(2, TINY_EN_DIMENSION)).astype(np.float32)
    path = tmp_path / "ep.audio.npy"
    write_embeddings(path, array, _metadata())
    loaded, metadata = read_embeddings(path)
    assert loaded.shape == (2, TINY_EN_DIMENSION)
    assert metadata.dimension == TINY_EN_DIMENSION
    assert np.allclose(loaded, array)


def test_tiny_en_embeddings_are_384_dimensional(tmp_path: Path):
    array = np.zeros((1, TINY_EN_DIMENSION), dtype=np.float32)
    path = tmp_path / "ep.audio.npy"
    write_embeddings(path, array, _metadata(rows=("a",)))
    _, metadata = read_embeddings(path)
    assert metadata.dimension == 384


def test_arrays_are_never_pickled(tmp_path: Path):
    """allow_pickle=False must be sufficient to read what we write."""
    path = tmp_path / "ep.audio.npy"
    write_embeddings(path, np.zeros((1, 4), dtype=np.float32), _metadata(dimension=4, rows=("a",)))
    assert np.load(path, allow_pickle=False).shape == (1, 4)


def test_a_row_count_mismatch_is_refused_at_write(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot be attributed"):
        write_embeddings(
            tmp_path / "ep.npy",
            np.zeros((3, TINY_EN_DIMENSION), dtype=np.float32),
            _metadata(rows=("a", "b")),
        )


def test_a_dimension_mismatch_is_refused_at_write(tmp_path: Path):
    with pytest.raises(ValueError, match="does not match the declared dimension"):
        write_embeddings(
            tmp_path / "ep.npy",
            np.zeros((2, 128), dtype=np.float32),
            _metadata(),
        )


def test_non_finite_embeddings_are_never_cached(tmp_path: Path):
    array = np.zeros((2, TINY_EN_DIMENSION), dtype=np.float32)
    array[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinity"):
        write_embeddings(tmp_path / "ep.npy", array, _metadata())


def test_a_missing_sidecar_makes_the_array_unusable(tmp_path: Path):
    path = tmp_path / "ep.audio.npy"
    write_embeddings(path, np.zeros((2, TINY_EN_DIMENSION), dtype=np.float32), _metadata())
    sidecar_path(path).unlink()
    with pytest.raises(StaleEmbeddingCache, match="no metadata sidecar"):
        read_embeddings(path)
    assert load_embeddings(path) is None


def test_a_stale_model_revision_invalidates_the_cache(tmp_path: Path):
    identity = CacheIdentity("audio_embedding", {"revision": "main"})
    other = CacheIdentity("audio_embedding", {"revision": "abc123"})
    path = tmp_path / "ep.audio.npy"
    write_embeddings(
        path,
        np.zeros((2, TINY_EN_DIMENSION), dtype=np.float32),
        _metadata(digest=identity.digest),
    )
    assert load_embeddings(path, identity) is not None
    assert load_embeddings(path, other) is None


def test_an_unexpected_dimension_invalidates_the_cache(tmp_path: Path):
    path = tmp_path / "ep.audio.npy"
    write_embeddings(
        path, np.zeros((2, 128), dtype=np.float32), _metadata(dimension=128)
    )
    assert load_embeddings(path, expected_dimension=128) is not None
    assert load_embeddings(path, expected_dimension=384) is None


def test_an_unsupported_store_version_is_rejected(tmp_path: Path):
    import json

    path = tmp_path / "ep.audio.npy"
    write_embeddings(path, np.zeros((2, TINY_EN_DIMENSION), dtype=np.float32), _metadata())
    sidecar = sidecar_path(path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["store_version"] = "embedding-store-v99"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StaleEmbeddingCache, match="Unsupported embedding store_version"):
        read_embeddings(path)


def test_rows_are_selected_by_id_not_by_position(tmp_path: Path):
    array = np.array([[1.0] * 4, [2.0] * 4], dtype=np.float32)
    path = tmp_path / "ep.npy"
    write_embeddings(path, array, _metadata(dimension=4, rows=("first", "second")))
    loaded, metadata = read_embeddings(path)
    assert select_rows(loaded, metadata, ["second"])[0][0] == pytest.approx(2.0)


def test_an_unknown_row_id_raises(tmp_path: Path):
    array = np.zeros((2, 4), dtype=np.float32)
    path = tmp_path / "ep.npy"
    write_embeddings(path, array, _metadata(dimension=4, rows=("a", "b")))
    loaded, metadata = read_embeddings(path)
    with pytest.raises(KeyError, match="no embedding row"):
        select_rows(loaded, metadata, ["missing"])


def test_duplicate_row_ids_are_rejected():
    from slotify_rank.embeddings.store import row_lookup

    with pytest.raises(ValueError, match="duplicate row id"):
        row_lookup(_metadata(rows=("a", "a")))


def test_an_absent_array_is_a_miss_not_an_error(tmp_path: Path):
    assert load_embeddings(tmp_path / "absent.npy") is None


def test_a_truncated_array_is_not_treated_as_complete(tmp_path: Path):
    """The atomic write is what prevents this; assert the detection too."""
    path = tmp_path / "ep.audio.npy"
    write_embeddings(path, np.zeros((2, TINY_EN_DIMENSION), dtype=np.float32), _metadata())
    # Simulate a partially written array by rewriting fewer rows underneath the
    # sidecar, which still claims two.
    np.save(path, np.zeros((1, TINY_EN_DIMENSION), dtype=np.float32), allow_pickle=False)
    with pytest.raises(StaleEmbeddingCache, match="row ids"):
        read_embeddings(path)
