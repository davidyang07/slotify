"""Shared builders for dataset tests.

Synthetic manifest records rather than real audio wherever the audio itself is
not what is under test. The repository's real fixtures are ~20 s each and belong
to four series, which is too small to exercise series-aware splitting honestly --
so splitting is tested against fabricated *manifests*, never by pretending the
real clips are something they are not.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord, make_episode_id

__all__ = [
    "make_episode",
    "make_candidate",
    "write_wav",
    "write_speech_like_wav",
]

_ZERO_SHA = "0" * 64


def make_episode(
    title: str = "Test episode",
    series_id: str = "test-series",
    sha_seed: str = "a",
    duration_ms: int = 600_000,
    content_type: str = "podcast",
    status: str = "normalized",
    **overrides,
) -> EpisodeRecord:
    digest = (sha_seed * 64)[:64]
    episode_id = overrides.pop("episode_id", make_episode_id(title, digest))
    payload = {
        "episode_id": episode_id,
        "series_id": series_id,
        "title": title,
        "source_type": "local_file",
        "source_uri": f"file:///audio/{episode_id}.mp3",
        "source_name": "Test source",
        "license_name": None,
        "license_url": None,
        "attribution": None,
        "language": "en",
        "content_type": content_type,
        "original_path": f"data/raw/{episode_id}.mp3",
        "sha256": digest,
        "status": status,
        "duration_ms": duration_ms,
        "sample_rate_hz": 44100,
        "channels": 2,
        "file_format": "mp3",
        "is_target_domain": content_type
        in ("podcast", "interview", "conversational", "narrated"),
    }
    if status == "normalized":
        payload["normalized_path"] = f"data/normalized/{episode_id}.wav"
        payload["normalized_sha256"] = digest
        payload["normalized_duration_ms"] = duration_ms
        payload["preprocessing_version"] = "preprocess-v1.0.0"
    payload.update(overrides)
    return EpisodeRecord(**payload)


def make_candidate(
    episode_id: str = "ep-test",
    timestamp_ms: int = 60_000,
    sources: tuple[str, ...] = ("silence",),
    **overrides,
) -> DatasetCandidate:
    payload = {
        "episode_id": episode_id,
        "candidate_id": f"{episode_id}:{timestamp_ms:09d}",
        "timestamp_ms": timestamp_ms,
        "candidate_sources": sources,
        "pause_duration_ms": 900,
        "silence_duration_ms": 900,
        "heuristic_score": 0.58,
        "baseline_version": "heuristic_offline_v1",
        "config_version": "heuristic-config-v1.0.0",
    }
    payload.update(overrides)
    return DatasetCandidate(**payload)


def write_wav(
    path: Path, samples: np.ndarray, sample_rate: int = 16_000, channels: int = 1
) -> Path:
    """Write int16 mono PCM, the format `normalize` produces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())
    return path


def write_speech_like_wav(
    path: Path,
    segments: list[tuple[int, bool]],
    sample_rate: int = 16_000,
    amplitude: int = 8000,
) -> Path:
    """Build a WAV from ``(duration_ms, is_loud)`` segments.

    Deterministic and dependency-free: loud segments are a fixed-frequency tone,
    quiet segments are true digital silence. That gives silence detection
    unambiguous boundaries to find, so a test can assert exact timestamps.
    """
    chunks: list[np.ndarray] = []
    for duration_ms, is_loud in segments:
        count = int(sample_rate * duration_ms / 1000)
        if is_loud:
            time = np.arange(count) / sample_rate
            chunks.append(amplitude * np.sin(2 * np.pi * 220.0 * time))
        else:
            chunks.append(np.zeros(count))
    return write_wav(path, np.concatenate(chunks) if chunks else np.zeros(0), sample_rate)
