"""Normalize source audio into a single ML format: 16 kHz, mono, PCM s16 WAV.

One format for everything downstream. 16 kHz is what speech models expect, mono
removes a channel-layout variable the ranker has no use for, and uncompressed
PCM means every later read is exact and cheap.

**This never touches the product path.** The Node ``/api/merge`` route and the
``ad_inserter`` pipeline keep operating on the user's original upload at its own
sample rate; these renders live under ``data/normalized/`` and exist only for
dataset work.

Caching is keyed on ``(source sha256, preprocessing version)``, both recorded on
the episode. An unchanged file is never re-rendered, and bumping
``PREPROCESSING_VERSION`` invalidates every cached render without anyone having
to remember to delete a directory.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from slotify_rank.config.versions import PREPROCESSING_VERSION
from slotify_rank.data.checksum import atomic_replace, sha256_file
from slotify_rank.data.ffmpeg import ffmpeg_path, run_tool
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.probe import UnreadableAudio, probe_audio
from slotify_rank.data.schema import EpisodeRecord

__all__ = [
    "TARGET_SAMPLE_RATE_HZ",
    "TARGET_CHANNELS",
    "TARGET_FORMAT",
    "NormalizeResult",
    "normalize_episode",
    "needs_normalization",
]

TARGET_SAMPLE_RATE_HZ = 16_000
TARGET_CHANNELS = 1
TARGET_FORMAT = "wav"
_TARGET_CODEC = "pcm_s16le"


@dataclass(frozen=True)
class NormalizeResult:
    episode: EpisodeRecord
    rendered: bool

    @property
    def action(self) -> str:
        return "rendered" if self.rendered else "cached"


def needs_normalization(episode: EpisodeRecord, paths: DataPaths) -> bool:
    """True when the cached render is missing, stale, or from another version."""
    if episode.normalized_path is None or episode.normalized_sha256 is None:
        return True
    if episode.preprocessing_version != PREPROCESSING_VERSION:
        return True
    target = paths.absolute(episode.normalized_path)
    if not target.is_file() or target.stat().st_size == 0:
        return True
    return sha256_file(target) != episode.normalized_sha256


def normalize_episode(
    episode: EpisodeRecord, paths: DataPaths, force: bool = False
) -> NormalizeResult:
    """Render ``episode`` to the canonical ML format, reusing a valid cache.

    The episode is returned with ``normalized_path``, ``normalized_sha256``,
    ``preprocessing_version`` and the *measured* metadata of the render filled
    in, and ``status`` advanced to ``normalized``.
    """
    source = paths.absolute(episode.original_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"{episode.episode_id}: original audio missing at {source}. Re-run "
            "`dataset import-local` or `dataset fetch`."
        )
    if source.stat().st_size == 0:
        raise UnreadableAudio(f"{episode.episode_id}: original audio is zero-length")

    actual_source_digest = sha256_file(source)
    if actual_source_digest != episode.sha256:
        raise ValueError(
            f"{episode.episode_id}: original audio at {source} no longer matches its "
            f"recorded SHA-256 (expected {episode.sha256}, found "
            f"{actual_source_digest}). The manifest and the disk disagree; re-import "
            "deliberately rather than normalizing unknown bytes."
        )

    destination = paths.normalized_dir / f"{episode.episode_id}.wav"
    if not force and not needs_normalization(episode, paths) and destination.is_file():
        return NormalizeResult(episode=episode, rendered=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp.wav"
    )
    temp_path = Path(temp_name)
    os.close(handle)
    try:
        run_tool(
            ffmpeg_path(),
            [
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-map",
                "0:a:0",
                "-ac",
                str(TARGET_CHANNELS),
                "-ar",
                str(TARGET_SAMPLE_RATE_HZ),
                "-acodec",
                _TARGET_CODEC,
                "-f",
                TARGET_FORMAT,
                str(temp_path),
            ],
        )
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise UnreadableAudio(
                f"{episode.episode_id}: ffmpeg produced an empty file from {source}"
            )
        metadata = probe_audio(temp_path)
        if (
            metadata.sample_rate_hz != TARGET_SAMPLE_RATE_HZ
            or metadata.channels != TARGET_CHANNELS
        ):
            raise UnreadableAudio(
                f"{episode.episode_id}: normalized render is "
                f"{metadata.sample_rate_hz} Hz / {metadata.channels}ch, expected "
                f"{TARGET_SAMPLE_RATE_HZ} Hz / {TARGET_CHANNELS}ch"
            )
        atomic_replace(temp_path, destination)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    # duration_ms / sample_rate_hz / channels / file_format describe the ORIGINAL
    # audio and are owned by `probe`. The render's own rate, channel count and
    # format are fixed by construction (and asserted above), so re-recording them
    # here would overwrite the source's real properties with three constants.
    # Duration is the one measurement the render can contradict, so it is checked
    # rather than copied.
    if episode.duration_ms is not None:
        drift_ms = abs(metadata.duration_ms - episode.duration_ms)
        tolerance_ms = max(500, int(episode.duration_ms * 0.01))
        if drift_ms > tolerance_ms:
            raise UnreadableAudio(
                f"{episode.episode_id}: normalized render is {metadata.duration_ms} ms "
                f"but the source probed at {episode.duration_ms} ms "
                f"(drift {drift_ms} ms > tolerance {tolerance_ms} ms). The decode is "
                "not faithful; investigate before using this episode."
            )

    updated = episode.replace(
        normalized_path=paths.relative(destination),
        normalized_sha256=sha256_file(destination),
        normalized_duration_ms=metadata.duration_ms,
        preprocessing_version=PREPROCESSING_VERSION,
        duration_ms=episode.duration_ms or metadata.duration_ms,
        status="normalized",
    )
    return NormalizeResult(episode=updated, rendered=True)
