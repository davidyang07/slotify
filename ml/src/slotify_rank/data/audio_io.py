"""Reading normalized WAV audio and reducing it to millisecond energy.

Everything downstream of this module works on a 1 ms energy envelope rather than
raw samples: silence detection, energy minima and the labelling UI's context
windows all need loudness over time, not waveforms. Computing the envelope once,
with :mod:`numpy`, keeps a 50-hour corpus tractable in pure Python.

Loudness is expressed in **dBFS with pydub's convention** -- ``20*log10(rms /
2**(8*sample_width - 1))`` -- because the canonical
``heuristic_offline_v1.silence_detection`` thresholds in
``config/heuristic_offline_v1.json`` are expressed against that scale by the
product's pydub-based analyzer. The arithmetic here is an independent
reimplementation with the same semantics, not a bit-exact port of
``pydub.silence.detect_silence``; Phase 1 parity covers the *scorer*, which is
the part that must match the product exactly.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AudioEnvelope",
    "load_normalized_wav",
    "samples_duration_ms",
    "NEGATIVE_INFINITY_DBFS",
]


def samples_duration_ms(sample_count: int, sample_rate_hz: int) -> int:
    """Duration of a sample buffer in whole milliseconds.

    The single definition, used by every stage. Rounding and truncation differ
    by one millisecond on most real files, and when two stages disagree the
    result is a transcript segment that ends one millisecond after the episode
    another stage believes exists -- which fails a bounds check with no
    plausible cause. Rounding is chosen because it is the nearer answer.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    return int(round(sample_count * 1000 / sample_rate_hz))

#: Stand-in for ``-inf`` dBFS (digital silence). Kept finite so the value can be
#: JSON-serialized and compared without special-casing every consumer.
NEGATIVE_INFINITY_DBFS = -120.0


@dataclass(frozen=True)
class AudioEnvelope:
    """Per-millisecond mean-square energy of a mono PCM file.

    ``mean_square`` has one entry per whole millisecond of audio. Everything
    else is derived from it, so a given file always yields identical candidates.
    """

    sample_rate_hz: int
    sample_width_bytes: int
    duration_ms: int
    mean_square: np.ndarray  # float64, shape (n_ms,)

    @property
    def full_scale(self) -> float:
        return float(1 << (8 * self.sample_width_bytes - 1))

    @property
    def n_ms(self) -> int:
        return int(self.mean_square.shape[0])

    def to_dbfs(self, mean_square: np.ndarray | float) -> np.ndarray | float:
        """Convert mean-square energy to dBFS, flooring digital silence."""
        values = np.asarray(mean_square, dtype=np.float64)
        with np.errstate(divide="ignore"):
            dbfs = 10.0 * np.log10(values / (self.full_scale**2))
        dbfs = np.where(np.isfinite(dbfs), dbfs, NEGATIVE_INFINITY_DBFS)
        dbfs = np.maximum(dbfs, NEGATIVE_INFINITY_DBFS)
        return dbfs if np.ndim(mean_square) else float(dbfs)

    @property
    def overall_dbfs(self) -> float:
        """Whole-file dBFS, the reference for the relative silence threshold."""
        if self.n_ms == 0:
            return NEGATIVE_INFINITY_DBFS
        return float(self.to_dbfs(float(self.mean_square.mean())))

    def window_mean_square(self, window_ms: int) -> np.ndarray:
        """Mean-square energy of every ``window_ms`` window at a 1 ms hop.

        Returns an array of length ``n_ms - window_ms + 1`` (empty when the file
        is shorter than the window).
        """
        if window_ms <= 0:
            raise ValueError(f"window_ms must be positive, got {window_ms}")
        if self.n_ms < window_ms:
            return np.empty(0, dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(self.mean_square)))
        sums = cumulative[window_ms:] - cumulative[:-window_ms]
        return sums / float(window_ms)

    def window_dbfs(self, window_ms: int) -> np.ndarray:
        return np.asarray(self.to_dbfs(self.window_mean_square(window_ms)))

    def region_dbfs(self, start_ms: int, end_ms: int) -> float:
        """dBFS of ``[start_ms, end_ms)``, clipped to the file."""
        start = max(0, min(int(start_ms), self.n_ms))
        end = max(start, min(int(end_ms), self.n_ms))
        if end <= start:
            return NEGATIVE_INFINITY_DBFS
        return float(self.to_dbfs(float(self.mean_square[start:end].mean())))


def load_normalized_wav(path: Path | str) -> AudioEnvelope:
    """Load a mono PCM WAV and reduce it to a millisecond energy envelope.

    Only mono PCM is accepted: this reads ``data/normalized/``, which
    :mod:`slotify_rank.data.normalize` guarantees is 16 kHz mono ``pcm_s16le``.
    Rejecting anything else here means a mis-rendered file surfaces immediately
    rather than as a strange candidate distribution later.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Normalized audio not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Normalized audio is zero-length: {file_path}")

    with wave.open(str(file_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        n_frames = handle.getnframes()
        if channels != 1:
            raise ValueError(
                f"{file_path} has {channels} channels; normalized audio must be mono. "
                "Re-run `dataset normalize --force`."
            )
        if sample_width != 2:
            raise ValueError(
                f"{file_path} has {sample_width * 8}-bit samples; normalized audio "
                "must be 16-bit PCM. Re-run `dataset normalize --force`."
            )
        if n_frames == 0:
            raise ValueError(f"{file_path} contains no audio frames")
        raw = handle.readframes(n_frames)

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    samples_per_ms = sample_rate // 1000
    if samples_per_ms < 1:
        raise ValueError(
            f"{file_path} has a sample rate of {sample_rate} Hz, which is below the "
            "1 kHz needed for millisecond resolution"
        )
    usable_ms = samples.shape[0] // samples_per_ms
    if usable_ms == 0:
        raise ValueError(f"{file_path} is shorter than one millisecond")
    trimmed = samples[: usable_ms * samples_per_ms].reshape(usable_ms, samples_per_ms)
    mean_square = np.square(trimmed).mean(axis=1)

    return AudioEnvelope(
        sample_rate_hz=sample_rate,
        sample_width_bytes=sample_width,
        duration_ms=int(round(n_frames * 1000 / sample_rate)),
        mean_square=mean_square,
    )
