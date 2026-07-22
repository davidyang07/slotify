"""Audio-driven candidate generators.

Four generators, each answering a different question about a moment in time:

``silence``
    Is there a product-length pause here? Uses the canonical
    ``heuristic_offline_v1`` thresholds, so what it finds is what the product
    would have found.
``pause``
    Is there a *short* pause here -- a breath, a beat -- that the product would
    reject? These are the informative near-misses.
``rms_minimum``
    Is this a locally quiet moment even though nothing is silent? Catches soft
    speech boundaries and music dips that pure silence detection misses.
``fixed_interval``
    A regular grid, ignoring the audio entirely. Most of these land mid-word,
    which is the point: they are the negative class.

All four return timestamps in integer milliseconds and are pure functions of the
:class:`~slotify_rank.data.audio_io.AudioEnvelope`, so the same file always
produces the same candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from slotify_rank.candidates.config import (
    FixedIntervalConfig,
    PauseConfig,
    RmsMinimumConfig,
)
from slotify_rank.data.audio_io import AudioEnvelope, NEGATIVE_INFINITY_DBFS

__all__ = [
    "RawCandidate",
    "detect_silence_ranges",
    "silence_candidates",
    "pause_candidates",
    "rms_minimum_candidates",
    "fixed_interval_candidates",
]


@dataclass(frozen=True)
class RawCandidate:
    """A proposed timestamp before merging, scoring or ID assignment."""

    ms: int
    source: str
    silence_ms: int = 0
    pause_ms: int = 0
    snippet: str = ""
    sentence_end: bool | None = None
    transcript_segment_id: str | None = None
    transcript_before: str | None = None
    transcript_after: str | None = None
    extra_sources: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.ms < 0:
            raise ValueError(f"RawCandidate.ms must be non-negative, got {self.ms}")

    @property
    def sources(self) -> tuple[str, ...]:
        return (self.source, *self.extra_sources)


def _silence_threshold_dbfs(
    envelope: AudioEnvelope, offset_db: float, fallback_dbfs: float
) -> float:
    """Relative threshold, matching the product's ``audio.dBFS + offset`` rule."""
    overall = envelope.overall_dbfs
    if overall <= NEGATIVE_INFINITY_DBFS:
        return fallback_dbfs
    return overall + offset_db


def detect_silence_ranges(
    envelope: AudioEnvelope,
    min_silence_len_ms: int,
    threshold_dbfs: float,
) -> list[tuple[int, int]]:
    """Half-open ``[start_ms, end_ms)`` ranges quieter than ``threshold_dbfs``.

    Mirrors the semantics of ``pydub.silence.detect_silence``: a 1 ms hop over
    windows of ``min_silence_len_ms``, keeping windows whose loudness is below
    the threshold, then joining overlapping windows into maximal ranges.
    """
    if min_silence_len_ms <= 0:
        raise ValueError("min_silence_len_ms must be positive")
    window_dbfs = envelope.window_dbfs(min_silence_len_ms)
    if window_dbfs.size == 0:
        return []
    quiet = np.flatnonzero(window_dbfs < threshold_dbfs)
    if quiet.size == 0:
        return []

    # Split the sorted window-start indices into runs of consecutive integers;
    # each run is one contiguous silent region.
    breaks = np.flatnonzero(np.diff(quiet) != 1)
    starts = np.concatenate(([quiet[0]], quiet[breaks + 1]))
    ends = np.concatenate((quiet[breaks], [quiet[-1]]))
    return [
        (int(start), int(end) + min_silence_len_ms) for start, end in zip(starts, ends)
    ]


def silence_candidates(
    envelope: AudioEnvelope,
    min_silence_len_ms: int,
    threshold_offset_db: float,
    fallback_threshold_dbfs: float,
) -> list[RawCandidate]:
    """Midpoint of every product-length silence, matching the canonical profile."""
    threshold = _silence_threshold_dbfs(
        envelope, threshold_offset_db, fallback_threshold_dbfs
    )
    ranges = detect_silence_ranges(envelope, min_silence_len_ms, threshold)
    return [
        RawCandidate(
            ms=(start + end) // 2,
            source="silence",
            silence_ms=end - start,
            pause_ms=end - start,
        )
        for start, end in ranges
    ]


def pause_candidates(
    envelope: AudioEnvelope, config: PauseConfig, fallback_threshold_dbfs: float
) -> list[RawCandidate]:
    """Midpoint of every short pause, excluding those long enough to be silences.

    The exclusion is what keeps the two generators from proposing the same
    timestamp twice under different names; anything at or above
    ``max_pause_ms`` belongs to ``silence``.
    """
    threshold = _silence_threshold_dbfs(
        envelope, config.threshold_offset_db, fallback_threshold_dbfs
    )
    ranges = detect_silence_ranges(envelope, config.min_pause_ms, threshold)
    candidates: list[RawCandidate] = []
    for start, end in ranges:
        length = end - start
        if length >= config.max_pause_ms:
            continue
        candidates.append(
            RawCandidate(
                ms=(start + end) // 2,
                source="pause",
                silence_ms=0,
                pause_ms=length,
            )
        )
    return candidates


def rms_minimum_candidates(
    envelope: AudioEnvelope, config: RmsMinimumConfig
) -> list[RawCandidate]:
    """Local energy minima below the configured percentile of the episode.

    Windows are evaluated at ``hop_ms``; a window qualifies when it is quieter
    than both neighbours and below the percentile threshold. Accepted minima are
    then thinned to ``min_spacing_ms``, quietest first, so a long quiet stretch
    contributes a few representative points rather than dozens.
    """
    window_ms = config.window_ms
    dbfs = envelope.window_dbfs(window_ms)
    if dbfs.size < 3:
        return []
    hop = max(1, config.hop_ms)
    positions = np.arange(0, dbfs.size, hop)
    sampled = dbfs[positions]
    if sampled.size < 3:
        return []

    threshold = float(np.percentile(sampled, config.percentile))
    interior = np.arange(1, sampled.size - 1)
    is_minimum = (sampled[interior] <= sampled[interior - 1]) & (
        sampled[interior] <= sampled[interior + 1]
    )
    below = sampled[interior] <= threshold
    selected = interior[is_minimum & below]
    if selected.size == 0:
        return []

    # Quietest first, so thinning keeps the strongest minimum in each region.
    order = selected[np.argsort(sampled[selected], kind="stable")]
    kept_ms: list[int] = []
    for index in order:
        centre = int(positions[index]) + window_ms // 2
        if any(abs(centre - existing) < config.min_spacing_ms for existing in kept_ms):
            continue
        kept_ms.append(centre)

    return [
        RawCandidate(ms=ms, source="rms_minimum")
        for ms in sorted(kept_ms)
    ]


def fixed_interval_candidates(
    duration_ms: int, config: FixedIntervalConfig
) -> list[RawCandidate]:
    """A regular grid over the episode, excluding both endpoints."""
    interval = config.interval_ms
    return [
        RawCandidate(ms=ms, source="fixed_interval")
        for ms in range(interval, max(0, int(duration_ms)), interval)
    ]


def energy_summary_values(
    envelope: AudioEnvelope, centre_ms: int, window_ms: int
) -> tuple[float, float, float]:
    """``(mean, min, max)`` dBFS over ``window_ms`` centred on ``centre_ms``."""
    half = max(1, window_ms // 2)
    start = max(0, centre_ms - half)
    end = min(envelope.n_ms, centre_ms + half)
    if end <= start:
        floor = NEGATIVE_INFINITY_DBFS
        return floor, floor, floor
    region = envelope.mean_square[start:end]
    mean_dbfs = float(envelope.to_dbfs(float(region.mean())))
    # Per-millisecond extremes are noisy; a 50 ms sub-window is the smallest
    # span over which "how quiet does it get here" is a meaningful question.
    sub = max(1, min(50, region.size))
    if region.size >= sub:
        cumulative = np.concatenate(([0.0], np.cumsum(region)))
        sub_means = (cumulative[sub:] - cumulative[:-sub]) / float(sub)
    else:  # pragma: no cover - guarded by the size clamp above
        sub_means = region
    sub_dbfs = np.asarray(envelope.to_dbfs(sub_means))
    return mean_dbfs, float(sub_dbfs.min()), float(sub_dbfs.max())


def apply_edge_guard(
    candidates: Sequence[RawCandidate], duration_ms: int, edge_guard_ms: int
) -> tuple[list[RawCandidate], int]:
    """Drop candidates too close to either end. Returns ``(kept, dropped)``."""
    low = edge_guard_ms
    high = duration_ms - edge_guard_ms
    kept = [
        candidate for candidate in candidates if low <= candidate.ms <= high
    ]
    return kept, len(candidates) - len(kept)
