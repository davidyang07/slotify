from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyloudnorm as pyln
from pydub import AudioSegment


# Loudness matching never moves the promo by more than this. A larger
# correction means one of the measurements is unreliable (near-silent context,
# a clip shorter than the meter's 400 ms block), not that the promo is that far
# off.
MAX_LOUDNESS_GAIN_DB = 20.0


@dataclass
class LoudnessMatch:
    matched: AudioSegment
    target_lufs: float
    promo_before_lufs: float
    promo_after_lufs: float


def _audiosegment_to_float(audio: AudioSegment) -> np.ndarray:
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
    max_val = float(1 << (8 * audio.sample_width - 1))
    return samples.astype(np.float32) / max_val


def measure_lufs(audio: AudioSegment) -> float:
    """Integrated loudness in LUFS; ``-inf`` for digital silence or too-short clips."""
    meter = pyln.Meter(audio.frame_rate)
    samples = _audiosegment_to_float(audio)
    if samples.size == 0:
        return -70.0
    try:
        return float(meter.integrated_loudness(samples))
    except ValueError:
        # pyloudnorm rejects clips shorter than one 400 ms gating block.
        return float("-inf")


def match_loudness(promo: AudioSegment, target_lufs: float) -> LoudnessMatch:
    """Gain ``promo`` toward ``target_lufs``.

    Matching is skipped (the promo is returned unchanged) when either loudness
    is not finite -- e.g. the context window is digital silence, which measures
    as ``-inf`` and would otherwise mute the promo. The applied gain is clamped
    to +/- ``MAX_LOUDNESS_GAIN_DB``.
    """
    promo_before = measure_lufs(promo)
    if not (math.isfinite(target_lufs) and math.isfinite(promo_before)):
        return LoudnessMatch(
            matched=promo,
            target_lufs=target_lufs,
            promo_before_lufs=promo_before,
            promo_after_lufs=promo_before,
        )
    gain_db = target_lufs - promo_before
    gain_db = max(-MAX_LOUDNESS_GAIN_DB, min(MAX_LOUDNESS_GAIN_DB, gain_db))
    matched = promo.apply_gain(gain_db)
    promo_after = measure_lufs(matched)
    return LoudnessMatch(
        matched=matched,
        target_lufs=target_lufs,
        promo_before_lufs=promo_before,
        promo_after_lufs=promo_after,
    )


def loop_to_length(audio: AudioSegment, target_ms: int) -> AudioSegment:
    if len(audio) == 0:
        return AudioSegment.silent(duration=target_ms, frame_rate=audio.frame_rate)
    output = AudioSegment.empty()
    while len(output) < target_ms:
        output += audio
    return output[:target_ms]


def apply_room_tone(
    promo: AudioSegment,
    room_tone: Optional[AudioSegment],
    gain_db: float = -26.0,
) -> AudioSegment:
    if room_tone is None:
        return promo
    bed = loop_to_length(room_tone, len(promo)).apply_gain(gain_db)
    return bed.overlay(promo)


def insert_with_crossfade(
    main: AudioSegment,
    promo: AudioSegment,
    insert_ms: int,
    crossfade_ms: int = 250,
) -> AudioSegment:
    """Splice ``promo`` into ``main`` at ``insert_ms`` with crossfaded joins.

    The result is ``main[:insert_ms] + promo + main[insert_ms:]``: the original
    audio pauses for the promo and resumes where it left off, so nothing plays
    under the promo and the promo is never truncated. Each join is a crossfade
    of up to ``crossfade_ms`` (limited by the audio on either side of the join,
    and to half the promo so the two joins never overlap), so the output is
    ``len(main) + len(promo)`` minus the two crossfade overlaps.
    """
    insert_ms = min(max(0, insert_ms), len(main))
    pre = main[:insert_ms]
    post = main[insert_ms:]
    crossfade_ms = max(0, crossfade_ms)
    half_promo = len(promo) // 2

    cf1 = min(crossfade_ms, len(pre), half_promo)
    merged = pre.append(promo, crossfade=cf1)
    cf2 = min(crossfade_ms, len(post), half_promo)
    merged = merged.append(post, crossfade=cf2)
    return merged


def context_window(
    audio: AudioSegment, center_ms: int, window_ms: int = 4000
) -> AudioSegment:
    start = max(0, center_ms - window_ms)
    end = min(len(audio), center_ms + window_ms)
    return audio[start:end]
