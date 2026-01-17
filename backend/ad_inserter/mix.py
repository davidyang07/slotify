from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyloudnorm as pyln
from pydub import AudioSegment


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
    meter = pyln.Meter(audio.frame_rate)
    samples = _audiosegment_to_float(audio)
    if samples.size == 0:
        return -70.0
    return float(meter.integrated_loudness(samples))


def match_loudness(promo: AudioSegment, target_lufs: float) -> LoudnessMatch:
    promo_before = measure_lufs(promo)
    gain_db = target_lufs - promo_before
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
    duck_db: float = 0.0,
    crossfade_ms: int = 250,
) -> AudioSegment:
    insert_ms = max(0, insert_ms)
    pre = main[:insert_ms]
    post = main[insert_ms + len(promo) :]
    mid = main[insert_ms : insert_ms + len(promo)]

    if duck_db > 0:
        mid = mid.apply_gain(-duck_db).fade_in(100).fade_out(100)

    mid = mid.overlay(promo)

    cf1 = min(crossfade_ms, len(pre), len(mid))
    merged = pre.append(mid, crossfade=cf1)
    cf2 = min(crossfade_ms, len(merged), len(post))
    merged = merged.append(post, crossfade=cf2)
    return merged


def context_window(
    audio: AudioSegment, center_ms: int, window_ms: int = 4000
) -> AudioSegment:
    start = max(0, center_ms - window_ms)
    end = min(len(audio), center_ms + window_ms)
    return audio[start:end]
