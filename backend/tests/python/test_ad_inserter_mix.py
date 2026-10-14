"""mix.py: splice insertion and loudness matching."""

from __future__ import annotations

import math

import pytest
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

from ad_inserter import mix

# 48 frames per ms, so millisecond slices land on exact frame boundaries.
RATE = 48000


def _main(duration_ms: int) -> AudioSegment:
    return (
        WhiteNoise(sample_rate=RATE)
        .to_audio_segment(duration=duration_ms, volume=-12)
        .set_channels(2)
    )


def _promo(duration_ms: int) -> AudioSegment:
    return (
        Sine(330, sample_rate=RATE)
        .to_audio_segment(duration=duration_ms, volume=-6)
        .set_channels(2)
    )


def test_splice_lengths_and_preserves_audio_around_the_promo() -> None:
    main = _main(10_000)
    promo = _promo(3_000)
    insert_ms, crossfade_ms = 4_000, 250

    merged = mix.insert_with_crossfade(main, promo, insert_ms, crossfade_ms=crossfade_ms)

    # Two crossfaded joins overlap crossfade_ms each.
    assert merged.frame_count() == (
        main.frame_count() + promo.frame_count() - 2 * crossfade_ms * RATE / 1000
    )
    assert len(merged) == len(main) + len(promo) - 2 * crossfade_ms

    # Before the first join: the original, untouched.
    head = insert_ms - crossfade_ms
    assert merged[:head].raw_data == main[:head].raw_data
    # Between the joins: the promo alone -- no speech underneath it.
    body = merged[insert_ms : insert_ms + len(promo) - 2 * crossfade_ms]
    assert body.raw_data == promo[crossfade_ms : len(promo) - crossfade_ms].raw_data
    # After the second join: the original from the insertion point on.
    resumed = main[insert_ms + crossfade_ms :]
    assert merged[-len(resumed) :].raw_data == resumed.raw_data


def test_promo_running_past_the_end_is_not_truncated() -> None:
    main = _main(5_000)
    promo = _promo(3_000)

    merged = mix.insert_with_crossfade(main, promo, insert_ms=4_000, crossfade_ms=250)

    # 1 s of main after the insertion point; the promo still plays in full.
    assert len(merged) == len(main) + len(promo) - 2 * 250
    body = merged[4_000 : 4_000 + len(promo) - 500]
    assert body.raw_data == promo[250 : len(promo) - 250].raw_data


def test_insert_at_the_ends_and_beyond() -> None:
    main = _main(4_000)
    promo = _promo(2_000)

    at_start = mix.insert_with_crossfade(main, promo, insert_ms=0, crossfade_ms=250)
    assert len(at_start) == len(main) + len(promo) - 250
    assert at_start[: len(promo) - 250].raw_data == promo[: len(promo) - 250].raw_data

    past_end = mix.insert_with_crossfade(main, promo, insert_ms=99_000, crossfade_ms=250)
    assert len(past_end) == len(main) + len(promo) - 250
    assert past_end[-(len(promo) - 250) :].raw_data == promo[250:].raw_data


def test_crossfades_never_overlap_on_a_short_promo() -> None:
    main = _main(4_000)
    promo = _promo(300)

    merged = mix.insert_with_crossfade(main, promo, insert_ms=2_000, crossfade_ms=250)

    # Each join is capped at half the promo (150 ms), so the original before
    # the first join and after the second are intact.
    assert len(merged) == len(main) + len(promo) - 2 * 150
    assert merged[: 2_000 - 150].raw_data == main[: 2_000 - 150].raw_data
    resumed = main[2_000 + 150 :]
    assert merged[-len(resumed) :].raw_data == resumed.raw_data


def test_zero_crossfade_is_a_plain_splice() -> None:
    main = _main(3_000)
    promo = _promo(1_000)

    merged = mix.insert_with_crossfade(main, promo, insert_ms=1_500, crossfade_ms=0)

    assert merged.raw_data == (main[:1_500] + promo + main[1_500:]).raw_data


def test_digital_silence_context_does_not_mute_the_promo() -> None:
    promo = _promo(3_000)
    silent_context = AudioSegment.silent(duration=8_000, frame_rate=RATE).set_channels(2)

    target = mix.measure_lufs(silent_context)
    assert target == float("-inf")

    result = mix.match_loudness(promo, target)
    assert result.matched.raw_data == promo.raw_data
    assert math.isfinite(result.promo_after_lufs)


def test_too_short_promo_skips_matching() -> None:
    promo = _promo(200)  # shorter than pyloudnorm's 400 ms gating block

    result = mix.match_loudness(promo, -20.0)

    assert result.matched.raw_data == promo.raw_data


@pytest.mark.parametrize("target_lufs", [-90.0, 40.0])
def test_loudness_gain_is_clamped(target_lufs: float) -> None:
    # Quiet enough that +20 dB does not clip.
    promo = Sine(330, sample_rate=RATE).to_audio_segment(duration=3_000, volume=-40)

    result = mix.match_loudness(promo, target_lufs)

    applied = result.promo_after_lufs - result.promo_before_lufs
    assert abs(applied) == pytest.approx(mix.MAX_LOUDNESS_GAIN_DB, abs=0.5)


def test_loudness_matching_reaches_a_reasonable_target() -> None:
    promo = _promo(3_000)
    target = mix.measure_lufs(promo) - 6.0

    result = mix.match_loudness(promo, target)

    assert result.promo_after_lufs == pytest.approx(target, abs=0.5)
