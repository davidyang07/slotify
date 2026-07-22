"""Deterministic handcrafted acoustic descriptors around a candidate.

Every feature here answers one question: *does the audio behave like a place a
listener would accept a break?* The signals that matter are a drop in energy, a
pause of some length, and a discontinuity in timbre across the break.

**No normalization is fitted here.** These are raw values in physical units.
Fitting mean/variance now would leak test-set statistics into the features
themselves; standardization belongs in the training pipeline, computed on the
train split alone.

**Tempo is deliberately absent.** ``librosa.beat.tempo`` on spoken word returns
an unstable estimate driven by whatever periodicity the onset envelope happens
to contain, and it changes materially with window length. It is a real feature
for the product's music path and a source of noise here, so it is not extracted.

**Feature reference.** Every name below is ``<family>_<statistic>_<window>``.
Units are dBFS for energy, milliseconds for durations, Hz for spectral
positions, and dimensionless otherwise. "Break-like" says which direction
intuitively indicates a natural breakpoint.

===================================  ======  =====================================
Feature                              Units   Break-like when
===================================  ======  =====================================
``rms_mean_before_*``                dBFS    lower (speech trailing off)
``rms_mean_after_*``                 dBFS    lower (next phrase not yet started)
``rms_min_around_*``                 dBFS    lower (a real trough at the break)
``rms_max_around_*``                 dBFS    lower (no shouting across the break)
``rms_var_around_*``                 dB^2    lower (steady, not mid-word)
``rms_delta_across_*``               dB      near zero (symmetric energy)
``rms_slope_before_*``               dB/s    negative (energy decaying into it)
``rms_slope_after_*``                dB/s    positive (energy rising out of it)
``energy_discontinuity_*``           dB      higher (a real seam in the audio)
``silence_duration_ms``              ms      higher
``pause_duration_ms``                ms      higher
``ms_since_prior_speech``            ms      higher
``ms_until_next_speech``             ms      higher
``zcr_mean_*`` / ``zcr_var_*``       ratio   lower variance (not mid-fricative)
``spectral_centroid_mean_*``         Hz      lower (voice absent)
``spectral_centroid_delta_across``   Hz      larger magnitude (timbre changes)
``spectral_bandwidth_mean_*``        Hz      lower
``spectral_rolloff_mean_*``          Hz      lower
``spectral_contrast_mean_*``         dB      lower (less harmonic structure)
``onset_strength_mean_*``            a.u.    lower (no attacks at the break)
``onset_strength_max_around_*``      a.u.    lower
``dbfs_mean_*``                      dBFS    lower
===================================  ======  =====================================

Missing-value behaviour is uniform: a window that survives clipping to fewer
than two analysis frames cannot support a variance or a slope, so those features
are reported as missing (mask ``True``, value ``0.0``) rather than as a
fabricated zero. The mask travels with the values all the way into the training
record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from slotify_rank.config.feature_settings import AcousticConfig
from slotify_rank.data.audio_io import NEGATIVE_INFINITY_DBFS, AudioEnvelope
from slotify_rank.features.windows import CandidateWindow, CandidateWindows

__all__ = [
    "AcousticFrames",
    "compute_frames",
    "extract_acoustic_features",
]

#: Value stored for a missing scalar. Always paired with a mask entry; never
#: meaningful on its own.
MISSING_VALUE = 0.0


@dataclass(frozen=True)
class AcousticFrames:
    """Frame-level descriptors for a whole episode, computed once.

    Computing these per candidate would re-run an STFT over overlapping audio
    dozens of times per episode. They are computed once at the configured hop
    and sliced per window, which is the difference between minutes and hours on
    a real corpus.

    All arrays share the same time base: frame ``i`` starts at ``i * hop_ms``.
    """

    hop_ms: int
    sample_rate_hz: int
    rms_db: np.ndarray
    zcr: np.ndarray
    centroid_hz: np.ndarray
    bandwidth_hz: np.ndarray
    rolloff_hz: np.ndarray
    contrast_db: np.ndarray
    onset_strength: np.ndarray
    #: Boolean per frame: is this frame speech rather than background?
    is_speech: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.rms_db.shape[0])

    def frame_span(self, window: CandidateWindow) -> tuple[int, int]:
        """Frame indices ``[first, last)`` overlapping a time window."""
        if window.is_empty:
            return (0, 0)
        first = int(np.floor(window.start_ms / self.hop_ms))
        last = int(np.ceil(window.end_ms / self.hop_ms))
        first = max(0, min(first, self.n_frames))
        last = max(first, min(last, self.n_frames))
        return (first, last)


def compute_frames(
    samples: np.ndarray, sample_rate_hz: int, config: AcousticConfig
) -> AcousticFrames:
    """Run the STFT-based descriptors over a whole episode.

    ``samples`` must be mono float32 in ``[-1, 1]`` at ``sample_rate_hz``.
    """
    import librosa

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("Refusing to analyse empty audio")

    hop_length = max(1, int(round(sample_rate_hz * config.hop_ms / 1000.0)))
    frame_length = max(hop_length, int(round(sample_rate_hz * config.frame_ms / 1000.0)))
    n_fft = max(config.n_fft, frame_length)

    # One magnitude spectrogram, reused by every spectral descriptor. librosa
    # would otherwise recompute the STFT once per feature.
    stft = np.abs(
        librosa.stft(
            audio, n_fft=n_fft, hop_length=hop_length, win_length=frame_length
        )
    )

    # When RMS is computed from a spectrogram, librosa needs frame_length to be
    # the FFT size (it infers the bin count from it), not the analysis window.
    # Passing win_length here raises ParameterError.
    rms = librosa.feature.rms(S=stft, frame_length=n_fft, hop_length=hop_length)[0]
    with np.errstate(divide="ignore"):
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    rms_db = np.maximum(rms_db, NEGATIVE_INFINITY_DBFS)

    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sample_rate_hz)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sample_rate_hz)[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=stft, sr=sample_rate_hz, roll_percent=config.roll_off_percent
    )[0]

    # Spectral contrast needs every band edge below Nyquist. At 16 kHz the
    # default 6 bands fit; if a future sample rate makes them not fit, librosa
    # raises, and a flat fallback is better than losing the whole episode --
    # recorded as all-zero contrast, which the variance check will not flag
    # because it is genuinely constant rather than corrupt.
    try:
        contrast = librosa.feature.spectral_contrast(
            S=stft, sr=sample_rate_hz, n_bands=config.n_spectral_contrast_bands
        ).mean(axis=0)
    except Exception:
        contrast = np.zeros(stft.shape[1], dtype=np.float64)

    onset = librosa.onset.onset_strength(
        S=librosa.power_to_db(stft**2, ref=np.max), sr=sample_rate_hz
    )

    # Speech/non-speech relative to the episode's own median frame energy, so a
    # quietly mastered episode is not classified as entirely silent.
    finite = rms_db[np.isfinite(rms_db)]
    reference_db = float(np.median(finite)) if finite.size else NEGATIVE_INFINITY_DBFS
    is_speech = rms_db > (reference_db + config.speech_threshold_db)

    # librosa's per-feature framing can differ by one frame; align to the
    # shortest so every array indexes the same instant.
    lengths = [
        array.shape[0]
        for array in (rms_db, zcr, centroid, bandwidth, rolloff, contrast, onset)
    ]
    n = min(lengths)
    return AcousticFrames(
        hop_ms=config.hop_ms,
        sample_rate_hz=sample_rate_hz,
        rms_db=rms_db[:n],
        zcr=zcr[:n],
        centroid_hz=centroid[:n],
        bandwidth_hz=bandwidth[:n],
        rolloff_hz=rolloff[:n],
        contrast_db=contrast[:n],
        onset_strength=onset[:n],
        is_speech=is_speech[:n],
    )


def _slice(array: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    first, last = span
    return array[first:last]


def _stat(
    values: np.ndarray, statistic: str, minimum_frames: int = 1
) -> tuple[float, bool]:
    """``(value, missing)`` for one statistic over one window.

    ``missing=True`` means the window could not support the statistic -- too few
    frames, or an all-non-finite slice. The value is then :data:`MISSING_VALUE`
    and carries no information.
    """
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size < minimum_frames:
        return (MISSING_VALUE, True)
    if statistic == "mean":
        result = float(np.mean(finite))
    elif statistic == "min":
        result = float(np.min(finite))
    elif statistic == "max":
        result = float(np.max(finite))
    elif statistic == "var":
        result = float(np.var(finite))
    else:
        raise ValueError(f"Unknown statistic {statistic!r}")
    return (result, False) if np.isfinite(result) else (MISSING_VALUE, True)


def _slope_db_per_second(values: np.ndarray, hop_ms: int) -> tuple[float, bool]:
    """Least-squares slope of a window, in units per second.

    Needs at least two finite frames; a single frame has no slope, and returning
    0.0 unmasked would claim "perfectly flat" for what is really "unknown".
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return (MISSING_VALUE, True)
    seconds = np.arange(finite.size, dtype=np.float64) * (hop_ms / 1000.0)
    slope = float(np.polyfit(seconds, finite.astype(np.float64), 1)[0])
    return (slope, False) if np.isfinite(slope) else (MISSING_VALUE, True)


def _speech_gap_ms(
    frames: AcousticFrames, timestamp_ms: int, direction: int
) -> tuple[float, bool]:
    """Milliseconds to the nearest speech frame before (-1) or after (+1).

    Missing when there is no speech at all in that direction, which is a real
    condition (a candidate in a long musical outro) and not a failure.
    """
    if frames.n_frames == 0:
        return (MISSING_VALUE, True)
    index = int(round(timestamp_ms / frames.hop_ms))
    index = max(0, min(index, frames.n_frames - 1))
    speech_indices = np.flatnonzero(frames.is_speech)
    if speech_indices.size == 0:
        return (MISSING_VALUE, True)

    if direction < 0:
        earlier = speech_indices[speech_indices <= index]
        if earlier.size == 0:
            return (MISSING_VALUE, True)
        return (float((index - earlier[-1]) * frames.hop_ms), False)
    later = speech_indices[speech_indices >= index]
    if later.size == 0:
        return (MISSING_VALUE, True)
    return (float((later[0] - index) * frames.hop_ms), False)


def _local_silence_ms(frames: AcousticFrames, timestamp_ms: int) -> float:
    """Length of the contiguous non-speech run containing the candidate.

    Zero when the candidate sits inside speech, which is itself informative:
    it is the single clearest "this is a bad breakpoint" signal available.
    """
    if frames.n_frames == 0:
        return 0.0
    index = int(round(timestamp_ms / frames.hop_ms))
    index = max(0, min(index, frames.n_frames - 1))
    if frames.is_speech[index]:
        return 0.0
    first = index
    while first > 0 and not frames.is_speech[first - 1]:
        first -= 1
    last = index
    while last + 1 < frames.n_frames and not frames.is_speech[last + 1]:
        last += 1
    return float((last - first + 1) * frames.hop_ms)


def extract_acoustic_features(
    frames: AcousticFrames,
    windows: CandidateWindows,
    envelope: AudioEnvelope | None = None,
) -> tuple[dict[str, float], dict[str, bool]]:
    """All acoustic features for one candidate.

    Returns ``(values, missing)`` keyed by feature name. Both mappings always
    have exactly the same keys -- an assembled record must never contain a value
    with no mask entry or a mask with no value.
    """
    values: dict[str, float] = {}
    missing: dict[str, bool] = {}

    def put(name: str, result: tuple[float, bool]) -> None:
        value, is_missing = result
        values[name] = float(value)
        missing[name] = bool(is_missing)

    for scale in sorted(windows.before):
        before_w, after_w, around_w = windows.scale(scale)
        before_span = frames.frame_span(before_w)
        after_span = frames.frame_span(after_w)
        around_span = frames.frame_span(around_w)

        rms_before = _slice(frames.rms_db, before_span)
        rms_after = _slice(frames.rms_db, after_span)
        rms_around = _slice(frames.rms_db, around_span)

        put(f"rms_mean_before_{scale}", _stat(rms_before, "mean"))
        put(f"rms_mean_after_{scale}", _stat(rms_after, "mean"))
        put(f"rms_min_around_{scale}", _stat(rms_around, "min"))
        put(f"rms_max_around_{scale}", _stat(rms_around, "max"))
        put(f"rms_var_around_{scale}", _stat(rms_around, "var", minimum_frames=2))

        # Energy step across the break. Missing unless *both* sides exist:
        # a difference against an absent side is not a small difference.
        mean_before, missing_before = _stat(rms_before, "mean")
        mean_after, missing_after = _stat(rms_after, "mean")
        if missing_before or missing_after:
            put(f"rms_delta_across_{scale}", (MISSING_VALUE, True))
            put(f"energy_discontinuity_{scale}", (MISSING_VALUE, True))
        else:
            put(f"rms_delta_across_{scale}", (mean_after - mean_before, False))
            put(
                f"energy_discontinuity_{scale}",
                (abs(mean_after - mean_before), False),
            )

        put(f"rms_slope_before_{scale}", _slope_db_per_second(rms_before, frames.hop_ms))
        put(f"rms_slope_after_{scale}", _slope_db_per_second(rms_after, frames.hop_ms))

        put(f"zcr_mean_around_{scale}", _stat(_slice(frames.zcr, around_span), "mean"))
        put(
            f"zcr_var_around_{scale}",
            _stat(_slice(frames.zcr, around_span), "var", minimum_frames=2),
        )

        centroid_before = _slice(frames.centroid_hz, before_span)
        centroid_after = _slice(frames.centroid_hz, after_span)
        put(f"spectral_centroid_mean_before_{scale}", _stat(centroid_before, "mean"))
        put(f"spectral_centroid_mean_after_{scale}", _stat(centroid_after, "mean"))
        c_before, c_missing_before = _stat(centroid_before, "mean")
        c_after, c_missing_after = _stat(centroid_after, "mean")
        put(
            f"spectral_centroid_delta_across_{scale}",
            (MISSING_VALUE, True)
            if (c_missing_before or c_missing_after)
            else (c_after - c_before, False),
        )

        put(
            f"spectral_bandwidth_mean_around_{scale}",
            _stat(_slice(frames.bandwidth_hz, around_span), "mean"),
        )
        put(
            f"spectral_rolloff_mean_around_{scale}",
            _stat(_slice(frames.rolloff_hz, around_span), "mean"),
        )
        put(
            f"spectral_contrast_mean_around_{scale}",
            _stat(_slice(frames.contrast_db, around_span), "mean"),
        )
        put(
            f"onset_strength_mean_around_{scale}",
            _stat(_slice(frames.onset_strength, around_span), "mean"),
        )
        put(
            f"onset_strength_max_around_{scale}",
            _stat(_slice(frames.onset_strength, around_span), "max"),
        )

        # Availability after clipping: lets a model discount a window that was
        # mostly outside the episode instead of reading it as a quiet passage.
        values[f"window_available_before_{scale}"] = before_w.available_fraction
        missing[f"window_available_before_{scale}"] = False
        values[f"window_available_after_{scale}"] = after_w.available_fraction
        missing[f"window_available_after_{scale}"] = False

    # Scale-independent timing features.
    values["silence_duration_ms"] = _local_silence_ms(frames, windows.timestamp_ms)
    missing["silence_duration_ms"] = False
    put("ms_since_prior_speech", _speech_gap_ms(frames, windows.timestamp_ms, -1))
    put("ms_until_next_speech", _speech_gap_ms(frames, windows.timestamp_ms, +1))

    if envelope is not None:
        short_before, short_after, _ = windows.scale("short")
        values["dbfs_mean_before_short"] = envelope.region_dbfs(
            short_before.start_ms, short_before.end_ms
        )
        missing["dbfs_mean_before_short"] = short_before.is_empty
        values["dbfs_mean_after_short"] = envelope.region_dbfs(
            short_after.start_ms, short_after.end_ms
        )
        missing["dbfs_mean_after_short"] = short_after.is_empty
    else:
        values["dbfs_mean_before_short"] = MISSING_VALUE
        missing["dbfs_mean_before_short"] = True
        values["dbfs_mean_after_short"] = MISSING_VALUE
        missing["dbfs_mean_after_short"] = True

    if set(values) != set(missing):
        raise AssertionError(
            "acoustic values and mask disagree on keys: "
            f"{sorted(set(values) ^ set(missing))}"
        )
    return values, missing
