"""Mapping Whisper encoder frames to episode time, and pooling over windows.

This module is pure NumPy and imports no model code, so the alignment arithmetic
-- the part most likely to be subtly wrong and least likely to announce it --
can be tested exhaustively without downloading anything.

**Where the temporal resolution comes from.** It is *derived*, never assumed.
The Whisper feature extractor produces a log-mel spectrogram at
``hop_length / sampling_rate`` seconds per mel frame (160/16000 = 10 ms for
every Whisper variant). The encoder's second convolution has stride 2, so two
mel frames become one encoder frame::

    seconds_per_encoder_frame = hop_length / sampling_rate * conv_stride
                              = 160 / 16000 * 2
                              = 0.02 s  (20 ms)

:func:`derive_frame_duration_ms` computes this from the values the loaded model
actually reports and cross-checks it against the encoder's real output length,
raising if the two disagree. A hardcoded "0.2 s resolution" would be wrong by a
factor of ten and would still produce plausible-looking embeddings.

**Padding is the trap.** Whisper pads every input to a full 30 seconds, so the
encoder emits the same 1500 frames whether it was given 30 seconds of speech or
2. Frames past the end of the real audio describe silence the model invented.
:func:`valid_frame_count` bounds pooling to the real region; without it, a
candidate near the end of an episode gets an embedding dominated by padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FrameGrid",
    "derive_frame_duration_ms",
    "valid_frame_count",
    "frame_span_for_window",
    "mean_pool",
]

#: Stride of the Whisper encoder's second convolution. Fixed across every
#: published Whisper variant; asserted against the real output length at load.
WHISPER_CONV_STRIDE = 2


def derive_frame_duration_ms(
    hop_length: int,
    sampling_rate: int,
    chunk_ms: int,
    encoder_frames: int,
    conv_stride: int = WHISPER_CONV_STRIDE,
) -> float:
    """Milliseconds of audio per encoder frame, derived and cross-checked.

    Two independent routes to the same number:

    1. From preprocessing: ``hop_length / sampling_rate * conv_stride``.
    2. From the observed shape: ``chunk_ms / encoder_frames``.

    They must agree. When they do not, the model's preprocessing is not what
    this code assumes and every pooled embedding would be misaligned -- so this
    raises rather than picking one.
    """
    if hop_length <= 0 or sampling_rate <= 0:
        raise ValueError("hop_length and sampling_rate must be positive")
    if encoder_frames <= 0:
        raise ValueError("encoder_frames must be positive")
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")

    from_preprocessing = hop_length / sampling_rate * conv_stride * 1000.0
    from_shape = chunk_ms / encoder_frames

    # One part in 1000 -- generous enough for the rounding in the mel frame
    # count, tight enough that a stride or hop mismatch is caught.
    if abs(from_preprocessing - from_shape) > from_preprocessing * 1e-3:
        raise ValueError(
            "Whisper temporal resolution is ambiguous: preprocessing implies "
            f"{from_preprocessing:.4f} ms per encoder frame "
            f"(hop_length={hop_length}, sampling_rate={sampling_rate}, "
            f"conv_stride={conv_stride}) but the encoder emitted "
            f"{encoder_frames} frames for a {chunk_ms} ms chunk, implying "
            f"{from_shape:.4f} ms. Refusing to guess: every pooled embedding "
            "would be misaligned in time."
        )
    return from_preprocessing


@dataclass(frozen=True)
class FrameGrid:
    """Time base for one chunk's encoder output.

    ``chunk_start_ms`` is the chunk's position in the episode; frame ``i``
    therefore covers ``[chunk_start_ms + i*d, chunk_start_ms + (i+1)*d)`` for
    ``d = frame_duration_ms``.
    """

    chunk_start_ms: int
    chunk_end_ms: int
    frame_duration_ms: float
    total_frames: int

    def __post_init__(self) -> None:
        if self.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive")
        if self.chunk_end_ms <= self.chunk_start_ms:
            raise ValueError(
                f"chunk_end_ms ({self.chunk_end_ms}) must exceed chunk_start_ms "
                f"({self.chunk_start_ms})"
            )
        if self.total_frames <= 0:
            raise ValueError("total_frames must be positive")

    @property
    def valid_frames(self) -> int:
        return valid_frame_count(
            self.chunk_end_ms - self.chunk_start_ms,
            self.frame_duration_ms,
            self.total_frames,
        )

    def frame_start_ms(self, index: int) -> float:
        return self.chunk_start_ms + index * self.frame_duration_ms

    def covers(self, timestamp_ms: int) -> bool:
        return self.chunk_start_ms <= timestamp_ms < self.chunk_end_ms


def valid_frame_count(
    real_audio_ms: int, frame_duration_ms: float, total_frames: int
) -> int:
    """How many of ``total_frames`` describe real audio rather than padding.

    Whisper pads to 30 s, so ``total_frames`` is constant while the real content
    is not. Rounding *up* keeps the frame that straddles the end of the audio:
    it contains real signal, and dropping it would lose the final moments of
    every episode -- exactly where end-of-episode candidates live.
    """
    if real_audio_ms <= 0:
        return 0
    frames = int(np.ceil(real_audio_ms / frame_duration_ms))
    return int(max(1, min(frames, total_frames)))


def frame_span_for_window(
    grid: FrameGrid, window_start_ms: int, window_end_ms: int
) -> tuple[int, int]:
    """Half-open frame range ``[first, last)`` overlapping a time window.

    The window is clipped to the chunk's real (non-padded) region first. An
    empty range ``(i, i)`` is returned when the window lies entirely outside --
    a normal situation for a chunk that simply does not contain the candidate,
    and the caller treats it as "no contribution", not as an error.
    """
    if window_end_ms <= window_start_ms:
        return (0, 0)

    valid = grid.valid_frames
    real_end_ms = grid.chunk_start_ms + valid * grid.frame_duration_ms
    start = max(window_start_ms, grid.chunk_start_ms)
    end = min(window_end_ms, real_end_ms, grid.chunk_end_ms)
    if end <= start:
        return (0, 0)

    first = int(np.floor((start - grid.chunk_start_ms) / grid.frame_duration_ms))
    last = int(np.ceil((end - grid.chunk_start_ms) / grid.frame_duration_ms))
    first = max(0, min(first, valid))
    last = max(first, min(last, valid))
    return (first, last)


def mean_pool(states: np.ndarray, first: int, last: int) -> np.ndarray | None:
    """Mean of ``states[first:last]``, or ``None`` for an empty span.

    ``None`` rather than a zero vector on purpose: an empty span means *no
    audio was available*, which is a missing modality the caller must record in
    a mask. Returning zeros here would make "silence" and "no data"
    indistinguishable to every model trained on the result.
    """
    if last <= first:
        return None
    window = np.asarray(states)[first:last]
    if window.size == 0:
        return None
    pooled = window.mean(axis=0, dtype=np.float64)
    if not np.isfinite(pooled).all():
        raise ValueError(
            f"Pooled encoder states over frames [{first}, {last}) contain NaN or "
            "infinity; the encoder output is corrupt."
        )
    return pooled.astype(np.float32)


def pool_across_chunks(
    chunks: list[tuple[FrameGrid, np.ndarray]],
    window_start_ms: int,
    window_end_ms: int,
) -> np.ndarray | None:
    """Mean-pool a time window that may span several overlapping chunks.

    Frames are accumulated with their counts and averaged once at the end, so
    the result is the mean over every contributing frame rather than a mean of
    per-chunk means -- those differ whenever the chunks contribute unequal
    numbers of frames, which at a chunk boundary they always do.

    Overlap double-counts the frames in the shared region. That is accepted
    deliberately: the alternative is assigning each overlapping frame to exactly
    one chunk, which introduces a discontinuity at the seam, and the frames in
    an overlap are near-identical anyway since they describe the same audio.
    """
    total: np.ndarray | None = None
    count = 0
    for grid, states in chunks:
        first, last = frame_span_for_window(grid, window_start_ms, window_end_ms)
        if last <= first:
            continue
        window = np.asarray(states)[first:last]
        summed = window.sum(axis=0, dtype=np.float64)
        total = summed if total is None else total + summed
        count += last - first

    if total is None or count == 0:
        return None
    pooled = total / count
    if not np.isfinite(pooled).all():
        raise ValueError(
            f"Pooled encoder states over [{window_start_ms}, {window_end_ms}) ms "
            "contain NaN or infinity; the encoder output is corrupt."
        )
    return pooled.astype(np.float32)
