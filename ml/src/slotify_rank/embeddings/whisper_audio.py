"""Frozen Whisper encoder representations, cached per episode.

**The efficiency decision.** Running the encoder once per candidate would mean
re-encoding overlapping 30-second windows dozens of times per episode -- an
episode with 200 candidates would cost 200 forward passes over largely the same
audio. Instead the episode is encoded once, chunk by chunk, and the time-indexed
encoder states are kept; each candidate's windows are then *pooled* out of that
cache. Cost becomes proportional to episode length rather than candidate count.

**Dimensions.** ``openai/whisper-tiny.en`` has ``d_model = 384``, so a raw
encoder frame is a 384-vector and every pooled representation is 384-dimensional
too. That is the number recorded in the manifest. It is emphatically *not* 768;
that is the base model's width, and documenting it for tiny.en would be a claim
the artifacts contradict. :func:`encoder_dimension` reads it from the loaded
config rather than trusting this docstring.

No projection to 128 dimensions happens here. Projection is a learned layer and
belongs inside the future ranker, where it can be trained; doing it now with an
untrained matrix would destroy information for no benefit.

The model is never fine-tuned. It runs under ``torch.no_grad`` in ``eval`` mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from slotify_rank.config.feature_settings import AudioEmbeddingConfig
from slotify_rank.embeddings.device import resolve_device, resolve_dtype
from slotify_rank.embeddings.pooling import (
    FrameGrid,
    derive_frame_duration_ms,
    pool_across_chunks,
)
from slotify_rank.transcription.segments import plan_chunks

__all__ = [
    "WhisperAudioEncoder",
    "EpisodeAudioStates",
    "CANDIDATE_EMBEDDING_KINDS",
]

WHISPER_SAMPLE_RATE = 16_000

#: Pooled representations produced per candidate. ``difference`` is
#: ``after - before``: exactly derivable from the other two, kept because it
#: makes the discontinuity across the break directly visible to a linear probe.
CANDIDATE_EMBEDDING_KINDS: tuple[str, ...] = (
    "before",
    "after",
    "context",
    "difference",
)


@dataclass(frozen=True)
class EpisodeAudioStates:
    """Time-indexed encoder output for one episode."""

    episode_id: str
    chunks: tuple[tuple[FrameGrid, np.ndarray], ...]
    dimension: int
    frame_duration_ms: float

    def pool(self, start_ms: int, end_ms: int) -> np.ndarray | None:
        return pool_across_chunks(list(self.chunks), start_ms, end_ms)


@dataclass
class WhisperAudioEncoder:
    """The frozen Whisper encoder plus its chunking settings."""

    config: AudioEmbeddingConfig
    _model: Any = None
    _extractor: Any = None
    _device: str = "cpu"

    def load(self) -> "WhisperAudioEncoder":
        from transformers import WhisperFeatureExtractor, WhisperModel

        self._device = resolve_device(self.config.device)
        torch_dtype = resolve_dtype(self.config.dtype, self._device)
        self._extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.model_id, revision=self.config.model_revision
        )
        model = WhisperModel.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
            dtype=torch_dtype,
        )
        model.eval()
        # Only the encoder is needed. Dropping the decoder halves resident
        # memory and removes any possibility of a generation path running.
        self._model = model.get_encoder().to(self._device)
        return self

    @property
    def device(self) -> str:
        return self._device

    @property
    def dimension(self) -> int:
        """Raw encoder hidden width, read from the model rather than assumed."""
        if self._model is None:
            raise RuntimeError("WhisperAudioEncoder.load() must be called first")
        return int(self._model.config.d_model)

    def release(self) -> None:
        from slotify_rank.embeddings.device import free_model

        model, self._model = self._model, None
        self._extractor = None
        if model is not None:
            free_model(model)

    def encode_episode(
        self, episode_id: str, audio: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE
    ) -> EpisodeAudioStates:
        """Encode a whole episode into time-indexed chunk states."""
        import torch

        if self._model is None or self._extractor is None:
            raise RuntimeError("WhisperAudioEncoder.load() must be called first")

        duration_ms = int(round(audio.size * 1000 / sample_rate))
        if duration_ms <= 0:
            raise ValueError(f"{episode_id}: audio has zero duration")

        plans = plan_chunks(duration_ms, self.config.chunk_ms, self.config.overlap_ms)
        encoded: list[tuple[FrameGrid, np.ndarray]] = []
        frame_duration_ms: float | None = None

        for plan in plans:
            first_sample = int(plan.start_ms * sample_rate / 1000)
            last_sample = int(plan.end_ms * sample_rate / 1000)
            window = audio[first_sample:last_sample]
            if window.size == 0:
                continue

            features = self._extractor(
                window, sampling_rate=sample_rate, return_tensors="pt"
            )
            input_features = features.input_features.to(
                self._device, dtype=self._model.dtype
            )
            with torch.no_grad():
                states = self._model(input_features).last_hidden_state

            # (1, frames, d_model) -> (frames, d_model), always float32 on the
            # way out so the stored dtype does not depend on the compute dtype.
            array = states[0].to(torch.float32).cpu().numpy()

            if frame_duration_ms is None:
                frame_duration_ms = derive_frame_duration_ms(
                    hop_length=int(self._extractor.hop_length),
                    sampling_rate=int(self._extractor.sampling_rate),
                    chunk_ms=self.config.chunk_ms,
                    encoder_frames=int(array.shape[0]),
                )

            encoded.append(
                (
                    FrameGrid(
                        chunk_start_ms=plan.start_ms,
                        chunk_end_ms=plan.end_ms,
                        frame_duration_ms=frame_duration_ms,
                        total_frames=int(array.shape[0]),
                    ),
                    array,
                )
            )

        if not encoded or frame_duration_ms is None:
            raise ValueError(f"{episode_id}: produced no encoder states")

        return EpisodeAudioStates(
            episode_id=episode_id,
            chunks=tuple(encoded),
            dimension=int(encoded[0][1].shape[1]),
            frame_duration_ms=frame_duration_ms,
        )


def pool_candidate_embeddings(
    states: EpisodeAudioStates,
    windows: Any,
    include_difference: bool = True,
) -> dict[str, np.ndarray | None]:
    """Pool one candidate's windows out of the episode-level cache.

    ``before``/``after`` use the medium window (the phrase either side) and
    ``context`` uses the wide one (did the topic change?). Any of them may be
    ``None`` when the window fell outside the audio -- a real missing modality
    that the caller records in a mask rather than papering over with zeros.
    """
    before_window = windows.before["medium"]
    after_window = windows.after["medium"]
    context_window = windows.around["context"]

    pooled: dict[str, np.ndarray | None] = {
        "before": states.pool(before_window.start_ms, before_window.end_ms),
        "after": states.pool(after_window.start_ms, after_window.end_ms),
        "context": states.pool(context_window.start_ms, context_window.end_ms),
    }
    if include_difference:
        before_vector = pooled["before"]
        after_vector = pooled["after"]
        pooled["difference"] = (
            None
            if before_vector is None or after_vector is None
            else (after_vector - before_vector).astype(np.float32)
        )
    return pooled
