"""Local Whisper transcription through the Hugging Face ecosystem.

``openai/whisper-tiny.en`` on CPU. No paid API is involved and no key is
required; the only network access is the one-time model download into the
Hugging Face cache, after which the pipeline is fully offline.

**Why Transformers rather than ``openai-whisper``.** The product already uses
``openai-whisper`` optionally, so it was the obvious candidate. Transformers
wins for this pipeline on two grounds that matter here and not in the product:
the same ``from_pretrained`` call gives a *pinnable revision* (a cache identity
needs one), and the encoder used for speech representations is the same object
loaded the same way -- so the transcript and the audio embeddings provably come
from the same weights rather than from two independently-versioned copies.

**Chunking.** Audio is cut into overlapping windows by
:func:`~slotify_rank.transcription.segments.plan_chunks` and each is decoded
separately, so peak memory is set by the chunk length rather than the episode
length. A three-hour episode costs the same memory as a thirty-second one.
Timestamps are made episode-relative and the overlap is reconciled in
:mod:`slotify_rank.transcription.segments`, which is unit-tested without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from slotify_rank.config.feature_settings import TranscriptionConfig
from slotify_rank.data.audio_io import samples_duration_ms
from slotify_rank.config.versions import (
    FEATURE_PIPELINE_VERSION,
    TRANSCRIPTION_VERSION,
)
from slotify_rank.embeddings.device import resolve_device, resolve_dtype
from slotify_rank.transcription.schema import EpisodeTranscript
from slotify_rank.transcription.segments import (
    RawSegment,
    build_segments,
    offset_segments,
    plan_chunks,
    reconcile_overlapping_segments,
)

__all__ = ["WhisperTranscriber", "TranscriptionFailure", "load_audio_samples"]

WHISPER_SAMPLE_RATE = 16_000


class TranscriptionFailure(RuntimeError):
    """Transcription produced nothing usable for an episode."""


def load_audio_samples(path: Path, target_sample_rate: int = WHISPER_SAMPLE_RATE):
    """Read a normalized WAV as mono float32 in ``[-1, 1]``.

    ``data/normalized/`` is already 16 kHz mono PCM s16, so this is a plain read
    and a scale in the normal case. Resampling is only reached if the caller
    points this at something else, and it is loud about doing so rather than
    silently changing the time base every timestamp depends on.
    """
    import soundfile

    samples, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        import librosa

        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=target_sample_rate
        )
        sample_rate = target_sample_rate
    if audio.size == 0:
        raise TranscriptionFailure(f"{path} decoded to zero samples")
    return audio, sample_rate


@dataclass
class WhisperTranscriber:
    """A loaded Whisper model plus its decoding settings.

    Constructed once per run and reused across episodes -- loading tiny.en takes
    seconds, which is a large fraction of the cost of transcribing a short
    fixture. Call :meth:`release` before the next stage loads its own model.
    """

    config: TranscriptionConfig
    _model: Any = None
    _processor: Any = None
    _device: str = "cpu"

    def load(self) -> "WhisperTranscriber":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._device = resolve_device(self.config.device)
        torch_dtype = resolve_dtype(self.config.dtype, self._device)
        self._processor = WhisperProcessor.from_pretrained(
            self.config.model_id, revision=self.config.model_revision
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
            dtype=torch_dtype,
        )
        model.eval()
        self._model = model.to(self._device)
        return self

    @property
    def device(self) -> str:
        return self._device

    def release(self) -> None:
        from slotify_rank.embeddings.device import free_model

        model, self._model = self._model, None
        self._processor = None
        if model is not None:
            free_model(model)

    def _require_loaded(self) -> None:
        if self._model is None or self._processor is None:
            raise RuntimeError(
                "WhisperTranscriber.load() must be called before transcribing"
            )

    def _decode_chunk(self, chunk: np.ndarray) -> list[RawSegment]:
        """Decode one window into raw, chunk-relative segments."""
        import torch

        self._require_loaded()
        features = self._processor(
            chunk,
            sampling_rate=WHISPER_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features.input_features.to(
            self._device, dtype=self._model.dtype
        )

        with torch.no_grad():
            generated = self._model.generate(
                input_features,
                num_beams=self.config.num_beams,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
                return_timestamps=True,
            )

        decoded = self._processor.batch_decode(
            generated, skip_special_tokens=False, output_offsets=True
        )
        raw: list[RawSegment] = []
        for item in decoded:
            for offset in item.get("offsets", ()):
                start, end = offset.get("timestamp", (None, None))
                raw.append(
                    RawSegment(
                        start_seconds=start,
                        end_seconds=end,
                        text=str(offset.get("text", "")),
                    )
                )
            if not item.get("offsets"):
                # No timestamp offsets: keep the text anchored to the chunk
                # start rather than discarding real speech.
                text = str(item.get("text", "")).strip()
                if text:
                    raw.append(
                        RawSegment(start_seconds=0.0, end_seconds=None, text=text)
                    )
        return raw

    def transcribe(
        self,
        episode_id: str,
        audio: np.ndarray,
        audio_sha256: str,
        identity_digest: str,
        sample_rate: int = WHISPER_SAMPLE_RATE,
    ) -> EpisodeTranscript:
        """Transcribe a whole episode, chunked and reconciled.

        Raises :class:`TranscriptionFailure` when nothing usable comes back. An
        empty result is never stored as a success: it would mark every candidate
        on the episode "no text" and quietly shrink the dataset.
        """
        self._require_loaded()
        duration_ms = samples_duration_ms(audio.size, sample_rate)
        if duration_ms <= 0:
            raise TranscriptionFailure(f"{episode_id}: audio has zero duration")

        chunks = plan_chunks(
            duration_ms, self.config.chunk_ms, self.config.overlap_ms
        )
        collected: list[
            tuple[int, int, str, tuple, int, float]
        ] = []
        for chunk in chunks:
            first_sample = int(chunk.start_ms * sample_rate / 1000)
            last_sample = int(chunk.end_ms * sample_rate / 1000)
            window = audio[first_sample:last_sample]
            if window.size == 0:
                continue
            raw_segments = self._decode_chunk(window)
            for start_ms, end_ms, text, words in offset_segments(
                raw_segments, chunk.start_ms
            ):
                collected.append(
                    (start_ms, end_ms, text, words, chunk.index, chunk.centre_ms)
                )

        reconciled = reconcile_overlapping_segments(collected)
        segments = build_segments(episode_id, reconciled, duration_ms)
        if not segments:
            raise TranscriptionFailure(
                f"{episode_id}: transcription produced no usable segments over "
                f"{duration_ms} ms of audio across {len(chunks)} chunk(s)"
            )

        return EpisodeTranscript(
            episode_id=episode_id,
            language=self.config.language,
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            audio_sha256=audio_sha256,
            audio_duration_ms=duration_ms,
            segments=tuple(segments),
            transcription_config=self.config.to_dict(),
            identity_digest=identity_digest,
            has_word_timestamps=False,
            transcription_version=TRANSCRIPTION_VERSION,
            created_by_pipeline_version=FEATURE_PIPELINE_VERSION,
        )
