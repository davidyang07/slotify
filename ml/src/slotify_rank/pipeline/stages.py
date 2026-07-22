"""The five Phase 3 stages, each independently runnable and resumable.

Every stage follows the same contract:

1. Compute the cache identity for this episode under the current configuration.
2. Ask the ledger whether the work is already ``complete`` under *that* identity.
3. If not, mark ``partial``, do the work, write atomically, mark ``complete``.
4. On failure, record ``failed`` with the reason and carry on to the next
   episode -- one unreadable file must not abandon a corpus that took hours.

Stages are separate processes' worth of work but share one ledger directory, so
``pipeline status`` can report the whole corpus without loading a model.

**Model lifetime.** Each stage loads its model once, processes every episode,
then releases it before the next stage loads its own. Whisper and MiniLM
resident simultaneously is the difference between fitting in laptop RAM and
swapping, and the sequence here is what keeps peak memory to one model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from slotify_rank.config.feature_settings import (
    EmbeddingConfig,
    FeatureConfig,
    PipelineConfig,
    TranscriptionConfig,
)
from slotify_rank.config.versions import (
    CANDIDATE_GENERATION_VERSION,
    FEATURE_PIPELINE_VERSION,
    FEATURE_SPEC_VERSION,
    TRANSCRIPTION_VERSION,
)
from slotify_rank.data.audio_io import samples_duration_ms
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.embeddings.store import (
    EmbeddingMetadata,
    load_embeddings,
    write_embeddings,
)
from slotify_rank.pipeline.identity import CacheIdentity, library_versions
from slotify_rank.pipeline.state import StageLedger
from slotify_rank.transcription.cache import (
    load_transcript,
    transcript_identity,
    transcript_path,
    write_transcript,
)

__all__ = [
    "StageContext",
    "StageOutcome",
    "run_transcribe",
    "run_acoustic",
    "run_audio_embeddings",
    "run_text_embeddings",
    "eligible_candidates",
    "select_episodes",
]

#: Libraries whose version can change an artifact's bytes. Recorded in every
#: cache identity so a library upgrade invalidates what it could have altered.
_AUDIO_LIBRARIES = ("torch", "transformers")
_TEXT_LIBRARIES = ("torch", "transformers", "sentence-transformers")
_ACOUSTIC_LIBRARIES = ("librosa", "numpy", "scipy")


@dataclass
class StageOutcome:
    """What one stage did, for the statistics report."""

    stage: str
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "seconds": round(self.seconds, 3),
            **({"details": self.details} if self.details else {}),
        }


@dataclass
class StageContext:
    """Everything the stages need, resolved once."""

    paths: DataPaths
    transcription: TranscriptionConfig
    features: FeatureConfig
    embeddings: EmbeddingConfig
    pipeline: PipelineConfig
    force: bool = False
    retry_failed: bool = False
    #: Called with a human-readable progress line. Defaults to printing.
    log: Callable[[str], None] = print


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_episodes(
    episodes: Sequence[EpisodeRecord],
    episode_ids: Sequence[str] | None = None,
    content_types: Sequence[str] | None = None,
    limit: int | None = None,
    splits: Sequence[str] | None = None,
    split_lookup: Mapping[str, str] | None = None,
) -> list[EpisodeRecord]:
    """Filter the corpus, in deterministic manifest order.

    Only ``normalized`` episodes are eligible: every Phase 3 stage reads the
    16 kHz render, and an episode that has not reached that state has nothing to
    process.
    """
    selected = [episode for episode in episodes if episode.status == "normalized"]

    if episode_ids:
        wanted = set(episode_ids)
        known = {episode.episode_id for episode in selected}
        missing = sorted(wanted - known)
        if missing:
            raise KeyError(
                f"Unknown or un-normalized episode_id(s): {missing}. Run "
                "`dataset normalize` first."
            )
        selected = [e for e in selected if e.episode_id in wanted]

    if content_types:
        allowed = set(content_types)
        selected = [e for e in selected if e.content_type in allowed]

    if splits:
        if split_lookup is None:
            raise ValueError(
                "Filtering by split requires a split manifest; run `dataset split`."
            )
        allowed_splits = set(splits)
        selected = [
            e
            for e in selected
            if split_lookup.get(e.episode_id, "unassigned") in allowed_splits
        ]

    selected.sort(key=lambda episode: episode.episode_id)
    return selected[:limit] if limit else selected


def eligible_candidates(
    candidates: Iterable[DatasetCandidate], episode_id: str
) -> list[DatasetCandidate]:
    """Candidates for one episode that may receive features.

    Synthetic product padding is excluded here and nowhere else, so there is a
    single place to audit. Those candidates exist in the manifest for
    traceability and must never enter a feature count.
    """
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.episode_id == episode_id and not candidate.is_synthetic
        ),
        key=lambda candidate: (candidate.timestamp_ms, candidate.candidate_id),
    )


def _normalized_audio(paths: DataPaths, episode: EpisodeRecord) -> Path:
    if not episode.normalized_path:
        raise FileNotFoundError(
            f"{episode.episode_id} has no normalized audio; run `dataset normalize`."
        )
    return paths.absolute(episode.normalized_path)


def _audio_checksum(episode: EpisodeRecord) -> str:
    if not episode.normalized_sha256:
        raise ValueError(
            f"{episode.episode_id} has no normalized_sha256; the cache identity "
            "cannot be computed without it. Re-run `dataset normalize`."
        )
    return episode.normalized_sha256


# ---------------------------------------------------------------------------
# Stage 1: transcription
# ---------------------------------------------------------------------------


def _transcription_identity(
    context: StageContext, episode: EpisodeRecord
) -> CacheIdentity:
    return transcript_identity(
        episode_id=episode.episode_id,
        normalized_audio_sha256=_audio_checksum(episode),
        model_id=context.transcription.model_id,
        model_revision=context.transcription.model_revision,
        transcription_config_digest=context.transcription.digest,
        transcription_version=TRANSCRIPTION_VERSION,
        feature_pipeline_version=FEATURE_PIPELINE_VERSION,
        library_versions=library_versions(*_AUDIO_LIBRARIES),
    )


def run_transcribe(
    context: StageContext, episodes: Sequence[EpisodeRecord]
) -> tuple[StageOutcome, StageLedger]:
    """Transcribe every selected episode, skipping unchanged completed work."""
    from slotify_rank.transcription.local_whisper import (
        TranscriptionFailure,
        WhisperTranscriber,
        load_audio_samples,
    )

    outcome = StageOutcome(stage="transcribe")
    ledger = StageLedger.read(context.paths.stage_ledger("transcribe"), "transcribe")
    started = time.perf_counter()

    pending: list[tuple[EpisodeRecord, CacheIdentity]] = []
    for episode in episodes:
        identity = _transcription_identity(context, episode)
        needs = context.force or ledger.needs_work(
            episode.episode_id, identity.digest, context.retry_failed
        )
        if not needs:
            outcome.skipped += 1
            outcome.cache_hits += 1
            continue
        pending.append((episode, identity))

    if not pending:
        outcome.seconds = time.perf_counter() - started
        return outcome, ledger

    transcriber = WhisperTranscriber(config=context.transcription).load()
    context.log(
        f"  transcribe: {context.transcription.model_id} on {transcriber.device}, "
        f"{len(pending)} episode(s)"
    )
    try:
        for episode, identity in pending:
            outcome.cache_misses += 1
            ledger.mark_partial(episode.episode_id)
            ledger.write(context.paths.stage_ledger("transcribe"))
            try:
                audio, sample_rate = load_audio_samples(
                    _normalized_audio(context.paths, episode)
                )
                transcript = transcriber.transcribe(
                    episode_id=episode.episode_id,
                    audio=audio,
                    audio_sha256=_audio_checksum(episode),
                    identity_digest=identity.digest,
                    sample_rate=sample_rate,
                )
                write_transcript(
                    transcript_path(context.paths, episode.episode_id), transcript
                )
                ledger.mark_complete(
                    episode.episode_id,
                    identity.digest,
                    segments=transcript.segment_count,
                    transcribed_ms=transcript.transcribed_duration_ms,
                    audio_duration_ms=transcript.audio_duration_ms,
                )
                outcome.processed += 1
                context.log(
                    f"    {episode.episode_id}: {transcript.segment_count} segment(s)"
                )
            except (TranscriptionFailure, ValueError, OSError, RuntimeError) as error:
                ledger.mark_failed(episode.episode_id, str(error))
                outcome.failed += 1
                context.log(f"    {episode.episode_id}: FAILED -- {error}")
                if context.pipeline.fail_fast:
                    raise
            finally:
                ledger.write(context.paths.stage_ledger("transcribe"))
    finally:
        transcriber.release()

    outcome.seconds = time.perf_counter() - started
    return outcome, ledger


# ---------------------------------------------------------------------------
# Stage 2: handcrafted features
# ---------------------------------------------------------------------------


def handcrafted_path(paths: DataPaths, episode_id: str) -> Path:
    return paths.handcrafted_dir / f"{episode_id}.handcrafted.json"


def _acoustic_identity(
    context: StageContext, episode: EpisodeRecord, transcript_digest: str
) -> CacheIdentity:
    return CacheIdentity(
        kind="handcrafted_features",
        inputs={
            "episode_id": episode.episode_id,
            "normalized_audio_sha256": _audio_checksum(episode),
            "candidate_generation_version": CANDIDATE_GENERATION_VERSION,
            "feature_config_sha256": context.features.digest,
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "transcript_identity": transcript_digest,
            "library_versions": library_versions(*_ACOUSTIC_LIBRARIES),
        },
    )


def run_acoustic(
    context: StageContext,
    episodes: Sequence[EpisodeRecord],
    candidates_by_episode: Mapping[str, Sequence[DatasetCandidate]],
    edge_guard_ms: int = 5_000,
) -> tuple[StageOutcome, StageLedger]:
    """Extract acoustic, structural and text-scalar features per candidate.

    Transcript context text is extracted here too and stored alongside, so the
    text-embedding stage encodes exactly the strings these features describe
    rather than re-deriving them and risking a mismatch.
    """
    from slotify_rank.data.audio_io import load_normalized_wav
    from slotify_rank.features.acoustic import compute_frames, extract_acoustic_features
    from slotify_rank.features.structural import extract_structural_features
    from slotify_rank.features.transcript import (
        build_transcript_context,
        extract_text_features,
    )
    from slotify_rank.features.windows import build_windows
    from slotify_rank.transcription.local_whisper import load_audio_samples

    outcome = StageOutcome(stage="acoustic")
    ledger = StageLedger.read(context.paths.stage_ledger("acoustic"), "acoustic")
    started = time.perf_counter()
    scales = context.features.windows.as_pairs()

    for episode in episodes:
        transcript = load_transcript(
            transcript_path(context.paths, episode.episode_id)
        )
        transcript_digest = transcript.identity_digest if transcript else "absent"
        identity = _acoustic_identity(context, episode, transcript_digest)

        if not context.force and not ledger.needs_work(
            episode.episode_id, identity.digest, context.retry_failed
        ):
            outcome.skipped += 1
            outcome.cache_hits += 1
            continue

        outcome.cache_misses += 1
        ledger.mark_partial(episode.episode_id)
        ledger.write(context.paths.stage_ledger("acoustic"))
        try:
            candidates = eligible_candidates(
                candidates_by_episode.get(episode.episode_id, ()), episode.episode_id
            )
            if not candidates:
                ledger.mark_complete(episode.episode_id, identity.digest, candidates=0)
                outcome.processed += 1
                continue

            audio_path = _normalized_audio(context.paths, episode)
            samples, sample_rate = load_audio_samples(audio_path)
            envelope = load_normalized_wav(audio_path)
            duration_ms = samples_duration_ms(samples.size, sample_rate)
            frames = compute_frames(samples, sample_rate, context.features.acoustic)
            segments = transcript.segments if transcript else ()

            records: dict[str, Any] = {}
            for candidate in candidates:
                if not 0 <= candidate.timestamp_ms <= duration_ms:
                    raise ValueError(
                        f"{candidate.candidate_id}: timestamp "
                        f"{candidate.timestamp_ms} ms is outside the episode "
                        f"[0, {duration_ms}] ms"
                    )
                windows = build_windows(candidate.timestamp_ms, duration_ms, scales)
                values, missing = extract_acoustic_features(frames, windows, envelope)

                structural_values, structural_missing = extract_structural_features(
                    candidate, duration_ms, edge_guard_ms
                )
                values.update(structural_values)
                missing.update(structural_missing)

                text_context = build_transcript_context(
                    segments, candidate.timestamp_ms, context.features.transcript_context
                )
                text_values, text_missing = extract_text_features(
                    text_context, transcript_available=transcript is not None
                )
                values.update(text_values)
                missing.update(text_missing)

                records[candidate.candidate_id] = {
                    "timestamp_ms": candidate.timestamp_ms,
                    "values": values,
                    "missing": missing,
                    "context_before_text": text_context.before_text,
                    "context_after_text": text_context.after_text,
                    "context_segment_ids_before": list(
                        text_context.before_segment_ids
                    ),
                    "context_segment_ids_after": list(text_context.after_segment_ids),
                }

            payload = {
                "episode_id": episode.episode_id,
                "identity_digest": identity.digest,
                "feature_spec_version": FEATURE_SPEC_VERSION,
                "audio_sha256": _audio_checksum(episode),
                "transcript_available": transcript is not None,
                "episode_duration_ms": duration_ms,
                "candidates": records,
            }
            atomic_write_bytes(
                handcrafted_path(context.paths, episode.episode_id),
                (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                ),
            )
            ledger.mark_complete(
                episode.episode_id, identity.digest, candidates=len(records)
            )
            outcome.processed += 1
            context.log(f"    {episode.episode_id}: {len(records)} candidate(s)")
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            ledger.mark_failed(episode.episode_id, str(error))
            outcome.failed += 1
            context.log(f"    {episode.episode_id}: FAILED -- {error}")
            if context.pipeline.fail_fast:
                raise
        finally:
            ledger.write(context.paths.stage_ledger("acoustic"))

    outcome.seconds = time.perf_counter() - started
    return outcome, ledger


def read_handcrafted(paths: DataPaths, episode_id: str) -> dict[str, Any] | None:
    path = handcrafted_path(paths, episode_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage 3: audio embeddings
# ---------------------------------------------------------------------------


def audio_embedding_path(paths: DataPaths, episode_id: str) -> Path:
    return paths.audio_embeddings_dir / f"{episode_id}.audio.npy"


def audio_row_id(candidate_id: str, kind: str) -> str:
    """One row per (candidate, pooled-window kind)."""
    return f"{candidate_id}#{kind}"


def _audio_identity(context: StageContext, episode: EpisodeRecord) -> CacheIdentity:
    audio = context.embeddings.audio
    return CacheIdentity(
        kind="audio_embedding",
        inputs={
            "episode_id": episode.episode_id,
            "normalized_audio_sha256": _audio_checksum(episode),
            "candidate_generation_version": CANDIDATE_GENERATION_VERSION,
            "model_id": audio.model_id,
            "model_revision": audio.model_revision,
            "model_config_sha256": audio.digest,
            "window_configuration": context.features.windows.to_dict(),
            "pooling_configuration": {
                "pooling": audio.pooling,
                "chunk_ms": audio.chunk_ms,
                "overlap_ms": audio.overlap_ms,
                "include_difference": audio.include_difference,
            },
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "library_versions": library_versions(*_AUDIO_LIBRARIES),
        },
    )


def run_audio_embeddings(
    context: StageContext,
    episodes: Sequence[EpisodeRecord],
    candidates_by_episode: Mapping[str, Sequence[DatasetCandidate]],
) -> tuple[StageOutcome, StageLedger]:
    """Encode each episode once and pool every candidate's windows out of it."""
    from slotify_rank.embeddings.whisper_audio import (
        CANDIDATE_EMBEDDING_KINDS,
        WhisperAudioEncoder,
        pool_candidate_embeddings,
    )
    from slotify_rank.features.windows import build_windows
    from slotify_rank.transcription.local_whisper import load_audio_samples

    outcome = StageOutcome(stage="audio_embedding")
    ledger = StageLedger.read(
        context.paths.stage_ledger("audio_embedding"), "audio_embedding"
    )
    started = time.perf_counter()
    scales = context.features.windows.as_pairs()
    audio_config = context.embeddings.audio

    pending: list[tuple[EpisodeRecord, CacheIdentity]] = []
    for episode in episodes:
        identity = _audio_identity(context, episode)
        if not context.force and not ledger.needs_work(
            episode.episode_id, identity.digest, context.retry_failed
        ):
            outcome.skipped += 1
            outcome.cache_hits += 1
            continue
        pending.append((episode, identity))

    if not pending:
        outcome.seconds = time.perf_counter() - started
        return outcome, ledger

    encoder = WhisperAudioEncoder(config=audio_config).load()
    context.log(
        f"  audio embeddings: {audio_config.model_id} on {encoder.device}, "
        f"dimension {encoder.dimension}, {len(pending)} episode(s)"
    )
    outcome.details["dimension"] = encoder.dimension
    outcome.details["device"] = encoder.device
    try:
        for episode, identity in pending:
            outcome.cache_misses += 1
            ledger.mark_partial(episode.episode_id)
            ledger.write(context.paths.stage_ledger("audio_embedding"))
            try:
                candidates = eligible_candidates(
                    candidates_by_episode.get(episode.episode_id, ()),
                    episode.episode_id,
                )
                if not candidates:
                    ledger.mark_complete(
                        episode.episode_id, identity.digest, rows=0
                    )
                    outcome.processed += 1
                    continue

                samples, sample_rate = load_audio_samples(
                    _normalized_audio(context.paths, episode)
                )
                duration_ms = samples_duration_ms(samples.size, sample_rate)
                states = encoder.encode_episode(
                    episode.episode_id, samples, sample_rate
                )
                outcome.details["frame_duration_ms"] = states.frame_duration_ms

                rows: list[np.ndarray] = []
                row_ids: list[str] = []
                for candidate in candidates:
                    windows = build_windows(
                        candidate.timestamp_ms, duration_ms, scales
                    )
                    pooled = pool_candidate_embeddings(
                        states, windows, audio_config.include_difference
                    )
                    for kind in CANDIDATE_EMBEDDING_KINDS:
                        vector = pooled.get(kind)
                        if vector is None:
                            # Missing modality: no row is written at all, so the
                            # absence is visible rather than encoded as zeros.
                            continue
                        rows.append(vector)
                        row_ids.append(audio_row_id(candidate.candidate_id, kind))

                matrix = (
                    np.vstack(rows)
                    if rows
                    else np.zeros((0, states.dimension), dtype=np.float32)
                )
                write_embeddings(
                    audio_embedding_path(context.paths, episode.episode_id),
                    matrix,
                    EmbeddingMetadata(
                        kind="audio_embedding",
                        episode_id=episode.episode_id,
                        model_id=audio_config.model_id,
                        model_revision=audio_config.model_revision,
                        dimension=states.dimension,
                        row_ids=tuple(row_ids),
                        identity_digest=identity.digest,
                        extra={
                            "frame_duration_ms": states.frame_duration_ms,
                            "pooling": audio_config.pooling,
                            "kinds": list(CANDIDATE_EMBEDDING_KINDS),
                        },
                    ),
                )
                ledger.mark_complete(
                    episode.episode_id,
                    identity.digest,
                    rows=len(row_ids),
                    dimension=states.dimension,
                )
                outcome.processed += 1
                context.log(
                    f"    {episode.episode_id}: {len(row_ids)} row(s) x "
                    f"{states.dimension}"
                )
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                ledger.mark_failed(episode.episode_id, str(error))
                outcome.failed += 1
                context.log(f"    {episode.episode_id}: FAILED -- {error}")
                if context.pipeline.fail_fast:
                    raise
            finally:
                ledger.write(context.paths.stage_ledger("audio_embedding"))
    finally:
        encoder.release()

    outcome.seconds = time.perf_counter() - started
    return outcome, ledger


# ---------------------------------------------------------------------------
# Stage 4: text embeddings
# ---------------------------------------------------------------------------


def text_embedding_path(paths: DataPaths, episode_id: str) -> Path:
    return paths.text_embeddings_dir / f"{episode_id}.text.npy"


def text_row_id(candidate_id: str, side: str) -> str:
    return f"{candidate_id}#{side}"


def _text_identity(
    context: StageContext, episode: EpisodeRecord, handcrafted_digest: str
) -> CacheIdentity:
    text = context.embeddings.text
    return CacheIdentity(
        kind="text_embedding",
        inputs={
            "episode_id": episode.episode_id,
            "normalized_audio_sha256": _audio_checksum(episode),
            "model_id": text.model_id,
            "model_revision": text.model_revision,
            "model_config_sha256": text.digest,
            "transcript_context_configuration": (
                context.features.transcript_context.to_dict()
            ),
            # The context strings come from the handcrafted stage, so its
            # identity is an input here: re-deriving context must re-embed it.
            "handcrafted_identity": handcrafted_digest,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "library_versions": library_versions(*_TEXT_LIBRARIES),
        },
    )


def run_text_embeddings(
    context: StageContext, episodes: Sequence[EpisodeRecord]
) -> tuple[StageOutcome, StageLedger]:
    """Embed the stored transcript context either side of each candidate."""
    from slotify_rank.embeddings.minilm_text import MiniLmTextEncoder

    outcome = StageOutcome(stage="text_embedding")
    ledger = StageLedger.read(
        context.paths.stage_ledger("text_embedding"), "text_embedding"
    )
    started = time.perf_counter()
    text_config = context.embeddings.text

    pending: list[tuple[EpisodeRecord, CacheIdentity, dict[str, Any]]] = []
    for episode in episodes:
        handcrafted = read_handcrafted(context.paths, episode.episode_id)
        if handcrafted is None:
            # Nothing to embed yet; the acoustic stage has not run. Not a
            # failure -- the pipeline runs the stages in order.
            continue
        identity = _text_identity(
            context, episode, str(handcrafted.get("identity_digest", ""))
        )
        if not context.force and not ledger.needs_work(
            episode.episode_id, identity.digest, context.retry_failed
        ):
            outcome.skipped += 1
            outcome.cache_hits += 1
            continue
        pending.append((episode, identity, handcrafted))

    if not pending:
        outcome.seconds = time.perf_counter() - started
        return outcome, ledger

    encoder = MiniLmTextEncoder(config=text_config).load()
    context.log(
        f"  text embeddings: {text_config.model_id} on {encoder.device}, "
        f"dimension {encoder.dimension}, {len(pending)} episode(s)"
    )
    outcome.details["dimension"] = encoder.dimension
    outcome.details["device"] = encoder.device
    try:
        for episode, identity, handcrafted in pending:
            outcome.cache_misses += 1
            ledger.mark_partial(episode.episode_id)
            ledger.write(context.paths.stage_ledger("text_embedding"))
            try:
                texts: list[str] = []
                row_ids: list[str] = []
                for candidate_id, record in sorted(
                    handcrafted.get("candidates", {}).items()
                ):
                    for side in ("before", "after"):
                        text = str(record.get(f"context_{side}_text", "")).strip()
                        if not text:
                            # Empty context: no row, so "no text" stays
                            # distinguishable from "an embedding of nothing".
                            continue
                        texts.append(text)
                        row_ids.append(text_row_id(candidate_id, side))

                matrix = encoder.encode(texts)
                write_embeddings(
                    text_embedding_path(context.paths, episode.episode_id),
                    matrix,
                    EmbeddingMetadata(
                        kind="text_embedding",
                        episode_id=episode.episode_id,
                        model_id=text_config.model_id,
                        model_revision=text_config.model_revision,
                        dimension=encoder.dimension,
                        row_ids=tuple(row_ids),
                        identity_digest=identity.digest,
                        extra={
                            "normalize_embeddings": text_config.normalize_embeddings,
                            "max_seq_length": text_config.max_seq_length,
                        },
                    ),
                )
                ledger.mark_complete(
                    episode.episode_id,
                    identity.digest,
                    rows=len(row_ids),
                    dimension=encoder.dimension,
                )
                outcome.processed += 1
                context.log(
                    f"    {episode.episode_id}: {len(row_ids)} row(s) x "
                    f"{encoder.dimension}"
                )
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                ledger.mark_failed(episode.episode_id, str(error))
                outcome.failed += 1
                context.log(f"    {episode.episode_id}: FAILED -- {error}")
                if context.pipeline.fail_fast:
                    raise
            finally:
                ledger.write(context.paths.stage_ledger("text_embedding"))
    finally:
        encoder.release()

    outcome.seconds = time.perf_counter() - started
    return outcome, ledger
