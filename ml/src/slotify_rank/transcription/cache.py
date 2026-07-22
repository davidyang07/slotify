"""Reading and writing transcripts, with cache identity enforced on load.

The load path is the interesting half. A transcript on disk is only reusable if
it was produced from the same audio, by the same model, under the same
transcription settings -- so :func:`load_transcript` takes the identity the
caller *expects* and refuses to return a transcript that does not match it.
Returning a stale transcript would be worse than returning nothing: it would
produce embeddings that silently describe the wrong audio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.paths import DataPaths
from slotify_rank.pipeline.identity import CacheIdentity
from slotify_rank.transcription.schema import EpisodeTranscript

__all__ = [
    "transcript_path",
    "write_transcript",
    "read_transcript",
    "load_transcript",
    "transcript_identity",
]


def transcript_path(paths: DataPaths, episode_id: str) -> Path:
    return paths.transcripts_dir / f"{episode_id}.transcript.json"


def transcript_identity(
    *,
    episode_id: str,
    normalized_audio_sha256: str,
    model_id: str,
    model_revision: str,
    transcription_config_digest: str,
    transcription_version: str,
    feature_pipeline_version: str,
    library_versions: Mapping[str, str],
) -> CacheIdentity:
    """Everything that determines a transcript's bytes.

    ``normalized_audio_sha256`` rather than the source checksum: the model reads
    the 16 kHz render, so a re-normalization must invalidate the transcript even
    when the original file is untouched.
    """
    return CacheIdentity(
        kind="transcript",
        inputs={
            "episode_id": episode_id,
            "normalized_audio_sha256": normalized_audio_sha256,
            "model_id": model_id,
            "model_revision": model_revision,
            "transcription_config_sha256": transcription_config_digest,
            "transcription_version": transcription_version,
            "feature_pipeline_version": feature_pipeline_version,
            "library_versions": dict(library_versions),
        },
    )


def write_transcript(path: Path, transcript: EpisodeTranscript) -> None:
    payload = json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False) + "\n"
    atomic_write_bytes(Path(path), payload.encode("utf-8"))


def read_transcript(path: Path) -> EpisodeTranscript:
    """Parse a transcript, with no cache-validity judgement."""
    file_path = Path(path)
    try:
        raw: Any = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Transcript not found: {file_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{file_path} is not valid JSON: {error}") from error
    return EpisodeTranscript.from_mapping(raw)


def load_transcript(
    path: Path, expected: CacheIdentity | None = None
) -> EpisodeTranscript | None:
    """Return a cached transcript, or ``None`` when it is absent or stale.

    ``None`` deliberately conflates "never transcribed" and "transcribed under
    different settings" *for the caller's purposes* -- both mean "do the work".
    The distinction is preserved where it matters, in the stage ledger, which is
    what the status report reads.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return None
    transcript = read_transcript(file_path)
    if expected is None:
        return transcript
    if transcript.identity_digest != expected.digest:
        return None
    return transcript
