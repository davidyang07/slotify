"""Minimal local labelling service (FastAPI + plain HTML/CSS/JS).

Runs on localhost only. There is no build step, no framework on the client, and
no separate JavaScript application -- one HTML file, one stylesheet, one script.

Bias control is the reason several obvious features are absent by default:

* the heuristic score and the candidate's source flags are **not sent to the
  client** unless ``reveal_hints`` is enabled, because an annotator who knows the
  baseline liked a candidate will rate it higher, and the resulting labels would
  be partly a measurement of the baseline rather than of the audio;
* candidate order is a deterministic shuffle keyed on the annotator, so two
  annotators see different orders (removing position effects from agreement
  figures) while each one's own order is stable across restarts -- which is what
  makes a session resumable.

Audio is served as a context window cut with FFmpeg: ~10 s before and ~10 s
after the candidate, with the boundary at a known offset the client marks. Clips
are cached under ``data/cache/clips/``.

Transcript context is resolved the same way the feature pipeline does it -- the
text either side of the candidate is selected by :func:`build_transcript_context`
from the cached episode transcript under ``data/transcripts/``, so the annotator
reads exactly what the model consumes. Candidates carry no inline transcript
(they are generated before transcription runs), so without this the UI would
always claim "no transcript" even for a fully transcribed episode.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slotify_rank.config.feature_settings import TranscriptContextConfig
from slotify_rank.config.versions import LABEL_RUBRIC_VERSION, PACKAGE_VERSION
from slotify_rank.data.ffmpeg import ffmpeg_path, run_tool
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.features.transcript import build_transcript_context
from slotify_rank.labelling.database import (
    MAX_QUALITY_SCORE,
    MIN_QUALITY_SCORE,
    LabelDatabase,
)
from slotify_rank.transcription.cache import read_transcript, transcript_path

__all__ = ["LabellingSettings", "create_app", "annotator_order_key", "extract_clip"]

_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class LabellingSettings:
    context_before_ms: int = 10_000
    context_after_ms: int = 10_000
    #: Reveal heuristic score and candidate sources to the annotator. Off by
    #: default: see the module docstring.
    reveal_hints: bool = False


def annotator_order_key(annotator_id: str, candidate_id: str) -> str:
    """Deterministic per-annotator shuffle key.

    Stable across restarts (so a session resumes exactly where it stopped) and
    different per annotator (so ordering effects do not correlate between them).
    """
    return hashlib.sha256(
        f"{annotator_id}|{candidate_id}".encode("utf-8")
    ).hexdigest()


def extract_clip(
    normalized_audio: Path,
    destination: Path,
    start_ms: int,
    duration_ms: int,
) -> Path:
    """Cut a context window with FFmpeg, caching the result."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".part.wav")
    run_tool(
        ffmpeg_path(),
        [
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-i",
            str(normalized_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(temp_path),
        ],
        timeout=120.0,
    )
    temp_path.replace(destination)
    return destination


def _candidate_payload(
    candidate: DatasetCandidate,
    episode: EpisodeRecord,
    settings: LabellingSettings,
    existing_score: int | None,
    existing_notes: str | None,
    existing_unusable: bool,
    transcript_before: str | None,
    transcript_after: str | None,
) -> dict[str, Any]:
    start_ms = max(0, candidate.timestamp_ms - settings.context_before_ms)
    end_ms = candidate.timestamp_ms + settings.context_after_ms
    if episode.duration_ms is not None:
        end_ms = min(end_ms, episode.duration_ms)
    payload: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "episode_id": candidate.episode_id,
        "episode_title": episode.title,
        "timestamp_ms": candidate.timestamp_ms,
        "timestamp_label": _format_timestamp(candidate.timestamp_ms),
        "clip_start_ms": start_ms,
        "clip_duration_ms": max(0, end_ms - start_ms),
        "boundary_offset_ms": candidate.timestamp_ms - start_ms,
        "transcript_before": transcript_before,
        "transcript_after": transcript_after,
        "has_transcript": bool(transcript_before or transcript_after),
        "existing_quality_score": existing_score,
        "existing_notes": existing_notes,
        "existing_is_unusable": existing_unusable,
    }
    if settings.reveal_hints:
        payload["heuristic_score"] = candidate.heuristic_score
        payload["candidate_sources"] = list(candidate.candidate_sources)
    return payload


def _format_timestamp(ms: int) -> str:
    total_seconds, milliseconds = divmod(int(ms), 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes:d}:{seconds:02d}.{milliseconds:03d}"


def create_app(
    database: LabelDatabase,
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    paths: DataPaths,
    settings: LabellingSettings | None = None,
):
    """Build the FastAPI application.

    Everything the app serves is passed in, so tests construct it against
    in-memory manifests and a temporary database with no filesystem discovery.
    """
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:  # pragma: no cover - exercised by the CLI path
        raise ImportError(
            "The labelling service needs FastAPI and Uvicorn. Install them with:\n"
            '    .\\.venv\\Scripts\\python.exe -m pip install -e ".[label]"'
        ) from error

    settings = settings or LabellingSettings()
    eligible = [
        candidate
        for candidate in candidates
        if candidate.eligible_for_labelling and not candidate.is_synthetic
    ]
    by_id = {candidate.candidate_id: candidate for candidate in eligible}
    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    database.register_candidates(eligible)

    # Transcript context is resolved from the cached episode transcript with the
    # same selection the feature pipeline uses, so the annotator reads what the
    # model reads. Transcripts are loaded once per episode and cached; an episode
    # with no transcript file resolves to ``None`` and the UI shows its clean
    # "rate from the audio alone" message.
    transcript_context_config = TranscriptContextConfig()
    _transcript_cache: dict[str, Any] = {}

    def _episode_segments(episode_id: str):
        if episode_id not in _transcript_cache:
            path = transcript_path(paths, episode_id)
            try:
                _transcript_cache[episode_id] = (
                    read_transcript(path).segments if path.is_file() else None
                )
            except (ValueError, OSError):
                # A corrupt transcript must not take the whole session down;
                # fall back to audio-only labelling for that episode.
                _transcript_cache[episode_id] = None
        return _transcript_cache[episode_id]

    def _resolve_transcript(
        candidate: DatasetCandidate,
    ) -> tuple[str | None, str | None]:
        # An inline transcript on the candidate wins (future-proofing for
        # transcript-aware candidate generation); otherwise resolve from cache.
        if candidate.transcript_before or candidate.transcript_after:
            return candidate.transcript_before, candidate.transcript_after
        segments = _episode_segments(candidate.episode_id)
        if not segments:
            return None, None
        context = build_transcript_context(
            segments, candidate.timestamp_ms, transcript_context_config
        )
        return (context.before_text or None, context.after_text or None)

    app = FastAPI(title="Slotify labelling", version=PACKAGE_VERSION)
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    def _ordered_for(annotator_id: str) -> list[DatasetCandidate]:
        return sorted(
            eligible,
            key=lambda candidate: annotator_order_key(
                annotator_id, candidate.candidate_id
            ),
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "package_version": PACKAGE_VERSION,
            "rubric_version": LABEL_RUBRIC_VERSION,
            "eligible_candidates": len(eligible),
            "episodes": len(episodes_by_id),
            "acceptable_threshold": database.acceptable_threshold,
            "reveal_hints": settings.reveal_hints,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        template = _TEMPLATE_DIR / "index.html"
        if not template.is_file():  # pragma: no cover - packaging error
            raise HTTPException(status_code=500, detail="index.html is missing")
        return HTMLResponse(template.read_text(encoding="utf-8"))

    @app.get("/api/progress")
    def progress(annotator_id: str) -> dict[str, Any]:
        if not annotator_id.strip():
            raise HTTPException(status_code=422, detail="annotator_id is required")
        stats = database.progress(annotator_id)
        stats["total_candidates"] = len(eligible)
        stats["remaining"] = len(eligible) - stats["labelled"]
        stats["rubric_version"] = LABEL_RUBRIC_VERSION
        return stats

    @app.get("/api/next")
    def next_candidate(annotator_id: str, include_labelled: bool = False):
        """The annotator's next unlabelled candidate, in their stable order."""
        if not annotator_id.strip():
            raise HTTPException(status_code=422, detail="annotator_id is required")
        done = database.labelled_candidate_ids(annotator_id)
        for candidate in _ordered_for(annotator_id):
            if candidate.candidate_id in done and not include_labelled:
                continue
            episode = episodes_by_id.get(candidate.episode_id)
            if episode is None:
                continue
            existing = database.get_label(candidate.candidate_id, annotator_id)
            transcript_before, transcript_after = _resolve_transcript(candidate)
            return {
                "candidate": _candidate_payload(
                    candidate,
                    episode,
                    settings,
                    existing.quality_score if existing else None,
                    existing.notes if existing else None,
                    bool(existing.is_unusable) if existing else False,
                    transcript_before,
                    transcript_after,
                ),
                "progress": database.progress(annotator_id) | {
                    "total_candidates": len(eligible),
                    "remaining": len(eligible)
                    - database.progress(annotator_id)["labelled"],
                },
            }
        return JSONResponse(
            {"candidate": None, "message": "All candidates have been labelled."},
            status_code=200,
        )

    @app.get("/api/clip/{candidate_id}")
    def clip(candidate_id: str):
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Unknown candidate")
        episode = episodes_by_id.get(candidate.episode_id)
        if episode is None or not episode.normalized_path:
            raise HTTPException(
                status_code=409,
                detail="Episode has no normalized audio; run `dataset normalize`",
            )
        source = paths.absolute(episode.normalized_path)
        if not source.is_file():
            raise HTTPException(status_code=410, detail="Normalized audio is missing")
        start_ms = max(0, candidate.timestamp_ms - settings.context_before_ms)
        end_ms = candidate.timestamp_ms + settings.context_after_ms
        if episode.duration_ms is not None:
            end_ms = min(end_ms, episode.duration_ms)
        duration_ms = max(0, end_ms - start_ms)
        if duration_ms <= 0:
            raise HTTPException(status_code=409, detail="Empty context window")
        destination = (
            paths.clips_dir
            / f"{candidate_id.replace(':', '_')}_{start_ms}_{duration_ms}.wav"
        )
        extract_clip(source, destination, start_ms, duration_ms)
        return FileResponse(
            str(destination), media_type="audio/wav", filename=destination.name
        )

    # The body is taken as a plain mapping and validated here rather than through
    # a Pydantic model: this module uses `from __future__ import annotations`, and
    # a model class defined inside this factory is not resolvable from the
    # module globals FastAPI inspects, so it would be silently demoted to a query
    # parameter. Validation is the same either way -- `upsert_label` is the real
    # gate on score range, annotator and candidate existence.
    @app.post("/api/label")
    def save_label(payload: dict = Body(...)) -> dict[str, Any]:
        candidate_id = str(payload.get("candidate_id") or "")
        if candidate_id not in by_id:
            raise HTTPException(status_code=404, detail="Unknown candidate")
        score = payload.get("quality_score")
        if not isinstance(score, int) or isinstance(score, bool):
            raise HTTPException(
                status_code=422,
                detail=f"quality_score must be an integer, got {score!r}",
            )
        if not MIN_QUALITY_SCORE <= score <= MAX_QUALITY_SCORE:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"quality_score must be between {MIN_QUALITY_SCORE} and "
                    f"{MAX_QUALITY_SCORE}, got {score}"
                ),
            )
        annotator_id = str(payload.get("annotator_id") or "")
        notes = payload.get("notes")
        try:
            record = database.upsert_label(
                candidate_id=candidate_id,
                annotator_id=annotator_id,
                quality_score=score,
                is_unusable=bool(payload.get("is_unusable", False)),
                notes=None if notes is None else str(notes),
            )
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        stats = database.progress(annotator_id)
        stats["total_candidates"] = len(eligible)
        stats["remaining"] = len(eligible) - stats["labelled"]
        return {"label": record.to_dict(), "progress": stats}

    return app
