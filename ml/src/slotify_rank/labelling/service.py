"""Local labelling service (FastAPI + plain HTML/CSS/JS).

Runs on localhost only. There is no build step, no framework on the client, and
no separate JavaScript application -- one HTML file, one stylesheet, one script.

This app exists to make one specific thing fast: a person judging a few thousand
short audio clips. Everything in it is downstream of that.

**Throughput.** Rating an item is one keystroke, and the keystroke both saves and
advances -- there is no confirm step. The next several items' audio and
transcript are already in the browser before the current one is judged
(``/api/batch``), so the wait between items is a repaint rather than a round
trip plus an FFmpeg cut. ``prerender_clips`` cuts every clip up front so even the
first pass never blocks on FFmpeg.

**Bias control.** The heuristic score and the candidate's source flags are *not
sent to the client* unless ``reveal_hints`` is enabled. An annotator who knows
the baseline liked a candidate rates it higher, and the resulting labels would be
partly a measurement of the baseline rather than of the audio -- which would make
the headline comparison circular.

**Order.** When a queue artifact is supplied, the queue decides which candidates
are shown and in what stage; it is stratified by split and heuristic-score
tertile and spread across episodes, so the labelled set is diverse by
construction rather than by luck (see :mod:`slotify_rank.labelling.queue`). The
per-annotator order within it is a deterministic shuffle keyed on the annotator,
so two annotators see different orders (removing position effects from agreement
figures) while each one's own order is stable across restarts -- which is what
makes a session resumable. Consistency repeats are held back until the annotator
is far enough in that they will not remember the first showing, and they are
indistinguishable from a first showing in the payload.

**Resumption.** Nothing is buffered client-side. A judgement is a POST that
either commits or does not, and "where was I" is recomputed from the database on
every request, so closing the tab loses at most the item on screen.

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
from typing import Any, Callable, Mapping, Sequence

from slotify_rank.config.feature_settings import TranscriptContextConfig
from slotify_rank.config.versions import (
    LABEL_RUBRIC_VERSION,
    LABEL_SCHEMA_VERSION,
    PACKAGE_VERSION,
)
from slotify_rank.data.ffmpeg import ffmpeg_path, run_tool
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.features.transcript import build_transcript_context
from slotify_rank.labelling.database import (
    MAX_QUALITY_SCORE,
    MIN_QUALITY_SCORE,
    LabelDatabase,
)
from slotify_rank.labelling.queue import LabellingQueue, QueuePresentation
from slotify_rank.transcription.cache import read_transcript, transcript_path

__all__ = [
    "LabellingSettings",
    "create_app",
    "annotator_order_key",
    "extract_clip",
    "clip_path_for",
    "prerender_clips",
    "build_plan",
    "PlannedItem",
]

_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class LabellingSettings:
    context_before_ms: int = 10_000
    context_after_ms: int = 10_000
    #: Reveal heuristic score and candidate sources to the annotator. Off by
    #: default: see the module docstring.
    reveal_hints: bool = False
    #: How many items the client is given at once so it can prefetch their audio.
    batch_size: int = 6
    #: The number of unique candidates this round is aiming for. Drives the
    #: "1234 / 2400" display; ``None`` means "every eligible candidate".
    target_unique: int | None = None
    #: A consistency repeat is withheld until this many first showings have been
    #: judged, so the repeat measures memory-free agreement rather than recall.
    repeat_delay_items: int = 150


def annotator_order_key(annotator_id: str, presentation_id: str) -> str:
    """Deterministic per-annotator shuffle key.

    Stable across restarts (so a session resumes exactly where it stopped) and
    different per annotator (so ordering effects do not correlate between them).
    """
    return hashlib.sha256(
        f"{annotator_id}|{presentation_id}".encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Clip extraction
# ---------------------------------------------------------------------------


def clip_window(
    candidate: DatasetCandidate,
    episode: EpisodeRecord,
    settings: LabellingSettings,
) -> tuple[int, int]:
    """``(start_ms, duration_ms)`` of the context window for this candidate."""
    start_ms = max(0, candidate.timestamp_ms - settings.context_before_ms)
    end_ms = candidate.timestamp_ms + settings.context_after_ms
    if episode.duration_ms is not None:
        end_ms = min(end_ms, episode.duration_ms)
    return start_ms, max(0, end_ms - start_ms)


def clip_path_for(
    paths: DataPaths, candidate_id: str, start_ms: int, duration_ms: int
) -> Path:
    return paths.clips_dir / f"{candidate_id.replace(':', '_')}_{start_ms}_{duration_ms}.wav"


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


def prerender_clips(
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    paths: DataPaths,
    settings: LabellingSettings | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Cut every candidate's clip ahead of the session.

    Called by ``label resume-experiment`` so the annotator never waits on FFmpeg.
    Already-cut clips are left alone, so this is cheap to re-run and safe to
    interrupt.
    """
    settings = settings or LabellingSettings()
    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    counts = {"rendered": 0, "cached": 0, "skipped": 0, "failed": 0}
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        episode = episodes_by_id.get(candidate.episode_id)
        if episode is None or not episode.normalized_path:
            counts["skipped"] += 1
            continue
        source = paths.absolute(episode.normalized_path)
        if not source.is_file():
            counts["skipped"] += 1
            continue
        start_ms, duration_ms = clip_window(candidate, episode, settings)
        if duration_ms <= 0:
            counts["skipped"] += 1
            continue
        destination = clip_path_for(
            paths, candidate.candidate_id, start_ms, duration_ms
        )
        if destination.is_file() and destination.stat().st_size > 0:
            counts["cached"] += 1
        else:
            try:
                extract_clip(source, destination, start_ms, duration_ms)
            except (OSError, RuntimeError) as error:
                counts["failed"] += 1
                log(f"  clip failed for {candidate.candidate_id}: {error}")
                continue
            counts["rendered"] += 1
        if index % 250 == 0 or index == total:
            log(
                f"  clips {index}/{total} "
                f"(rendered {counts['rendered']}, cached {counts['cached']})"
            )
    return counts


# ---------------------------------------------------------------------------
# The per-annotator plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedItem:
    """One item in an annotator's order, before any database lookup."""

    presentation_id: str
    candidate_id: str
    stage: str
    is_repeat: bool


def build_plan(
    presentations: Sequence[QueuePresentation],
    annotator_id: str,
    repeat_delay_items: int,
) -> list[PlannedItem]:
    """Order the queue for one annotator.

    First showings come first, shuffled deterministically on the annotator. Each
    consistency repeat is then placed at least ``repeat_delay_items`` items after
    its own first showing, so the repeat is far enough away to be a genuine
    re-judgement rather than a memory test.

    Guaranteeing that gap needs one extra step, because the shuffle does not know
    which candidates will be repeated: a repeat whose first showing landed near
    the end of the queue has nowhere far enough to go. Those first showings are
    therefore pulled forward into the opening stretch first, at a deterministic
    slot keyed on the annotator, and only then are the repeats appended in
    anchor order. Pulling one item forward is a smaller distortion of the
    annotator's order than a repeat arriving two items after its original, which
    would silently make the consistency figure meaningless.

    When the queue is shorter than twice the delay, no such gap exists; the delay
    falls back to half the queue, which is the largest one that can be honoured.
    """
    firsts = [p for p in presentations if not p.is_repeat]
    repeats = [p for p in presentations if p.is_repeat]
    firsts.sort(key=lambda p: annotator_order_key(annotator_id, p.presentation_id))

    plan = [
        PlannedItem(p.presentation_id, p.candidate_id, p.stage, False) for p in firsts
    ]
    if not plan:
        return plan

    repeated_ids = {p.candidate_id for p in repeats}
    delay = min(max(1, repeat_delay_items), max(1, len(plan) // 2))
    limit = max(1, len(plan) - delay)

    # Pull any repeated candidate's first showing into the opening stretch.
    position_of = {item.candidate_id: index for index, item in enumerate(plan)}
    late = sorted(
        (position_of[cid] for cid in repeated_ids if position_of.get(cid, 0) >= limit)
    )
    for source in late:
        key = annotator_order_key(annotator_id, plan[source].presentation_id)
        target = int(key[:8], 16) % limit
        while plan[target].candidate_id in repeated_ids and target + 1 < limit:
            target += 1
        plan[target], plan[source] = plan[source], plan[target]

    position_of = {item.candidate_id: index for index, item in enumerate(plan)}
    ordered_repeats = sorted(
        (p for p in repeats if p.candidate_id in position_of),
        key=lambda p: position_of[p.candidate_id],
    )
    plan.extend(
        PlannedItem(p.presentation_id, p.candidate_id, p.stage, True)
        for p in ordered_repeats
    )
    return plan


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


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
    queue: LabellingQueue | None = None,
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
    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    series_by_episode = {e.episode_id: e.series_id for e in episodes}

    if queue is not None:
        queued_ids = set(queue.unique_candidate_ids)
        eligible = [c for c in eligible if c.candidate_id in queued_ids]
        # The set is built once. Rebuilding it inside the comprehension would
        # make this quadratic, which at a few thousand candidates is the
        # difference between instant startup and a visible stall.
        served_ids = {candidate.candidate_id for candidate in eligible}
        presentations: list[QueuePresentation] = [
            p for p in queue.presentations if p.candidate_id in served_ids
        ]
        queue_version = queue.queue_version
    else:
        # No queue: every eligible candidate is its own presentation, which is
        # exactly the pre-queue behaviour.
        presentations = [
            QueuePresentation(
                presentation_id=candidate.candidate_id,
                candidate_id=candidate.candidate_id,
                episode_id=candidate.episode_id,
                series_id=series_by_episode.get(candidate.episode_id, ""),
                dataset_split=candidate.dataset_split,
                stage="primary",
                score_stratum="unknown",
                is_repeat=False,
                repeat_of=None,
            )
            for candidate in eligible
        ]
        queue_version = ""

    by_id = {candidate.candidate_id: candidate for candidate in eligible}
    database.register_candidates(eligible, series_by_episode=series_by_episode)
    database.register_presentations(presentations, queue_version=queue_version)

    unique_total = len({p.candidate_id for p in presentations if not p.is_repeat})
    target_unique = settings.target_unique or unique_total

    # Transcript context is resolved from the cached episode transcript with the
    # same selection the feature pipeline uses, so the annotator reads what the
    # model reads. Transcripts are loaded once per episode and cached; an episode
    # with no transcript file resolves to ``None`` and the UI shows its clean
    # "rate from the audio alone" message.
    transcript_context_config = TranscriptContextConfig()
    _transcript_cache: dict[str, Any] = {}
    _plan_cache: dict[str, list[PlannedItem]] = {}

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

    def _plan_for(annotator_id: str) -> list[PlannedItem]:
        if annotator_id not in _plan_cache:
            _plan_cache[annotator_id] = build_plan(
                presentations, annotator_id, settings.repeat_delay_items
            )
        return _plan_cache[annotator_id]

    def _payload(item: PlannedItem, annotator_id: str) -> dict[str, Any] | None:
        candidate = by_id.get(item.candidate_id)
        if candidate is None:
            return None
        episode = episodes_by_id.get(candidate.episode_id)
        if episode is None:
            return None
        start_ms, duration_ms = clip_window(candidate, episode, settings)
        transcript_before, transcript_after = _resolve_transcript(candidate)
        existing = database.get_label(
            candidate.candidate_id, annotator_id, presentation_id=item.presentation_id
        )
        payload: dict[str, Any] = {
            # `presentation_id` is what a label is keyed on. It is deliberately
            # opaque to the client: nothing in the payload says whether this is a
            # repeat, because knowing that would defeat the measurement.
            "presentation_id": item.presentation_id,
            "candidate_id": candidate.candidate_id,
            "episode_id": candidate.episode_id,
            "episode_title": episode.title,
            "timestamp_ms": candidate.timestamp_ms,
            "timestamp_label": _format_timestamp(candidate.timestamp_ms),
            "clip_url": f"/api/clip/{item.presentation_id}",
            "clip_start_ms": start_ms,
            "clip_duration_ms": duration_ms,
            "boundary_offset_ms": candidate.timestamp_ms - start_ms,
            "transcript_before": transcript_before,
            "transcript_after": transcript_after,
            "has_transcript": bool(transcript_before or transcript_after),
            "existing_quality_score": existing.quality_score if existing else None,
            "existing_notes": existing.notes if existing else None,
            "existing_is_unusable": bool(existing.is_unusable) if existing else False,
        }
        if settings.reveal_hints:
            payload["heuristic_score"] = candidate.heuristic_score
            payload["candidate_sources"] = list(candidate.candidate_sources)
        return payload

    def _progress(annotator_id: str) -> dict[str, Any]:
        stats = database.progress(annotator_id, target=target_unique)
        stats["queue_unique_candidates"] = unique_total
        stats["rubric_version"] = LABEL_RUBRIC_VERSION
        stats["label_schema_version"] = LABEL_SCHEMA_VERSION
        return stats

    def _upcoming(annotator_id: str, count: int) -> list[dict[str, Any]]:
        done = database.labelled_presentation_ids(annotator_id)
        skipped = database.skipped_presentation_ids(annotator_id)
        items: list[dict[str, Any]] = []
        for item in _plan_for(annotator_id):
            if len(items) >= count:
                break
            if item.presentation_id in done or item.presentation_id in skipped:
                continue
            payload = _payload(item, annotator_id)
            if payload is not None:
                items.append(payload)
        return items

    app = FastAPI(title="Slotify labelling", version=PACKAGE_VERSION)
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Presentation ids are the client's handle on everything, including audio, so
    # a repeat's clip URL differs from its original's and the browser cache
    # cannot leak the fact that the two are the same clip.
    presentation_by_id = {p.presentation_id: p for p in presentations}
    presentation_to_candidate = {
        presentation_id: presentation.candidate_id
        for presentation_id, presentation in presentation_by_id.items()
    }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "package_version": PACKAGE_VERSION,
            "rubric_version": LABEL_RUBRIC_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "eligible_candidates": len(eligible),
            "queue_version": queue_version,
            "presentation_count": len(presentations),
            "target_unique": target_unique,
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
        return _progress(annotator_id)

    @app.get("/api/batch")
    def batch(annotator_id: str, count: int = 0):
        """The next few items, so the client can prefetch their audio."""
        if not annotator_id.strip():
            raise HTTPException(status_code=422, detail="annotator_id is required")
        size = count or settings.batch_size
        if size < 1 or size > 50:
            raise HTTPException(status_code=422, detail="count must be in [1, 50]")
        return {
            "items": _upcoming(annotator_id, size),
            "progress": _progress(annotator_id),
        }

    @app.get("/api/next")
    def next_candidate(annotator_id: str):
        """The annotator's next unjudged item, in their stable order."""
        if not annotator_id.strip():
            raise HTTPException(status_code=422, detail="annotator_id is required")
        items = _upcoming(annotator_id, 1)
        if not items:
            return JSONResponse(
                {
                    "candidate": None,
                    "progress": _progress(annotator_id),
                    "message": "Nothing left in this queue.",
                },
                status_code=200,
            )
        return {"candidate": items[0], "progress": _progress(annotator_id)}

    @app.get("/api/clip/{presentation_id}")
    def clip(presentation_id: str):
        candidate_id = presentation_to_candidate.get(presentation_id)
        candidate = by_id.get(candidate_id) if candidate_id else None
        if candidate is None:
            raise HTTPException(status_code=404, detail="Unknown presentation")
        episode = episodes_by_id.get(candidate.episode_id)
        if episode is None or not episode.normalized_path:
            raise HTTPException(
                status_code=409,
                detail="Episode has no normalized audio; run `dataset normalize`",
            )
        source = paths.absolute(episode.normalized_path)
        if not source.is_file():
            raise HTTPException(status_code=410, detail="Normalized audio is missing")
        start_ms, duration_ms = clip_window(candidate, episode, settings)
        if duration_ms <= 0:
            raise HTTPException(status_code=409, detail="Empty context window")
        destination = clip_path_for(
            paths, candidate.candidate_id, start_ms, duration_ms
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
        presentation_id = str(payload.get("presentation_id") or "")
        candidate_id = presentation_to_candidate.get(presentation_id)
        if candidate_id is None:
            raise HTTPException(status_code=404, detail="Unknown presentation")
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
        elapsed = payload.get("elapsed_ms")
        presentation = presentation_by_id[presentation_id]
        try:
            record = database.upsert_label(
                candidate_id=candidate_id,
                annotator_id=annotator_id,
                quality_score=score,
                is_unusable=bool(payload.get("is_unusable", False)),
                notes=None if notes is None else str(notes),
                presentation_id=presentation_id,
                stage=presentation.stage,
                is_repeat=presentation.is_repeat,
                queue_version=queue_version,
                elapsed_ms=(
                    int(elapsed) if isinstance(elapsed, (int, float)) else None
                ),
            )
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"label": record.to_dict(), "progress": _progress(annotator_id)}

    @app.post("/api/skip")
    def skip_item(payload: dict = Body(...)) -> dict[str, Any]:
        """Defer an item. Explicitly not a judgement -- see LabelDatabase.skip."""
        presentation_id = str(payload.get("presentation_id") or "")
        candidate_id = presentation_to_candidate.get(presentation_id)
        if candidate_id is None:
            raise HTTPException(status_code=404, detail="Unknown presentation")
        annotator_id = str(payload.get("annotator_id") or "")
        reason = payload.get("reason")
        try:
            record = database.skip(
                candidate_id=candidate_id,
                annotator_id=annotator_id,
                presentation_id=presentation_id,
                reason=None if reason is None else str(reason),
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"skip": record.to_dict(), "progress": _progress(annotator_id)}

    @app.post("/api/skips/clear")
    def clear_skips(payload: dict = Body(...)) -> dict[str, Any]:
        annotator_id = str(payload.get("annotator_id") or "").strip()
        if not annotator_id:
            raise HTTPException(status_code=422, detail="annotator_id is required")
        cleared = database.clear_skips(annotator_id)
        return {"cleared": cleared, "progress": _progress(annotator_id)}

    return app
