"""``infer serve``: one long-lived ranker process, many requests.

``infer rank`` pays for torch, the checkpoint, Whisper, the Whisper encoder and
MiniLM on every request -- most of a short clip's latency, before any of its own
work starts. The worker loads all of them once and then answers requests one at
a time, over newline-delimited JSON:

stdin, one request per line::

    {"id": "r1", "audio_path": "/tmp/x/upload.mp3", "work_dir": "/tmp/x/work",
     "episode_id": "upload", "skip_transcription": false}

stdout, exactly one line per request, in order::

    {"id": "r1", "ok": true, "payload": {<the infer-rank document>}}
    {"id": "r1", "ok": false, "error": "<the last meaningful line of the error>"}

Nothing else is ever written to stdout -- the CLI moves file descriptor 1 onto
stderr before anything is imported that could print, and keeps a private copy
for this protocol. Progress, warnings and tracebacks go to stderr.

**One request at a time.** The process is single-threaded and the caller queues.
A request runs entirely inside its own ``work_dir`` (see
:func:`slotify_rank.inference.service.request_scratch`) and leaves nothing
behind but the response.

**Failures.** A request that fails the way ``infer rank`` reports failures (a
bad file, an incompatible checkpoint, a feature stage that could not run) gets
``ok: false`` and the worker carries on. Anything else -- an exception the
package does not raise on purpose -- is answered too, and then the worker exits
non-zero: whatever went wrong may have left state behind, and the next request
deserves a fresh process. A line that is not a valid request gets ``ok: false``
(with ``id: null`` when not even the id could be read).

**Stopping.** End of stdin, SIGTERM or SIGINT end the process with exit code 0;
a request in flight at that moment has its scratch directory removed and gets
no response.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from slotify_rank.inference.episode import DEFAULT_EPISODE_ID, PipelineConfigs
from slotify_rank.inference.predictor import RankerPredictor
from slotify_rank.inference.service import rank_audio, request_scratch

__all__ = [
    "EXPECTED_ERRORS",
    "RankerWorker",
    "RequestError",
    "WorkerRequest",
    "last_meaningful_line",
    "preload_models",
    "serve",
]

#: The failures ``slotify-rank`` reports as a runtime error (exit 1) rather
#: than a crash -- the same set ``cli.main`` catches. In the worker they become
#: ``ok: false`` and the next request is served by the same process.
EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    KeyError,
    TypeError,
    OSError,
    RuntimeError,
)

_REQUEST_FIELDS = frozenset(
    {"id", "audio_path", "work_dir", "episode_id", "skip_transcription"}
)

#: Episode ids become file names inside the scratch corpus, so a request may not
#: use one that is a path.
_EPISODE_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")

_MAX_ID_LENGTH = 200


class RequestError(ValueError):
    """A line that is not a valid request. Carries the id when it was readable."""

    def __init__(self, message: str, request_id: str | int | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


def _request_id(raw: Mapping[str, Any]) -> str | int | None:
    value = raw.get("id")
    # bool is an int subclass; an id of `true` is a bug, not request 1.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and 0 < len(value) <= _MAX_ID_LENGTH:
        return value
    return None


@dataclass(frozen=True)
class WorkerRequest:
    """One validated request line."""

    id: str | int
    audio_path: Path
    work_dir: Path
    episode_id: str = DEFAULT_EPISODE_ID
    skip_transcription: bool = False

    @classmethod
    def from_line(cls, line: str) -> "WorkerRequest":
        """Parse and validate. Strict: an unknown key is refused, not ignored --
        a misspelt ``skip_transcription`` silently ignored would transcribe."""
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise RequestError(f"request is not JSON: {error}") from error
        if not isinstance(raw, dict):
            raise RequestError(
                f"request must be a JSON object, got {type(raw).__name__}"
            )
        request_id = _request_id(raw)
        if request_id is None:
            raise RequestError(
                "request needs an 'id': a non-empty string or an integer"
            )

        def fail(message: str) -> RequestError:
            return RequestError(message, request_id)

        unknown = sorted(set(raw) - _REQUEST_FIELDS)
        if unknown:
            raise fail(
                f"unknown request field(s) {unknown}; known: {sorted(_REQUEST_FIELDS)}"
            )

        audio_path = raw.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise fail("'audio_path' must be a non-empty string")

        work_dir = raw.get("work_dir")
        if not isinstance(work_dir, str) or not work_dir:
            raise fail(
                "'work_dir' must be a non-empty string: every request's scratch "
                "files go in a directory its caller owns"
            )
        work = Path(work_dir)
        if not work.is_absolute():
            raise fail(f"'work_dir' must be an absolute path, got {work_dir!r}")
        if not work.is_dir():
            raise fail(f"'work_dir' {work_dir!r} is not an existing directory")

        episode_id = raw.get("episode_id", DEFAULT_EPISODE_ID)
        if not isinstance(episode_id, str) or not _EPISODE_ID.match(episode_id):
            raise fail(
                "'episode_id' must be 1-128 characters of letters, digits, '_', "
                f"'.' or '-' (not starting with '.' or '-'), got {episode_id!r}"
            )

        skip = raw.get("skip_transcription", False)
        if not isinstance(skip, bool):
            raise fail("'skip_transcription' must be true or false")

        return cls(
            id=request_id,
            audio_path=Path(audio_path),
            work_dir=work,
            episode_id=episode_id,
            skip_transcription=skip,
        )


def last_meaningful_line(text: str, fallback: str = "unknown error") -> str:
    """The last non-blank line of an error message -- the part that names the problem.

    The same line the Node API shows from a one-shot run's stderr, so a client
    sees the same text whichever path produced the failure.
    """
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else fallback


def _respond(stream: TextIO, message: Mapping[str, Any]) -> None:
    stream.write(json.dumps(message, ensure_ascii=False) + "\n")
    stream.flush()


def serve(
    handle: Callable[[WorkerRequest], Mapping[str, Any]],
    stdin: TextIO,
    stdout: TextIO,
    log: Callable[[str], None] | None = None,
) -> int:
    """Answer requests from ``stdin`` until it ends. Returns the exit code.

    ``handle`` turns a request into the ``infer rank`` document. Exceptions in
    :data:`EXPECTED_ERRORS` become ``ok: false`` and serving continues; any other
    exception is answered and ends the loop with exit code 1. ``BaseException``
    (SIGTERM via :class:`~slotify_rank.inference.service.Terminated`, Ctrl-C)
    propagates to the caller unanswered.
    """
    logger = log or (lambda _message: None)
    while True:
        line = stdin.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            request = WorkerRequest.from_line(line)
        except RequestError as error:
            logger(f"  worker: rejected a request: {error}")
            _respond(stdout, {"id": error.request_id, "ok": False, "error": str(error)})
            continue

        started = time.perf_counter()
        try:
            payload = handle(request)
        except EXPECTED_ERRORS as error:
            message = str(error) or type(error).__name__
            logger(f"error: {message}")
            _respond(
                stdout,
                {"id": request.id, "ok": False, "error": last_meaningful_line(message)},
            )
            continue
        except Exception as error:  # noqa: BLE001 -- answered, then the process ends
            logger(traceback.format_exc().rstrip())
            logger(
                "  worker: unexpected failure; exiting so the next request gets a "
                "fresh process"
            )
            message = f"{type(error).__name__}: {error}"
            _respond(
                stdout,
                {"id": request.id, "ok": False, "error": last_meaningful_line(message)},
            )
            return 1
        _respond(stdout, {"id": request.id, "ok": True, "payload": dict(payload)})
        logger(
            f"  worker: request {request.id!r} answered in "
            f"{time.perf_counter() - started:.1f}s"
        )


def preload_models(configs: PipelineConfigs) -> None:
    """Load Whisper, the Whisper encoder and MiniLM into the resident scope.

    Must run inside :func:`slotify_rank.embeddings.resident.resident_models`;
    the stages' own ``load()`` calls then find these and reuse them. The wrapper
    classes are looked up on their modules at call time, so whatever the stages
    will construct is what is preloaded.
    """
    from slotify_rank.embeddings import minilm_text, resident, whisper_audio
    from slotify_rank.transcription import local_whisper

    if not resident.is_open():
        raise RuntimeError(
            "preload_models() outside resident_models() would load three models "
            "and immediately discard them"
        )
    for wrapper in (
        local_whisper.WhisperTranscriber(config=configs.transcription),
        whisper_audio.WhisperAudioEncoder(config=configs.embeddings.audio),
        minilm_text.MiniLmTextEncoder(config=configs.embeddings.text),
    ):
        wrapper.load()


@dataclass
class RankerWorker:
    """A loaded predictor plus the pipeline configs it serves under."""

    predictor: RankerPredictor
    configs: PipelineConfigs
    log: Callable[[str], None] = field(default=lambda _message: None)

    def handle(self, request: WorkerRequest) -> dict[str, Any]:
        """Rank one request inside its own work directory."""
        with request_scratch(request.work_dir) as scratch:
            result = rank_audio(
                self.predictor,
                self.configs,
                request.audio_path,
                scratch,
                cleanup=True,
                episode_id=request.episode_id,
                transcribe=not request.skip_transcription,
                model_load_seconds=0.0,
                log=self.log,
            )
        self.log(
            f"  ranked {len(result.ranked)} of {result.candidate_count} candidate(s)"
        )
        return result.to_dict()
