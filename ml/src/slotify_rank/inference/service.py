"""One ranking routine for both entry points, and where its scratch files go.

``infer rank`` (one process per request) and ``infer serve`` (one process for
many) must produce the same document for the same audio, so both call
:func:`rank_audio`. The only things that differ between them are how long the
model and the encoders have been loaded, and who owns the scratch directory.

**Scratch files.** Every request builds a throwaway single-episode corpus (see
:mod:`slotify_rank.inference.episode`), which holds a copy of the user's upload.
It must never land somewhere nobody cleans: a ``mkdtemp`` in the system temp
directory leaks the upload whenever the process is killed before its
``finally`` runs. So the caller names a work directory it owns -- the Node API
creates one per request and removes it, whatever happens to this process -- and
:func:`request_scratch` puts the corpus inside it and points :mod:`tempfile` at
it for the duration, so a library that asks for a temp file gets one there too.

**SIGTERM.** Python's default is to die without running ``finally`` blocks.
:func:`terminate_on_sigterm` turns the signal into an exception, so cleanup
runs and the process exits on its own terms.
"""

from __future__ import annotations

import signal
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from slotify_rank.inference.episode import (
    DEFAULT_EPISODE_ID,
    PipelineConfigs,
    episode_examples,
    prepare_episode,
    workspace,
)
from slotify_rank.inference.predictor import RankerPredictor, rank_examples
from slotify_rank.inference.schema import ExcludedCandidate, InferenceResult

__all__ = [
    "Terminated",
    "rank_audio",
    "request_scratch",
    "terminate_on_sigterm",
]


class Terminated(BaseException):
    """SIGTERM arrived. A ``BaseException`` so no ``except Exception`` eats it."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum


@contextmanager
def terminate_on_sigterm() -> Iterator[None]:
    """Raise :class:`Terminated` on SIGTERM inside this block; restore after."""

    def _raise(signum: int, _frame: Any) -> None:
        raise Terminated(signum)

    previous = signal.signal(signal.SIGTERM, _raise)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def request_scratch(work_dir: Path | str) -> Iterator[Path]:
    """A fresh scratch directory inside ``work_dir``, with :mod:`tempfile` aimed there.

    ``work_dir`` must already exist: its owner creates it, and its owner removes
    it. The scratch directory inside it is created here and is the caller's to
    clean (``workspace(..., cleanup=True)`` does). ``tempfile.tempdir`` is
    process-wide, which is fine for a process that serves one request at a time,
    and is restored on the way out.
    """
    root = Path(work_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"work directory {root} does not exist; its owner must create it"
        )
    previous = tempfile.tempdir
    tempfile.tempdir = str(root)
    try:
        yield Path(tempfile.mkdtemp(prefix="slotify-infer-", dir=root))
    finally:
        tempfile.tempdir = previous


def rank_audio(
    predictor: RankerPredictor,
    configs: PipelineConfigs,
    audio_path: Path | str,
    scratch: Path | str,
    *,
    cleanup: bool = True,
    episode_id: str = DEFAULT_EPISODE_ID,
    transcribe: bool = True,
    top: int | None = None,
    model_load_seconds: float = 0.0,
    log: Callable[[str], None] | None = None,
) -> InferenceResult:
    """Featurize one audio file in ``scratch`` and rank it: the ``infer rank`` document.

    ``model_load_seconds`` is what *this* request spent loading the checkpoint,
    reported as ``timings_seconds.model_load``: the load time for a one-shot run,
    ``0.0`` for a worker request (the worker loaded it once, at startup, and
    says so on stderr). Every other timing likewise measures this request's work.
    """
    logger = log or (lambda _message: None)
    with workspace(Path(scratch), cleanup=cleanup) as space:
        prepared = prepare_episode(
            space,
            audio_path,
            configs,
            episode_id=episode_id,
            transcribe=transcribe,
            log=logger,
        )
        examples = list(episode_examples(prepared))
        started = time.perf_counter()
        ranking = rank_examples(
            predictor,
            examples,
            prepared.dataset.schema,
            prepared.timestamps_ms,
            prepared.feature_status,
        )
        ranked = ranking.ranked
        score_seconds = round(time.perf_counter() - started, 3)
        if ranking.withheld_columns:
            logger(
                f"  {len(ranking.withheld_columns)} served column(s) not used by "
                "this checkpoint were withheld from it"
            )

        timings = dict(prepared.timings_seconds)
        timings["model_load"] = model_load_seconds
        timings["score"] = score_seconds

        if top is not None and top > 0:
            ranked = ranked[:top]

        return InferenceResult(
            episode_id=prepared.episode.episode_id,
            duration_seconds=prepared.duration_seconds,
            ranked=tuple(ranked),
            excluded=tuple(
                ExcludedCandidate(
                    candidate_id=exclusion.candidate_id,
                    reason=exclusion.reason,
                    detail=exclusion.detail,
                )
                for exclusion in prepared.dataset.exclusions
            ),
            model=predictor.identity,
            candidate_count=len(prepared.candidates),
            # The training disclosure (weak / synthetic / unknown supervision)
            # is prepended by InferenceResult itself.
            warnings=(*prepared.warnings, *ranking.warnings),
            timings_seconds=timings,
        )
