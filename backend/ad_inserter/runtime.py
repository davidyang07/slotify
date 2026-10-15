"""Process lifecycle shared by the ``ad_inserter`` CLIs: stop signals and temp files.

The Node API spawns one CLI per request and stops it with SIGTERM, then
SIGKILL once a grace period (default 5 s) has passed. Each CLI's ``run()``
wraps its work in :func:`cli_session` so that stop is clean:

* SIGTERM and SIGINT raise ``SystemExit(128 + signum)`` -- exit code 143 for
  SIGTERM, 130 for SIGINT -- so ``finally`` blocks and ``with`` exits run:
  temp-file removal, and ``insert_ad``'s deletion of the voices it cloned.
  Only the first signal raises; later ones are ignored so that they cannot cut
  that cleanup short. Blocking calls the pipeline makes (ffmpeg subprocess
  waits, HTTP requests with a timeout, sleeps) return to the interpreter when
  a signal arrives, so the exit is prompt.
* Every temp file the process creates goes in one private directory, removed
  on the way out. That covers library temp files too: ``tempfile.tempdir`` and
  ``TMPDIR`` point at it for the session, and pydub's MP3 export, for one,
  leaves its temp files behind when it is interrupted. With ``--work-dir`` the
  private directory is created inside the work dir and nothing is written to
  the system temp dir; without it, it is created in the system temp dir.

:func:`shielded` lets a short critical step (creating or deleting a cloned
voice) finish before a signal received during it takes effect.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGINT)

# The first stop signal of the current session, if any.
_received: Optional[int] = None
# Whether that signal's SystemExit has been raised yet (it waits while shielded).
_exit_raised = False
_shield_depth = 0


def exit_code(signum: int) -> int:
    """The conventional shell exit status for a process stopped by ``signum``."""
    return 128 + int(signum)


def termination_requested() -> bool:
    """True once a stop signal has arrived in the current session."""
    return _received is not None


def _handle_signal(signum: int, _frame: object) -> None:
    global _received, _exit_raised
    if _received is not None:
        return  # already stopping: let the cleanup in progress finish
    _received = signum
    if _shield_depth:
        return  # raised when the shielded step ends
    _exit_raised = True
    raise SystemExit(exit_code(signum))


@contextmanager
def shielded() -> Iterator[None]:
    """Let a short step finish even if a stop signal arrives during it.

    A signal received inside the block raises ``SystemExit`` as the block
    ends, rather than part-way through it.
    """
    global _shield_depth, _exit_raised
    _shield_depth += 1
    try:
        yield
    finally:
        _shield_depth -= 1
        if not _shield_depth and _received is not None and not _exit_raised:
            _exit_raised = True
            raise SystemExit(exit_code(_received))


def _existing_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"{value!r} is not an existing directory")
    return path


def add_work_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--work-dir``, shared by every CLI."""
    parser.add_argument(
        "--work-dir",
        type=_existing_dir,
        default=None,
        help=(
            "Existing directory for every temp file of the run (the caller "
            "creates and removes it); default: the system temp dir."
        ),
    )


@contextmanager
def cli_session(work_dir: Optional[Path] = None) -> Iterator[Path]:
    """Install the stop-signal handlers and a private temp dir for one CLI run.

    Yields the private temp dir. On the way out, the directory is removed and
    the previous signal handlers and temp-dir settings are restored.
    """
    global _received, _exit_raised
    _received = None
    _exit_raised = False

    previous_handlers: Dict[int, Any] = {}
    previous_tempdir = tempfile.tempdir
    previous_tmpdir_env = os.environ.get("TMPDIR")
    private: Optional[Path] = None
    try:
        # signal.signal() works only in the main thread; elsewhere (an embedding
        # caller's worker thread) the run keeps the default behaviour.
        if threading.current_thread() is threading.main_thread():
            for signum in HANDLED_SIGNALS:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _handle_signal)
        private = Path(
            tempfile.mkdtemp(
                prefix="ad_inserter-", dir=str(work_dir) if work_dir else None
            )
        )
        tempfile.tempdir = str(private)
        os.environ["TMPDIR"] = str(private)
        yield private
    finally:
        try:
            with shielded():
                tempfile.tempdir = previous_tempdir
                if previous_tmpdir_env is None:
                    os.environ.pop("TMPDIR", None)
                else:
                    os.environ["TMPDIR"] = previous_tmpdir_env
                if private is not None:
                    shutil.rmtree(private, ignore_errors=True)
        finally:
            for signum, handler in previous_handlers.items():
                # None: a handler installed outside Python; restore the default.
                signal.signal(signum, signal.SIG_DFL if handler is None else handler)
            _received = None
            _exit_raised = False
