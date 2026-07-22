"""Locating and invoking FFmpeg / FFprobe.

The product already requires both binaries on ``PATH`` (``backend/src/services/
ffmpeg.ts`` and the ``ad_inserter`` pipeline), so this reuses that dependency
rather than adding an audio-decoding library. ``FFMPEG_BIN`` / ``FFPROBE_BIN``
override the executable names, mirroring how the backend treats ``PYTHON_BIN``.

The one thing this module adds over ``subprocess.run`` is a genuinely useful
failure message on Windows, where "file not found" for a missing binary is
otherwise indistinguishable from a missing input file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

__all__ = [
    "FFmpegNotFound",
    "FFmpegError",
    "ffmpeg_path",
    "ffprobe_path",
    "require_ffmpeg",
    "run_tool",
]

_INSTALL_HINT_WINDOWS = (
    "Install it with `winget install Gyan.FFmpeg` (then open a new terminal so "
    "PATH is refreshed), or set the FFMPEG_BIN / FFPROBE_BIN environment "
    "variables to the full path of the executable, e.g.\n"
    '    $env:FFMPEG_BIN = "C:\\tools\\ffmpeg\\bin\\ffmpeg.exe"'
)
_INSTALL_HINT_POSIX = (
    "Install it with your package manager (`brew install ffmpeg`, "
    "`apt install ffmpeg`), or set FFMPEG_BIN / FFPROBE_BIN to the executable path."
)


class FFmpegNotFound(RuntimeError):
    """FFmpeg or FFprobe is not installed or not on PATH."""


class FFmpegError(RuntimeError):
    """FFmpeg or FFprobe ran and failed."""

    def __init__(self, tool: str, returncode: int, stderr: str):
        self.tool = tool
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-8:])
        super().__init__(f"{tool} exited with code {returncode}:\n{tail}")


def _resolve(env_var: str, executable: str) -> str:
    override = os.environ.get(env_var)
    if override:
        if Path(override).is_file() or shutil.which(override):
            return override
        raise FFmpegNotFound(
            f"{env_var}={override!r} does not point at an executable.\n"
            + (_INSTALL_HINT_WINDOWS if os.name == "nt" else _INSTALL_HINT_POSIX)
        )
    found = shutil.which(executable)
    if found:
        return found
    raise FFmpegNotFound(
        f"{executable} was not found on PATH, and it is required to probe and "
        f"normalize audio.\n"
        + (_INSTALL_HINT_WINDOWS if os.name == "nt" else _INSTALL_HINT_POSIX)
    )


def ffmpeg_path() -> str:
    return _resolve("FFMPEG_BIN", "ffmpeg")


def ffprobe_path() -> str:
    return _resolve("FFPROBE_BIN", "ffprobe")


def require_ffmpeg() -> tuple[str, str]:
    """Resolve both binaries up front so a batch job fails on item 0, not 400."""
    return ffmpeg_path(), ffprobe_path()


def run_tool(executable: str, args: Sequence[str], timeout: float = 900.0) -> str:
    """Run an FFmpeg-family tool and return stdout, raising on failure."""
    command = [executable, *args]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as error:  # pragma: no cover - guarded by _resolve
        raise FFmpegNotFound(f"Could not execute {executable}: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise FFmpegError(
            Path(executable).name, -1, f"timed out after {timeout}s"
        ) from error
    if completed.returncode != 0:
        raise FFmpegError(
            Path(executable).name, completed.returncode, completed.stderr or ""
        )
    return completed.stdout
