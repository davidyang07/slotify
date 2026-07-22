"""Read real audio metadata with ``ffprobe``.

Duration, sample rate, channel count and format are **measured**, never inferred
from the file extension or from what a source declared. A corpus-hours figure
computed from anything other than the decoder's own answer is not evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slotify_rank.data.ffmpeg import FFmpegError, ffprobe_path, run_tool

__all__ = ["AudioMetadata", "probe_audio", "UnreadableAudio"]


class UnreadableAudio(ValueError):
    """The file is empty, is not audio, or has no usable duration."""


@dataclass(frozen=True)
class AudioMetadata:
    duration_ms: int
    sample_rate_hz: int
    channels: int
    file_format: str
    codec_name: str
    bit_rate: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "file_format": self.file_format,
            "codec_name": self.codec_name,
            "bit_rate": self.bit_rate,
        }


def probe_audio(path: Path | str) -> AudioMetadata:
    """Probe ``path`` and return its measured metadata.

    Raises :class:`UnreadableAudio` for zero-length files, files with no audio
    stream, and files whose duration is missing or non-positive -- all three
    would otherwise contribute a silent zero to the corpus totals.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise UnreadableAudio(f"Not a readable file: {file_path}")
    if file_path.stat().st_size == 0:
        raise UnreadableAudio(f"File is zero-length: {file_path}")

    try:
        stdout = run_tool(
            ffprobe_path(),
            [
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-select_streams",
                "a:0",
                str(file_path),
            ],
            timeout=120.0,
        )
    except FFmpegError as error:
        raise UnreadableAudio(f"ffprobe could not read {file_path}: {error}") from error

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise UnreadableAudio(
            f"ffprobe returned non-JSON output for {file_path}: {error}"
        ) from error

    streams = payload.get("streams") or []
    if not streams:
        raise UnreadableAudio(f"{file_path} contains no audio stream")
    stream = streams[0]
    container = payload.get("format") or {}

    duration_seconds = _first_float(stream.get("duration"), container.get("duration"))
    if duration_seconds is None or duration_seconds <= 0:
        raise UnreadableAudio(
            f"{file_path} has no positive duration (ffprobe reported "
            f"{stream.get('duration')!r} / {container.get('duration')!r})"
        )

    sample_rate = _first_int(stream.get("sample_rate"))
    channels = _first_int(stream.get("channels"))
    if not sample_rate or not channels:
        raise UnreadableAudio(
            f"{file_path} is missing sample_rate or channels in the audio stream"
        )

    format_name = str(container.get("format_name") or "").split(",")[0] or "unknown"
    bit_rate = _first_int(container.get("bit_rate"), stream.get("bit_rate"))

    return AudioMetadata(
        duration_ms=int(round(duration_seconds * 1000)),
        sample_rate_hz=sample_rate,
        channels=channels,
        file_format=format_name,
        codec_name=str(stream.get("codec_name") or "unknown"),
        bit_rate=bit_rate,
    )


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and parsed not in (float("inf"), float("-inf")):
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    parsed = _first_float(*values)
    return None if parsed is None else int(parsed)
