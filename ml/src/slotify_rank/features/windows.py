"""Candidate-centred analysis windows, clipped to the episode.

Every window is a half-open ``[start_ms, end_ms)`` interval. Two rules apply
everywhere:

**Clip, never pad.** A candidate 2 seconds into an episode has no 10-second
"before" context. The window is clipped to the audio and the *fraction actually
available* is recorded alongside it, so a model can tell a genuinely quiet
lead-in from a window that was three-quarters missing. Zero-padding the audio
instead would manufacture silence and make every early candidate look like it
sits in a long pause -- a systematic bias towards the start of every episode.

**Availability is a feature, not an error.** Short windows near an edge are
normal and expected. They are reported through
:attr:`CandidateWindow.available_fraction`, never through a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CandidateWindow", "CandidateWindows", "build_windows"]


@dataclass(frozen=True)
class CandidateWindow:
    """One side of one window scale, already clipped to the episode."""

    name: str
    start_ms: int
    end_ms: int
    #: How much of the requested span survived clipping, in ``[0, 1]``.
    available_fraction: float
    requested_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def is_empty(self) -> bool:
        return self.duration_ms <= 0


def _clip(
    name: str, start_ms: int, end_ms: int, requested_ms: int, duration_ms: int
) -> CandidateWindow:
    clipped_start = max(0, min(start_ms, duration_ms))
    clipped_end = max(clipped_start, min(end_ms, duration_ms))
    available = 0.0 if requested_ms <= 0 else (clipped_end - clipped_start) / requested_ms
    return CandidateWindow(
        name=name,
        start_ms=clipped_start,
        end_ms=clipped_end,
        available_fraction=float(min(1.0, max(0.0, available))),
        requested_ms=requested_ms,
    )


@dataclass(frozen=True)
class CandidateWindows:
    """All windows for one candidate, at every scale and on both sides."""

    timestamp_ms: int
    episode_duration_ms: int
    before: dict[str, CandidateWindow]
    after: dict[str, CandidateWindow]
    #: Symmetric windows spanning the candidate, used for the "around" features
    #: (minimum energy across the break, spectral contrast either side as one).
    around: dict[str, CandidateWindow]

    def scale(self, name: str) -> tuple[CandidateWindow, CandidateWindow, CandidateWindow]:
        try:
            return (self.before[name], self.after[name], self.around[name])
        except KeyError as error:
            raise KeyError(
                f"Unknown window scale {name!r}; available: {sorted(self.before)}"
            ) from error

    @property
    def edge_clipped(self) -> bool:
        """True when any window lost content to an episode boundary."""
        return any(
            window.available_fraction < 1.0
            for group in (self.before, self.after)
            for window in group.values()
        )


def build_windows(
    timestamp_ms: int,
    episode_duration_ms: int,
    scales: tuple[tuple[str, int], ...],
) -> CandidateWindows:
    """Build clipped windows at every configured scale.

    ``scales`` is ``(name, half_width_ms)`` pairs, e.g. ``(("short", 1000),
    ("medium", 3000))``. The candidate timestamp must lie inside the episode:
    a candidate outside its own audio is a data-integrity failure, not a window
    to clip, and is rejected here so it cannot reach the feature extractors.
    """
    if episode_duration_ms <= 0:
        raise ValueError(f"episode_duration_ms must be positive, got {episode_duration_ms}")
    if not 0 <= timestamp_ms <= episode_duration_ms:
        raise ValueError(
            f"Candidate timestamp {timestamp_ms} ms lies outside the episode "
            f"[0, {episode_duration_ms}] ms. This is a manifest integrity failure, "
            "not a window that can be clipped."
        )
    if not scales:
        raise ValueError("At least one window scale is required")

    before: dict[str, CandidateWindow] = {}
    after: dict[str, CandidateWindow] = {}
    around: dict[str, CandidateWindow] = {}
    for name, half_width_ms in scales:
        if half_width_ms <= 0:
            raise ValueError(f"Window {name!r} half-width must be positive")
        before[name] = _clip(
            f"{name}_before",
            timestamp_ms - half_width_ms,
            timestamp_ms,
            half_width_ms,
            episode_duration_ms,
        )
        after[name] = _clip(
            f"{name}_after",
            timestamp_ms,
            timestamp_ms + half_width_ms,
            half_width_ms,
            episode_duration_ms,
        )
        around[name] = _clip(
            f"{name}_around",
            timestamp_ms - half_width_ms,
            timestamp_ms + half_width_ms,
            2 * half_width_ms,
            episode_duration_ms,
        )

    return CandidateWindows(
        timestamp_ms=timestamp_ms,
        episode_duration_ms=episode_duration_ms,
        before=before,
        after=after,
        around=around,
    )
