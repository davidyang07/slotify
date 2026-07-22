"""Deterministic JSONL manifests for episodes and dataset candidates.

JSONL rather than a database: manifests are regenerated often, must diff
readably, and must be appendable without a migration step. Every write is
atomic and every write re-sorts the whole file into canonical order, so
regenerating a manifest after adding one episode produces a one-line diff
instead of a reshuffle.

Reads are strict. An unknown field, an unsupported schema version or a
duplicate ID raises rather than being dropped, because every downstream number
-- corpus hours, candidate counts, split membership -- is only as trustworthy
as the parse that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord, sort_candidates

__all__ = [
    "read_episodes",
    "write_episodes",
    "upsert_episodes",
    "read_candidates",
    "write_candidates",
    "read_jsonl",
    "write_jsonl",
    "ManifestError",
]


class ManifestError(ValueError):
    """A manifest file is unreadable, inconsistent, or internally duplicated."""


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)`` pairs, skipping blank lines."""
    file_path = Path(path)
    if not file_path.is_file():
        return
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"{file_path}:{line_number} is not valid JSON: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ManifestError(
                    f"{file_path}:{line_number} must be a JSON object, got "
                    f"{type(payload).__name__}"
                )
            yield line_number, payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomically write ``rows`` as JSONL. Returns the row count."""
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_bytes(Path(path), body.encode("utf-8"))
    return body.count("\n")


def read_episodes(path: Path) -> list[EpisodeRecord]:
    """Read the episode manifest, rejecting duplicate episode IDs."""
    episodes: list[EpisodeRecord] = []
    seen: dict[str, int] = {}
    for line_number, payload in read_jsonl(path):
        try:
            episode = EpisodeRecord.from_mapping(payload)
        except (ValueError, TypeError) as error:
            raise ManifestError(f"{path}:{line_number}: {error}") from error
        if episode.episode_id in seen:
            raise ManifestError(
                f"{path}:{line_number}: duplicate episode_id {episode.episode_id!r} "
                f"(first seen on line {seen[episode.episode_id]})"
            )
        seen[episode.episode_id] = line_number
        episodes.append(episode)
    return episodes


def write_episodes(path: Path, episodes: Sequence[EpisodeRecord]) -> int:
    """Atomically write episodes in canonical (series, episode) order."""
    ordered = sorted(episodes, key=lambda e: (e.series_id, e.episode_id))
    ids = [episode.episode_id for episode in ordered]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ManifestError(f"Refusing to write duplicate episode_id(s): {duplicates}")
    return write_jsonl(path, (episode.to_dict() for episode in ordered))


def upsert_episodes(
    path: Path, episodes: Sequence[EpisodeRecord]
) -> tuple[int, int, int]:
    """Merge ``episodes`` into the manifest by ``episode_id``.

    Returns ``(added, updated, total)``. Upsert rather than append is what makes
    ``import``/``probe``/``normalize`` resumable: each stage rewrites the rows it
    touched and leaves the rest byte-identical.
    """
    existing = {episode.episode_id: episode for episode in read_episodes(path)}
    added = 0
    updated = 0
    for episode in episodes:
        if episode.episode_id in existing:
            if existing[episode.episode_id] != episode:
                updated += 1
        else:
            added += 1
        existing[episode.episode_id] = episode
    write_episodes(path, list(existing.values()))
    return added, updated, len(existing)


def read_candidates(path: Path) -> list[DatasetCandidate]:
    """Read the candidate manifest, rejecting duplicate candidate IDs."""
    candidates: list[DatasetCandidate] = []
    seen: dict[str, int] = {}
    for line_number, payload in read_jsonl(path):
        try:
            candidate = DatasetCandidate.from_mapping(payload)
        except (ValueError, TypeError, KeyError) as error:
            raise ManifestError(f"{path}:{line_number}: {error}") from error
        if candidate.candidate_id in seen:
            raise ManifestError(
                f"{path}:{line_number}: duplicate candidate_id "
                f"{candidate.candidate_id!r} (first seen on line "
                f"{seen[candidate.candidate_id]})"
            )
        seen[candidate.candidate_id] = line_number
        candidates.append(candidate)
    return candidates


def write_candidates(path: Path, candidates: Sequence[DatasetCandidate]) -> int:
    """Atomically write candidates in canonical (episode, timestamp, id) order."""
    ordered = sort_candidates(candidates)
    ids = [candidate.candidate_id for candidate in ordered]
    if len(ids) != len(set(ids)):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise ManifestError(f"Refusing to write duplicate candidate_id(s): {duplicates}")
    return write_jsonl(path, (candidate.to_dict() for candidate in ordered))


def replace_episode_candidates(
    path: Path,
    episode_ids: Sequence[str],
    candidates: Sequence[DatasetCandidate],
) -> tuple[int, int]:
    """Replace all candidates belonging to ``episode_ids``, keep the rest.

    Returns ``(removed, written_total)``. Regenerating candidates for one
    episode must not disturb another episode's rows -- their IDs are stable and
    their labels point at them.
    """
    keep = [
        candidate
        for candidate in read_candidates(path)
        if candidate.episode_id not in set(episode_ids)
    ]
    removed = len(read_candidates(path)) - len(keep)
    total = write_candidates(path, [*keep, *candidates])
    return removed, total
