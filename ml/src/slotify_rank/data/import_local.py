"""Register local audio as episodes.

Two flavours, distinguished by what ends up on disk:

``existing_repository_fixture``
    Already inside the checkout (``backend/audio_tests/*.mp3``). Referenced in
    place -- copying a file that is already committed would only waste space.

``local_file``
    A file the user supplied from anywhere on their machine. Copied into
    ``data/raw/`` so the manifest can reference a repository-relative path and
    the corpus stays reconstructible if the original moves. The original
    location is retained as ``source_uri`` for provenance.

Neither flavour is assumed to carry a public licence. Both are private by
default and are never redistributed; the licence fields stay null unless the
operator declares them.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from slotify_rank.data.checksum import sha256_file
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import EpisodeRecord, make_episode_id
from slotify_rank.data.sources import SourceEntry

__all__ = ["ImportResult", "import_source", "AUDIO_SUFFIXES"]

AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac")


@dataclass(frozen=True)
class ImportResult:
    episode: EpisodeRecord
    copied: bool
    already_present: bool

    @property
    def action(self) -> str:
        if self.already_present:
            return "unchanged"
        return "copied" if self.copied else "referenced"


def import_source(
    entry: SourceEntry,
    paths: DataPaths,
    existing: dict[str, EpisodeRecord] | None = None,
) -> ImportResult:
    """Turn one local/fixture source declaration into an episode record."""
    if entry.source_type == "direct_download":
        raise ValueError(
            f"{entry.id}: import_source handles local sources only; use fetch for "
            "direct_download"
        )
    assert entry.path is not None

    declared = Path(entry.path)
    origin = declared if declared.is_absolute() else paths.repo_root / declared
    origin = origin.expanduser()
    if not origin.is_file():
        raise FileNotFoundError(
            f"{entry.id}: audio file not found at {origin}. Paths in the source "
            "registry are absolute, or relative to the repository root."
        )
    if origin.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(
            f"{entry.id}: {origin.name} does not have a recognised audio extension "
            f"{list(AUDIO_SUFFIXES)}"
        )

    digest = sha256_file(origin)
    episode_id = make_episode_id(entry.title, digest)

    if entry.source_type == "existing_repository_fixture":
        if not _is_within(origin, paths.repo_root):
            raise ValueError(
                f"{entry.id}: source_type is existing_repository_fixture but "
                f"{origin} is outside the repository. Use local_file instead."
            )
        original_path = paths.relative(origin)
        copied = False
    else:
        destination = paths.raw_dir / f"{episode_id}{origin.suffix.lower()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) == digest:
            copied = False  # identical bytes already imported; nothing to do
        else:
            if destination.exists():
                raise FileExistsError(
                    f"{destination} exists with different content. Refusing to "
                    "overwrite an imported original; delete it deliberately first."
                )
            shutil.copy2(origin, destination)
            copied = True
        original_path = paths.relative(destination)

    transcript_path = None
    if entry.transcript_path:
        transcript = Path(entry.transcript_path)
        transcript_abs = (
            transcript if transcript.is_absolute() else paths.repo_root / transcript
        )
        if not transcript_abs.is_file():
            raise FileNotFoundError(
                f"{entry.id}: declared transcript not found at {transcript_abs}"
            )
        transcript_path = paths.relative(transcript_abs)

    episode = EpisodeRecord(
        episode_id=episode_id,
        series_id=entry.series_id,
        title=entry.title,
        source_type=entry.source_type,
        source_uri=entry.source_uri,
        source_name=entry.source_name,
        license_name=entry.license_name,
        license_url=entry.license_url,
        attribution=entry.attribution,
        language=entry.language,
        content_type=entry.content_type,
        original_path=original_path,
        sha256=digest,
        status="fetched",
        transcript_path=transcript_path,
        is_target_domain=entry.is_target_domain,
        notes=entry.notes,
    )

    previous = (existing or {}).get(episode_id)
    if previous is not None:
        # Preserve everything probe/normalize already established rather than
        # resetting the episode to 'fetched' on every re-import.
        episode = previous.replace(
            series_id=episode.series_id,
            title=episode.title,
            source_type=episode.source_type,
            source_uri=episode.source_uri,
            source_name=episode.source_name,
            license_name=episode.license_name,
            license_url=episode.license_url,
            attribution=episode.attribution,
            language=episode.language,
            content_type=episode.content_type,
            original_path=episode.original_path,
            transcript_path=episode.transcript_path or previous.transcript_path,
            is_target_domain=episode.is_target_domain,
            notes=episode.notes,
        )
        return ImportResult(episode=episode, copied=copied, already_present=True)

    return ImportResult(episode=episode, copied=copied, already_present=False)


def import_sources(
    entries: Sequence[SourceEntry],
    paths: DataPaths,
    existing: dict[str, EpisodeRecord] | None = None,
) -> list[ImportResult]:
    return [import_source(entry, paths, existing) for entry in entries]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
