"""Find generated files under ``data/`` that no manifest episode references.

WHY THIS EXISTS. ``dataset reconcile`` removes episodes the committed registries
no longer declare, and it deliberately stops there: its own docstring records
that it never deletes audio, because ``data/raw`` and ``data/normalized`` are
caches keyed by checksum and re-downloading a show is expensive where a manifest
edit is not. That is the right call for reconciliation, but it leaves a real
gap. After a corpus version bump the manifest holds 77 episodes while the disk
holds the renders and transcripts of every episode ever processed -- so the two
disagree, and anything that walks the directory instead of the manifest counts
the strays.

WHAT IT DOES. Reads the episode manifest, collects every path the episodes
actually reference, walks the generated directories, and reports the difference.

Two directories are referenced two different ways, and conflating them would
make this report worse than useless. ``data/raw`` and ``data/normalized`` are
named explicitly by ``original_path`` and ``normalized_path`` on each episode.
``data/transcripts`` is not: ``transcript_path`` is ``None`` on every episode in
the corpus, because transcripts are a cache addressed by episode id through
``transcription.cache.transcript_path`` rather than a field the manifest
records. Matching transcripts on the manifest field would report all 84 of them
as unreferenced -- every transcript the corpus depends on -- so they are matched
by the same episode-id convention the reader uses.

WHAT IT DOES NOT DO. It does not delete anything, and it takes no flag that
would. This is a reporting command by design: the canonical statistics and
evidence reports derive from the manifest, never from a directory walk, so an
unreferenced render is a disk-space question and not a correctness one. Deciding
that a particular file is genuinely dead is a judgement about whether a corpus
version might come back, and that judgement belongs to a person, who can remove
the file themselves once this report has named it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from slotify_rank.data import manifests

__all__ = ["OrphanReport", "scan_orphans"]

#: Episode fields that name a generated file, and the directory each lives in.
_REFERENCE_FIELDS = ("original_path", "normalized_path", "transcript_path")


@dataclass
class OrphanReport:
    """Generated files on disk that no manifest episode references."""

    episode_count: int = 0
    scanned: dict[str, int] = field(default_factory=dict)
    orphans: dict[str, list[str]] = field(default_factory=dict)

    @property
    def orphan_count(self) -> int:
        return sum(len(paths) for paths in self.orphans.values())

    @property
    def clean(self) -> bool:
        return self.orphan_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "scanned_file_counts": dict(self.scanned),
            "orphan_count": self.orphan_count,
            "orphans": {name: list(paths) for name, paths in self.orphans.items()},
        }

    def lines(self) -> Iterable[str]:
        yield (
            f"Scanned {sum(self.scanned.values())} generated file(s) against "
            f"{self.episode_count} manifest episode(s)."
        )
        for name in sorted(self.scanned):
            found = self.orphans.get(name, [])
            yield f"  {name:<12} {self.scanned[name]:>4} file(s), {len(found)} unreferenced"
            for path in found:
                yield f"      orphan {path}"
        if self.clean:
            yield "Every generated file is referenced by the episode manifest."
        else:
            yield (
                f"{self.orphan_count} generated file(s) are referenced by no "
                "manifest episode. Nothing has been deleted: review them and "
                "remove them by hand if the corpus version that produced them "
                "is not coming back."
            )


def _referenced_paths(episodes: Iterable[Any], repo_root: Path) -> set[Path]:
    """Every existing file the manifest episodes point at, resolved absolutely."""
    referenced: set[Path] = set()
    for episode in episodes:
        for attribute in _REFERENCE_FIELDS:
            value = getattr(episode, attribute, None)
            if not value or str(value) == "None":
                continue
            referenced.add((repo_root / str(value)).resolve())
    return referenced


def scan_orphans(paths: Any) -> OrphanReport:
    """Report generated files under ``data/`` that no episode references."""
    from slotify_rank.transcription.cache import transcript_path

    episodes = manifests.read_episodes(paths.episodes_manifest)
    referenced = _referenced_paths(episodes, paths.repo_root)
    # Transcripts are addressed by episode id, not by a manifest field.
    referenced.update(
        transcript_path(paths, episode.episode_id).resolve() for episode in episodes
    )

    directories = {
        "raw": paths.raw_dir,
        "normalized": paths.normalized_dir,
        "transcripts": paths.transcripts_dir,
    }

    report = OrphanReport(episode_count=len(episodes))
    for name, directory in directories.items():
        if not directory.is_dir():
            report.scanned[name] = 0
            continue
        found = sorted(path for path in directory.iterdir() if path.is_file())
        report.scanned[name] = len(found)
        unreferenced = [
            str(path.relative_to(paths.repo_root).as_posix())
            for path in found
            if path.resolve() not in referenced
        ]
        if unreferenced:
            report.orphans[name] = unreferenced
    return report
