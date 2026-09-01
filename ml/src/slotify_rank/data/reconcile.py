"""Bring the episode manifest back into line with the committed registries.

WHY THIS EXISTS. ``dataset fetch`` is additive, and it has to be: it is
cache-aware, resumable, and must never delete audio somebody is part-way through
processing. The consequence is that the episode manifest is a record of
everything ever fetched, not of what the corpus plan currently declares. Those
two are the same thing right up until the plan changes, and then they silently
are not.

That silence is the problem. The dataset card promises that "a clone reproduces
exactly this corpus", and the experiment manifest hashes the source registries to
make the promise checkable. Neither is worth anything if the manifest on the
machine that actually ran the pipeline holds episodes no registry declares. When
the benchmark corpus went from plan v1 to plan v2 the manifest kept:

* three extra episodes of *Houston We Have a Podcast*, because v1 allowed six and
  v2 caps the show at three -- so the show would have entered the split at twice
  its declared weight; and
* seven synthetic smoke fixtures from earlier development runs, which no
  registry has ever declared.

Neither would have failed a check. The split would have been computed over them,
the statistics would have counted them, and the numbers would have described a
corpus that exists on one laptop.

WHAT IT DOES. Compares the manifest against the registries it is given, joining
on ``source_uri`` -- the URL is the thing a registry declares and an episode
records, and unlike the episode id it is stable across id-scheme changes. An
episode whose URI no registry declares is removed from the manifest, along with
its candidate and feature rows, and every removal is reported with its reason.

WHAT IT DOES NOT DO. It never deletes audio. ``data/raw`` and ``data/normalized``
are caches keyed by checksum; leaving a file there costs disk and nothing else,
and deleting it would make re-adding a show an expensive re-download rather than
a manifest edit. Reconciliation is a statement about what the corpus *is*, not a
disk-cleaning tool.

Locally imported material (``local_file`` sources and repository fixtures) is
kept when ``keep_local`` is set, because no remote registry can declare it and
dropping it would make ``--fixtures`` useless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from slotify_rank.data.manifests import read_jsonl, write_jsonl

__all__ = ["ReconcileReport", "declared_source_uris", "reconcile_manifest"]

#: Source types that no downloadable registry can declare.
_LOCAL_SOURCE_TYPES = frozenset({"local_file", "existing_repository_fixture"})


@dataclass
class ReconcileReport:
    """What reconciliation removed, and why."""

    registries: tuple[str, ...] = ()
    episodes_before: int = 0
    episodes_after: int = 0
    removed: list[dict[str, Any]] = field(default_factory=list)
    candidates_removed: int = 0
    features_removed: int = 0
    kept_local: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.removed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "registries": list(self.registries),
            "episodes_before": self.episodes_before,
            "episodes_after": self.episodes_after,
            "episodes_removed": len(self.removed),
            "candidates_removed": self.candidates_removed,
            "features_removed": self.features_removed,
            "kept_local": self.kept_local,
            "removed": list(self.removed),
        }

    def lines(self) -> Iterable[str]:
        yield (
            f"Reconciled {self.episodes_before} manifest episode(s) against "
            f"{len(self.registries)} registry/registries: "
            f"{self.episodes_after} kept, {len(self.removed)} removed."
        )
        if self.kept_local:
            yield f"  kept {self.kept_local} locally imported episode(s)"
        by_series: dict[str, int] = {}
        for entry in self.removed:
            by_series[str(entry["series_id"])] = (
                by_series.get(str(entry["series_id"]), 0) + 1
            )
        for series, count in sorted(by_series.items()):
            yield f"  removed {count:>3} episode(s) from {series}"
        if self.candidates_removed or self.features_removed:
            yield (
                f"  removed {self.candidates_removed} candidate row(s) and "
                f"{self.features_removed} feature row(s) belonging to them"
            )


def declared_source_uris(registries: Sequence[Path | str]) -> set[str]:
    """Every ``url`` the given source registries declare."""
    import yaml

    declared: set[str] = set()
    for registry in registries:
        path = Path(registry)
        if not path.is_file():
            raise FileNotFoundError(f"Source registry not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in loaded.get("sources") or ():
            url = entry.get("url") if isinstance(entry, dict) else None
            if url:
                declared.add(str(url))
    return declared


def reconcile_manifest(
    *,
    episodes_path: Path,
    candidates_path: Path | None,
    features_path: Path | None,
    registries: Sequence[Path | str],
    keep_local: bool = True,
    dry_run: bool = False,
) -> ReconcileReport:
    """Drop manifest episodes that no registry declares, and their rows."""
    declared = declared_source_uris(registries)
    report = ReconcileReport(registries=tuple(str(r) for r in registries))

    rows = [row for _, row in read_jsonl(Path(episodes_path))]
    report.episodes_before = len(rows)

    kept: list[dict[str, Any]] = []
    dropped_ids: set[str] = set()
    for row in rows:
        source_type = str(row.get("source_type") or "")
        if keep_local and source_type in _LOCAL_SOURCE_TYPES:
            report.kept_local += 1
            kept.append(row)
            continue
        uri = str(row.get("source_uri") or "")
        if uri in declared:
            kept.append(row)
            continue
        dropped_ids.add(str(row.get("episode_id")))
        report.removed.append(
            {
                "episode_id": str(row.get("episode_id")),
                "series_id": str(row.get("series_id") or ""),
                "source_uri": uri,
                "source_type": source_type,
                "reason": (
                    "no committed source registry declares this URL; it is left "
                    "over from an earlier corpus version"
                ),
            }
        )
    report.episodes_after = len(kept)

    if not dropped_ids:
        return report

    def _prune(path: Path | None) -> int:
        if path is None or not Path(path).is_file():
            return 0
        surviving = [
            row
            for _, row in read_jsonl(Path(path))
            if str(row.get("episode_id")) not in dropped_ids
        ]
        removed = sum(1 for _ in read_jsonl(Path(path))) - len(surviving)
        if not dry_run:
            write_jsonl(Path(path), surviving)
        return removed

    report.candidates_removed = _prune(candidates_path)
    report.features_removed = _prune(features_path)

    if not dry_run:
        write_jsonl(Path(episodes_path), kept)

    return report


def write_report(path: Path, report: ReconcileReport) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
