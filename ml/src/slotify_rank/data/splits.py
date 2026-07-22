"""Deterministic, group-aware train/validation/test splitting.

The failure this module exists to prevent is leakage. Two episodes of the same
podcast share hosts, room, mic chain, editing rhythm and vocabulary; a model that
sees one in training and the other in test is graded partly on memorisation. So
the unit of assignment is the **series**, never the episode and never the
candidate.

Algorithm ``split-grouped-greedy-v1.0.0``:

1. Group episodes by ``series_id`` (which is never null -- an episode with no
   series is its own series).
2. Order groups by a seeded hash of the group ID. Deterministic and independent
   of manifest order, so adding an episode does not reshuffle everything.
3. Walk the groups largest-duration-first and assign each to whichever partition
   is furthest below its target *duration* share. Balancing on duration rather
   than episode count matters because a 90-minute interview and a 4-minute clip
   are not interchangeable.

Balancing on duration and ordering by hash pull in different directions; the
hash decides ties among equal-duration groups, which is what keeps the result
stable rather than dependent on sort implementation.

Two guard rails:

* **Not enough groups.** With fewer groups than partitions, a three-way split is
  a fiction. The splitter emits a single ``development`` partition and says so,
  rather than producing a "test set" of one clip.
* **Target-domain test sets.** ``require_target_domain_test`` keeps out-of-domain
  material (music; supplemental meeting corpora such as AMI) out of the test
  partition entirely, so headline numbers are always measured on podcast-like
  audio.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import SPLIT_ALGORITHM_VERSION
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.schema import TARGET_DOMAIN_CONTENT_TYPES, EpisodeRecord

__all__ = [
    "SplitConfig",
    "SplitAssignment",
    "SplitManifest",
    "compute_splits",
    "write_split_manifest",
    "read_split_manifest",
    "load_split_config",
    "InsufficientGroups",
]

_PARTITIONS = ("train", "validation", "test")


class InsufficientGroups(RuntimeError):
    """Not enough independent groups to form a meaningful split."""


@dataclass(frozen=True)
class SplitConfig:
    version: str = "v1"
    seed: int = 42
    group_by: str = "series"
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    #: Keep non-target-domain audio (music, meetings) out of the test partition.
    require_target_domain_test: bool = True
    #: Below this many groups, emit a single ``development`` partition instead of
    #: pretending a three-way split is meaningful.
    min_groups_for_split: int = 6

    def __post_init__(self) -> None:
        if self.group_by not in ("series", "episode"):
            raise ValueError(
                f"group_by must be 'series' or 'episode', got {self.group_by!r}"
            )
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train/validation/test ratios must sum to 1.0, got {total}"
            )
        if min(self.train, self.validation, self.test) <= 0:
            raise ValueError("every partition ratio must be positive")
        if self.min_groups_for_split < len(_PARTITIONS):
            raise ValueError(
                f"min_groups_for_split must be at least {len(_PARTITIONS)}"
            )

    @property
    def ratios(self) -> dict[str, float]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class SplitAssignment:
    group_id: str
    split: str
    episode_ids: tuple[str, ...]
    duration_ms: int
    is_target_domain: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "split": self.split,
            "episode_ids": list(self.episode_ids),
            "duration_ms": self.duration_ms,
            "is_target_domain": self.is_target_domain,
        }


@dataclass(frozen=True)
class SplitManifest:
    version: str
    algorithm_version: str
    seed: int
    group_by: str
    ratios: Mapping[str, float]
    degraded: bool
    reason: str | None
    assignments: tuple[SplitAssignment, ...] = field(default_factory=tuple)

    @property
    def by_episode(self) -> dict[str, str]:
        return {
            episode_id: assignment.split
            for assignment in self.assignments
            for episode_id in assignment.episode_ids
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_version": self.version,
            "algorithm_version": self.algorithm_version,
            "seed": self.seed,
            "group_by": self.group_by,
            "ratios": dict(self.ratios),
            "degraded": self.degraded,
            "reason": self.reason,
            "assignments": [a.to_dict() for a in self.assignments],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SplitManifest":
        return cls(
            version=str(raw["split_version"]),
            algorithm_version=str(raw["algorithm_version"]),
            seed=int(raw["seed"]),
            group_by=str(raw["group_by"]),
            ratios=dict(raw["ratios"]),
            degraded=bool(raw["degraded"]),
            reason=raw.get("reason"),
            assignments=tuple(
                SplitAssignment(
                    group_id=str(entry["group_id"]),
                    split=str(entry["split"]),
                    episode_ids=tuple(entry["episode_ids"]),
                    duration_ms=int(entry["duration_ms"]),
                    is_target_domain=bool(entry["is_target_domain"]),
                )
                for entry in raw.get("assignments", ())
            ),
        )


def _group_order_key(group_id: str, seed: int) -> str:
    """Stable pseudo-random ordering key for a group."""
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def compute_splits(
    episodes: Sequence[EpisodeRecord], config: SplitConfig | None = None
) -> SplitManifest:
    """Assign every episode to a partition via its group."""
    config = config or SplitConfig()
    if not episodes:
        raise InsufficientGroups("No episodes to split")

    grouped: dict[str, list[EpisodeRecord]] = {}
    for episode in episodes:
        key = episode.series_id if config.group_by == "series" else episode.episode_id
        grouped.setdefault(key, []).append(episode)

    groups = [
        {
            "group_id": group_id,
            "episode_ids": tuple(sorted(e.episode_id for e in members)),
            "duration_ms": sum(e.duration_ms or 0 for e in members),
            "is_target_domain": all(
                e.is_target_domain and e.content_type in TARGET_DOMAIN_CONTENT_TYPES
                for e in members
            ),
        }
        for group_id, members in grouped.items()
    ]

    if len(groups) < config.min_groups_for_split:
        reason = (
            f"only {len(groups)} independent group(s) under group_by="
            f"{config.group_by!r}; a train/validation/test split needs at least "
            f"{config.min_groups_for_split} to be meaningful. Everything is "
            "assigned to 'development' -- do not report held-out metrics from it."
        )
        assignments = tuple(
            SplitAssignment(
                group_id=str(group["group_id"]),
                split="development",
                episode_ids=tuple(group["episode_ids"]),
                duration_ms=int(group["duration_ms"]),
                is_target_domain=bool(group["is_target_domain"]),
            )
            for group in sorted(
                groups, key=lambda g: _group_order_key(str(g["group_id"]), config.seed)
            )
        )
        return SplitManifest(
            version=config.version,
            algorithm_version=SPLIT_ALGORITHM_VERSION,
            seed=config.seed,
            group_by=config.group_by,
            ratios=config.ratios,
            degraded=True,
            reason=reason,
            assignments=assignments,
        )

    total_duration = sum(int(group["duration_ms"]) for group in groups) or len(groups)
    targets = {
        name: ratio * total_duration for name, ratio in config.ratios.items()
    }
    assigned_duration = {name: 0.0 for name in _PARTITIONS}
    assigned_count = {name: 0 for name in _PARTITIONS}

    # Largest groups first (they constrain the balance most); the seeded hash
    # breaks ties so equal-duration groups land deterministically.
    ordered = sorted(
        groups,
        key=lambda g: (
            -int(g["duration_ms"]),
            _group_order_key(str(g["group_id"]), config.seed),
        ),
    )

    assignments: list[SplitAssignment] = []
    for group in ordered:
        eligible = list(_PARTITIONS)
        if config.require_target_domain_test and not group["is_target_domain"]:
            eligible.remove("test")
        # Every partition must end up non-empty, so a partition with nothing in
        # it outranks pure deficit ordering.
        empty = [name for name in eligible if assigned_count[name] == 0]
        pool = empty or eligible
        chosen = min(
            pool,
            key=lambda name: (
                (assigned_duration[name] - targets[name]) / max(targets[name], 1.0),
                name,
            ),
        )
        assigned_duration[chosen] += int(group["duration_ms"])
        assigned_count[chosen] += 1
        assignments.append(
            SplitAssignment(
                group_id=str(group["group_id"]),
                split=chosen,
                episode_ids=tuple(group["episode_ids"]),
                duration_ms=int(group["duration_ms"]),
                is_target_domain=bool(group["is_target_domain"]),
            )
        )

    empty_partitions = [name for name in _PARTITIONS if assigned_count[name] == 0]
    if empty_partitions:
        raise InsufficientGroups(
            f"Partition(s) {empty_partitions} ended up empty. With "
            f"{len(groups)} group(s) and require_target_domain_test="
            f"{config.require_target_domain_test}, a usable split is not possible; "
            "add more target-domain series."
        )

    assignments.sort(key=lambda assignment: assignment.group_id)
    return SplitManifest(
        version=config.version,
        algorithm_version=SPLIT_ALGORITHM_VERSION,
        seed=config.seed,
        group_by=config.group_by,
        ratios=config.ratios,
        degraded=False,
        reason=None,
        assignments=tuple(assignments),
    )


def write_split_manifest(path: Path, manifest: SplitManifest, force: bool = False) -> None:
    """Write a split manifest, refusing to silently mutate an existing one.

    A split is a commitment: models are trained against it and results are
    reported against it. Overwriting one in place would invalidate every number
    already produced, so an existing manifest with the same version is only
    replaced when ``force`` is set -- otherwise, bump the version.
    """
    destination = Path(path)
    if destination.exists() and not force:
        existing = read_split_manifest(destination)
        if existing.to_dict() == manifest.to_dict():
            return
        raise FileExistsError(
            f"{destination} already exists with different content. A split manifest "
            f"is immutable for a given version: bump split version "
            f"{manifest.version!r} (e.g. to 'v2'), or pass --force if you are "
            "certain no results depend on the current split."
        )
    payload = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n"
    atomic_write_bytes(destination, payload.encode("utf-8"))


def read_split_manifest(path: Path) -> SplitManifest:
    file_path = Path(path)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Split manifest not found: {file_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{file_path} is not valid JSON: {error}") from error
    manifest = SplitManifest.from_mapping(raw)
    if manifest.algorithm_version != SPLIT_ALGORITHM_VERSION:
        raise ValueError(
            f"{file_path} was produced by split algorithm "
            f"{manifest.algorithm_version!r}, but this build implements "
            f"{SPLIT_ALGORITHM_VERSION!r}. Regenerate the split under a new version."
        )
    return manifest


def load_split_config(path: Path | str | None) -> SplitConfig:
    """Load ``ml/configs/splits_v1.yaml``; with no path, use the defaults."""
    if path is None:
        return SplitConfig()

    import yaml

    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    section = loaded.get("split") or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{config_path}: 'split' must be a mapping")
    ratios = section.get("ratios") or {}
    if not isinstance(ratios, Mapping):
        raise ValueError(f"{config_path}: 'split.ratios' must be a mapping")
    payload: dict[str, Any] = {
        key: value for key, value in section.items() if key != "ratios"
    }
    payload.update({key: float(value) for key, value in ratios.items()})
    known = set(SplitConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"{config_path}: unknown split setting(s) {unknown}. Known: {sorted(known)}"
        )
    return SplitConfig(**payload)
