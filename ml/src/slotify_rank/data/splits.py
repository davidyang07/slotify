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

**Stratification** (``stratify_by: content_type``). The greedy walk above
balances total duration and nothing else, which is right when the corpus is
homogeneous and wrong when it is not. A corpus of twenty audiobook series and
fourteen podcasts has one content type worth most of its hours and another worth
most of its *meaning*: the product ranks ad breaks in podcasts. Balancing on
duration alone can put every podcast in one partition -- which is exactly what
split v3 did, leaving a test set of one show and a training set with no podcast
audio in it at all.

Stratifying runs the same greedy assignment once per content type, against
targets computed within that content type: each content type's duration
targets, count targets and running totals are its own, so each partition
receives its share of the podcasts *and* its share of the audiobooks. A content
type barred from a partition (``test_content_types``) has its ratios
renormalized over the partitions it may enter, so audiobooks split 70:15 between
train and validation rather than being measured against a test share they can
never fill. Grouping is untouched: the unit of assignment is still the series, a
series still lands in exactly one partition, and the leakage guarantee is
character-for-character the same one. Only the order of consideration and the
deficit each choice is measured against change.

Algorithm ``split-stratified-greedy-v1.0.0`` (split v4) said the above but
measured every content type against *corpus-wide* targets and totals. Whichever
type went first could then fill a partition's whole budget and leave nothing
for the next: twenty narrated series (60 h) and fourteen podcasts (14 h) put
zero podcasts in validation, because the narrated series had already met
validation's corpus-wide target. ``split-stratified-greedy-v1.1.0`` is the
corrected algorithm; v4 manifests stay readable, but this build cannot
regenerate them, which is why the corrected split is a new split version.

Two guard rails:

* **Not enough groups.** With fewer groups than partitions, a three-way split is
  a fiction. The splitter emits a single ``development`` partition and says so,
  rather than producing a "test set" of one clip.
* **Target-domain test sets.** ``require_target_domain_test`` keeps out-of-domain
  material (music; supplemental meeting corpora such as AMI) out of the test
  partition entirely, so headline numbers are always measured on podcast-like
  audio.
* **Too few series in a partition.** ``min_series_per_partition`` fails the
  split when any partition holds fewer independent series than that. A metric
  macro-averaged over the episodes of a single show is a measurement of that
  show, and a bootstrap over its handful of episodes is an interval around one
  programme's editing style. This is a hard failure rather than a warning
  because the number it protects is the headline one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import (
    SPLIT_ALGORITHM_VERSION,
    STRATIFIED_SPLIT_ALGORITHM_VERSION,
    SUPPORTED_SPLIT_ALGORITHM_VERSIONS,
)
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.schema import TARGET_DOMAIN_CONTENT_TYPES, EpisodeRecord

__all__ = [
    "CANONICAL_SPLIT_CONFIG",
    "CANONICAL_SPLIT_VERSION",
    "SplitConfig",
    "SplitAssignment",
    "SplitManifest",
    "compute_splits",
    "write_split_manifest",
    "read_split_manifest",
    "load_split_config",
    "resolve_split_config",
    "InsufficientGroups",
]

_PARTITIONS = ("train", "validation", "test")

#: The split every command defaults to, as a repository-relative path. One
#: place, so ``dataset split`` cannot quietly default to something different
#: from ``dataset validate`` / ``stats`` / ``prepare-experiment`` -- which is how
#: a bare ``dataset split`` once restamped every candidate with an unstratified
#: v1 split while every reader was looking for v4.
CANONICAL_SPLIT_CONFIG = "ml/configs/splits_v5.yaml"
#: The ``split.version`` that :data:`CANONICAL_SPLIT_CONFIG` declares (a test
#: keeps the two in step).
CANONICAL_SPLIT_VERSION = "v5"

#: The checkout this module was imported from (``<repo>/ml/src/slotify_rank/
#: data/splits.py``), used as the last place a repository-relative config path
#: is looked for.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[4]


def resolve_split_config(
    path: Path | str | None, repo_root: Path | None = None
) -> Path:
    """Where a ``--config`` value (or its absence) points.

    ``None`` means :data:`CANONICAL_SPLIT_CONFIG`. A relative path is tried as
    given (relative to the working directory, which is what an operator typing
    it means), then against ``repo_root``, then against the checkout this code
    lives in -- so the canonical default resolves the same from the repository
    root, from ``ml/``, and under a scratch ``SLOTIFY_REPO_ROOT``. The first
    existing file wins; if none exists the as-given path is returned so the
    error names what was asked for.
    """
    raw = Path(CANONICAL_SPLIT_CONFIG if path is None else path)
    if raw.is_absolute():
        return raw
    for base in (Path.cwd(), repo_root, _CHECKOUT_ROOT):
        if base is None:
            continue
        candidate = Path(base) / raw
        if candidate.is_file():
            return candidate
    return raw


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
    #: Balance each value of this episode field across the partitions separately.
    #: ``None`` reproduces the plain grouped-greedy algorithm exactly.
    stratify_by: str | None = None
    #: Refuse to emit a split where any partition holds fewer independent series
    #: than this. 1 preserves the historical behaviour.
    min_series_per_partition: int = 1
    #: Content types allowed into the *test* partition. ``None`` means "anything
    #: ``require_target_domain_test`` permits", which is the historical
    #: behaviour. Naming the target format here is stronger: it makes the
    #: headline number a measurement of the task the product performs rather
    #: than of the corpus that happened to be available.
    test_content_types: tuple[str, ...] | None = None
    #: The largest share of a partition's duration budget that any one group may
    #: occupy. ``None`` disables the rule. It is the corpus plan's "no show may
    #: dominate" stated where it can be enforced: a partition whose hours are
    #: mostly one series measures that series, and the smaller the partition the
    #: more easily one long show swallows it. A group barred from every
    #: partition by this rule keeps its eligibility rather than becoming
    #: unplaceable -- the rule shapes the split, it can never break it.
    max_group_share_of_partition: float | None = None

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
        if self.stratify_by not in (None, "content_type"):
            raise ValueError(
                f"stratify_by must be None or 'content_type', got "
                f"{self.stratify_by!r}"
            )
        if self.min_series_per_partition < 1:
            raise ValueError("min_series_per_partition must be at least 1")
        if self.test_content_types is not None:
            unknown = sorted(set(self.test_content_types) - TARGET_DOMAIN_CONTENT_TYPES)
            if unknown:
                raise ValueError(
                    f"test_content_types {unknown} are outside the target domain "
                    f"{sorted(TARGET_DOMAIN_CONTENT_TYPES)}; a test partition built "
                    "from them could not support a headline claim"
                )
            if not self.test_content_types:
                raise ValueError("test_content_types, when set, may not be empty")
        if self.max_group_share_of_partition is not None and not (
            0.0 < self.max_group_share_of_partition <= 1.0
        ):
            raise ValueError(
                "max_group_share_of_partition must be in (0, 1], got "
                f"{self.max_group_share_of_partition}"
            )

    @property
    def algorithm_version(self) -> str:
        """The algorithm this configuration selects."""
        return (
            STRATIFIED_SPLIT_ALGORITHM_VERSION
            if self.stratify_by
            else SPLIT_ALGORITHM_VERSION
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


def _count_targets(
    groups: Sequence[Mapping[str, Any]],
    config: "SplitConfig",
    duration_targets: Mapping[str, float],
) -> dict[str, float]:
    """How many groups each partition should expect to hold.

    Not simply ``ratio * len(groups)``. When ``test_content_types`` narrows what
    may enter the test partition, test can only be filled from part of the
    corpus, and asking it for its share of *every* group would make it swallow
    that part whole -- 15 % of forty-four groups drawn from a pool of fourteen
    is half the pool. So each partition's expected count is its duration target
    divided by the mean duration of the groups that are actually eligible for
    it: how many typical *eligible* shows it takes to fill the hours it is owed.
    """
    expected: dict[str, float] = {}
    for name in _PARTITIONS:
        pool = [
            group
            for group in groups
            if not (
                name == "test"
                and (
                    (config.require_target_domain_test and not group["is_target_domain"])
                    or (
                        config.test_content_types
                        and str(group["stratum"]) not in config.test_content_types
                    )
                )
            )
        ]
        mean_duration = (
            sum(int(group["duration_ms"]) for group in pool) / len(pool)
            if pool
            else 0.0
        )
        expected[name] = (
            duration_targets[name] / mean_duration if mean_duration > 0 else 0.0
        )
    return expected


def _group_stratum(
    members: Sequence[EpisodeRecord], stratify_by: str | None
) -> str:
    """Which stratum a group belongs to.

    A group is one series, and a series has a single content type in every
    corpus this project builds. A mixed one is still handled rather than assumed
    away: it lands in a stratum named for its whole sorted set, so it is
    balanced against other identically-mixed series and is never silently
    counted as one of its halves.
    """
    if not stratify_by:
        return "all"
    values = sorted({str(episode.content_type) for episode in members})
    return values[0] if len(values) == 1 else "+".join(values)


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
            "stratum": _group_stratum(members, config.stratify_by),
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
            algorithm_version=config.algorithm_version,
            seed=config.seed,
            group_by=config.group_by,
            ratios=config.ratios,
            degraded=True,
            reason=reason,
            assignments=assignments,
        )

    # Without stratification there is a single stratum holding every group, and
    # everything below reduces to the original grouped-greedy walk.
    strata: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        strata.setdefault(str(group["stratum"]), []).append(group)

    def _eligible(group: Mapping[str, Any]) -> list[str]:
        """Partitions this group may be assigned to."""
        names = list(_PARTITIONS)
        if config.require_target_domain_test and not group["is_target_domain"]:
            names.remove("test")
        if (
            config.test_content_types
            and "test" in names
            and str(group["stratum"]) not in config.test_content_types
        ):
            names.remove("test")
        return names

    def _stratum_targets(members: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Duration each partition is owed *from this stratum*.

        The stratum's own hours times the partition ratios. Under stratification
        the ratios are renormalized over the partitions the stratum may enter at
        all, so a content type barred from test is split 70:15 between train and
        validation instead of being measured against hours it can never place.
        Without stratification the single stratum is the whole corpus and the
        targets are exactly the original algorithm's.
        """
        total = sum(int(group["duration_ms"]) for group in members) or len(members)
        if not config.stratify_by:
            return {name: ratio * total for name, ratio in config.ratios.items()}
        open_names = [
            name
            for name in _PARTITIONS
            if any(name in _eligible(group) for group in members)
        ]
        open_ratio = sum(config.ratios[name] for name in open_names)
        return {
            name: (
                config.ratios[name] / open_ratio * total
                if name in open_names and open_ratio > 0
                else 0.0
            )
            for name in _PARTITIONS
        }

    stratum_targets = {
        stratum: _stratum_targets(members) for stratum, members in strata.items()
    }
    stratum_count_targets = {
        stratum: _count_targets(members, config, stratum_targets[stratum])
        for stratum, members in strata.items()
    }
    #: A partition's whole budget is what every stratum owes it. Used by the
    #: no-series-may-dominate rule, which is about the partition, not a stratum.
    partition_targets = {
        name: sum(targets[name] for targets in stratum_targets.values())
        for name in _PARTITIONS
    }
    assigned_count = {name: 0 for name in _PARTITIONS}
    assignments: list[SplitAssignment] = []

    # Per-stratum state, rebound for each stratum in the walk below. Measuring a
    # stratum's deficit against its own targets and its own running totals is
    # the whole of stratification; measuring it against corpus-wide ones lets an
    # earlier stratum exhaust a partition the later one was owed a share of.
    targets: dict[str, float] = {}
    count_targets: dict[str, float] = {}
    assigned_duration: dict[str, float] = {}
    stratum_count: dict[str, int] = {}

    def _fullness(name: str) -> tuple[float, float, str]:
        """How full a partition is, as the sort key that picks the emptiest.

        Every quantity here is the current stratum's: its targets, its hours
        and groups placed so far. Under stratification the *group count* leads
        and duration breaks the tie; without it, duration alone decides, exactly
        as the original grouped-greedy algorithm did.

        Count has to lead because of an arithmetic trap: one show can be longer
        than a whole partition's share of the hours -- a two-hour series against
        a 2.8-hour test target -- so pure duration balancing hands that partition
        most of its quota in a single group and effectively closes it. A
        partition holding one show is the failure v3 shipped.

        The *larger* of the two measures is taken, so a partition that has met
        either its share of the shows or its share of the hours stops attracting
        more. Count alone would let a partition restricted to long-form shows
        overshoot its hours badly; duration alone is the trap above.

        Ties break on remaining capacity, largest first, rather than
        alphabetically. At the start every partition is equally empty, and an
        alphabetic tie-break hands the single longest show in the corpus to
        'test' -- which then spends most of its budget on one series and has room
        for nothing else. Sending the longest shows where there is the most room
        leaves the small partitions to be filled by several small series, which
        is the point.
        """
        by_duration = (assigned_duration[name] - targets[name]) / max(
            targets[name], 1.0
        )
        if not config.stratify_by:
            return (by_duration, 0.0, name)
        by_count = (stratum_count[name] - count_targets[name]) / max(
            count_targets[name], 1.0
        )
        return (
            max(by_count, by_duration),
            assigned_duration[name] - targets[name],
            name,
        )

    def _stratum_key(name: str) -> tuple[int, int, str]:
        """Most-constrained stratum first, then longest, then by name.

        A stratum barred from the test partition can only compete for train and
        validation, so letting it go first means the strata that *are* eligible
        for test see how much room is left and fill it. The other order fills
        test from whatever happened to be largest and then overfills train with
        the material that had nowhere else to go.

        Eligibility is taken as the narrowest any member has, not the first
        member's: a stratum is one content type today, but nothing here should
        depend on that staying true.
        """
        members = strata[name]
        return (
            min(len(_eligible(group)) for group in members),
            -sum(int(group["duration_ms"]) for group in members),
            name,
        )

    for stratum in sorted(strata, key=_stratum_key):
        targets = stratum_targets[stratum]
        count_targets = stratum_count_targets[stratum]
        assigned_duration = {name: 0.0 for name in _PARTITIONS}
        stratum_count = {name: 0 for name in _PARTITIONS}
        # Largest groups first (they constrain the balance most); the seeded
        # hash breaks ties so equal-duration groups land deterministically.
        ordered = sorted(
            strata[stratum],
            key=lambda g: (
                -int(g["duration_ms"]),
                _group_order_key(str(g["group_id"]), config.seed),
            ),
        )

        for group in ordered:
            eligible = _eligible(group)
            if not eligible:
                raise InsufficientGroups(
                    f"group {group['group_id']!r} (content type "
                    f"{group['stratum']!r}) is eligible for no partition"
                )

            share = config.max_group_share_of_partition
            if share is not None:
                roomy = [
                    name
                    for name in eligible
                    if int(group["duration_ms"]) <= share * partition_targets[name]
                ]
                # Only when some partition can still take it. A group larger
                # than every partition's allowance is a corpus problem, not a
                # reason to fail the split.
                eligible = roomy or eligible

            # Every partition must end up non-empty, so one with nothing in it
            # outranks pure deficit ordering.
            empty = [name for name in eligible if assigned_count[name] == 0]
            chosen = min(empty or eligible, key=_fullness)

            assigned_duration[chosen] += int(group["duration_ms"])
            stratum_count[chosen] += 1
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

    thin = {
        name: assigned_count[name]
        for name in _PARTITIONS
        if assigned_count[name] < config.min_series_per_partition
    }
    if thin:
        raise InsufficientGroups(
            f"Partition(s) {thin} hold fewer than "
            f"{config.min_series_per_partition} independent "
            f"{config.group_by} group(s). A metric macro-averaged over one "
            "group's episodes measures that group and not the task; widen the "
            "corpus rather than lowering min_series_per_partition."
        )

    assignments.sort(key=lambda assignment: assignment.group_id)
    return SplitManifest(
        version=config.version,
        algorithm_version=config.algorithm_version,
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
    if manifest.algorithm_version not in SUPPORTED_SPLIT_ALGORITHM_VERSIONS:
        raise ValueError(
            f"{file_path} was produced by split algorithm "
            f"{manifest.algorithm_version!r}, but this build implements "
            f"{list(SUPPORTED_SPLIT_ALGORITHM_VERSIONS)}. Regenerate the split "
            "under a new version."
        )
    return manifest


def load_split_config(path: Path | str | None) -> SplitConfig:
    """Load a split config such as ``ml/configs/splits_v5.yaml``.

    With no path, returns ``SplitConfig()`` -- the historical *unstratified* v1
    defaults. Commands do not rely on that: they default to
    :data:`CANONICAL_SPLIT_CONFIG`.

    A config may pin ``split.algorithm_version``. When it does, loading fails
    unless this build implements exactly that algorithm for it, so a split
    version can never be regenerated by a different algorithm than the one it
    was frozen under -- the manifest it produced would silently differ.
    """
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
    pinned_algorithm = section.get("algorithm_version")
    payload: dict[str, Any] = {
        key: value
        for key, value in section.items()
        if key not in ("ratios", "algorithm_version")
    }
    payload.update({key: float(value) for key, value in ratios.items()})
    if isinstance(payload.get("test_content_types"), list):
        payload["test_content_types"] = tuple(
            str(value) for value in payload["test_content_types"]
        )
    known = set(SplitConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"{config_path}: unknown split setting(s) {unknown}. Known: {sorted(known)}"
        )
    config = SplitConfig(**payload)
    if (
        pinned_algorithm is not None
        and str(pinned_algorithm) != config.algorithm_version
    ):
        raise ValueError(
            f"{config_path} pins split algorithm {pinned_algorithm!r}, but this "
            f"build implements {config.algorithm_version!r} for these settings. "
            "A split version is frozen under the algorithm that produced it; "
            "create a new split version rather than regenerating this one."
        )
    return config
