"""The v4 split rules, each tested against the failure it exists to prevent.

Split v3 was grouped, leakage-free and non-degraded, and it still shipped a test
partition containing one show and a training set containing none of the target
format. Every check passed. These tests exist so that specific failure cannot
pass again, so each one reconstructs a corpus shaped like the one that produced
it rather than asserting on a convenient abstraction.

Built on fabricated manifests for the same reason `test_splits.py` is: the
repository's real fixtures are a handful of ~20 s clips and cannot honestly
exercise series-aware splitting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slotify_rank.config.versions import (
    SPLIT_ALGORITHM_VERSION,
    STRATIFIED_SPLIT_ALGORITHM_VERSION,
)
from slotify_rank.data.splits import (
    InsufficientGroups,
    SplitConfig,
    compute_splits,
    load_split_config,
)

from tests.dataset_fixtures import make_episode


def _config(name):
    """Repo-relative config path, so this works from either working directory."""
    return Path(__file__).resolve().parents[1] / "configs" / name


_SEEDS = "0123456789abcdef"


def _series(series_id, content_type, minutes, episodes=2, offset=0):
    """One series' worth of episode records."""
    return [
        make_episode(
            title=f"{series_id} {index}",
            series_id=series_id,
            content_type=content_type,
            sha_seed=_SEEDS[(offset + index) % 16],
            duration_ms=int(minutes * 60 * 1000),
            episode_id=f"{series_id}-e{index}",
        )
        for index in range(episodes)
    ]


def _v3_shaped_corpus():
    """The corpus that broke v3: one podcast series, many audiobook series.

    The podcast is also the longest thing in the corpus, which is what made the
    duration-balancing greedy walk hand it a whole partition.
    """
    episodes = _series("the-only-podcast", "podcast", 25.0, episodes=6)
    for index in range(12):
        episodes += _series(
            f"audiobook-{index}", "narrated", 6.0, episodes=2, offset=index
        )
    return episodes


def _v4_shaped_corpus():
    """The corpus v2 of the plan builds: many podcast series of varied length."""
    episodes = []
    # Fourteen podcast series, one of them far longer than the others -- the
    # 59-minute weekly interview show that would otherwise fill a partition.
    lengths = [59.0, 23.0, 20.0, 20.0, 19.0, 18.0, 16.0, 15.0, 14.0, 13.0, 11.0, 10.0, 8.0, 5.0]
    for index, minutes in enumerate(lengths):
        episodes += _series(f"podcast-{index:02d}", "podcast", minutes, offset=index)
    for index in range(10):
        episodes += _series(f"dramatic-{index}", "conversational", 12.0, offset=index)
    for index in range(10):
        episodes += _series(f"narrated-{index}", "narrated", 5.0, offset=index)
    return episodes


def _v4_config(**overrides):
    settings = dict(
        version="v4",
        seed=42,
        group_by="series",
        stratify_by="content_type",
        test_content_types=("podcast",),
        max_group_share_of_partition=0.5,
        min_series_per_partition=3,
    )
    settings.update(overrides)
    return SplitConfig(**settings)


def _by_split(manifest):
    grouped: dict[str, list] = {"train": [], "validation": [], "test": []}
    for assignment in manifest.assignments:
        grouped.setdefault(assignment.split, []).append(assignment)
    return grouped


# ---------------------------------------------------------------------------
# The leakage guarantee must survive every new rule
# ---------------------------------------------------------------------------


def test_stratification_does_not_weaken_the_leakage_guarantee():
    """The entire point of grouping still holds: no series spans partitions."""
    manifest = compute_splits(_v4_shaped_corpus(), _v4_config())
    splits_by_series: dict[str, set[str]] = {}
    for assignment in manifest.assignments:
        splits_by_series.setdefault(assignment.group_id, set()).add(assignment.split)
    assert all(len(splits) == 1 for splits in splits_by_series.values())

    seen: dict[str, str] = {}
    for assignment in manifest.assignments:
        for episode_id in assignment.episode_ids:
            assert episode_id not in seen, f"{episode_id} is in two partitions"
            seen[episode_id] = assignment.split


def test_stratified_splits_are_deterministic():
    corpus = _v4_shaped_corpus()
    first = compute_splits(corpus, _v4_config())
    second = compute_splits(list(reversed(corpus)), _v4_config())
    assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# Rule 1: stratification
# ---------------------------------------------------------------------------


def test_without_stratification_a_minority_format_can_pool_into_one_partition():
    """The v3 failure, reproduced. This is what the other tests are about."""
    manifest = compute_splits(
        _v3_shaped_corpus(),
        SplitConfig(version="v3-shaped", seed=42, group_by="series"),
    )
    podcast_splits = {
        assignment.split
        for assignment in manifest.assignments
        if assignment.group_id == "the-only-podcast"
    }
    # One series can only be in one partition -- that is not the bug. The bug is
    # that nothing noticed the partition it landed in held nothing else like it.
    assert len(podcast_splits) == 1
    landed = podcast_splits.pop()
    others = [
        assignment
        for assignment in manifest.assignments
        if assignment.split == landed and assignment.group_id != "the-only-podcast"
    ]
    assert others == [], (
        "this corpus is supposed to reproduce v3's failure: the lone podcast "
        "series taking a partition to itself"
    )


def test_stratification_spreads_the_target_format_across_partitions():
    manifest = compute_splits(_v4_shaped_corpus(), _v4_config())
    podcast_partitions = {
        assignment.split
        for assignment in manifest.assignments
        if assignment.group_id.startswith("podcast-")
    }
    assert podcast_partitions == {"train", "validation", "test"}


def test_the_unstratified_algorithm_is_untouched():
    """v1-v3 manifests must stay reproducible: same config, same assignments."""
    corpus = _v4_shaped_corpus()
    plain = SplitConfig(version="plain", seed=42, group_by="series")
    manifest = compute_splits(corpus, plain)
    assert manifest.algorithm_version == SPLIT_ALGORITHM_VERSION
    # Recomputing gives the same answer, and the stratified flag is what selects
    # the other algorithm -- nothing else does.
    assert compute_splits(corpus, plain).to_dict() == manifest.to_dict()
    assert (
        compute_splits(corpus, _v4_config()).algorithm_version
        == STRATIFIED_SPLIT_ALGORITHM_VERSION
    )


# ---------------------------------------------------------------------------
# Rule 2: the test partition holds the target format only
# ---------------------------------------------------------------------------


def test_the_test_partition_holds_only_the_declared_content_types():
    manifest = compute_splits(_v4_shaped_corpus(), _v4_config())
    test_groups = [a.group_id for a in _by_split(manifest)["test"]]
    assert test_groups, "the test partition must not be empty"
    assert all(group.startswith("podcast-") for group in test_groups), test_groups


def test_test_content_types_must_be_inside_the_target_domain():
    with pytest.raises(ValueError, match="outside the target domain"):
        SplitConfig(test_content_types=("music",))


# ---------------------------------------------------------------------------
# Rule 3: no series may dominate a partition
# ---------------------------------------------------------------------------


def test_one_long_series_cannot_swallow_a_small_partition():
    manifest = compute_splits(_v4_shaped_corpus(), _v4_config())
    for name, assignments in _by_split(manifest).items():
        total = sum(a.duration_ms for a in assignments)
        largest = max(a.duration_ms for a in assignments)
        assert largest <= 0.75 * total, (
            f"{name} is {largest / total:.0%} one series; the cap is meant to "
            "stop a partition being a measurement of a single show"
        )


def test_the_longest_series_goes_where_there_is_room_for_it():
    """The 59-minute show must not end up in the smallest partition."""
    manifest = compute_splits(_v4_shaped_corpus(), _v4_config())
    placement = {a.group_id: a.split for a in manifest.assignments}
    assert placement["podcast-00"] == "train"


def test_a_series_too_long_for_every_partition_is_still_placed():
    """The rule shapes the split; it must never make one impossible."""
    corpus = _series("enormous", "podcast", 600.0, episodes=2)
    for index in range(8):
        corpus += _series(f"podcast-{index}", "podcast", 10.0, offset=index)
    manifest = compute_splits(
        corpus, _v4_config(min_series_per_partition=1)
    )
    assert "enormous" in {a.group_id for a in manifest.assignments}


def test_max_group_share_is_validated():
    with pytest.raises(ValueError, match="max_group_share_of_partition"):
        SplitConfig(max_group_share_of_partition=0.0)
    with pytest.raises(ValueError, match="max_group_share_of_partition"):
        SplitConfig(max_group_share_of_partition=1.5)


# ---------------------------------------------------------------------------
# Rule 4: a thin partition is a failure, not a warning
# ---------------------------------------------------------------------------


def test_a_single_show_test_partition_fails_the_split():
    """v3's exact outcome, now unrepresentable."""
    corpus = _series("lone-podcast", "podcast", 40.0, episodes=6)
    for index in range(12):
        corpus += _series(f"narrated-{index}", "narrated", 6.0, offset=index)
    with pytest.raises(InsufficientGroups, match="fewer than 3"):
        compute_splits(corpus, _v4_config())


def test_the_gate_names_the_partitions_that_are_thin():
    corpus = _series("lone-podcast", "podcast", 40.0, episodes=6)
    for index in range(12):
        corpus += _series(f"narrated-{index}", "narrated", 6.0, offset=index)
    with pytest.raises(InsufficientGroups) as error:
        compute_splits(corpus, _v4_config())
    assert "test" in str(error.value)
    # And it must not suggest lowering itself.
    assert "widen the corpus" in str(error.value)


def test_min_series_per_partition_defaults_to_the_historical_behaviour():
    assert SplitConfig().min_series_per_partition == 1
    assert SplitConfig().stratify_by is None
    assert SplitConfig().test_content_types is None
    assert SplitConfig().max_group_share_of_partition is None


# ---------------------------------------------------------------------------
# The committed configuration
# ---------------------------------------------------------------------------


def test_the_committed_v4_config_parses_and_selects_the_stratified_algorithm():
    config = load_split_config(_config("splits_v4.yaml"))
    assert config.version == "v4"
    assert config.group_by == "series"
    assert config.seed == 42
    assert config.stratify_by == "content_type"
    assert config.test_content_types == ("podcast",)
    assert config.min_series_per_partition >= 3
    assert config.algorithm_version == STRATIFIED_SPLIT_ALGORITHM_VERSION


def test_the_committed_v3_config_still_selects_the_original_algorithm():
    """v3 is history, and history has to keep reading."""
    config = load_split_config(_config("splits_v3.yaml"))
    assert config.algorithm_version == SPLIT_ALGORITHM_VERSION
