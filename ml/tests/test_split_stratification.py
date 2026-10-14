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

import json
from pathlib import Path

import pytest

from slotify_rank.config.settings import find_repo_root
from slotify_rank.config.versions import (
    LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION,
    SPLIT_ALGORITHM_VERSION,
    STRATIFIED_SPLIT_ALGORITHM_VERSION,
)
from slotify_rank.data.splits import (
    CANONICAL_SPLIT_CONFIG,
    CANONICAL_SPLIT_VERSION,
    InsufficientGroups,
    SplitConfig,
    compute_splits,
    load_split_config,
    read_split_manifest,
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


def _v5_config(**overrides):
    """v4's rules under the corrected per-content-type algorithm, which is
    opt-in: only a pinned config gets it."""
    settings = dict(
        version="v5", pinned_algorithm_version=STRATIFIED_SPLIT_ALGORITHM_VERSION
    )
    settings.update(overrides)
    return _v4_config(**settings)


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


def _hours_by_split_and_type(manifest, content_type_of):
    hours: dict[str, dict[str, float]] = {}
    for assignment in manifest.assignments:
        kind = content_type_of(assignment.group_id)
        bucket = hours.setdefault(assignment.split, {})
        bucket[kind] = bucket.get(kind, 0.0) + assignment.duration_ms / 3_600_000
    return hours


def _minority_podcast_corpus():
    """Twenty narrated series (60 h) and fourteen podcast series (14 h).

    The corpus on which corpus-wide targets put zero podcasts in validation:
    narrated series, barred from test and so assigned first, met validation's
    corpus-wide target on their own, and every podcast went to test or train.
    """
    episodes = []
    for index in range(20):
        episodes += _series(f"narrated-{index:02d}", "narrated", 90.0, offset=index)
    for index in range(14):
        episodes += _series(f"podcast-{index:02d}", "podcast", 30.0, offset=index)
    return episodes


def test_targets_are_computed_within_each_content_type():
    """Under v5's algorithm. (v4's, frozen, is the one that fails this.)"""
    manifest = compute_splits(
        _minority_podcast_corpus(), _v5_config(min_series_per_partition=1)
    )
    hours = _hours_by_split_and_type(manifest, lambda group: group.split("-")[0])

    # Every partition gets its share of the podcasts -- validation included.
    for split in ("train", "validation", "test"):
        assert hours[split].get("podcast", 0.0) > 0.0, (split, hours)
    # Each partition's podcast hours are its ratio of the *podcast* hours (14 h),
    # to within one one-hour series.
    assert abs(hours["validation"]["podcast"] - 0.15 * 14) <= 1.0, hours
    assert abs(hours["test"]["podcast"] - 0.15 * 14) <= 1.0, hours
    assert abs(hours["train"]["podcast"] - 0.70 * 14) <= 1.0, hours
    # Narrated series cannot enter test, so their 70/15 is renormalized over
    # train and validation rather than measured against a share they can never
    # place -- and they do not swallow validation's whole budget.
    assert "narrated" not in hours["test"]
    narrated_validation = hours["validation"]["narrated"] / 60.0
    assert abs(narrated_validation - 0.15 / 0.85) <= 0.05, hours


def test_one_content_type_cannot_exhaust_a_partition_owed_to_another():
    """The order strata are walked in must not decide who gets validation."""
    manifest = compute_splits(
        _minority_podcast_corpus(), _v5_config(min_series_per_partition=1)
    )
    validation = [a.group_id for a in manifest.assignments if a.split == "validation"]
    assert any(group.startswith("podcast-") for group in validation)
    assert any(group.startswith("narrated-") for group in validation)


def test_a_thin_podcast_share_still_fails_the_series_gate_honestly():
    """Fifteen percent of fourteen one-hour podcasts is two series. Under v4's
    three-series gate that is a corpus too thin for a podcast-only test set, and
    the splitter says so instead of padding test with other formats' share."""
    with pytest.raises(InsufficientGroups, match="test"):
        compute_splits(_minority_podcast_corpus(), _v5_config())


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
    # An unpinned stratified config keeps the algorithm v4 was written under;
    # only a pin selects the per-content-type one.
    assert (
        compute_splits(corpus, _v4_config()).algorithm_version
        == LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION
    )
    assert (
        compute_splits(corpus, _v5_config()).algorithm_version
        == STRATIFIED_SPLIT_ALGORITHM_VERSION
    )


def test_v4s_corpus_wide_targets_are_what_v5_corrects():
    """The same rules on the minority-podcast corpus: v4's frozen algorithm
    leaves validation without a podcast, v5's does not."""
    corpus = _minority_podcast_corpus()
    v4 = compute_splits(corpus, _v4_config(min_series_per_partition=1))
    v5 = compute_splits(corpus, _v5_config(min_series_per_partition=1))

    def podcasts_in(manifest, split):
        return [
            a.group_id
            for a in manifest.assignments
            if a.split == split and a.group_id.startswith("podcast-")
        ]

    assert podcasts_in(v4, "validation") == []
    assert podcasts_in(v5, "validation")
    assert v4.algorithm_version == LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION
    assert v5.algorithm_version == STRATIFIED_SPLIT_ALGORITHM_VERSION


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
    # Unpinned, so it keeps the algorithm it was frozen under.
    assert config.pinned_algorithm_version is None
    assert config.algorithm_version == LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION


def test_the_committed_v5_config_pins_the_corrected_algorithm():
    config = load_split_config(_config("splits_v5.yaml"))
    assert config.version == "v5"
    assert config.algorithm_version == STRATIFIED_SPLIT_ALGORITHM_VERSION
    assert STRATIFIED_SPLIT_ALGORITHM_VERSION == "split-stratified-greedy-v1.1.0"
    # v5 carries v4's rules unchanged; only the algorithm moved.
    v4 = load_split_config(_config("splits_v4.yaml"))
    for name in (
        "seed",
        "group_by",
        "train",
        "validation",
        "test",
        "stratify_by",
        "require_target_domain_test",
        "test_content_types",
        "max_group_share_of_partition",
        "min_series_per_partition",
        "min_groups_for_split",
    ):
        assert getattr(config, name) == getattr(v4, name), name


def _pinned(tmp_path: Path, algorithm: str, stratify: bool = True) -> Path:
    path = tmp_path / "splits_pinned.yaml"
    path.write_text(
        "split:\n"
        "  version: pinned\n"
        f"  algorithm_version: {algorithm}\n"
        + ("  stratify_by: content_type\n" if stratify else ""),
        encoding="utf-8",
    )
    return path


def test_a_pin_this_build_does_not_implement_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="not one this build implements"):
        load_split_config(_pinned(tmp_path, "split-stratified-greedy-v9.9.9"))


def test_a_pin_that_contradicts_stratify_by_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="stratify_by"):
        load_split_config(
            _pinned(tmp_path, STRATIFIED_SPLIT_ALGORITHM_VERSION, stratify=False)
        )
    with pytest.raises(ValueError, match="stratify_by"):
        SplitConfig(pinned_algorithm_version=LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION)


def test_a_pinned_algorithm_is_used_exactly(tmp_path: Path):
    for algorithm in (
        LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION,
        STRATIFIED_SPLIT_ALGORITHM_VERSION,
    ):
        assert load_split_config(_pinned(tmp_path, algorithm)).algorithm_version == (
            algorithm
        )
    plain = load_split_config(
        _pinned(tmp_path, SPLIT_ALGORITHM_VERSION, stratify=False)
    )
    assert plain.algorithm_version == SPLIT_ALGORITHM_VERSION


def test_an_unpinned_stratified_config_keeps_the_legacy_algorithm():
    """So `dataset split --config splits_v4.yaml --force` cannot quietly change
    what v4 means."""
    assert (
        SplitConfig(stratify_by="content_type").algorithm_version
        == LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION
    )
    assert SplitConfig().algorithm_version == SPLIT_ALGORITHM_VERSION


def test_the_canonical_split_config_declares_the_canonical_version():
    config = load_split_config(find_repo_root() / CANONICAL_SPLIT_CONFIG)
    assert config.version == CANONICAL_SPLIT_VERSION
    assert config.stratify_by == "content_type"


def test_a_v4_manifest_from_the_legacy_algorithm_still_reads(tmp_path: Path):
    """History stays inspectable even though this build cannot regenerate it."""
    manifest = compute_splits(_v4_shaped_corpus(), _v4_config())
    payload = manifest.to_dict()
    payload["algorithm_version"] = LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION
    path = tmp_path / "splits_v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_split_manifest(path).algorithm_version == (
        LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION
    )


def test_the_committed_v3_config_still_selects_the_original_algorithm():
    """v3 is history, and history has to keep reading."""
    config = load_split_config(_config("splits_v3.yaml"))
    assert config.algorithm_version == SPLIT_ALGORITHM_VERSION


# ---------------------------------------------------------------------------
# v1-v4 regenerate byte for byte
# ---------------------------------------------------------------------------

#: The commit whose ``data/splits.py`` produced splits v1-v4.
_PRE_V5_COMMIT = "2f80944"


def _fuzz_corpora():
    """Seeded corpora of mixed content types, series sizes and durations."""
    import random

    rng = random.Random(20261014)
    kinds = ["podcast", "narrated", "conversational", "music"]
    for _ in range(60):
        episodes = []
        for series in range(rng.randint(6, 40)):
            kind = rng.choice(kinds)
            for index in range(rng.randint(1, 3)):
                episodes.append(
                    make_episode(
                        title=f"s{series} {index}",
                        series_id=f"s{series}",
                        content_type=kind,
                        sha_seed=f"{rng.randint(0, 15):x}",
                        duration_ms=rng.randint(60_000, 7_200_000),
                        episode_id=f"s{series}-e{index}",
                    )
                )
        yield episodes


def _outcome(module, episodes, config_path):
    try:
        return module.compute_splits(episodes, module.load_split_config(config_path))
    except Exception as error:  # the refusal is part of the behaviour too
        return ("error", type(error).__name__, str(error))


def _rendered(outcome) -> str:
    if isinstance(outcome, tuple):
        return json.dumps(list(outcome))
    return json.dumps(outcome.to_dict(), sort_keys=True)


#: SHA-256 over every fuzz corpus's manifest under each committed pre-v5
#: config, as produced by the code at ``_PRE_V5_COMMIT``. Pinned so the check
#: runs without git history; the test below re-derives it from git when it can.
_PRE_V5_DIGESTS = {
    "splits_v1.yaml": "c0b63ad5b2510b0aa111d9b21a72ce6210b16f761e71e0a5150937a0ae61ef62",
    "splits_v2.yaml": "eddf2a521becf6100bfa73dd62b357a4798a87c4f20d76ddacdf60fb4d0624b2",
    "splits_v3.yaml": "b0bad9bf2ceb8f84c551cda09523cfef918a048fae5af42a0e73f55dd7d999e5",
    "splits_v4.yaml": "a8128a220ef546242b1d27c2b9205f334d97dddc41528f91bd5cd28bfb8f7064",
}


def _digest(module, config_name: str) -> str:
    import hashlib

    lines = [
        _rendered(_outcome(module, episodes, _config(config_name)))
        for episodes in _fuzz_corpora()
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "config_name",
    ["splits_v1.yaml", "splits_v2.yaml", "splits_v3.yaml", "splits_v4.yaml"],
)
def test_committed_pre_v5_configs_regenerate_exactly(config_name):
    from slotify_rank.data import splits as current

    assert _digest(current, config_name) == _PRE_V5_DIGESTS[config_name]


def _module_at_pre_v5_commit():
    """``data/splits.py`` as of ``_PRE_V5_COMMIT``, imported in isolation."""
    import shutil
    import subprocess
    import sys
    import types

    from slotify_rank.config import versions

    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    result = subprocess.run(
        [git, "show", f"{_PRE_V5_COMMIT}:ml/src/slotify_rank/data/splits.py"],
        cwd=find_repo_root(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"{_PRE_V5_COMMIT} is not in this checkout's history")
    name = "_slotify_splits_pre_v5"
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        code = compile(result.stdout, f"{_PRE_V5_COMMIT}:splits.py", "exec")
        exec(code, module.__dict__)
    finally:
        sys.modules.pop(name, None)
    # That code stamped its stratified manifests with the constant's value at the
    # time, which is today's legacy name.
    module.STRATIFIED_SPLIT_ALGORITHM_VERSION = (
        versions.LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION
    )
    return module


@pytest.mark.parametrize(
    "config_name",
    ["splits_v1.yaml", "splits_v2.yaml", "splits_v3.yaml", "splits_v4.yaml"],
)
def test_the_restored_legacy_path_matches_the_pre_v5_code(config_name):
    """Manifest for manifest, refusal for refusal, against the old module."""
    from slotify_rank.data import splits as current

    old = _module_at_pre_v5_commit()
    for episodes in _fuzz_corpora():
        assert _rendered(_outcome(current, episodes, _config(config_name))) == (
            _rendered(_outcome(old, episodes, _config(config_name)))
        )


def test_v5_differs_from_v4_only_by_algorithm(tmp_path: Path):
    v4 = load_split_config(_config("splits_v4.yaml"))
    v5 = load_split_config(_config("splits_v5.yaml"))
    assert v4.algorithm_version == LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION
    assert v5.algorithm_version == STRATIFIED_SPLIT_ALGORITHM_VERSION
    manifest = compute_splits(_v4_shaped_corpus(), v5)
    assert manifest.algorithm_version == STRATIFIED_SPLIT_ALGORITHM_VERSION


def test_the_canonical_split_is_defined_once_and_is_the_experiments():
    from slotify_rank.data import splits
    from slotify_rank.experiment import canonical

    assert canonical.CANONICAL_SPLIT_VERSION is splits.CANONICAL_SPLIT_VERSION
    assert "CANONICAL_SPLIT_VERSION =" not in Path(canonical.__file__).read_text(
        encoding="utf-8"
    )
    assert splits.CANONICAL_SPLIT_CONFIG == "ml/configs/splits_v4.yaml"
