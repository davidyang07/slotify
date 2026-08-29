"""Split determinism and leakage prevention.

Deliberately built on fabricated manifests. The repository's real fixtures are
four series of ~20 s clips, which cannot honestly exercise series-aware
splitting; inventing manifest records is more truthful than pretending they can.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from slotify_rank.config.settings import find_repo_root
from slotify_rank.data.splits import (
    InsufficientGroups,
    SplitConfig,
    compute_splits,
    load_split_config,
    read_split_manifest,
    write_split_manifest,
)

from tests.dataset_fixtures import make_episode


def _corpus(series_count: int = 8, per_series: int = 3):
    episodes = []
    for series_index in range(series_count):
        for episode_index in range(per_series):
            seed = "0123456789abcdef"[(series_index * per_series + episode_index) % 16]
            episodes.append(
                make_episode(
                    title=f"Series {series_index} episode {episode_index}",
                    series_id=f"series-{series_index}",
                    sha_seed=seed,
                    duration_ms=(1200 + series_index * 100) * 1000,
                    episode_id=f"s{series_index}-e{episode_index}",
                )
            )
    return episodes


def test_no_episode_appears_in_two_partitions():
    manifest = compute_splits(_corpus())
    seen: dict[str, str] = {}
    for assignment in manifest.assignments:
        for episode_id in assignment.episode_ids:
            assert episode_id not in seen, f"{episode_id} is in two partitions"
            seen[episode_id] = assignment.split


def test_no_series_spans_partitions():
    """The leakage this whole module exists to prevent."""
    manifest = compute_splits(_corpus())
    splits_by_series: dict[str, set[str]] = {}
    for assignment in manifest.assignments:
        splits_by_series.setdefault(assignment.group_id, set()).add(assignment.split)
    assert all(len(splits) == 1 for splits in splits_by_series.values())


def test_every_episode_of_a_series_lands_together():
    episodes = _corpus()
    by_episode = compute_splits(episodes).by_episode
    for series_index in range(8):
        series_episodes = [e for e in episodes if e.series_id == f"series-{series_index}"]
        splits = {by_episode[e.episode_id] for e in series_episodes}
        assert len(splits) == 1


def test_splitting_is_deterministic_for_a_seed():
    episodes = _corpus()
    assert compute_splits(episodes).to_dict() == compute_splits(episodes).to_dict()


def test_splitting_is_independent_of_manifest_order():
    episodes = _corpus()
    forward = compute_splits(episodes).by_episode
    reverse = compute_splits(list(reversed(episodes))).by_episode
    assert forward == reverse


def test_changing_the_seed_can_change_the_assignment():
    episodes = _corpus()
    first = compute_splits(episodes, SplitConfig(seed=1)).by_episode
    second = compute_splits(episodes, SplitConfig(seed=999)).by_episode
    assert first.keys() == second.keys()


def test_all_three_partitions_are_populated():
    manifest = compute_splits(_corpus())
    assert {a.split for a in manifest.assignments} == {"train", "validation", "test"}
    assert not manifest.degraded


def test_train_receives_the_largest_duration_share():
    manifest = compute_splits(_corpus())
    duration: dict[str, int] = {}
    for assignment in manifest.assignments:
        duration[assignment.split] = (
            duration.get(assignment.split, 0) + assignment.duration_ms
        )
    assert duration["train"] == max(duration.values())


def test_out_of_domain_series_are_kept_out_of_the_test_split():
    episodes = _corpus()
    episodes += [
        make_episode(
            title="Music one",
            series_id="music-series",
            content_type="music",
            episode_id="music-e0",
            sha_seed="f",
        )
    ]
    manifest = compute_splits(episodes)
    test_groups = {a.group_id for a in manifest.assignments if a.split == "test"}
    assert "music-series" not in test_groups


def test_too_few_groups_degrades_to_development():
    """A smoke dataset must not pretend to have a held-out test set."""
    manifest = compute_splits(_corpus(series_count=2, per_series=2))
    assert manifest.degraded is True
    assert {a.split for a in manifest.assignments} == {"development"}
    assert "development" in manifest.reason


def test_empty_corpus_is_rejected():
    with pytest.raises(InsufficientGroups, match="No episodes"):
        compute_splits([])


def test_grouping_by_episode_is_available():
    manifest = compute_splits(_corpus(), SplitConfig(group_by="episode"))
    assert manifest.group_by == "episode"
    assert len(manifest.assignments) == 24


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train=0.5, validation=0.2, test=0.2)


def test_invalid_group_by_is_rejected():
    with pytest.raises(ValueError, match="group_by"):
        SplitConfig(group_by="candidate")


# --------------------------------------------------------------------------
# Manifest persistence and immutability
# --------------------------------------------------------------------------


def test_manifest_round_trips(tmp_path: Path):
    manifest = compute_splits(_corpus())
    path = tmp_path / "splits_v1.json"
    write_split_manifest(path, manifest)
    assert read_split_manifest(path).to_dict() == manifest.to_dict()


def test_rewriting_an_identical_manifest_is_a_no_op(tmp_path: Path):
    manifest = compute_splits(_corpus())
    path = tmp_path / "splits_v1.json"
    write_split_manifest(path, manifest)
    write_split_manifest(path, manifest)
    assert read_split_manifest(path).to_dict() == manifest.to_dict()


def test_a_split_manifest_is_immutable_without_force(tmp_path: Path):
    path = tmp_path / "splits_v1.json"
    write_split_manifest(path, compute_splits(_corpus(), SplitConfig(seed=1)))
    with pytest.raises(FileExistsError, match="immutable"):
        write_split_manifest(path, compute_splits(_corpus(), SplitConfig(seed=2)))


def test_force_overwrites_a_split_manifest(tmp_path: Path):
    path = tmp_path / "splits_v1.json"
    write_split_manifest(path, compute_splits(_corpus(), SplitConfig(seed=1)))
    replacement = compute_splits(_corpus(), SplitConfig(seed=2))
    write_split_manifest(path, replacement, force=True)
    assert read_split_manifest(path).seed == 2


def test_a_foreign_algorithm_version_is_rejected(tmp_path: Path):
    import json

    path = tmp_path / "splits_v1.json"
    write_split_manifest(path, compute_splits(_corpus()))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["algorithm_version"] = "split-something-else-v9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split algorithm"):
        read_split_manifest(path)


def test_repository_split_config_loads():
    config = load_split_config(find_repo_root() / "ml" / "configs" / "splits_v1.yaml")
    assert (config.train, config.validation, config.test) == (0.70, 0.15, 0.15)
    assert config.group_by == "series"
    assert config.require_target_domain_test is True


def test_unknown_split_setting_is_rejected(tmp_path: Path):
    path = tmp_path / "splits.yaml"
    path.write_text("split:\n  nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown split setting"):
        load_split_config(path)


def test_every_committed_split_config_loads_and_groups_on_series():
    """A split that grouped on the episode would let one show span partitions."""
    configs = sorted(
        (find_repo_root() / "ml" / "configs").glob("splits_v*.yaml")
    )
    assert len(configs) >= 3, "expected at least v1, v2 and v3"
    for path in configs:
        config = load_split_config(path)
        assert config.group_by == "series", path.name
        assert config.require_target_domain_test is True, path.name
        assert (config.train, config.validation, config.test) == (0.70, 0.15, 0.15)


def test_the_experiments_split_config_matches_what_the_experiment_declares():
    """Two files describe this split. They must not disagree."""
    from slotify_rank.experiment.canonical import load_experiment_config

    experiment = load_experiment_config(
        find_repo_root() / "ml" / "configs" / "experiment_resume_v1.yaml"
    )
    config = load_split_config(find_repo_root() / str(experiment.split["config"]))
    assert config.version == experiment.split_version
    assert config.group_by == experiment.split["group_by"]
    assert config.seed == experiment.split["seed"]
    for name, ratio in experiment.split["ratios"].items():
        assert getattr(config, name) == ratio


def test_regenerating_a_split_from_the_committed_config_is_byte_identical(
    tmp_path: Path,
):
    """Determinism the experiment manifest's split hash depends on."""
    config = load_split_config(find_repo_root() / "ml" / "configs" / "splits_v3.yaml")
    episodes = [
        make_episode(
            title=f"Episode {index}",
            series_id=f"series-{index % 9}",
            sha_seed=chr(ord("a") + index),
            duration_ms=300_000 + index * 11_000,
        )
        for index in range(18)
    ]
    first = compute_splits(episodes, config)
    second = compute_splits(list(reversed(episodes)), config)
    assert first.to_dict() == second.to_dict()

    path = tmp_path / "splits_v3.json"
    write_split_manifest(path, first)
    before = path.read_bytes()
    write_split_manifest(path, second)  # identical content is a no-op, not an error
    assert path.read_bytes() == before
