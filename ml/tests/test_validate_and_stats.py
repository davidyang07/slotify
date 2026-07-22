"""Validation failure cases and statistics arithmetic.

Every validation check that can fail is made to fail here, because a validator
that has never rejected anything is not evidence of anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate
from slotify_rank.data.splits import SplitAssignment, SplitManifest, compute_splits
from slotify_rank.data.stats import compute_statistics, write_statistics
from slotify_rank.data.validate import validate_dataset
from slotify_rank.labelling.database import LabelDatabase

from tests.dataset_fixtures import make_candidate, make_episode, write_speech_like_wav


@pytest.fixture()
def paths(tmp_path: Path) -> DataPaths:
    instance = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    instance.mkdirs()
    return instance


def _episode_with_file(paths: DataPaths, **overrides):
    episode = make_episode(**overrides)
    write_speech_like_wav(
        paths.absolute(episode.normalized_path), [(1000, True), (1000, False)]
    )
    original = paths.repo_root / episode.original_path
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"fake original audio")
    return episode


def _checks(report):
    return {finding.check for finding in report.errors}


# --------------------------------------------------------------------------
# Passing case
# --------------------------------------------------------------------------


def test_a_healthy_dataset_passes(paths):
    episode = _episode_with_file(paths)
    candidates = [make_candidate(episode.episode_id, 60_000)]
    report = validate_dataset([episode], candidates, paths)
    assert report.ok, [f.to_dict() for f in report.errors]
    assert report.checks_run


def test_warnings_do_not_fail_validation(paths):
    """A smoke dataset is incomplete, not broken."""
    episode = _episode_with_file(paths)
    report = validate_dataset([episode], [], paths)
    assert report.ok
    assert report.warnings


# --------------------------------------------------------------------------
# Failure cases
# --------------------------------------------------------------------------


def test_duplicate_candidate_ids_fail(paths):
    episode = _episode_with_file(paths)
    candidate = make_candidate(episode.episode_id, 60_000)
    report = validate_dataset([episode], [candidate, candidate], paths)
    assert "duplicate_candidate_ids" in _checks(report)


def test_duplicate_episode_ids_fail(paths):
    episode = _episode_with_file(paths)
    report = validate_dataset([episode, episode], [], paths)
    assert "duplicate_episode_ids" in _checks(report)


def test_candidate_past_the_end_of_the_episode_fails(paths):
    episode = _episode_with_file(paths, duration_ms=60_000)
    candidate = make_candidate(episode.episode_id, 999_000)
    report = validate_dataset([episode], [candidate], paths)
    assert "candidate_within_episode_bounds" in _checks(report)


def test_candidate_referencing_an_unknown_episode_fails(paths):
    report = validate_dataset([], [make_candidate("ghost", 1000)], paths)
    assert "candidate_episode_exists" in _checks(report)


def test_synthetic_candidate_marked_eligible_fails(paths):
    """Constructed by hand to bypass the constructor, so the validator is tested."""
    episode = _episode_with_file(paths)
    candidate = object.__new__(DatasetCandidate)
    payload = make_candidate(
        episode.episode_id,
        60_000,
        sources=("product_padding",),
        is_synthetic=True,
        eligible_for_labelling=False,
        eligible_for_evaluation=False,
    ).to_dict()
    payload["eligible_for_evaluation"] = True
    for key, value in payload.items():
        object.__setattr__(candidate, key, value)
    object.__setattr__(candidate, "candidate_sources", ("product_padding",))
    object.__setattr__(candidate, "merged_from_ms", ())
    report = validate_dataset([episode], [candidate], paths)
    assert "synthetic_candidate_eligibility" in _checks(report)


def test_product_padding_flag_without_synthetic_fails(paths):
    episode = _episode_with_file(paths)
    candidate = make_candidate(episode.episode_id, 60_000, sources=("product_padding",))
    report = validate_dataset([episode], [candidate], paths)
    assert "candidate_source_consistency" in _checks(report)


def test_missing_preprocessing_version_fails(paths):
    episode = _episode_with_file(paths)
    candidate = make_candidate(episode.episode_id, 60_000, preprocessing_version="")
    report = validate_dataset([episode], [candidate], paths)
    assert "candidate_preprocessing_version" in _checks(report)


def test_missing_normalized_audio_fails(paths):
    episode = make_episode()
    report = validate_dataset([episode], [], paths)
    assert "episode_audio_readable" in _checks(report)


def test_checksum_mismatch_fails_a_deep_run(paths):
    episode = _episode_with_file(paths)
    report = validate_dataset([episode], [], paths, deep=True)
    assert "episode_checksums" in _checks(report)


def test_label_on_a_missing_candidate_fails(paths, tmp_path: Path):
    episode = _episode_with_file(paths)
    candidate = make_candidate(episode.episode_id, 60_000)
    database = LabelDatabase(tmp_path / "labels.sqlite3")
    database.register_candidates([candidate])
    database.upsert_label(candidate.candidate_id, "a", 4)
    report = validate_dataset([episode], [], paths, database.all_labels())
    assert "label_candidate_exists" in _checks(report)


def test_episode_split_leakage_fails(paths):
    episode = _episode_with_file(paths)
    manifest = SplitManifest(
        version="v1",
        algorithm_version="split-grouped-greedy-v1.0.0",
        seed=42,
        group_by="series",
        ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
        degraded=False,
        reason=None,
        assignments=(
            SplitAssignment("g1", "train", (episode.episode_id,), 1000, True),
            SplitAssignment("g2", "test", (episode.episode_id,), 1000, True),
        ),
    )
    report = validate_dataset([episode], [], paths, split_manifest=manifest)
    assert "split_leakage_by_episode" in _checks(report)


def test_series_split_leakage_fails(paths):
    """Two episodes of one series in different partitions is the classic leak."""
    first = _episode_with_file(paths, title="Ep one", sha_seed="a", series_id="show")
    second = _episode_with_file(paths, title="Ep two", sha_seed="b", series_id="show")
    manifest = SplitManifest(
        version="v1",
        algorithm_version="split-grouped-greedy-v1.0.0",
        seed=42,
        group_by="episode",
        ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
        degraded=False,
        reason=None,
        assignments=(
            SplitAssignment("g1", "train", (first.episode_id,), 1000, True),
            SplitAssignment("g2", "test", (second.episode_id,), 1000, True),
        ),
    )
    report = validate_dataset([first, second], [], paths, split_manifest=manifest)
    assert "split_leakage_by_series" in _checks(report)


def test_a_test_partition_without_target_domain_audio_fails(paths):
    music = _episode_with_file(paths, content_type="music", series_id="music-show")
    manifest = SplitManifest(
        version="v1",
        algorithm_version="split-grouped-greedy-v1.0.0",
        seed=42,
        group_by="series",
        ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
        degraded=False,
        reason=None,
        assignments=(SplitAssignment("music-show", "test", (music.episode_id,), 1000, False),),
    )
    report = validate_dataset([music], [], paths, split_manifest=manifest)
    assert "test_partition_target_domain" in _checks(report)


def test_candidate_split_inconsistency_fails(paths):
    episode = _episode_with_file(paths)
    candidate = make_candidate(episode.episode_id, 60_000, dataset_split="test")
    manifest = SplitManifest(
        version="v1",
        algorithm_version="split-grouped-greedy-v1.0.0",
        seed=42,
        group_by="series",
        ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
        degraded=False,
        reason=None,
        assignments=(
            SplitAssignment(
                episode.series_id, "train", (episode.episode_id,), 1000, True
            ),
        ),
    )
    report = validate_dataset([episode], [candidate], paths, split_manifest=manifest)
    assert "split_candidate_consistency" in _checks(report)


def test_report_serialises(paths):
    episode = _episode_with_file(paths)
    report = validate_dataset([episode], [], paths)
    destination = paths.artifacts_dir / "validation_report.json"
    report.write(destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["checks_run"]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_statistics_separate_the_headline_quantities(paths, tmp_path: Path):
    podcast = _episode_with_file(paths, title="Pod", sha_seed="a", duration_ms=3_600_000)
    music = _episode_with_file(
        paths, title="Song", sha_seed="b", content_type="music", duration_ms=1_800_000
    )
    candidates = [
        make_candidate(podcast.episode_id, 60_000),
        make_candidate(podcast.episode_id, 120_000),
        make_candidate(
            podcast.episode_id,
            180_000,
            sources=("product_padding",),
            is_synthetic=True,
            eligible_for_labelling=False,
            eligible_for_evaluation=False,
        ),
    ]
    database = LabelDatabase(tmp_path / "labels.sqlite3")
    database.register_candidates([c for c in candidates if not c.is_synthetic])
    database.upsert_label(candidates[0].candidate_id, "a", 4)

    bundle = compute_statistics(
        [podcast, music], candidates, database.all_labels()
    )

    assert bundle.dataset["processed_episode_count"] == 2
    assert bundle.dataset["processed_audio_hours"] == 1.5
    assert bundle.dataset["target_domain_audio_hours"] == 1.0
    assert bundle.dataset["out_of_domain_audio_hours"] == 0.5

    assert bundle.candidates["generated_candidate_count"] == 2
    assert bundle.candidates["synthetic_product_padding_count"] == 1

    assert bundle.labels["human_labelled_candidate_count"] == 1
    assert bundle.labels["weakly_labelled_candidate_count"] == 0
    assert bundle.labels["unlabelled_candidate_count"] == 1
    assert bundle.labels["held_out_evaluation_candidate_count"] == 0
    assert bundle.labels["human_labelled_audio_hours"] == 1.0


def test_synthetic_candidates_never_enter_the_generated_count(paths):
    episode = _episode_with_file(paths)
    synthetic = [
        make_candidate(
            episode.episode_id,
            ms,
            sources=("product_padding",),
            is_synthetic=True,
            eligible_for_labelling=False,
            eligible_for_evaluation=False,
        )
        for ms in (10_000, 20_000, 30_000)
    ]
    bundle = compute_statistics([episode], synthetic)
    assert bundle.candidates["generated_candidate_count"] == 0
    assert bundle.candidates["synthetic_product_padding_count"] == 3
    assert bundle.candidates["synthetic_excluded_from_all_counts"] is True


def test_weak_labels_are_never_counted_as_human(paths):
    episode = _episode_with_file(paths)
    weak = make_candidate(episode.episode_id, 60_000, label_status="weak_heuristic")
    bundle = compute_statistics([episode], [weak])
    assert bundle.labels["human_labelled_candidate_count"] == 0
    assert bundle.labels["weakly_labelled_candidate_count"] == 1


def test_held_out_count_requires_both_human_label_and_test_split(paths, tmp_path: Path):
    episodes = [
        _episode_with_file(paths, title=f"Ep {index}", sha_seed=seed, series_id=f"s{index}")
        for index, seed in enumerate("abcdefgh")
    ]
    manifest = compute_splits(episodes)
    test_episode = next(
        episode
        for episode in episodes
        if manifest.by_episode[episode.episode_id] == "test"
    )
    candidate = make_candidate(test_episode.episode_id, 60_000)
    database = LabelDatabase(tmp_path / "labels.sqlite3")
    database.register_candidates([candidate])
    database.upsert_label(candidate.candidate_id, "a", 5)

    bundle = compute_statistics(
        episodes, [candidate], database.all_labels(), manifest
    )
    assert bundle.labels["held_out_evaluation_candidate_count"] == 1


def test_statistics_write_all_five_artifacts(paths):
    episode = _episode_with_file(paths)
    bundle = compute_statistics([episode], [make_candidate(episode.episode_id, 60_000)])
    written = write_statistics(bundle, paths.artifacts_dir)
    names = {path.name for path in written}
    assert names == {
        "dataset_statistics.json",
        "candidate_statistics.json",
        "label_statistics.json",
        "split_statistics.json",
        "dataset_summary.md",
    }
    summary = (paths.artifacts_dir / "dataset_summary.md").read_text(encoding="utf-8")
    assert "processed_audio_hours" in summary
    assert "GENERATED" in summary


def test_summary_reports_a_degraded_split(paths):
    episode = _episode_with_file(paths)
    manifest = compute_splits([episode])
    bundle = compute_statistics([episode], [], split_manifest=manifest)
    assert bundle.splits["degraded"] is True
    write_statistics(bundle, paths.artifacts_dir)
    summary = (paths.artifacts_dir / "dataset_summary.md").read_text(encoding="utf-8")
    assert "Degraded split" in summary
