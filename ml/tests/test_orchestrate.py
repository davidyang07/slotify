"""The one-command corpus preparation: step order, failure handling, readiness.

Nothing here runs a real stage. The steps are replaced with recorders, so what
is under test is the orchestration -- ordering, stopping, and the readiness
*measurement*, which is the part an operator acts on before spending hours
labelling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slotify_rank.data import manifests
from slotify_rank.data.paths import DataPaths
from slotify_rank.experiment.orchestrate import (
    PreparationStep,
    build_steps,
    run_preparation,
    summarise_readiness,
)
from tests.dataset_fixtures import make_candidate, make_episode


def _step(name: str, code: int, log: list[str]) -> PreparationStep:
    def run(_args: argparse.Namespace) -> int:
        log.append(name)
        return code

    return PreparationStep(
        name=name, description=name, run=run, args=argparse.Namespace()
    )


# --------------------------------------------------------------------------
# Step sequencing
# --------------------------------------------------------------------------


def test_every_step_runs_in_order_when_all_succeed():
    log: list[str] = []
    steps = [_step(name, 0, log) for name in ("a", "b", "c")]
    report = run_preparation(steps, "corpus-v1", log=lambda _: None)
    assert log == ["a", "b", "c"]
    assert report.ok is True
    assert [entry.name for entry in report.steps] == ["a", "b", "c"]


def test_a_failing_step_stops_the_run_because_later_steps_depend_on_it():
    log: list[str] = []
    steps = [_step("a", 0, log), _step("b", 1, log), _step("c", 0, log)]
    report = run_preparation(steps, "corpus-v1", log=lambda _: None)
    assert log == ["a", "b"]
    assert report.ok is False


def test_a_raising_step_is_recorded_rather_than_propagated():
    def boom(_args: argparse.Namespace) -> int:
        raise RuntimeError("ffprobe exploded")

    steps = [
        PreparationStep("probe", "probe", boom, argparse.Namespace()),
    ]
    report = run_preparation(steps, "corpus-v1", log=lambda _: None)
    assert report.ok is False
    assert "ffprobe exploded" in report.steps[0].error


def test_the_default_step_list_is_in_dependency_order():
    steps = build_steps(
        data_root=None,
        source_registries=["ml/configs/sources_resume_v1.yaml"],
        dataset_config=None,
        split_config=None,
        split_version="v3",
        queue_config=None,
        queue_output=None,
        fixtures_registry=None,
        fetch_timeout=60.0,
        skip_fetch=False,
        skip_features=False,
        force_split=False,
        force_queue=False,
        limit=None,
    )
    names = [step.name for step in steps]
    assert names == [
        "fetch:sources_resume_v1",
        "probe",
        "normalize",
        "candidates",
        "split",
        "features",
        "validate",
        "stats",
        "queue",
    ]
    # Candidates must exist before the split stamps a partition onto them, and
    # the queue must come after both.
    assert names.index("candidates") < names.index("split") < names.index("queue")


def test_fetching_and_featurising_can_be_skipped_independently():
    common = dict(
        data_root=None,
        source_registries=["a.yaml"],
        dataset_config=None,
        split_config=None,
        split_version="v3",
        queue_config=None,
        queue_output=None,
        fixtures_registry=None,
        fetch_timeout=60.0,
        force_split=False,
        force_queue=False,
        limit=None,
    )
    without_fetch = [
        step.name for step in build_steps(**common, skip_fetch=True, skip_features=False)
    ]
    without_features = [
        step.name for step in build_steps(**common, skip_fetch=False, skip_features=True)
    ]
    assert not any(name.startswith("fetch") for name in without_fetch)
    assert "features" in without_fetch
    assert "features" not in without_features


# --------------------------------------------------------------------------
# Readiness measurement
# --------------------------------------------------------------------------


def _corpus(tmp_path: Path, splits: dict[str, str], per_episode: int = 5):
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episodes = []
    candidates = []
    for index, (series, split) in enumerate(sorted(splits.items())):
        episode = make_episode(
            title=f"Episode {index}",
            series_id=series,
            sha_seed=chr(ord("a") + index),
        )
        episodes.append(episode)
        for step in range(per_episode):
            candidates.append(
                make_candidate(
                    episode.episode_id,
                    timestamp_ms=30_000 + step * 20_000,
                    dataset_split=split,
                )
            )
    manifests.write_episodes(paths.episodes_manifest, episodes)
    manifests.write_candidates(paths.candidates_manifest, candidates)
    return paths, episodes, candidates


def test_readiness_counts_what_exists_and_names_the_shortfall(tmp_path: Path):
    paths, _, candidates = _corpus(
        tmp_path, {"s1": "train", "s2": "validation", "s3": "test"}
    )
    summary = summarise_readiness(
        paths, tmp_path / "no-queue.json", target_labels=2400, split_version="v3"
    )
    assert summary.generated_candidate_count == len(candidates)
    assert summary.labellable_candidate_count == len(candidates)
    assert summary.queued_unique_count == 0
    assert summary.ready_to_label is False
    assert any("2400" in reason for reason in summary.blocking_reasons)


def test_readiness_reports_a_missing_partition_as_blocking(tmp_path: Path):
    paths, _, _ = _corpus(tmp_path, {"s1": "train", "s2": "validation"})
    summary = summarise_readiness(
        paths, tmp_path / "q.json", target_labels=1, split_version="v3"
    )
    assert any("'test' split" in reason for reason in summary.blocking_reasons)


def test_readiness_reads_the_queue_when_one_exists(tmp_path: Path):
    from slotify_rank.labelling.queue import QueueConfig, build_queue, write_queue

    paths, episodes, candidates = _corpus(
        tmp_path,
        {f"s{i}": ["train", "validation", "test"][i % 3] for i in range(9)},
        per_episode=6,
    )
    queue = build_queue(
        candidates,
        episodes,
        config=QueueConfig(
            target_unique=20,
            pilot_size=2,
            overlap_size=2,
            consistency_size=2,
            max_per_episode=10,
            include_fixtures=True,
        ),
    )
    destination = tmp_path / "queue.json"
    write_queue(destination, queue)
    summary = summarise_readiness(
        paths, destination, target_labels=20, split_version="v3"
    )
    assert summary.queued_unique_count == len(queue.unique_candidate_ids)


def test_the_readiness_summary_serialises(tmp_path: Path):
    paths, _, _ = _corpus(tmp_path, {"s1": "train", "s2": "validation", "s3": "test"})
    summary = summarise_readiness(
        paths, tmp_path / "q.json", target_labels=10, split_version="v3"
    )
    payload = json.loads(json.dumps(summary.to_dict()))
    assert payload["target_labels"] == 10
    assert set(payload["series_by_split"]) == {"train", "validation", "test"}
