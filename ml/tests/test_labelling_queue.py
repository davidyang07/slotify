"""Stratified labelling-queue construction."""

from __future__ import annotations

import json

import pytest

from slotify_rank.labelling.queue import (
    QueueConfig,
    QueueError,
    build_queue,
    queue_summary,
    read_queue,
    write_queue,
    write_queue_summary,
)
from tests.dataset_fixtures import make_candidate, make_episode
from tests.phase5_fixtures import build_corpus


def _corpus():
    series_splits = {
        "series-a": "train",
        "series-b": "train",
        "series-c": "train",
        "series-d": "validation",
        "series-e": "test",
        "series-f": "train",
    }
    corpus = build_corpus(series_splits, candidates_per_episode=40, labelled=False)
    return corpus.episodes, corpus.candidates


def _config(**overrides):
    base = dict(
        target_unique=120,
        pilot_size=12,
        overlap_size=20,
        consistency_size=8,
        max_per_episode=40,
    )
    base.update(overrides)
    return QueueConfig(**base)


def test_queue_is_deterministic():
    episodes, candidates = _corpus()
    a = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")
    b = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")
    assert a.to_dict() == b.to_dict()
    assert a.content_hash() == b.content_hash()


def test_stages_are_separated_and_union_to_unique():
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")
    pilot = set(queue.pilot_candidate_ids)
    primary = set(queue.primary_candidate_ids)
    unique = set(queue.unique_candidate_ids)
    assert pilot.isdisjoint(primary)
    assert pilot | primary == unique
    assert set(queue.overlap_candidate_ids) <= unique
    assert set(queue.consistency_candidate_ids) <= unique
    assert len(queue.pilot_candidate_ids) == 12


def test_unique_export_excludes_repeated_consistency_items():
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")
    # Consistency re-checks are extra presentations, never extra unique candidates.
    repeats = [p for p in queue.presentations if p.is_repeat]
    assert len(repeats) == len(queue.consistency_candidate_ids)
    assert len(queue.unique_candidate_ids) == len(set(queue.unique_candidate_ids))
    for repeat in repeats:
        assert repeat.presentation_id != repeat.candidate_id
        assert repeat.repeat_of == repeat.candidate_id
        # The repeat's candidate is in the unique set exactly once.
        assert queue.unique_candidate_ids.count(repeat.candidate_id) == 1


def test_stratifies_across_splits_and_score_tiers():
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")
    cov = queue.coverage
    # Every split that has candidates is represented.
    assert set(cov["by_split"]) == {"train", "validation", "test"}
    # All three score tiers appear.
    assert set(cov["by_score_stratum"]) == {"low", "medium", "high"}
    # Split allocation roughly follows the configured weights (train dominates).
    assert cov["by_split"]["train"] > cov["by_split"]["validation"]


def test_per_episode_cap_is_respected():
    episodes, candidates = _corpus()
    queue = build_queue(
        candidates, episodes, _config(max_per_episode=15), candidate_manifest_hash="h"
    )
    assert queue.coverage["max_per_episode_selected"] <= 15


def test_series_are_spread_not_dominated_by_one():
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")
    by_series = queue.coverage["by_series"]
    # More than one series contributes, and no series supplies everything.
    assert len(by_series) >= 4
    assert max(by_series.values()) < queue.unique_count


def test_presentations_do_not_leak_heuristic_score():
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")
    for presentation in queue.presentations:
        keys = presentation.to_dict().keys()
        assert "heuristic_score" not in keys
        assert "score" not in keys
        # The bias-sensitive stratum label is not on the presentation the UI would
        # render as an annotator-facing field beyond stage/split.
        assert "primary_source" not in keys


def test_synthetic_padding_is_never_queued():
    episodes = [make_episode(series_id="s1", content_type="podcast")]
    good = make_candidate(
        episode_id=episodes[0].episode_id, timestamp_ms=20_000, dataset_split="train"
    )
    synthetic = make_candidate(
        episode_id=episodes[0].episode_id,
        timestamp_ms=30_000,
        is_synthetic=True,
        eligible_for_labelling=False,
        eligible_for_evaluation=False,
        sources=("product_padding",),
        dataset_split="train",
    )
    queue = build_queue(
        [good, synthetic],
        episodes,
        _config(target_unique=1, pilot_size=0, overlap_size=0, consistency_size=0),
        candidate_manifest_hash="h",
    )
    assert synthetic.candidate_id not in queue.unique_candidate_ids
    assert good.candidate_id in queue.unique_candidate_ids


def test_fixtures_are_excluded_by_default():
    fixture_ep = make_episode(
        series_id="fixture-x",
        content_type="podcast",
        source_type="existing_repository_fixture",
        original_path="backend/audio_tests/x.mp3",
    )
    real_ep = make_episode(series_id="real-x", content_type="podcast", sha_seed="z")
    cands = [
        make_candidate(episode_id=fixture_ep.episode_id, timestamp_ms=20_000, dataset_split="train"),
        make_candidate(episode_id=real_ep.episode_id, timestamp_ms=20_000, dataset_split="train"),
    ]
    queue = build_queue(
        cands,
        [fixture_ep, real_ep],
        _config(target_unique=2, pilot_size=0, overlap_size=0, consistency_size=0),
        candidate_manifest_hash="h",
    )
    assert all(
        not cid.startswith(fixture_ep.episode_id) for cid in queue.unique_candidate_ids
    )


def test_empty_pool_is_an_error():
    episodes = [make_episode(series_id="s1", content_type="music")]
    cands = [make_candidate(episode_id=episodes[0].episode_id, dataset_split="train")]
    with pytest.raises(QueueError):
        build_queue(cands, episodes, _config(), candidate_manifest_hash="h")


def test_write_read_round_trip_and_immutability(tmp_path):
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")
    path = tmp_path / "queue.json"
    write_queue(path, queue)
    loaded = read_queue(path)
    assert loaded.to_dict() == queue.to_dict()

    # Re-writing identical content is a no-op; different content is refused.
    write_queue(path, queue)
    other = build_queue(
        candidates, episodes, _config(seed=999), candidate_manifest_hash="h"
    )
    with pytest.raises(FileExistsError):
        write_queue(path, other)
    write_queue(path, other, force=True)  # explicit override allowed


def test_read_rejects_wrong_schema_version(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "labelling-queue-v0.0.0"}), encoding="utf-8")
    with pytest.raises(QueueError):
        read_queue(path)


def test_stage_candidate_ids_restricts_a_serve_session():
    """`label serve --stage pilot` must expose exactly the pilot candidates."""
    from slotify_rank.dataset_cli import stage_candidate_ids

    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, _config(), candidate_manifest_hash="h")

    pilot = stage_candidate_ids(queue, "pilot")
    primary = stage_candidate_ids(queue, "primary")
    every = stage_candidate_ids(queue, "all")

    assert pilot == set(queue.pilot_candidate_ids)
    assert primary == set(queue.primary_candidate_ids)
    assert every == set(queue.unique_candidate_ids)
    # A controlled pilot is exactly the pilot stage: disjoint from primary,
    # smaller than the whole queue, and never a consistency re-check.
    assert pilot.isdisjoint(primary)
    assert pilot | primary == every
    assert len(pilot) < len(every)
    with pytest.raises(ValueError):
        stage_candidate_ids(queue, "bogus")


def test_require_complete_features_excludes_unfeaturisable_candidates():
    """A candidate the model cannot consume must not cost human attention.

    The readiness gate refuses a labelled set containing one, so queuing it
    guarantees the judgement is wasted.
    """
    from slotify_rank.labelling.queue import QueueConfig, build_candidate_views

    from tests.dataset_fixtures import make_candidate, make_episode

    class _Record:
        def __init__(self, status: str):
            self.feature_status = status
            self.transcript_available = status == "complete"
            self.audio_embedding_available = True
            self.text_embedding_available = status == "complete"

    episode = make_episode(title="Featurised", series_id="s1")
    candidates = [
        make_candidate(episode.episode_id, timestamp_ms=30_000 + i * 20_000)
        for i in range(4)
    ]
    features = {
        candidates[0].candidate_id: _Record("complete"),
        candidates[1].candidate_id: _Record("audio_only"),
        candidates[2].candidate_id: _Record("failed"),
        # candidates[3] has no feature record at all.
    }

    permissive = build_candidate_views(
        candidates, [episode], QueueConfig(include_fixtures=True), features
    )
    assert len(permissive) == 4

    strict = build_candidate_views(
        candidates,
        [episode],
        QueueConfig(include_fixtures=True, require_complete_features=True),
        features,
    )
    assert [view.candidate_id for view in strict] == [candidates[0].candidate_id]
    assert strict[0].feature_status == "complete"


def test_the_setting_is_recorded_in_the_queue_config_digest():
    """Two queues that filtered differently must not share a config hash."""
    from slotify_rank.labelling.queue import QueueConfig

    permissive = QueueConfig()
    strict = QueueConfig(require_complete_features=True)
    assert permissive.digest() != strict.digest()
    assert strict.to_dict()["require_complete_features"] is True


def test_queue_summary_carries_the_shape_without_the_candidate_ids(tmp_path):
    """The reduction that ships as evidence.

    The queue itself lives under the uncommitted data tree, so this is all a
    reviewer of a fresh clone ever sees. It has to carry enough to establish
    that the round was built -- size, target, blind repeats, the hashes it is
    pinned to -- and none of the candidate ids, which would make it a copy of
    the queue rather than a summary of it.
    """
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, config=_config())
    summary = queue_summary(queue)

    assert summary["unique_candidate_count"] == queue.unique_count
    assert summary["target_unique"] == queue.config["target_unique"]
    assert summary["presentation_count"] == len(queue.presentations)
    assert summary["blind_repeat_presentations"] == len(
        queue.consistency_candidate_ids
    )
    assert summary["queue_content_hash"] == queue.content_hash()
    assert summary["config_hash"] == queue.config_hash

    serialized = json.dumps(summary)
    for candidate_id in queue.unique_candidate_ids:
        assert candidate_id not in serialized


def test_queue_summary_counts_repeats_as_presentations_not_as_labels(tmp_path):
    """A blind repeat is a quality control, never an extra unique candidate."""
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, config=_config(consistency_size=8))
    summary = queue_summary(queue)
    assert summary["blind_repeat_presentations"] == 8
    assert (
        summary["presentation_count"]
        == summary["unique_candidate_count"] + summary["blind_repeat_presentations"]
    )


def test_write_queue_summary_round_trips(tmp_path):
    episodes, candidates = _corpus()
    queue = build_queue(candidates, episodes, config=_config())
    path = write_queue_summary(tmp_path / "queue_summary.json", queue)
    assert json.loads(path.read_text(encoding="utf-8")) == queue_summary(queue)
