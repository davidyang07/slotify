"""Label-quality controls: warnings, never mutation."""

from __future__ import annotations

from dataclasses import replace

from slotify_rank.labelling.database import LabelRecord
from slotify_rank.labelling.quality import check_label_quality
from slotify_rank.labelling.queue import QueueConfig, build_queue
from tests.dataset_fixtures import make_candidate, make_episode
from tests.phase5_fixtures import build_corpus


def _label(candidate_id, episode_id, quality, acceptable, annotator="a", unusable=False):
    return LabelRecord(
        label_id=1,
        candidate_id=candidate_id,
        episode_id=episode_id,
        annotator_id=annotator,
        quality_score=quality,
        is_acceptable=acceptable,
        is_unusable=unusable,
        notes=None,
        rubric_version="rubric-v1.0.0",
        created_at="2026-07-23T00:00:00+00:00",
        updated_at="2026-07-23T00:00:00+00:00",
    )


def test_clean_labels_pass():
    corpus = build_corpus({"s-a": "train", "s-b": "train"}, candidates_per_episode=5)
    report = check_label_quality(
        corpus.raw_labels, corpus.candidates, corpus.episodes
    )
    assert report.ok
    assert report.label_count == len(corpus.raw_labels)


def test_orphan_label_is_an_error():
    ep = make_episode(series_id="s1")
    cand = make_candidate(episode_id=ep.episode_id)
    labels = [_label("ghost:000000001", ep.episode_id, 4, True)]
    report = check_label_quality(labels, [cand], [ep])
    assert not report.ok
    assert any(f.check == "missing_candidate_reference" for f in report.errors)


def test_acceptability_inconsistency_is_flagged_not_fixed():
    ep = make_episode(series_id="s1")
    cand = make_candidate(episode_id=ep.episode_id)
    # quality 5 but marked unacceptable: the human value is kept, the discrepancy
    # is warned about.
    labels = [_label(cand.candidate_id, ep.episode_id, 5, acceptable=False)]
    report = check_label_quality(labels, [cand], [ep], acceptable_threshold=3)
    assert report.ok  # a warning, not an error
    assert any(f.check == "acceptability_inconsistent" for f in report.warnings)


def test_duplicate_active_label_is_an_error():
    ep = make_episode(series_id="s1")
    cand = make_candidate(episode_id=ep.episode_id)
    labels = [
        _label(cand.candidate_id, ep.episode_id, 4, True, annotator="a"),
        _label(cand.candidate_id, ep.episode_id, 2, False, annotator="a"),
    ]
    report = check_label_quality(labels, [cand], [ep])
    assert any(f.check == "duplicate_active_label" for f in report.errors)


def test_unusable_in_trainable_split_is_warned():
    ep = make_episode(series_id="s1")
    cand = make_candidate(episode_id=ep.episode_id, dataset_split="train")
    labels = [_label(cand.candidate_id, ep.episode_id, 1, False, unusable=True)]
    report = check_label_quality(labels, [cand], [ep])
    assert any(f.check == "unusable_in_training" for f in report.warnings)


def test_candidate_manifest_drift_is_warned():
    corpus = build_corpus({"s-a": "train"}, candidates_per_episode=6, labelled=False)
    queue = build_queue(
        corpus.candidates,
        corpus.episodes,
        QueueConfig(target_unique=3, pilot_size=0, overlap_size=0, consistency_size=0),
        candidate_manifest_hash="original-hash",
    )
    labels = [_label(corpus.candidates[0].candidate_id, corpus.episodes[0].episode_id, 4, True)]
    report = check_label_quality(
        labels,
        corpus.candidates,
        corpus.episodes,
        queue=queue,
        current_candidate_manifest_hash="a-different-hash",
    )
    assert any(f.check == "candidate_manifest_drift" for f in report.warnings)


def test_label_outside_queue_is_warned():
    corpus = build_corpus({"s-a": "train"}, candidates_per_episode=6, labelled=False)
    queue = build_queue(
        corpus.candidates,
        corpus.episodes,
        QueueConfig(target_unique=2, pilot_size=0, overlap_size=0, consistency_size=0),
        candidate_manifest_hash="h",
    )
    outside = next(
        c for c in corpus.candidates if c.candidate_id not in queue.unique_candidate_ids
    )
    labels = [_label(outside.candidate_id, outside.episode_id, 3, True)]
    report = check_label_quality(labels, corpus.candidates, corpus.episodes, queue=queue)
    assert any(f.check == "label_outside_queue" for f in report.warnings)


def test_split_mismatch_is_warned():
    corpus = build_corpus({"s-a": "train"}, candidates_per_episode=4, labelled=False)
    queue = build_queue(
        corpus.candidates,
        corpus.episodes,
        QueueConfig(target_unique=2, pilot_size=0, overlap_size=0, consistency_size=0),
        candidate_manifest_hash="h",
    )
    # Move a candidate to a different split after the queue froze.
    moved = [replace(c, dataset_split="test") for c in corpus.candidates]
    report = check_label_quality([], moved, corpus.episodes, queue=queue)
    assert any(f.check == "split_mismatch" for f in report.warnings)
