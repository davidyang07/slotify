"""The labelling session: queue-driven order, prefetch, skips and blind repeats.

The properties that matter for a session of a few thousand items:

* the annotator is served the *queue*, not whatever the manifest listed first;
* each annotator's order is stable across restarts and different from another
  annotator's;
* a blind repeat is served far from its first showing and is indistinguishable
  from a first showing in the payload -- otherwise the consistency measurement
  measures memory, not judgement;
* skipping defers without judging;
* nothing about the heuristic reaches the client.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slotify_rank.data.paths import DataPaths
from slotify_rank.labelling.database import LabelDatabase
from slotify_rank.labelling.queue import QueueConfig, build_queue
from slotify_rank.labelling.service import (
    LabellingSettings,
    build_plan,
    clip_path_for,
    clip_window,
    create_app,
    prerender_clips,
)
from tests.dataset_fixtures import make_candidate, make_episode, write_speech_like_wav


@pytest.fixture()
def corpus(tmp_path: Path):
    """Three series, two episodes each, thirty candidates apiece."""
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episodes = []
    candidates = []
    letters = "abcdef"
    for index in range(6):
        episode = make_episode(
            title=f"Episode {index}",
            series_id=f"series-{index // 2}",
            sha_seed=letters[index],
            duration_ms=600_000,
        )
        episodes.append(episode)
        for step in range(30):
            candidates.append(
                make_candidate(
                    episode.episode_id,
                    timestamp_ms=20_000 + step * 15_000,
                    heuristic_score=float((step * 7) % 100),
                )
            )
    return paths, episodes, candidates


def _queue(candidates, episodes, **overrides):
    config = QueueConfig(
        target_unique=60,
        pilot_size=6,
        overlap_size=6,
        consistency_size=6,
        max_per_episode=20,
        allocation={"train": 0.34, "validation": 0.33, "test": 0.33},
        include_fixtures=True,
        **overrides,
    )
    return build_queue(candidates, episodes, config=config)


def _client(paths, episodes, candidates, queue=None, settings=None):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    database = LabelDatabase(paths.label_database)
    app = create_app(
        database,
        candidates,
        episodes,
        paths,
        settings or LabellingSettings(batch_size=5),
        queue=queue,
    )
    return TestClient(app), database


# --------------------------------------------------------------------------
# Plan construction
# --------------------------------------------------------------------------


def test_the_plan_is_stable_per_annotator_and_differs_between_annotators(
    corpus,
):
    _, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    first = [item.presentation_id for item in build_plan(queue.presentations, "a", 10)]
    again = [item.presentation_id for item in build_plan(queue.presentations, "a", 10)]
    other = [item.presentation_id for item in build_plan(queue.presentations, "b", 10)]
    assert first == again
    assert first != other
    assert sorted(first) == sorted(other)


def test_a_repeat_is_spliced_well_after_its_first_showing(corpus):
    """A repeat shown immediately after the original measures memory, not
    agreement."""
    _, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    delay = 20
    plan = build_plan(queue.presentations, "annotator-a", delay)
    positions = {item.presentation_id: index for index, item in enumerate(plan)}
    firsts = {
        item.candidate_id: positions[item.presentation_id]
        for item in plan
        if not item.is_repeat
    }
    repeats = [item for item in plan if item.is_repeat]
    assert repeats, "the queue should contain consistency repeats"
    for repeat in repeats:
        gap = positions[repeat.presentation_id] - firsts[repeat.candidate_id]
        assert gap >= delay


def test_every_unique_candidate_appears_exactly_once_as_a_first_showing(corpus):
    _, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    plan = build_plan(queue.presentations, "a", 5)
    firsts = [item.candidate_id for item in plan if not item.is_repeat]
    assert sorted(firsts) == sorted(queue.unique_candidate_ids)
    assert len(firsts) == len(set(firsts))


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------


def test_only_queued_candidates_are_served(corpus):
    paths, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    http, _ = _client(paths, episodes, candidates, queue)
    health = http.get("/api/health").json()
    assert health["eligible_candidates"] == len(queue.unique_candidate_ids)
    assert health["eligible_candidates"] < len(candidates)
    assert health["queue_version"] == queue.queue_version


def test_progress_counts_against_the_configured_target(corpus):
    paths, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    http, _ = _client(
        paths,
        episodes,
        candidates,
        queue,
        LabellingSettings(target_unique=2400, batch_size=4),
    )
    progress = http.get("/api/progress", params={"annotator_id": "a"}).json()
    assert progress["target"] == 2400
    assert progress["labelled"] == 0
    assert progress["remaining"] == 2400


def test_the_batch_endpoint_returns_upcoming_items_for_prefetch(corpus):
    paths, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    http, _ = _client(paths, episodes, candidates, queue)
    payload = http.get(
        "/api/batch", params={"annotator_id": "a", "count": 5}
    ).json()
    assert len(payload["items"]) == 5
    ids = [item["presentation_id"] for item in payload["items"]]
    assert len(set(ids)) == 5
    for item in payload["items"]:
        assert item["clip_url"].endswith(item["presentation_id"])
    # And it is the head of that annotator's plan.
    first = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    assert first["presentation_id"] == ids[0]


def test_a_repeat_looks_exactly_like_a_first_showing(corpus):
    """Nothing in the payload may reveal that this clip has been seen before."""
    paths, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    http, _ = _client(paths, episodes, candidates, queue)
    payload = http.get(
        "/api/batch", params={"annotator_id": "a", "count": 50}
    ).json()
    keys = {frozenset(item.keys()) for item in payload["items"]}
    assert len(keys) == 1
    forbidden = {"is_repeat", "repeat_of", "stage", "score_stratum"}
    assert not (next(iter(keys)) & forbidden)
    # The clip URL is keyed on the presentation, so the browser cache cannot
    # reveal that two presentations share audio.
    urls = [item["clip_url"] for item in payload["items"]]
    assert len(set(urls)) == len(urls)


def test_no_heuristic_hint_reaches_the_client(corpus):
    paths, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    http, _ = _client(paths, episodes, candidates, queue)
    item = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    assert "heuristic_score" not in item
    assert "candidate_sources" not in item


def test_labelling_advances_and_resumes(corpus):
    paths, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    http, database = _client(paths, episodes, candidates, queue)
    first = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    response = http.post(
        "/api/label",
        json={
            "presentation_id": first["presentation_id"],
            "annotator_id": "a",
            "quality_score": 4,
            "elapsed_ms": 3200,
        },
    )
    assert response.status_code == 200
    assert response.json()["progress"]["labelled"] == 1

    second = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    assert second["presentation_id"] != first["presentation_id"]

    stored = database.get_label(
        first["candidate_id"], "a", presentation_id=first["presentation_id"]
    )
    assert stored.quality_score == 4
    assert stored.elapsed_ms == 3200
    assert stored.queue_version == queue.queue_version
    assert stored.dataset_split == "unassigned" or stored.dataset_split
    assert stored.series_id


def test_skipping_defers_without_judging(corpus):
    paths, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    http, database = _client(paths, episodes, candidates, queue)
    first = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    skipped = http.post(
        "/api/skip",
        json={
            "presentation_id": first["presentation_id"],
            "annotator_id": "a",
            "reason": "phone rang",
        },
    )
    assert skipped.status_code == 200
    assert skipped.json()["progress"]["skipped"] == 1
    assert skipped.json()["progress"]["labelled"] == 0
    assert database.all_labels() == []

    following = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    assert following["presentation_id"] != first["presentation_id"]

    cleared = http.post("/api/skips/clear", json={"annotator_id": "a"})
    assert cleared.json()["cleared"] == 1
    back = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    assert back["presentation_id"] == first["presentation_id"]


def test_a_blind_repeat_produces_a_second_row_not_an_overwrite(corpus):
    paths, episodes, candidates = corpus
    queue = _queue(candidates, episodes)
    http, database = _client(paths, episodes, candidates, queue)

    repeat = next(p for p in queue.presentations if p.is_repeat)
    original = next(
        p
        for p in queue.presentations
        if not p.is_repeat and p.candidate_id == repeat.candidate_id
    )
    for presentation, score in ((original, 5), (repeat, 2)):
        http.post(
            "/api/label",
            json={
                "presentation_id": presentation.presentation_id,
                "annotator_id": "a",
                "quality_score": score,
            },
        )
    labels = [
        label
        for label in database.all_labels()
        if label.candidate_id == repeat.candidate_id
    ]
    assert len(labels) == 2
    assert {label.is_repeat for label in labels} == {True, False}
    # ...and the candidate counts once toward the target.
    assert database.progress("a")["labelled"] == 1

    from slotify_rank.labelling.quality import measure_repeat_consistency

    consistency = measure_repeat_consistency(database.all_labels())
    assert consistency["measured"] is True
    assert consistency["pair_count"] == 1
    assert consistency["mean_absolute_difference"] == 3.0


def test_an_unknown_presentation_is_rejected(corpus):
    paths, episodes, candidates = corpus
    http, _ = _client(paths, episodes, candidates, _queue(candidates, episodes))
    for path in ("/api/label", "/api/skip"):
        response = http.post(
            path,
            json={
                "presentation_id": "nope:000000001",
                "annotator_id": "a",
                "quality_score": 3,
            },
        )
        assert response.status_code == 404


def test_serving_without_a_queue_still_works(corpus):
    """The pre-queue behaviour: every eligible candidate, once."""
    paths, episodes, candidates = corpus
    http, _ = _client(paths, episodes, candidates, queue=None)
    health = http.get("/api/health").json()
    assert health["eligible_candidates"] == len(candidates)
    assert health["queue_version"] == ""


# --------------------------------------------------------------------------
# Clip pre-rendering
# --------------------------------------------------------------------------


def test_clips_are_pre_rendered_so_no_rating_waits_on_ffmpeg(tmp_path: Path):
    from slotify_rank.data.ffmpeg import FFmpegNotFound, ffmpeg_path

    try:
        ffmpeg_path()
    except FFmpegNotFound:
        pytest.skip("ffmpeg is not available")

    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episode = make_episode(title="Prerender", duration_ms=60_000)
    write_speech_like_wav(
        paths.absolute(episode.normalized_path), [(3000, True), (1000, False)] * 15
    )
    candidates = [
        make_candidate(episode.episode_id, timestamp_ms=ms)
        for ms in (15_000, 25_000, 35_000)
    ]
    settings = LabellingSettings()

    counts = prerender_clips(candidates, [episode], paths, settings, log=lambda _: None)
    assert counts["rendered"] == 3
    for candidate in candidates:
        start_ms, duration_ms = clip_window(candidate, episode, settings)
        assert clip_path_for(
            paths, candidate.candidate_id, start_ms, duration_ms
        ).is_file()

    # Re-running is free, which is what makes it safe to interrupt.
    again = prerender_clips(candidates, [episode], paths, settings, log=lambda _: None)
    assert again["cached"] == 3
    assert again["rendered"] == 0
