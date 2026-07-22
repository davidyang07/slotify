"""Label storage, export, and the labelling service endpoints.

Nothing here starts a real server, downloads anything, or calls a paid API. The
service is exercised through FastAPI's ``TestClient``; the one endpoint that
shells out to FFmpeg (clip extraction) is skipped when FFmpeg is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slotify_rank.labelling.database import (
    DEFAULT_ACCEPTABLE_THRESHOLD,
    LabelDatabase,
)
from slotify_rank.labelling.export import export_labels
from slotify_rank.labelling.service import annotator_order_key

from tests.dataset_fixtures import make_candidate, make_episode, write_speech_like_wav


@pytest.fixture()
def candidates():
    return [
        make_candidate("ep-one", timestamp_ms=ms)
        for ms in (30_000, 60_000, 90_000, 120_000)
    ]


@pytest.fixture()
def database(tmp_path: Path, candidates):
    db = LabelDatabase(tmp_path / "labels.sqlite3")
    db.register_candidates(candidates)
    return db


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_insert_then_read_back(database, candidates):
    record = database.upsert_label(candidates[0].candidate_id, "annotator-a", 4)
    assert record.quality_score == 4
    assert record.is_acceptable is True
    stored = database.get_label(candidates[0].candidate_id, "annotator-a")
    assert stored.label_id == record.label_id


def test_update_replaces_rather_than_appends(database, candidates):
    first = database.upsert_label(candidates[0].candidate_id, "annotator-a", 2)
    second = database.upsert_label(
        candidates[0].candidate_id, "annotator-a", 5, notes="changed my mind"
    )
    assert first.label_id == second.label_id
    assert second.quality_score == 5
    assert second.notes == "changed my mind"
    assert len(database.all_labels()) == 1


def test_two_annotators_may_both_rate_one_candidate(database, candidates):
    database.upsert_label(candidates[0].candidate_id, "annotator-a", 4)
    database.upsert_label(candidates[0].candidate_id, "annotator-b", 2)
    assert len(database.all_labels()) == 2
    assert set(database.annotators()) == {"annotator-a", "annotator-b"}


def test_acceptability_follows_the_configured_threshold(tmp_path: Path, candidates):
    strict = LabelDatabase(tmp_path / "strict.sqlite3", acceptable_threshold=4)
    strict.register_candidates(candidates)
    assert strict.upsert_label(candidates[0].candidate_id, "a", 3).is_acceptable is False
    assert strict.upsert_label(candidates[1].candidate_id, "a", 4).is_acceptable is True


def test_default_threshold_is_three(database, candidates):
    assert DEFAULT_ACCEPTABLE_THRESHOLD == 3
    assert database.upsert_label(candidates[0].candidate_id, "a", 3).is_acceptable is True
    assert database.upsert_label(candidates[1].candidate_id, "a", 2).is_acceptable is False


@pytest.mark.parametrize("score", [0, 6, -1])
def test_out_of_range_scores_are_rejected(database, candidates, score):
    with pytest.raises(ValueError, match="between 1 and 5"):
        database.upsert_label(candidates[0].candidate_id, "a", score)


def test_non_integer_score_is_rejected(database, candidates):
    with pytest.raises(TypeError, match="must be an int"):
        database.upsert_label(candidates[0].candidate_id, "a", 3.5)


def test_label_on_an_unknown_candidate_is_rejected(database):
    with pytest.raises(KeyError, match="Unknown candidate_id"):
        database.upsert_label("nope:000000001", "a", 3)


def test_blank_annotator_is_rejected(database, candidates):
    with pytest.raises(ValueError, match="annotator_id is required"):
        database.upsert_label(candidates[0].candidate_id, "   ", 3)


def test_synthetic_candidates_cannot_be_registered(tmp_path: Path):
    db = LabelDatabase(tmp_path / "labels.sqlite3")
    synthetic = make_candidate(
        sources=("product_padding",),
        is_synthetic=True,
        eligible_for_labelling=False,
        eligible_for_evaluation=False,
    )
    with pytest.raises(ValueError, match="not eligible for labelling"):
        db.register_candidates([synthetic])


def test_progress_tracks_completion(database, candidates):
    assert database.progress("a") == {
        "total_candidates": 4,
        "labelled": 0,
        "remaining": 4,
        "marked_unusable": 0,
    }
    database.upsert_label(candidates[0].candidate_id, "a", 3)
    database.upsert_label(candidates[1].candidate_id, "a", 1, is_unusable=True)
    assert database.progress("a") == {
        "total_candidates": 4,
        "labelled": 2,
        "remaining": 2,
        "marked_unusable": 1,
    }


def test_progress_is_per_annotator(database, candidates):
    database.upsert_label(candidates[0].candidate_id, "a", 3)
    assert database.progress("b")["labelled"] == 0


def test_reopening_the_database_preserves_labels(tmp_path: Path, candidates):
    path = tmp_path / "labels.sqlite3"
    first = LabelDatabase(path)
    first.register_candidates(candidates)
    first.upsert_label(candidates[0].candidate_id, "a", 5)
    assert len(LabelDatabase(path).all_labels()) == 1


def test_registering_candidates_twice_is_idempotent(database, candidates):
    database.register_candidates(candidates)
    assert len(database.registered_candidate_ids()) == 4


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def test_export_writes_jsonl_and_metadata(tmp_path: Path, database, candidates):
    database.upsert_label(candidates[0].candidate_id, "annotator-a", 4, notes="clean")
    database.upsert_label(candidates[1].candidate_id, "annotator-a", 2)
    result = export_labels(database, candidates, tmp_path / "labels_v1.jsonl")

    assert result.row_count == 2
    assert result.annotator_count == 1
    lines = result.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    import json

    row = json.loads(lines[0])
    assert row["label_source"] == "human"
    assert row["timestamp_ms"] in (30_000, 60_000)
    assert "heuristic_score" in row

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["acceptable_rule"] == "quality_score >= 3"
    assert metadata["row_count"] == 2


def test_export_reports_orphans_without_exporting_them(tmp_path: Path, database, candidates):
    database.upsert_label(candidates[0].candidate_id, "a", 4)
    result = export_labels(database, candidates[1:], tmp_path / "labels_v1.jsonl")
    assert result.row_count == 0
    assert result.orphan_count == 1


def test_export_is_deterministic(tmp_path: Path, database, candidates):
    database.upsert_label(candidates[2].candidate_id, "b", 3)
    database.upsert_label(candidates[0].candidate_id, "a", 5)
    first = export_labels(database, candidates, tmp_path / "one.jsonl").path.read_bytes()
    second = export_labels(database, candidates, tmp_path / "two.jsonl").path.read_bytes()
    assert first == second


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


def test_annotator_order_is_stable_and_annotator_specific():
    ids = [f"ep:{index:09d}" for index in range(20)]
    order_a = sorted(ids, key=lambda cid: annotator_order_key("a", cid))
    assert order_a == sorted(ids, key=lambda cid: annotator_order_key("a", cid))
    assert order_a != sorted(ids, key=lambda cid: annotator_order_key("b", cid))


@pytest.fixture()
def client(tmp_path: Path):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from slotify_rank.data.paths import DataPaths
    from slotify_rank.labelling.service import LabellingSettings, create_app

    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episode = make_episode(title="Service episode", duration_ms=60_000)
    write_speech_like_wav(
        paths.absolute(episode.normalized_path),
        [(3000, True), (1000, False)] * 15,
    )
    candidate_list = [
        make_candidate(episode.episode_id, timestamp_ms=ms)
        for ms in (15_000, 25_000, 35_000)
    ]
    database = LabelDatabase(tmp_path / "labels.sqlite3")
    app = create_app(
        database, candidate_list, [episode], paths, LabellingSettings()
    )
    return TestClient(app), database, candidate_list


def test_health_endpoint(client):
    http, _, candidate_list = client
    payload = http.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["eligible_candidates"] == len(candidate_list)
    assert payload["reveal_hints"] is False


def test_index_is_served(client):
    http, _, _ = client
    response = http.get("/")
    assert response.status_code == 200
    assert "Breakpoint labelling" in response.text


def test_next_returns_a_candidate_without_bias_hints(client):
    http, _, _ = client
    payload = http.get("/api/next", params={"annotator_id": "a"}).json()
    candidate = payload["candidate"]
    assert candidate is not None
    assert candidate["boundary_offset_ms"] > 0
    assert candidate["clip_duration_ms"] > 0
    # Bias control: the annotator must not see what the baseline thought.
    assert "heuristic_score" not in candidate
    assert "candidate_sources" not in candidate


def test_reveal_hints_opt_in_exposes_the_score(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from slotify_rank.data.paths import DataPaths
    from slotify_rank.labelling.service import LabellingSettings, create_app

    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episode = make_episode(title="Hinted", duration_ms=60_000)
    candidate_list = [make_candidate(episode.episode_id, timestamp_ms=20_000)]
    app = create_app(
        LabelDatabase(tmp_path / "l.sqlite3"),
        candidate_list,
        [episode],
        paths,
        LabellingSettings(reveal_hints=True),
    )
    payload = TestClient(app).get("/api/next", params={"annotator_id": "a"}).json()
    assert "heuristic_score" in payload["candidate"]


def test_posting_a_label_persists_and_advances(client):
    http, database, _ = client
    first = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    response = http.post(
        "/api/label",
        json={
            "candidate_id": first["candidate_id"],
            "annotator_id": "a",
            "quality_score": 4,
            "is_unusable": False,
            "notes": "clean break",
        },
    )
    assert response.status_code == 200
    assert response.json()["progress"]["labelled"] == 1
    assert database.get_label(first["candidate_id"], "a").quality_score == 4

    second = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    assert second["candidate_id"] != first["candidate_id"]


def test_a_session_resumes_where_it_stopped(client):
    """The core resumability guarantee, asserted rather than assumed."""
    http, _, _ = client
    first = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    http.post(
        "/api/label",
        json={"candidate_id": first["candidate_id"], "annotator_id": "a", "quality_score": 3},
    )
    resumed = http.get("/api/next", params={"annotator_id": "a"}).json()
    assert resumed["candidate"]["candidate_id"] != first["candidate_id"]
    assert resumed["progress"]["labelled"] == 1
    # A different annotator starts from the beginning of their own order.
    other = http.get("/api/next", params={"annotator_id": "b"}).json()
    assert other["progress"]["labelled"] == 0


def test_exhausting_the_pool_returns_no_candidate(client):
    http, _, candidate_list = client
    for candidate in candidate_list:
        http.post(
            "/api/label",
            json={
                "candidate_id": candidate.candidate_id,
                "annotator_id": "a",
                "quality_score": 3,
            },
        )
    assert http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"] is None


def test_invalid_score_is_rejected_by_the_endpoint(client):
    http, _, candidate_list = client
    response = http.post(
        "/api/label",
        json={
            "candidate_id": candidate_list[0].candidate_id,
            "annotator_id": "a",
            "quality_score": 9,
        },
    )
    assert response.status_code == 422


def test_unknown_candidate_is_rejected_by_the_endpoint(client):
    http, _, _ = client
    response = http.post(
        "/api/label",
        json={"candidate_id": "nope:000000001", "annotator_id": "a", "quality_score": 3},
    )
    assert response.status_code == 404


def test_progress_endpoint_requires_an_annotator(client):
    http, _, _ = client
    assert http.get("/api/progress", params={"annotator_id": " "}).status_code == 422


def test_clip_endpoint_returns_audio(client):
    from slotify_rank.data.ffmpeg import FFmpegNotFound, ffmpeg_path

    try:
        ffmpeg_path()
    except FFmpegNotFound:
        pytest.skip("ffmpeg is not available")
    http, _, candidate_list = client
    response = http.get(f"/api/clip/{candidate_list[0].candidate_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 44  # larger than a bare WAV header


def test_clip_endpoint_404s_for_an_unknown_candidate(client):
    http, _, _ = client
    assert http.get("/api/clip/nope:000000001").status_code == 404
