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
        "target": 4,
        "labelled": 0,
        "judgements": 0,
        "repeat_judgements": 0,
        "remaining": 4,
        "marked_unusable": 0,
        "skipped": 0,
    }
    database.upsert_label(candidates[0].candidate_id, "a", 3)
    database.upsert_label(candidates[1].candidate_id, "a", 1, is_unusable=True)
    assert database.progress("a") == {
        "total_candidates": 4,
        "target": 4,
        "labelled": 2,
        "judgements": 2,
        "repeat_judgements": 0,
        "remaining": 2,
        "marked_unusable": 1,
        "skipped": 0,
    }


def test_progress_counts_against_an_explicit_target(database, candidates):
    """The UI shows "n / 2400", not "n / everything generated"."""
    database.upsert_label(candidates[0].candidate_id, "a", 3)
    progress = database.progress("a", target=2400)
    assert progress["target"] == 2400
    assert progress["labelled"] == 1
    assert progress["remaining"] == 2399


def test_a_blind_repeat_is_a_second_row_not_an_overwrite(database, candidates):
    """The measurement this schema exists for: the same candidate, judged twice."""
    candidate_id = candidates[0].candidate_id
    database.upsert_label(candidate_id, "a", 4)
    database.upsert_label(
        candidate_id,
        "a",
        2,
        presentation_id=f"{candidate_id}__recheck",
        is_repeat=True,
        stage="consistency",
    )
    labels = database.all_labels()
    assert len(labels) == 2
    # ...but the candidate has been labelled once, not twice.
    assert database.labelled_candidate_ids("a") == {candidate_id}
    assert database.progress("a")["labelled"] == 1
    assert database.progress("a")["repeat_judgements"] == 1
    first, repeat = database.repeat_pairs()[0]
    assert (first.quality_score, repeat.quality_score) == (4, 2)


def test_repeats_are_excluded_from_the_export(tmp_path: Path, database, candidates):
    """A quality control must never become extra supervision."""
    candidate_id = candidates[0].candidate_id
    database.upsert_label(candidate_id, "a", 4)
    database.upsert_label(
        candidate_id,
        "a",
        2,
        presentation_id=f"{candidate_id}__recheck",
        is_repeat=True,
        stage="consistency",
    )
    result = export_labels(database, candidates, tmp_path / "labels_v1.jsonl")
    assert result.row_count == 1
    assert result.repeat_count == 1

    import json

    row = json.loads(result.path.read_text(encoding="utf-8").strip())
    assert row["quality_score"] == 4
    assert row["is_repeat"] is False
    assert row["graded_relevance"] == 3.0


def test_skipping_is_not_a_judgement(database, candidates):
    candidate_id = candidates[0].candidate_id
    database.skip(candidate_id, "a", reason="phone rang")
    assert database.skipped_presentation_ids("a") == {candidate_id}
    assert database.progress("a")["labelled"] == 0
    assert database.all_labels() == []
    # Labelling it later resolves the deferral.
    database.upsert_label(candidate_id, "a", 3)
    assert database.skipped_presentation_ids("a") == set()


def test_clearing_skips_puts_items_back(database, candidates):
    database.skip(candidates[0].candidate_id, "a")
    database.skip(candidates[1].candidate_id, "a")
    assert database.clear_skips("a") == 2
    assert database.skipped_presentation_ids("a") == set()


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
            "presentation_id": first["presentation_id"],
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
        json={
            "presentation_id": first["presentation_id"],
            "annotator_id": "a",
            "quality_score": 3,
        },
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
                "presentation_id": candidate.candidate_id,
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
            "presentation_id": candidate_list[0].candidate_id,
            "annotator_id": "a",
            "quality_score": 9,
        },
    )
    assert response.status_code == 422


def test_unknown_candidate_is_rejected_by_the_endpoint(client):
    http, _, _ = client
    response = http.post(
        "/api/label",
        json={
            "presentation_id": "nope:000000001",
            "annotator_id": "a",
            "quality_score": 3,
        },
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


# --------------------------------------------------------------------------
# Transcript context surfacing (resolved from the cache, as the model reads it)
# --------------------------------------------------------------------------


def _transcript_app(tmp_path: Path, *, write_transcript_file: bool):
    """Build a service over one episode + candidate with no inline transcript.

    Candidates are generated before transcription, so their inline transcript
    fields are null; the service must resolve context from the cached episode
    transcript instead. This helper writes (or omits) that cache file.
    """
    from fastapi.testclient import TestClient

    from slotify_rank.data.paths import DataPaths
    from slotify_rank.labelling.service import LabellingSettings, create_app
    from slotify_rank.transcription.cache import transcript_path, write_transcript
    from slotify_rank.transcription.schema import (
        EpisodeTranscript,
        TranscriptSegment,
        make_segment_id,
    )

    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episode = make_episode(title="Transcript episode", duration_ms=60_000)
    candidate = make_candidate(episode.episode_id, timestamp_ms=20_000)
    assert candidate.transcript_before is None  # the whole point

    if write_transcript_file:
        segments = (
            TranscriptSegment(
                segment_id=make_segment_id(episode.episode_id, 0, 15_000),
                start_ms=15_000,
                end_ms=19_000,
                text="So that wraps up the first topic.",
                normalized_text="so that wraps up the first topic",
                sentence_end=True,
                terminal_punctuation="period",
            ),
            TranscriptSegment(
                segment_id=make_segment_id(episode.episode_id, 1, 21_000),
                start_ms=21_000,
                end_ms=25_000,
                text="Now for something completely different.",
                normalized_text="now for something completely different",
                sentence_end=True,
                terminal_punctuation="period",
            ),
        )
        write_transcript(
            transcript_path(paths, episode.episode_id),
            EpisodeTranscript(
                episode_id=episode.episode_id,
                language="en",
                model_id="whisper-tiny.en",
                model_revision="test",
                audio_sha256="a" * 64,
                audio_duration_ms=60_000,
                segments=segments,
            ),
        )

    app = create_app(
        LabelDatabase(tmp_path / "labels.sqlite3"),
        [candidate],
        [episode],
        paths,
        LabellingSettings(),
    )
    return TestClient(app)


def test_transcript_context_is_resolved_from_the_cache(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    http = _transcript_app(tmp_path, write_transcript_file=True)
    candidate = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    assert candidate["has_transcript"] is True
    assert "first topic" in candidate["transcript_before"]
    assert "completely different" in candidate["transcript_after"]


def test_missing_transcript_is_handled_cleanly(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    http = _transcript_app(tmp_path, write_transcript_file=False)
    candidate = http.get("/api/next", params={"annotator_id": "a"}).json()["candidate"]
    assert candidate["has_transcript"] is False
    assert candidate["transcript_before"] is None
    assert candidate["transcript_after"] is None


def test_every_exported_row_carries_its_full_provenance(tmp_path: Path, database, candidates):
    """A label is only interpretable years later if it says where it came from."""
    database.upsert_label(
        candidates[0].candidate_id,
        "annotator-a",
        4,
        stage="primary",
        queue_version="full-v1",
        elapsed_ms=4200,
        notes="clean topic change",
    )
    result = export_labels(database, candidates, tmp_path / "labels_full-v1.jsonl")

    import json

    row = json.loads(result.path.read_text(encoding="utf-8").strip())
    for field in (
        "annotator_id",
        "candidate_id",
        "episode_id",
        "series_id",
        "dataset_split",
        "presentation_id",
        "stage",
        "queue_version",
        "rubric_version",
        "schema_version",
        "created_at",
        "updated_at",
        "elapsed_ms",
        "quality_score",
        "graded_relevance",
        "label_source",
        "timestamp_ms",
        "candidate_sources",
        "heuristic_score",
        "baseline_version",
        "candidate_generation_version",
    ):
        assert field in row, field
    assert row["label_source"] == "human"
    assert row["queue_version"] == "full-v1"
    assert row["elapsed_ms"] == 4200
    assert row["created_at"].endswith("+00:00")

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["label_schema_version"] == row["schema_version"]
    assert metadata["graded_relevance_rule"].startswith("graded_relevance = quality_score - 1")
