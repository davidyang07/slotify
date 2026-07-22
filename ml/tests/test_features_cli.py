"""Feature CLI exit codes, stage filtering, resumability and status.

The models are stubbed at the stage boundary, so the whole pipeline runs -- and
its resume and cache behaviour is asserted -- without a download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slotify_rank.cli import main
from slotify_rank.data import manifests
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.pipeline.state import StageLedger

EPISODE = "fixture-smoke_0123456789ab"
SAMPLE_RATE = 16_000
DURATION_MS = 20_000


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch):
    """A tiny on-disk corpus: one normalized episode with three candidates."""
    import wave

    repo = tmp_path
    data_root = repo / "data"
    # The CLI builds its own DataPaths and only accepts --data-root, so the
    # repository root has to be redirected here or manifest paths (which are
    # repo-relative by design) would resolve against the real checkout.
    monkeypatch.setattr(
        "slotify_rank.data.paths.find_repo_root", lambda: repo, raising=True
    )
    paths = DataPaths(repo_root=repo, data_root=data_root)
    paths.mkdirs()

    # A real mono 16 kHz PCM WAV so audio_io and librosa both work.
    rng = np.random.default_rng(0)
    samples = (rng.normal(0, 0.2, SAMPLE_RATE * DURATION_MS // 1000) * 32767).astype(
        "<i2"
    )
    wav_path = paths.normalized_dir / f"{EPISODE}.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())

    from slotify_rank.data.checksum import sha256_file

    checksum = sha256_file(wav_path)
    episode = EpisodeRecord(
        episode_id=EPISODE,
        series_id="fixture-series",
        title="Smoke fixture",
        source_type="existing_repository_fixture",
        source_uri="backend/audio_tests/convo.mp3",
        source_name="fixture",
        license_name=None,
        license_url=None,
        attribution=None,
        language="en",
        content_type="podcast",
        original_path="backend/audio_tests/convo.mp3",
        sha256="a" * 64,
        status="normalized",
        normalized_path=paths.relative(wav_path),
        normalized_sha256=checksum,
        normalized_duration_ms=DURATION_MS,
        duration_ms=DURATION_MS,
        sample_rate_hz=SAMPLE_RATE,
        channels=1,
        file_format="wav",
    )
    manifests.write_episodes(paths.episodes_manifest, [episode])

    candidates = [
        DatasetCandidate(
            episode_id=EPISODE,
            candidate_id=f"{EPISODE}:{ms:09d}",
            timestamp_ms=ms,
            candidate_sources=("silence",),
            heuristic_score=0.5,
            raw_component_scores={
                "silence": 0.5,
                "position": 0.5,
                "sentence": 0.5,
                "spacing": 0.5,
            },
        )
        for ms in (6_000, 10_000, 14_000)
    ]
    manifests.write_candidates(paths.candidates_manifest, candidates)

    _stub_models(monkeypatch)
    return repo, paths, episode, candidates


def _stub_models(monkeypatch):
    """Replace the two model wrappers with deterministic fakes.

    The stage code, cache identities, stores and assembly all run for real; only
    the weights are fake. That is the boundary that keeps this test offline.
    """
    from slotify_rank.embeddings import minilm_text, whisper_audio
    from slotify_rank.embeddings.pooling import FrameGrid
    from slotify_rank.transcription import local_whisper
    from slotify_rank.transcription.schema import EpisodeTranscript
    from slotify_rank.transcription.segments import build_segments

    class FakeTranscriber:
        def __init__(self, config):
            self.config = config

        def load(self):
            return self

        @property
        def device(self):
            return "cpu"

        def release(self):
            return None

        def transcribe(self, episode_id, audio, audio_sha256, identity_digest, sample_rate=SAMPLE_RATE):
            duration_ms = int(audio.size * 1000 / sample_rate)
            spans = [
                (0, 5_000, "The first topic finishes here."),
                (5_500, 9_500, "A second thought begins now."),
                (10_500, 15_000, "And a third one after that."),
                (15_500, min(19_000, duration_ms), "Finally we wrap up."),
            ]
            segments = build_segments(
                episode_id,
                [(s, e, t, ()) for s, e, t in spans if s < duration_ms],
                duration_ms,
            )
            return EpisodeTranscript(
                episode_id=episode_id,
                language="en",
                model_id=self.config.model_id,
                model_revision=self.config.model_revision,
                audio_sha256=audio_sha256,
                audio_duration_ms=duration_ms,
                segments=tuple(segments),
                transcription_config=self.config.to_dict(),
                identity_digest=identity_digest,
            )

    class FakeAudioEncoder:
        def __init__(self, config):
            self.config = config

        def load(self):
            return self

        @property
        def device(self):
            return "cpu"

        @property
        def dimension(self):
            return 384

        def release(self):
            return None

        def encode_episode(self, episode_id, audio, sample_rate=SAMPLE_RATE):
            duration_ms = int(audio.size * 1000 / sample_rate)
            frames = 1_500
            states = np.tile(
                np.arange(frames, dtype=np.float32).reshape(frames, 1), (1, 384)
            )
            grid = FrameGrid(
                chunk_start_ms=0,
                chunk_end_ms=min(30_000, duration_ms),
                frame_duration_ms=20.0,
                total_frames=frames,
            )
            return whisper_audio.EpisodeAudioStates(
                episode_id=episode_id,
                chunks=((grid, states),),
                dimension=384,
                frame_duration_ms=20.0,
            )

    class FakeTextEncoder:
        def __init__(self, config):
            self.config = config

        def load(self):
            return self

        @property
        def device(self):
            return "cpu"

        @property
        def dimension(self):
            return 384

        def release(self):
            return None

        def encode(self, texts):
            if not texts:
                return np.zeros((0, 384), dtype=np.float32)
            rows = []
            for text in texts:
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                vector = rng.normal(size=384).astype(np.float32)
                rows.append(vector / np.linalg.norm(vector))
            return np.vstack(rows)

    monkeypatch.setattr(local_whisper, "WhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(whisper_audio, "WhisperAudioEncoder", FakeAudioEncoder)
    monkeypatch.setattr(minilm_text, "MiniLmTextEncoder", FakeTextEncoder)


def _pipeline(repo: Path, *extra: str) -> int:
    return main(
        ["pipeline", "features", "--data-root", str(repo / "data"), *extra]
    )


# ---------------------------------------------------------------------------
# Exit codes and the happy path
# ---------------------------------------------------------------------------


def test_the_full_pipeline_exits_zero(corpus, capsys):
    repo, paths, _, _ = corpus
    assert _pipeline(repo) == 0
    assert paths.features_manifest.is_file()


def test_the_pipeline_produces_complete_multimodal_records(corpus, capsys):
    repo, paths, _, candidates = corpus
    _pipeline(repo)
    from slotify_rank.features.assemble import read_feature_manifest

    header, records = read_feature_manifest(paths.features_manifest)
    assert len(records) == len(candidates)
    assert all(record.audio_embedding_dimension == 384 for record in records)
    assert any(record.feature_status == "complete" for record in records)


def test_validation_exits_zero_on_a_good_manifest(corpus, capsys):
    repo, _, _, _ = corpus
    _pipeline(repo)
    assert (
        main(["features", "validate", "--data-root", str(repo / "data"), "--deep"]) == 0
    )


def test_validation_exits_non_zero_on_a_corrupt_manifest(corpus, capsys):
    repo, paths, _, _ = corpus
    _pipeline(repo)

    lines = paths.features_manifest.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["handcrafted_feature_values"] = [1.0, 2.0]
    record["handcrafted_missing_mask"] = [False, False]
    lines[1] = json.dumps(record)
    paths.features_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert main(["features", "validate", "--data-root", str(repo / "data")]) == 1


def test_validation_exits_non_zero_when_a_timestamp_is_out_of_bounds(corpus, capsys):
    repo, paths, _, _ = corpus
    _pipeline(repo)
    lines = paths.features_manifest.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["timestamp_ms"] = 5_000_000
    lines[1] = json.dumps(record)
    paths.features_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["features", "validate", "--data-root", str(repo / "data")]) == 1


def test_transcript_validation_exits_zero(corpus, capsys):
    repo, _, _, _ = corpus
    _pipeline(repo)
    assert main(["transcribe", "validate", "--data-root", str(repo / "data")]) == 0


def test_an_unknown_episode_id_is_a_usage_failure(corpus, capsys):
    repo, _, _, _ = corpus
    assert _pipeline(repo, "--episode-id", "no-such-episode") == 1


# ---------------------------------------------------------------------------
# Stage filtering
# ---------------------------------------------------------------------------


def test_a_single_stage_can_be_run_alone(corpus, capsys):
    repo, paths, _, _ = corpus
    assert _pipeline(repo, "--stage", "transcribe") == 0
    ledger = StageLedger.read(paths.stage_ledger("transcribe"), "transcribe")
    assert ledger.counts()["complete"] == 1
    # Nothing downstream ran.
    assert StageLedger.read(paths.stage_ledger("acoustic"), "acoustic").counts()[
        "complete"
    ] == 0
    assert not paths.features_manifest.is_file()


def test_stages_can_be_run_one_at_a_time_to_the_same_result(corpus, capsys):
    repo, paths, _, _ = corpus
    for stage in ("transcribe", "acoustic", "audio_embedding", "text_embedding", "assemble"):
        assert _pipeline(repo, "--stage", stage) == 0
    from slotify_rank.features.assemble import read_feature_manifest

    _, records = read_feature_manifest(paths.features_manifest)
    assert len(records) == 3


def test_individual_subcommands_match_the_pipeline(corpus, capsys):
    repo, paths, _, _ = corpus
    root = str(repo / "data")
    assert main(["transcribe", "run", "--data-root", root]) == 0
    assert main(["features", "acoustic", "--data-root", root]) == 0
    assert main(["embeddings", "audio", "--data-root", root]) == 0
    assert main(["embeddings", "text", "--data-root", root]) == 0
    assert main(["features", "assemble", "--data-root", root]) == 0
    assert paths.features_manifest.is_file()


# ---------------------------------------------------------------------------
# Caching and resumability
# ---------------------------------------------------------------------------


def test_a_second_run_is_entirely_cache_hits(corpus, capsys):
    repo, paths, _, _ = corpus
    _pipeline(repo)
    capsys.readouterr()
    _pipeline(repo)
    output = capsys.readouterr().out
    assert "transcribe       processed=0 skipped=1" in output
    assert "audio_embedding  processed=0 skipped=1" in output


def test_an_interrupted_stage_is_reprocessed(corpus, capsys):
    """A 'partial' entry means the process died mid-stage; output is untrusted."""
    repo, paths, _, _ = corpus
    _pipeline(repo)

    ledger = StageLedger.read(paths.stage_ledger("acoustic"), "acoustic")
    ledger.mark_partial(EPISODE)
    ledger.write(paths.stage_ledger("acoustic"))

    capsys.readouterr()
    assert _pipeline(repo) == 0
    assert "acoustic         processed=1" in capsys.readouterr().out


def test_a_changed_configuration_invalidates_the_cache(corpus, tmp_path, capsys):
    repo, paths, _, _ = corpus
    _pipeline(repo)

    config = tmp_path / "features_changed.yaml"
    config.write_text(
        "features:\n  windows:\n    short_ms: 500\n    medium_ms: 3000\n"
        "    context_ms: 10000\n",
        encoding="utf-8",
    )
    capsys.readouterr()
    assert _pipeline(repo, "--features-config", str(config)) == 0
    assert "acoustic         processed=1" in capsys.readouterr().out


def test_force_recomputes_a_valid_cache(corpus, capsys):
    repo, _, _, _ = corpus
    _pipeline(repo)
    capsys.readouterr()
    assert _pipeline(repo, "--force") == 0
    assert "transcribe       processed=1" in capsys.readouterr().out


def test_validate_offers_no_force_escape_hatch(corpus, capsys):
    """--force re-runs work; it must never be a way to silence an integrity check."""
    repo, _, _, _ = corpus
    _pipeline(repo)
    with pytest.raises(SystemExit) as exit_info:
        main(["features", "validate", "--data-root", str(repo / "data"), "--force"])
    assert exit_info.value.code == 2


def test_a_corrupt_manifest_fails_validation_even_after_a_forced_run(corpus, capsys):
    repo, paths, _, _ = corpus
    _pipeline(repo, "--force")
    lines = paths.features_manifest.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["candidate_id"] = "not-in-the-candidate-manifest"
    lines[1] = json.dumps(record)
    paths.features_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["features", "validate", "--data-root", str(repo / "data")]) == 1


def test_a_failed_episode_is_skipped_until_retry_is_requested(corpus, capsys):
    repo, paths, _, _ = corpus
    ledger = StageLedger.read(paths.stage_ledger("transcribe"), "transcribe")
    ledger.mark_failed(EPISODE, "synthetic failure")
    ledger.write(paths.stage_ledger("transcribe"))

    capsys.readouterr()
    _pipeline(repo, "--stage", "transcribe")
    assert "transcribe       processed=0" in capsys.readouterr().out

    _pipeline(repo, "--stage", "transcribe", "--retry-failed")
    assert (
        StageLedger.read(paths.stage_ledger("transcribe"), "transcribe").counts()[
            "complete"
        ]
        == 1
    )


# ---------------------------------------------------------------------------
# Filters, status and statistics
# ---------------------------------------------------------------------------


def test_the_episode_limit_is_respected(corpus, capsys):
    repo, _, _, _ = corpus
    assert _pipeline(repo, "--limit", "1") == 0


def test_a_content_type_filter_can_select_nothing(corpus, capsys):
    repo, paths, _, _ = corpus
    assert _pipeline(repo, "--content-type", "music") == 0
    assert not paths.features_manifest.is_file()


def test_the_status_command_reports_progress(corpus, capsys):
    repo, _, _, _ = corpus
    _pipeline(repo)
    capsys.readouterr()
    assert main(["pipeline", "status", "--data-root", str(repo / "data")]) == 0
    output = capsys.readouterr().out
    assert "transcribe" in output
    assert "Feature manifest: 3 record(s)" in output


def test_the_status_command_works_before_anything_has_run(corpus, capsys):
    repo, _, _, _ = corpus
    assert main(["pipeline", "status", "--data-root", str(repo / "data")]) == 0
    assert "not yet assembled" in capsys.readouterr().out


def test_statistics_are_generated(corpus, capsys):
    repo, paths, _, _ = corpus
    _pipeline(repo)
    assert main(["features", "stats", "--data-root", str(repo / "data")]) == 0
    directory = paths.feature_artifacts_dir
    for name in (
        "feature_statistics.json",
        "transcription_statistics.json",
        "embedding_statistics.json",
        "cache_statistics.json",
        "feature_summary.md",
    ):
        assert (directory / name).is_file(), name


def test_statistics_json_is_machine_readable(corpus, capsys):
    repo, paths, _, _ = corpus
    _pipeline(repo)
    main(["features", "stats", "--data-root", str(repo / "data")])
    payload = json.loads(
        (paths.feature_artifacts_dir / "feature_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["candidates"]["processed"] == 3
    assert payload["embeddings"]["constructed_text_vector_dimension"] == 1536
