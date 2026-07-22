"""End-to-end CLI behaviour and exit codes for the Phase 2 commands.

Every command is driven through :func:`slotify_rank.cli.main` against a scratch
data root, so the tests cover argument wiring, error handling and exit codes --
not just the library functions underneath.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slotify_rank.cli import main
from slotify_rank.data.ffmpeg import FFmpegNotFound, require_ffmpeg


def _has_ffmpeg() -> bool:
    try:
        require_ffmpeg()
    except FFmpegNotFound:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="FFmpeg is not installed")


@pytest.fixture()
def workspace(tmp_path: Path):
    """A scratch data root plus a sources file pointing at a synthesised clip."""
    from tests.dataset_fixtures import write_speech_like_wav

    audio = tmp_path / "audio" / "clip-one.wav"
    segments = []
    for _ in range(8):
        segments += [(4000, True), (1200, False)]
    write_speech_like_wav(audio, segments)

    sources = tmp_path / "sources.yaml"
    sources.write_text(
        "source_manifest_version: source-manifest-v1.0.0\n"
        "sources:\n"
        "  - id: clip-one\n"
        "    source_type: local_file\n"
        f"    path: {audio.as_posix()}\n"
        '    title: "Clip one"\n'
        "    series_id: clip-series\n"
        '    source_name: "Test"\n'
        "    content_type: podcast\n",
        encoding="utf-8",
    )
    return tmp_path, sources


def _run(args: list[str], data_root: Path) -> int:
    return main([*args, "--data-root", str(data_root)])


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_full_pipeline_exits_zero(workspace, capsys):
    tmp_path, sources = workspace
    data_root = tmp_path / "data"

    assert _run(["dataset", "import-local", "--sources", str(sources)], data_root) == 0
    assert _run(["dataset", "probe"], data_root) == 0
    assert _run(["dataset", "normalize"], data_root) == 0
    assert _run(["candidates", "generate"], data_root) == 0
    assert _run(["dataset", "split"], data_root) == 0
    assert _run(["dataset", "validate"], data_root) == 0
    assert _run(["dataset", "stats"], data_root) == 0

    output = capsys.readouterr().out
    assert "processed_audio_hours" in output
    assert (data_root / "manifests" / "episodes.jsonl").is_file()
    assert (data_root / "manifests" / "candidates.jsonl").is_file()


@needs_ffmpeg
def test_commands_are_resumable(workspace):
    """Re-running a completed stage must be a cheap no-op, not a re-do."""
    tmp_path, sources = workspace
    data_root = tmp_path / "data"
    _run(["dataset", "import-local", "--sources", str(sources)], data_root)
    _run(["dataset", "probe"], data_root)
    _run(["dataset", "normalize"], data_root)

    from slotify_rank.data import manifests

    before = (data_root / "manifests" / "episodes.jsonl").read_bytes()
    assert _run(["dataset", "import-local", "--sources", str(sources)], data_root) == 0
    assert _run(["dataset", "probe"], data_root) == 0
    assert _run(["dataset", "normalize"], data_root) == 0
    assert (data_root / "manifests" / "episodes.jsonl").read_bytes() == before
    assert len(manifests.read_episodes(data_root / "manifests" / "episodes.jsonl")) == 1


@needs_ffmpeg
def test_generate_writes_a_generation_report(workspace):
    tmp_path, sources = workspace
    data_root = tmp_path / "data"
    _run(["dataset", "import-local", "--sources", str(sources)], data_root)
    _run(["dataset", "probe"], data_root)
    _run(["dataset", "normalize"], data_root)
    _run(["candidates", "generate"], data_root)

    reports = json.loads(
        (data_root / "manifests" / "generation_reports.json").read_text(encoding="utf-8")
    )
    assert len(reports) == 1
    assert reports[0]["count_before_merge"] >= reports[0]["count_after_merge"]
    assert reports[0]["transcript_available"] is False


@needs_ffmpeg
def test_stats_artifacts_are_machine_readable(workspace):
    tmp_path, sources = workspace
    data_root = tmp_path / "data"
    for args in (
        ["dataset", "import-local", "--sources", str(sources)],
        ["dataset", "probe"],
        ["dataset", "normalize"],
        ["candidates", "generate"],
        ["dataset", "stats", "--output-dir", str(tmp_path / "artifacts")],
    ):
        assert _run(args, data_root) == 0

    for name in (
        "dataset_statistics.json",
        "candidate_statistics.json",
        "label_statistics.json",
        "split_statistics.json",
    ):
        payload = json.loads((tmp_path / "artifacts" / name).read_text(encoding="utf-8"))
        assert payload["package_version"]


@needs_ffmpeg
def test_label_export_produces_an_empty_but_valid_file(workspace):
    tmp_path, sources = workspace
    data_root = tmp_path / "data"
    for args in (
        ["dataset", "import-local", "--sources", str(sources)],
        ["dataset", "probe"],
        ["dataset", "normalize"],
        ["candidates", "generate"],
        ["label", "export"],
    ):
        assert _run(args, data_root) == 0
    exported = data_root / "labels" / "labels_v1.jsonl"
    assert exported.is_file()
    assert exported.read_text(encoding="utf-8") == ""
    metadata = json.loads(
        (data_root / "labels" / "labels_v1.jsonl.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["row_count"] == 0
    assert metadata["label_source"] == "human"


# --------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------


def test_missing_sources_file_exits_one(tmp_path: Path, capsys):
    code = _run(
        ["dataset", "import-local", "--sources", str(tmp_path / "nope.yaml")],
        tmp_path / "data",
    )
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_malformed_sources_file_exits_one(tmp_path: Path, capsys):
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        "source_manifest_version: source-manifest-v1.0.0\n"
        "sources:\n"
        "  - id: broken\n"
        "    source_type: direct_download\n"
        "    url: https://example.org/a.mp3\n"
        '    title: "Broken"\n'
        '    source_name: "Example"\n'
        "    content_type: podcast\n",
        encoding="utf-8",
    )
    assert _run(["dataset", "import-local", "--sources", str(sources)], tmp_path / "d") == 1
    assert "license_name" in capsys.readouterr().err


def test_generate_without_normalized_audio_exits_one(tmp_path: Path, capsys):
    assert _run(["candidates", "generate"], tmp_path / "data") == 1
    assert "dataset normalize" in capsys.readouterr().err


def test_split_without_episodes_exits_one(tmp_path: Path, capsys):
    assert _run(["dataset", "split"], tmp_path / "data") == 1
    assert "dataset normalize" in capsys.readouterr().err


def test_unknown_episode_id_on_an_empty_manifest_exits_one(tmp_path: Path, capsys):
    assert _run(["dataset", "probe", "--episode-id", "ghost"], tmp_path / "data") == 1
    assert "Unknown episode_id" in capsys.readouterr().err


@needs_ffmpeg
def test_unknown_episode_id_with_a_populated_manifest_exits_one(workspace, capsys):
    tmp_path, sources = workspace
    data_root = tmp_path / "data"
    _run(["dataset", "import-local", "--sources", str(sources)], data_root)
    assert _run(["dataset", "probe", "--episode-id", "ghost"], data_root) == 1
    assert "Unknown episode_id" in capsys.readouterr().err


def test_validate_exits_nonzero_on_an_integrity_error(tmp_path: Path):
    """A manifest that references missing audio must fail the gate."""
    from tests.dataset_fixtures import make_candidate, make_episode
    from slotify_rank.data import manifests

    data_root = tmp_path / "data"
    (data_root / "manifests").mkdir(parents=True)
    episode = make_episode()
    manifests.write_episodes(data_root / "manifests" / "episodes.jsonl", [episode])
    manifests.write_candidates(
        data_root / "manifests" / "candidates.jsonl",
        [make_candidate(episode.episode_id, 999_999_999)],
    )
    assert _run(["dataset", "validate"], data_root) == 1


def test_unknown_command_exits_two():
    with pytest.raises(SystemExit) as error:
        main(["dataset", "teleport"])
    assert error.value.code == 2


def test_missing_required_argument_exits_two():
    with pytest.raises(SystemExit) as error:
        main(["heuristic", "rank"])
    assert error.value.code == 2


def test_label_serve_without_candidates_exits_one(tmp_path: Path, capsys):
    assert _run(["label", "serve"], tmp_path / "data") == 1
    assert "candidates generate" in capsys.readouterr().err


def test_split_is_immutable_without_force(tmp_path: Path, capsys):
    from slotify_rank.data import manifests
    from tests.dataset_fixtures import make_episode

    data_root = tmp_path / "data"
    (data_root / "manifests").mkdir(parents=True)
    episodes = [
        make_episode(
            title=f"Ep {index}",
            sha_seed=seed,
            series_id=f"series-{index}",
            episode_id=f"ep-{index}",
        )
        for index, seed in enumerate("abcdefgh")
    ]
    manifests.write_episodes(data_root / "manifests" / "episodes.jsonl", episodes)

    assert _run(["dataset", "split", "--seed", "1"], data_root) == 0
    assert _run(["dataset", "split", "--seed", "1"], data_root) == 0  # identical: no-op
    assert _run(["dataset", "split", "--seed", "77"], data_root) == 1
    assert "immutable" in capsys.readouterr().err
    assert _run(["dataset", "split", "--seed", "77", "--force"], data_root) == 0
