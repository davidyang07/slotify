"""End-to-end CLI behaviour and exit codes for the Phase 2 commands.

Every command is driven through :func:`slotify_rank.cli.main` against a scratch
data root, so the tests cover argument wiring, error handling and exit codes --
not just the library functions underneath.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from slotify_rank.cli import main
from slotify_rank.config.settings import canonical_config_path, find_repo_root
from slotify_rank.data.ffmpeg import FFmpegNotFound, require_ffmpeg


def _has_ffmpeg() -> bool:
    try:
        require_ffmpeg()
    except FFmpegNotFound:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="FFmpeg is not installed")


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A scratch repository root holding a data root and a sources file.

    Manifests record repository-relative paths, so a data root outside the
    repository root cannot be described by one. The scratch directory therefore
    becomes its own root: the canonical heuristic config is copied in and
    ``SLOTIFY_REPO_ROOT`` points at it, which is the documented way to run the
    package from somewhere other than the checkout. Without this the tests only
    pass when pytest happens to place ``tmp_path`` inside the checkout.
    """
    from tests.dataset_fixtures import write_speech_like_wav

    canonical_config = canonical_config_path()
    scratch_config = tmp_path / "config" / canonical_config.name
    scratch_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(canonical_config, scratch_config)
    monkeypatch.setenv("SLOTIFY_REPO_ROOT", str(tmp_path))

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
    # Eight one-episode series is too thin for the canonical split's
    # three-series gate, and immutability is not about which rules are used.
    config = ["--config", str(find_repo_root() / "ml" / "configs" / "splits_v1.yaml")]

    assert _run(["dataset", "split", "--seed", "1", *config], data_root) == 0
    assert _run(["dataset", "split", "--seed", "1", *config], data_root) == 0  # no-op
    assert _run(["dataset", "split", "--seed", "77", *config], data_root) == 1
    assert "immutable" in capsys.readouterr().err
    assert (
        _run(["dataset", "split", "--seed", "77", "--force", *config], data_root) == 0
    )


def _canonical_splittable_corpus(data_root: Path):
    """Enough podcast and narrated series for the canonical split's gates."""
    from slotify_rank.data import manifests
    from tests.dataset_fixtures import make_candidate, make_episode

    (data_root / "manifests").mkdir(parents=True)
    episodes = []
    for index in range(24):
        episodes.append(
            make_episode(
                title=f"Podcast {index}",
                sha_seed=f"{index % 16:x}",
                series_id=f"podcast-{index:02d}",
                episode_id=f"podcast-{index:02d}",
                duration_ms=600_000,
            )
        )
    for index in range(6):
        episodes.append(
            make_episode(
                title=f"Narrated {index}",
                sha_seed=f"{index:x}",
                series_id=f"narrated-{index}",
                episode_id=f"narrated-{index}",
                content_type="narrated",
                duration_ms=600_000,
            )
        )
    manifests.write_episodes(data_root / "manifests" / "episodes.jsonl", episodes)
    manifests.write_candidates(
        data_root / "manifests" / "candidates.jsonl",
        [make_candidate(episode_id=e.episode_id, timestamp_ms=60_000) for e in episodes],
    )
    return episodes


def test_split_without_config_uses_the_canonical_split_and_says_so(
    tmp_path: Path, capsys
):
    """No --config used to mean `SplitConfig()` -- the unstratified v1 rules --
    and every candidate was restamped with a split no other command read."""
    from slotify_rank.config.versions import STRATIFIED_SPLIT_ALGORITHM_VERSION
    from slotify_rank.data import manifests
    from slotify_rank.data.splits import (
        CANONICAL_SPLIT_CONFIG,
        CANONICAL_SPLIT_VERSION,
        read_split_manifest,
    )

    data_root = tmp_path / "data"
    _canonical_splittable_corpus(data_root)

    assert _run(["dataset", "split"], data_root) == 0
    output = capsys.readouterr().out
    assert Path(CANONICAL_SPLIT_CONFIG).name in output
    assert "canonical default" in output
    assert STRATIFIED_SPLIT_ALGORITHM_VERSION in output

    manifest_path = data_root / "manifests" / f"splits_{CANONICAL_SPLIT_VERSION}.json"
    manifest = read_split_manifest(manifest_path)
    assert manifest.algorithm_version == STRATIFIED_SPLIT_ALGORITHM_VERSION
    assert not (data_root / "manifests" / "splits_v1.json").exists()
    by_episode = manifest.by_episode
    for candidate in manifests.read_candidates(data_root / "manifests" / "candidates.jsonl"):
        assert candidate.dataset_split == by_episode[candidate.episode_id]

    # And the readers default to the very same split. (The fabricated episodes
    # have no audio on disk, so validation itself fails; only which split it
    # looked for is under test.)
    _run(["dataset", "validate"], data_root)
    report = json.loads(
        (data_root.parent / "artifacts" / "dataset" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert not any("no split manifest" in f["message"] for f in report["findings"])


def test_an_explicit_split_config_is_named_without_the_default_note(
    tmp_path: Path, capsys
):
    data_root = tmp_path / "data"
    _canonical_splittable_corpus(data_root)
    config = find_repo_root() / "ml" / "configs" / "splits_v3.yaml"
    assert _run(["dataset", "split", "--config", str(config)], data_root) == 0
    output = capsys.readouterr().out
    assert "splits_v3.yaml" in output
    assert "canonical default" not in output
