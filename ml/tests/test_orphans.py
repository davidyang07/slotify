"""Unreferenced generated files: found, reported, and never deleted.

The gap this closes is the one `dataset reconcile` deliberately leaves. Reconcile
drops episodes no registry declares but never touches audio, so a corpus version
bump leaves the renders and transcripts of removed episodes on disk. Anything
that walks the directory instead of the manifest then counts them.
"""

from __future__ import annotations

from pathlib import Path

from slotify_rank.data.manifests import write_episodes
from slotify_rank.data.orphans import scan_orphans
from slotify_rank.data.paths import DataPaths
from slotify_rank.transcription.cache import transcript_path
from tests.dataset_fixtures import make_episode


def _corpus(tmp_path: Path, count: int = 3):
    """A manifest of `count` episodes with every generated file on disk."""
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episodes = [
        make_episode(title=f"Episode {index}", sha_seed=chr(ord("a") + index))
        for index in range(count)
    ]
    write_episodes(paths.episodes_manifest, episodes)
    for episode in episodes:
        (tmp_path / episode.original_path).write_bytes(b"raw")
        (tmp_path / episode.normalized_path).write_bytes(b"wav")
        transcript_path(paths, episode.episode_id).write_text("{}", encoding="utf-8")
    return paths, episodes


def test_a_clean_corpus_reports_no_orphans(tmp_path: Path):
    paths, episodes = _corpus(tmp_path)
    report = scan_orphans(paths)

    assert report.episode_count == len(episodes)
    assert report.clean
    assert report.orphan_count == 0


def test_a_render_no_episode_references_is_reported(tmp_path: Path):
    paths, _ = _corpus(tmp_path)
    stray = paths.normalized_dir / "dropped-episode_deadbeef.wav"
    stray.write_bytes(b"wav")

    report = scan_orphans(paths)
    assert not report.clean
    assert report.orphans["normalized"] == ["data/normalized/dropped-episode_deadbeef.wav"]


def test_transcripts_are_matched_by_episode_id_not_by_manifest_field(tmp_path: Path):
    """The bug that would make this report dangerous.

    `transcript_path` is None on every real episode -- transcripts are a cache
    addressed by episode id. Matching on the manifest field would report every
    transcript in the corpus as unreferenced, and a reader acting on that would
    delete the whole transcript set.
    """
    paths, episodes = _corpus(tmp_path)
    assert all(episode.transcript_path is None for episode in episodes)

    report = scan_orphans(paths)
    assert report.scanned["transcripts"] == len(episodes)
    assert "transcripts" not in report.orphans


def test_an_orphaned_transcript_is_still_reported(tmp_path: Path):
    paths, _ = _corpus(tmp_path)
    stray = paths.transcripts_dir / "fixture-smoke-clip_1234.transcript.json"
    stray.write_text("{}", encoding="utf-8")

    report = scan_orphans(paths)
    assert report.orphans["transcripts"] == [
        "data/transcripts/fixture-smoke-clip_1234.transcript.json"
    ]


def test_scanning_never_removes_anything(tmp_path: Path):
    """It is a report. The docstring promises it, so a test holds it."""
    paths, _ = _corpus(tmp_path)
    stray = paths.normalized_dir / "stray_00.wav"
    stray.write_bytes(b"wav")
    before = sorted(p.name for p in paths.normalized_dir.iterdir())

    scan_orphans(paths)

    assert sorted(p.name for p in paths.normalized_dir.iterdir()) == before
    assert stray.is_file()
