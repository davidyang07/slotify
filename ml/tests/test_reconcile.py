"""Reconciliation: the corpus on disk must be the corpus the plan declares.

The failure this prevents is quiet. `dataset fetch` is additive, so when a corpus
plan lowers a show's episode cap or drops a show entirely, the episodes it
already downloaded stay in the manifest and keep being split over, counted and
labelled. Nothing errors. The numbers simply describe a corpus that exists on one
machine and in no committed file -- which is exactly what the source-registry
hashing in the experiment manifest is supposed to make impossible.
"""

from __future__ import annotations

import json

import pytest
import yaml

from slotify_rank.data.reconcile import (
    declared_source_uris,
    reconcile_manifest,
)


def _registry(tmp_path, name, entries):
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            {
                "source_manifest_version": "source-manifest-v1.0.0",
                "sources": [
                    {
                        "id": entry["id"],
                        "source_type": "direct_download",
                        "url": entry["url"],
                        "title": entry["id"],
                        "series_id": entry["series"],
                        "content_type": "podcast",
                        "source_name": "Internet Archive",
                        "language": "en",
                        "license_name": "Public domain",
                        "license_url": "https://creativecommons.org/publicdomain/mark/1.0/",
                    }
                    for entry in entries
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _episode(episode_id, series, url, source_type="direct_download"):
    return {
        "episode_id": episode_id,
        "series_id": series,
        "source_uri": url,
        "source_type": source_type,
        "content_type": "podcast",
        "duration_ms": 600_000,
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _setup(tmp_path, declared, manifest_rows):
    registry = _registry(tmp_path, "sources.yaml", declared)
    episodes = tmp_path / "episodes.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    features = tmp_path / "features.jsonl"
    _write_jsonl(episodes, manifest_rows)
    _write_jsonl(
        candidates,
        [{"episode_id": row["episode_id"], "candidate_id": f"c-{i}"}
         for i, row in enumerate(manifest_rows)],
    )
    _write_jsonl(
        features,
        [{"episode_id": row["episode_id"], "ok": True} for row in manifest_rows],
    )
    return registry, episodes, candidates, features


def test_an_episode_the_plan_no_longer_declares_is_removed():
    """A show whose episode cap was lowered must not keep its old episodes."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        declared = [
            {"id": "a", "series": "show", "url": "https://archive.org/download/x/a.mp3"},
            {"id": "b", "series": "show", "url": "https://archive.org/download/x/b.mp3"},
        ]
        manifest_rows = [
            _episode("ep-a", "show", "https://archive.org/download/x/a.mp3"),
            _episode("ep-b", "show", "https://archive.org/download/x/b.mp3"),
            # Left over from a plan version that allowed three.
            _episode("ep-c", "show", "https://archive.org/download/x/c.mp3"),
        ]
        registry, episodes, candidates, features = _setup(
            tmp_path, declared, manifest_rows
        )

        report = reconcile_manifest(
            episodes_path=episodes,
            candidates_path=candidates,
            features_path=features,
            registries=[registry],
        )

        assert report.episodes_before == 3
        assert report.episodes_after == 2
        assert [entry["episode_id"] for entry in report.removed] == ["ep-c"]
        assert {row["episode_id"] for row in _read_jsonl(episodes)} == {"ep-a", "ep-b"}


def test_the_removed_episodes_candidates_and_features_go_with_it():
    """An orphan candidate row is a candidate that can still be counted."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        declared = [{"id": "a", "series": "show", "url": "https://example.org/a.mp3"}]
        manifest_rows = [
            _episode("ep-a", "show", "https://example.org/a.mp3"),
            _episode("ep-gone", "show", "https://example.org/gone.mp3"),
        ]
        registry, episodes, candidates, features = _setup(
            tmp_path, declared, manifest_rows
        )

        report = reconcile_manifest(
            episodes_path=episodes,
            candidates_path=candidates,
            features_path=features,
            registries=[registry],
        )

        assert report.candidates_removed == 1
        assert report.features_removed == 1
        assert all(row["episode_id"] == "ep-a" for row in _read_jsonl(candidates))
        assert all(row["episode_id"] == "ep-a" for row in _read_jsonl(features))


def test_locally_imported_material_is_kept_by_default_and_droppable_on_request():
    """No remote registry can declare a fixture, so keeping it must be a choice."""
    import tempfile
    from pathlib import Path

    for drop, expected in ((False, {"ep-a", "ep-fixture"}), (True, {"ep-a"})):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            declared = [{"id": "a", "series": "show", "url": "https://example.org/a.mp3"}]
            manifest_rows = [
                _episode("ep-a", "show", "https://example.org/a.mp3"),
                _episode(
                    "ep-fixture",
                    "fixture-podcastlike",
                    "",
                    source_type="existing_repository_fixture",
                ),
            ]
            registry, episodes, candidates, features = _setup(
                tmp_path, declared, manifest_rows
            )

            reconcile_manifest(
                episodes_path=episodes,
                candidates_path=candidates,
                features_path=features,
                registries=[registry],
                keep_local=not drop,
            )
            assert {row["episode_id"] for row in _read_jsonl(episodes)} == expected


def test_a_dry_run_reports_without_writing():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        declared = [{"id": "a", "series": "show", "url": "https://example.org/a.mp3"}]
        manifest_rows = [
            _episode("ep-a", "show", "https://example.org/a.mp3"),
            _episode("ep-gone", "show", "https://example.org/gone.mp3"),
        ]
        registry, episodes, candidates, features = _setup(
            tmp_path, declared, manifest_rows
        )
        before = episodes.read_text(encoding="utf-8")

        report = reconcile_manifest(
            episodes_path=episodes,
            candidates_path=candidates,
            features_path=features,
            registries=[registry],
            dry_run=True,
        )

        assert report.changed
        assert episodes.read_text(encoding="utf-8") == before


def test_a_clean_manifest_is_left_exactly_alone():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        declared = [{"id": "a", "series": "show", "url": "https://example.org/a.mp3"}]
        manifest_rows = [_episode("ep-a", "show", "https://example.org/a.mp3")]
        registry, episodes, candidates, features = _setup(
            tmp_path, declared, manifest_rows
        )
        before = episodes.read_text(encoding="utf-8")

        report = reconcile_manifest(
            episodes_path=episodes,
            candidates_path=candidates,
            features_path=features,
            registries=[registry],
        )

        assert not report.changed
        assert report.removed == []
        assert episodes.read_text(encoding="utf-8") == before


def test_a_missing_registry_is_a_hard_failure():
    """Silently treating an unreadable registry as 'declares nothing' would
    delete the entire corpus."""
    with pytest.raises(FileNotFoundError):
        declared_source_uris(["definitely/not/here.yaml"])


def test_the_committed_registries_declare_every_manifest_episode():
    """The state this repository must be committed in.

    If this fails, the corpus on disk is not the corpus the plan describes and
    every count downstream is about something uncommitted.
    """
    from pathlib import Path

    from slotify_rank.config.settings import find_repo_root

    root = find_repo_root()
    episodes_path = root / "data" / "manifests" / "episodes.jsonl"
    if not episodes_path.is_file():
        pytest.skip("no local corpus built")

    registries = [
        root / "ml" / "configs" / "sources_real_v1.yaml",
        root / "ml" / "configs" / "sources_v2.yaml",
    ]
    declared = declared_source_uris(registries)
    stray = [
        row["episode_id"]
        for row in _read_jsonl(episodes_path)
        if str(row.get("source_type")) == "direct_download"
        and str(row.get("source_uri") or "") not in declared
    ]
    assert stray == [], (
        f"{len(stray)} episode(s) in the manifest are declared by no committed "
        f"registry: {stray[:5]}. Run `dataset reconcile`."
    )
