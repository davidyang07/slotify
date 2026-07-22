"""Episode/candidate schema, deterministic IDs, checksums and manifests.

No network, no audio, no models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slotify_rank.data import manifests
from slotify_rank.data.checksum import (
    ChecksumMismatch,
    atomic_write_bytes,
    sha256_file,
    verify_sha256,
)
from slotify_rank.data.paths import DataPaths, resolve, to_repo_relative
from slotify_rank.data.schema import (
    DatasetCandidate,
    EnergySummary,
    EpisodeRecord,
    make_episode_id,
    slugify,
    sort_candidates,
)

from tests.dataset_fixtures import make_candidate, make_episode


# --------------------------------------------------------------------------
# Deterministic identity
# --------------------------------------------------------------------------


def test_episode_id_is_deterministic_for_the_same_content():
    digest = "a" * 64
    assert make_episode_id("My Show 001", digest) == make_episode_id("My Show 001", digest)


def test_episode_id_keys_on_content_not_path():
    """Two copies of the same audio collapse to one episode, whatever they are called."""
    digest = "b" * 64
    assert make_episode_id("Title", digest) == make_episode_id("Title", digest)
    assert make_episode_id("Title", "c" * 64) != make_episode_id("Title", digest)


def test_episode_id_is_filesystem_safe():
    episode_id = make_episode_id("Ünïcode: Show / Episode #3!", "d" * 64)
    assert set(episode_id) <= set("abcdefghijklmnopqrstuvwxyz0123456789-_")


def test_episode_id_rejects_a_short_hash():
    with pytest.raises(ValueError, match="64 hex"):
        make_episode_id("Title", "abc")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello World", "hello-world"),
        ("  spaced  out  ", "spaced-out"),
        ("Ünïcode", "unicode"),
        ("!!!", ""),
    ],
)
def test_slugify(text, expected):
    assert slugify(text) == expected


def test_candidate_ids_are_unique_and_ordered():
    candidates = [
        make_candidate("ep-a", 5000),
        make_candidate("ep-a", 1000),
        make_candidate("ep-b", 3000),
    ]
    ordered = sort_candidates(candidates)
    assert [c.candidate_id for c in ordered] == [
        "ep-a:000001000",
        "ep-a:000005000",
        "ep-b:000003000",
    ]
    assert len({c.candidate_id for c in candidates}) == 3


# --------------------------------------------------------------------------
# Episode schema validation
# --------------------------------------------------------------------------


def test_remote_source_requires_licence_metadata():
    with pytest.raises(ValueError, match="license_name and license_url"):
        make_episode(source_type="direct_download", source_uri="https://x/y.mp3")


def test_remote_source_accepts_declared_licence():
    episode = make_episode(
        source_type="direct_download",
        source_uri="https://x/y.mp3",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    )
    assert episode.license_name == "CC BY 4.0"


def test_local_source_may_omit_licence():
    """A private local file is not assumed to carry any public licence."""
    episode = make_episode(source_type="local_file")
    assert episode.license_name is None


def test_absolute_paths_are_rejected():
    with pytest.raises(ValueError, match="repository-relative"):
        make_episode(original_path="C:/Users/me/audio.mp3")


def test_backslash_paths_are_rejected():
    with pytest.raises(ValueError, match="forward slashes"):
        make_episode(original_path="data\\raw\\x.mp3")


def test_upward_traversal_is_rejected():
    with pytest.raises(ValueError, match="traverse upward"):
        make_episode(original_path="data/../../etc/passwd")


def test_non_positive_duration_is_rejected():
    with pytest.raises(ValueError, match="duration_ms must be positive"):
        make_episode(duration_ms=0, status="probed")


def test_normalized_status_requires_a_normalized_path():
    with pytest.raises(ValueError, match="requires normalized_path"):
        make_episode(status="normalized", normalized_path=None)


def test_unknown_content_type_is_rejected():
    with pytest.raises(ValueError, match="content_type"):
        make_episode(content_type="video")


def test_unknown_field_is_rejected_rather_than_dropped():
    payload = make_episode().to_dict()
    payload["mystery"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        EpisodeRecord.from_mapping(payload)


def test_unsupported_schema_version_is_rejected():
    payload = make_episode().to_dict()
    payload["schema_version"] = "episode-schema-v0.9.0"
    with pytest.raises(ValueError, match="Unsupported episode schema_version"):
        EpisodeRecord.from_mapping(payload)


def test_episode_round_trips_through_a_mapping():
    episode = make_episode()
    assert EpisodeRecord.from_mapping(episode.to_dict()) == episode


# --------------------------------------------------------------------------
# Candidate schema validation
# --------------------------------------------------------------------------


def test_synthetic_candidates_cannot_be_marked_eligible():
    """The exclusion invariant is enforced by the constructor, not by convention."""
    with pytest.raises(ValueError, match="eligible_for_labelling=False"):
        make_candidate(
            sources=("product_padding",),
            is_synthetic=True,
            eligible_for_labelling=True,
            eligible_for_evaluation=False,
        )
    with pytest.raises(ValueError, match="eligible_for_labelling=False"):
        make_candidate(
            sources=("product_padding",),
            is_synthetic=True,
            eligible_for_labelling=False,
            eligible_for_evaluation=True,
        )


def test_synthetic_candidate_with_both_flags_off_is_accepted():
    candidate = make_candidate(
        sources=("product_padding",),
        is_synthetic=True,
        eligible_for_labelling=False,
        eligible_for_evaluation=False,
    )
    assert candidate.is_synthetic and not candidate.eligible_for_evaluation


def test_float_timestamps_are_rejected():
    """Canonical identity is integer milliseconds; a float would break joins."""
    payload = make_candidate().to_dict()
    payload["timestamp_ms"] = 60_000.5
    with pytest.raises(TypeError, match="must be an int"):
        DatasetCandidate.from_mapping(payload)


def test_empty_sources_are_rejected():
    with pytest.raises(ValueError, match="candidate_sources must not be empty"):
        make_candidate(sources=())


def test_candidate_round_trips_with_an_energy_summary():
    candidate = make_candidate(
        local_energy_summary=EnergySummary(1000, -30.0, -55.0, -12.0)
    )
    restored = DatasetCandidate.from_mapping(candidate.to_dict())
    assert restored == candidate
    assert restored.local_energy_summary.min_dbfs == -55.0


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------


def test_sha256_matches_a_known_value(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"slotify")
    import hashlib

    assert sha256_file(path) == hashlib.sha256(b"slotify").hexdigest()


def test_zero_length_files_are_refused(tmp_path: Path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="zero-length"):
        sha256_file(path)


def test_verify_sha256_raises_on_mismatch(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"slotify")
    with pytest.raises(ChecksumMismatch) as error:
        verify_sha256(path, "0" * 64)
    assert error.value.expected == "0" * 64


def test_verify_sha256_rejects_a_malformed_expected_value(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"slotify")
    with pytest.raises(ValueError, match="64 hex"):
        verify_sha256(path, "not-a-hash")


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path):
    destination = tmp_path / "out.json"
    atomic_write_bytes(destination, b'{"a": 1}')
    assert json.loads(destination.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def test_to_repo_relative_rejects_paths_outside_the_repository():
    # Not `tmp_path`: conftest may relocate pytest's temp root inside the
    # repository on this machine, which would make the path legitimately inside.
    import tempfile

    with tempfile.TemporaryDirectory() as outside:
        with pytest.raises(ValueError, match="outside the repository"):
            to_repo_relative(Path(outside) / "elsewhere.mp3")


def test_resolve_rejects_traversal():
    with pytest.raises(ValueError, match="traverse upward"):
        resolve("data/../../secrets")


def test_data_root_can_be_relocated(tmp_path: Path):
    paths = DataPaths(data_root=tmp_path / "corpus")
    assert paths.episodes_manifest == tmp_path / "corpus" / "manifests" / "episodes.jsonl"


def test_artifacts_follow_the_data_root(tmp_path: Path):
    """Statistics describe one corpus, so relocating the corpus relocates them.

    Guards a real leak: with this pinned to the repository root, a test run
    against a scratch corpus overwrote the committed statistics.
    """
    paths = DataPaths(data_root=tmp_path / "corpus")
    assert paths.artifacts_dir == tmp_path / "artifacts" / "dataset"


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------


def test_episode_manifest_round_trips(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    episodes = [make_episode(title="A", sha_seed="a"), make_episode(title="B", sha_seed="b")]
    manifests.write_episodes(path, episodes)
    assert manifests.read_episodes(path) == sorted(
        episodes, key=lambda e: (e.series_id, e.episode_id)
    )


def test_manifest_write_refuses_duplicate_episode_ids(tmp_path: Path):
    episode = make_episode()
    with pytest.raises(manifests.ManifestError, match="duplicate episode_id"):
        manifests.write_episodes(tmp_path / "episodes.jsonl", [episode, episode])


def test_manifest_read_detects_duplicate_episode_ids(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    row = json.dumps(make_episode().to_dict())
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(manifests.ManifestError, match="duplicate episode_id"):
        manifests.read_episodes(path)


def test_manifest_read_detects_duplicate_candidate_ids(tmp_path: Path):
    path = tmp_path / "candidates.jsonl"
    row = json.dumps(make_candidate().to_dict())
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(manifests.ManifestError, match="duplicate candidate_id"):
        manifests.read_candidates(path)


def test_upsert_is_idempotent_and_reports_changes(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    episode = make_episode()
    assert manifests.upsert_episodes(path, [episode]) == (1, 0, 1)
    assert manifests.upsert_episodes(path, [episode]) == (0, 0, 1)
    changed = episode.replace(title="Renamed")
    assert manifests.upsert_episodes(path, [changed]) == (0, 1, 1)
    assert manifests.read_episodes(path)[0].title == "Renamed"


def test_manifest_output_is_stable_across_input_order(tmp_path: Path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    episodes = [
        make_episode(title="A", sha_seed="a"),
        make_episode(title="B", sha_seed="b"),
        make_episode(title="C", sha_seed="c"),
    ]
    manifests.write_episodes(first, episodes)
    manifests.write_episodes(second, list(reversed(episodes)))
    assert first.read_bytes() == second.read_bytes()


def test_replace_episode_candidates_leaves_other_episodes_untouched(tmp_path: Path):
    path = tmp_path / "candidates.jsonl"
    keep = make_candidate("ep-keep", 1000)
    replace_me = make_candidate("ep-replace", 2000)
    manifests.write_candidates(path, [keep, replace_me])
    removed, total = manifests.replace_episode_candidates(
        path, ["ep-replace"], [make_candidate("ep-replace", 9000)]
    )
    assert (removed, total) == (1, 2)
    remaining = manifests.read_candidates(path)
    assert {c.candidate_id for c in remaining} == {
        "ep-keep:000001000",
        "ep-replace:000009000",
    }


def test_manifest_read_reports_the_offending_line(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(manifests.ManifestError, match=r":1"):
        manifests.read_episodes(path)
