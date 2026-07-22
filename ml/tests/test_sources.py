"""Source registry parsing, licence enforcement and secret rejection.

These are all offline: ``load_sources`` performs no I/O beyond reading the YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slotify_rank.data.sources import SourceConfigError, load_sources

_HEADER = "source_manifest_version: source-manifest-v1.0.0\nsources:\n"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(_HEADER + body, encoding="utf-8")
    return path


_LOCAL = """
  - id: local-one
    source_type: local_file
    path: audio/one.mp3
    title: "One"
    series_id: my-series
    source_name: "Private"
    content_type: podcast
"""


def test_local_source_parses(tmp_path: Path):
    registry = load_sources(_write(tmp_path, _LOCAL))
    entry = registry.get("local-one")
    assert entry.source_type == "local_file"
    assert entry.series_id == "my-series"
    assert entry.license_name is None
    assert entry.is_target_domain is True


def test_music_is_not_target_domain(tmp_path: Path):
    registry = load_sources(
        _write(tmp_path, _LOCAL.replace("content_type: podcast", "content_type: music"))
    )
    assert registry.get("local-one").is_target_domain is False


def test_series_id_defaults_to_the_source_id(tmp_path: Path):
    body = _LOCAL.replace("    series_id: my-series\n", "")
    assert load_sources(_write(tmp_path, body)).get("local-one").series_id == "local-one"


def test_remote_source_requires_a_licence(tmp_path: Path):
    body = """
  - id: remote-one
    source_type: direct_download
    url: https://example.org/a.mp3
    title: "One"
    source_name: "Example"
    content_type: podcast
"""
    with pytest.raises(SourceConfigError, match="requires both license_name"):
        load_sources(_write(tmp_path, body))


def test_remote_source_with_a_licence_parses(tmp_path: Path):
    body = """
  - id: remote-one
    source_type: direct_download
    url: https://example.org/a.mp3
    title: "One"
    source_name: "Example"
    content_type: podcast
    license_name: "CC BY 4.0"
    license_url: https://creativecommons.org/licenses/by/4.0/
"""
    assert load_sources(_write(tmp_path, body)).get("remote-one").license_name == "CC BY 4.0"


@pytest.mark.parametrize(
    "url,message",
    [
        ("ftp://example.org/a.mp3", "http or https"),
        ("https://user:pw@example.org/a.mp3", "credentials"),
        ("https://example.org/a.mp3?api_key=abc", "credential-like"),
        ("https://example.org/show/episode-1", "not a recognised audio extension"),
    ],
)
def test_unsafe_or_indirect_urls_are_rejected(tmp_path: Path, url, message):
    body = f"""
  - id: remote-one
    source_type: direct_download
    url: {url}
    title: "One"
    source_name: "Example"
    content_type: podcast
    license_name: "CC BY 4.0"
    license_url: https://creativecommons.org/licenses/by/4.0/
"""
    with pytest.raises(SourceConfigError, match=message):
        load_sources(_write(tmp_path, body))


def test_committed_secrets_are_rejected(tmp_path: Path):
    body = _LOCAL + '    notes: "key sk-abcdefghijklmnopqrstuv"\n'
    with pytest.raises(SourceConfigError, match="credential"):
        load_sources(_write(tmp_path, body))


def test_duplicate_source_ids_are_rejected(tmp_path: Path):
    with pytest.raises(SourceConfigError, match="duplicate source id"):
        load_sources(_write(tmp_path, _LOCAL + _LOCAL))


def test_expected_sha256_must_be_hex(tmp_path: Path):
    body = _LOCAL + "    expected_sha256: nothex\n"
    with pytest.raises(SourceConfigError, match="64 hex"):
        load_sources(_write(tmp_path, body))


def test_local_source_rejects_a_url(tmp_path: Path):
    body = _LOCAL + "    url: https://example.org/a.mp3\n"
    with pytest.raises(SourceConfigError, match="'url' is not allowed"):
        load_sources(_write(tmp_path, body))


def test_remote_source_rejects_a_path(tmp_path: Path):
    body = """
  - id: remote-one
    source_type: direct_download
    url: https://example.org/a.mp3
    path: audio/a.mp3
    title: "One"
    source_name: "Example"
    content_type: podcast
    license_name: "CC BY 4.0"
    license_url: https://x/
"""
    with pytest.raises(SourceConfigError, match="'path' is not allowed"):
        load_sources(_write(tmp_path, body))


def test_unsupported_manifest_version_is_rejected(tmp_path: Path):
    path = tmp_path / "sources.yaml"
    path.write_text("source_manifest_version: v0\nsources: []\n", encoding="utf-8")
    with pytest.raises(SourceConfigError, match="unsupported source_manifest_version"):
        load_sources(path)


def test_defaults_are_applied(tmp_path: Path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "source_manifest_version: source-manifest-v1.0.0\n"
        'defaults:\n  source_name: "Shared"\n  language: fr\n'
        "sources:\n"
        "  - id: local-one\n"
        "    source_type: local_file\n"
        "    path: audio/one.mp3\n"
        '    title: "One"\n'
        "    content_type: podcast\n",
        encoding="utf-8",
    )
    entry = load_sources(path).get("local-one")
    assert (entry.source_name, entry.language) == ("Shared", "fr")


def test_missing_registry_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Source registry not found"):
        load_sources(tmp_path / "nope.yaml")


def test_repository_registry_is_valid():
    """The committed registry must always parse -- it is the on-ramp for new audio."""
    from slotify_rank.config.settings import find_repo_root

    registry = load_sources(find_repo_root() / "ml" / "configs" / "sources.yaml")
    assert len(registry) >= 5
    music = [entry for entry in registry if entry.content_type == "music"]
    assert music, "the smoke registry should include music, marked out of domain"
    assert all(not entry.is_target_domain for entry in music)
    assert all(
        entry.source_type != "direct_download" or (entry.license_name and entry.license_url)
        for entry in registry
    )
