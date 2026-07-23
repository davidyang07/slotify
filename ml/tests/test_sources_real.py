"""The committed real-corpus source manifest must validate and stay licensed."""

from __future__ import annotations

from slotify_rank.config.settings import find_repo_root
from slotify_rank.data.schema import TARGET_DOMAIN_CONTENT_TYPES
from slotify_rank.data.sources import load_sources

REAL_SOURCES = find_repo_root() / "ml" / "configs" / "sources_real_v1.yaml"


def test_real_manifest_loads_and_is_licensed():
    registry = load_sources(REAL_SOURCES)
    assert len(registry) >= 6
    for entry in registry.entries:
        assert entry.source_type == "direct_download"
        # direct_download without a licence would have failed load_sources; assert
        # the invariant here too so a regression is caught at this layer.
        assert entry.license_name and entry.license_url, entry.id
        assert entry.url and entry.url.startswith("https://")


def test_real_manifest_has_at_least_six_target_domain_series():
    registry = load_sources(REAL_SOURCES)
    series = {entry.series_id for entry in registry.entries}
    assert len(series) >= 6, f"only {len(series)} series"
    for entry in registry.entries:
        assert entry.content_type in TARGET_DOMAIN_CONTENT_TYPES, entry.id


def test_real_manifest_excludes_music_and_meetings():
    registry = load_sources(REAL_SOURCES)
    content_types = {entry.content_type for entry in registry.entries}
    assert "music" not in content_types
    assert "meeting" not in content_types


def test_real_manifest_licences_are_public_domain_or_cc():
    registry = load_sources(REAL_SOURCES)
    for entry in registry.entries:
        licence = (entry.license_name or "").lower()
        url = (entry.license_url or "").lower()
        assert (
            "public domain" in licence
            or "cc" in licence
            or "creativecommons.org" in url
        ), f"{entry.id}: unrecognised licence {entry.license_name!r}"
