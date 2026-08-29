"""Corpus discovery: licensing, access, deduplication and determinism.

No network. A fake :class:`InternetArchiveClient` serves fixture metadata, so
what is under test is the selection logic -- which is where the mistakes that
silently corrupt a corpus live.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slotify_rank.data.discover import (
    CORPUS_PLAN_VERSION,
    DiscoveryError,
    discover,
    is_public_domain_license_url,
    item_is_restricted,
    load_corpus_plan,
    render_sources_yaml,
)
from slotify_rank.data.sources import load_sources


class FakeClient:
    """Serves canned metadata and records what was asked for."""

    def __init__(self, items: dict[str, dict], search: dict[str, list] | None = None):
        self.items = items
        self.search = search or {}
        self.metadata_calls: list[str] = []

    def metadata(self, identifier: str) -> dict:
        self.metadata_calls.append(identifier)
        return self.items.get(identifier, {"metadata": {"identifier": identifier}})

    def scrape(self, query: str, count: int) -> list[dict]:
        return self.search.get(query, [])


def _mp3(name: str, length: float, fmt: str = "128Kbps MP3", **extra) -> dict:
    return {"name": name, "length": str(length), "format": fmt, "size": "1000", **extra}


def _librivox_item(identifier: str, files: list[dict]) -> dict:
    return {
        "metadata": {
            "identifier": identifier,
            "title": "A LibriVox item",
            "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/",
            "collection": ["librivoxaudio"],
            "uploader": "librivoxbooks@example.org",
        },
        "files": files,
    }


def _plan(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "plan.yaml"
    path.write_text(body, encoding="utf-8")
    return path


_HEADER = f"""
plan_version: {CORPUS_PLAN_VERSION}
corpus_version: test-corpus-v1
defaults:
  source_name: "Internet Archive"
  language: en
  min_duration_seconds: 60
  max_duration_seconds: 3000
licences:
  pd:
    basis: declared_public_domain
    name: "Public Domain Mark 1.0"
    url: https://creativecommons.org/publicdomain/mark/1.0/
  gov:
    basis: us_government_work
    name: "U.S. Government work"
    url: https://www.usa.gov/government-works
    agency: "NASA"
    agency_collections: [nasaaudiocollection]
"""


# --------------------------------------------------------------------------
# Licence recognition
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://creativecommons.org/publicdomain/mark/1.0/",
        "https://creativecommons.org/publicdomain/zero/1.0",
        "http://www.creativecommons.org/licenses/publicdomain/",
    ],
)
def test_public_domain_urls_are_recognised(url):
    assert is_public_domain_license_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://creativecommons.org/licenses/by-nc-nd/3.0/",
        "https://example.com/terms",
    ],
)
def test_non_public_domain_urls_are_not_recognised(url):
    assert is_public_domain_license_url(url) is False


def test_an_item_without_a_public_domain_licence_is_dropped(tmp_path: Path):
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: unlicensed
    title: "Unlicensed"
    content_type: narrated
    licence: pd
    selection:
      kind: item_files
      identifier: item-a
      max_episodes: 2
""",
    )
    client = FakeClient(
        {
            "item-a": {
                "metadata": {
                    "identifier": "item-a",
                    "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
                },
                "files": [_mp3("a.mp3", 300)],
            }
        }
    )
    report = discover(load_corpus_plan(plan), client)
    assert report.episodes == []
    assert report.rejections[0]["reason"] == "licence_not_established"


def test_a_government_item_is_accepted_on_its_agency_collection(tmp_path: Path):
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: nasa-show
    title: "NASA show"
    content_type: podcast
    licence: gov
    selection:
      kind: item_files
      identifier: nasa-item
      max_episodes: 1
""",
    )
    client = FakeClient(
        {
            "nasa-item": {
                "metadata": {
                    "identifier": "nasa-item",
                    "collection": ["nasaaudiocollection", "nasa"],
                },
                "files": [_mp3("Ep1.mp3", 1200)],
            }
        }
    )
    report = discover(load_corpus_plan(plan), client)
    assert len(report.episodes) == 1
    provenance = report.episodes[0].provenance
    assert provenance["licence_verified_by"] == "agency_collection"
    assert provenance["producing_agency"] == "NASA"


def test_a_government_licence_needs_a_collection_or_an_attestation(tmp_path: Path):
    """The claim is about the producer, so it must be evidenced somehow."""
    plan = _plan(
        tmp_path,
        f"""
plan_version: {CORPUS_PLAN_VERSION}
corpus_version: c
licences:
  gov:
    basis: us_government_work
    name: "U.S. Government work"
    url: https://www.usa.gov/government-works
    agency: "NASA"
shows: []
""",
    )
    with pytest.raises(DiscoveryError, match="agency_collections"):
        load_corpus_plan(plan)


# --------------------------------------------------------------------------
# Access
# --------------------------------------------------------------------------


def test_access_restricted_items_are_detected():
    assert item_is_restricted({"metadata": {"access-restricted-item": "true"}}) is True
    assert item_is_restricted({"is_dark": True}) is True
    assert item_is_restricted({"metadata": {"identifier": "x"}}) is False


def test_an_access_restricted_item_is_rejected_before_any_download(tmp_path: Path):
    """These list fine and 401 on download; the failure belongs at plan time."""
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: mirrored
    title: "Mirrored feed"
    content_type: podcast
    licence: pd
    selection:
      kind: item_files
      identifier: restricted
      max_episodes: 2
""",
    )
    client = FakeClient(
        {
            "restricted": {
                "metadata": {
                    "identifier": "restricted",
                    "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/",
                    "access-restricted-item": "true",
                },
                "files": [_mp3("a.mp3", 600)],
            }
        }
    )
    report = discover(load_corpus_plan(plan), client)
    assert report.episodes == []
    assert report.rejections[0]["reason"] == "access_restricted"


def test_private_files_are_never_selected(tmp_path: Path):
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: partly-private
    title: "Partly private"
    content_type: narrated
    licence: pd
    selection:
      kind: item_files
      identifier: item-a
      max_episodes: 5
""",
    )
    client = FakeClient(
        {
            "item-a": _librivox_item(
                "item-a",
                [
                    _mp3("secret.mp3", 300, private="true"),
                    _mp3("public.mp3", 300),
                ],
            )
        }
    )
    report = discover(load_corpus_plan(plan), client)
    assert [e.provenance["internet_archive_file"] for e in report.episodes] == [
        "public.mp3"
    ]


# --------------------------------------------------------------------------
# Deduplication across encodings -- the silent corpus corrupter
# --------------------------------------------------------------------------


def test_one_chapter_in_four_encodings_yields_one_episode(tmp_path: Path):
    """Otherwise "the first two files" is one chapter twice, counted twice."""
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: book
    title: "A book"
    content_type: conversational
    licence: pd
    selection:
      kind: item_files
      identifier: item-a
      max_episodes: 2
""",
    )
    client = FakeClient(
        {
            "item-a": _librivox_item(
                "item-a",
                [
                    _mp3("book_01.mp3", 600, "VBR MP3", track="1"),
                    _mp3("book_01.ogg", 600, "Ogg Vorbis"),
                    _mp3("book_01_128kb.mp3", 600, "128Kbps MP3", track="1"),
                    _mp3("book_01_64kb.mp3", 600, "64Kbps MP3", track="1"),
                    _mp3("book_02.mp3", 700, "VBR MP3", track="2"),
                    _mp3("book_02.ogg", 700, "Ogg Vorbis"),
                    _mp3("book_02_128kb.mp3", 700, "128Kbps MP3", track="2"),
                ],
            )
        }
    )
    report = discover(load_corpus_plan(plan), client)
    files = [e.provenance["internet_archive_file"] for e in report.episodes]
    assert files == ["book_01_128kb.mp3", "book_02_128kb.mp3"]


def test_the_preferred_rendition_is_chosen_deterministically(tmp_path: Path):
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: book
    title: "A book"
    content_type: narrated
    licence: pd
    selection:
      kind: item_files
      identifier: item-a
      max_episodes: 1
""",
    )
    files = [
        _mp3("t_01.ogg", 600, "Ogg Vorbis"),
        _mp3("t_01_64kb.mp3", 600, "64Kbps MP3"),
        _mp3("t_01.mp3", 600, "VBR MP3"),
    ]
    first = discover(
        load_corpus_plan(plan), FakeClient({"item-a": _librivox_item("item-a", files)})
    )
    second = discover(
        load_corpus_plan(plan),
        FakeClient({"item-a": _librivox_item("item-a", list(reversed(files)))}),
    )
    assert (
        first.episodes[0].provenance["internet_archive_file"]
        == second.episodes[0].provenance["internet_archive_file"]
        == "t_01.mp3"
    )


# --------------------------------------------------------------------------
# Duration filtering, determinism and emission
# --------------------------------------------------------------------------


def test_duration_bounds_are_applied_and_reported(tmp_path: Path):
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: book
    title: "A book"
    content_type: narrated
    licence: pd
    selection:
      kind: item_files
      identifier: item-a
      max_episodes: 5
    min_duration_seconds: 300
    max_duration_seconds: 900
""",
    )
    client = FakeClient(
        {
            "item-a": _librivox_item(
                "item-a",
                [
                    _mp3("a_01.mp3", 100),
                    _mp3("a_02.mp3", 600),
                    _mp3("a_03.mp3", 5000),
                ],
            )
        }
    )
    report = discover(load_corpus_plan(plan), client)
    assert [e.provenance["internet_archive_file"] for e in report.episodes] == [
        "a_02.mp3"
    ]
    assert {r["reason"] for r in report.rejections} == {"duration_out_of_range"}


def test_hms_durations_are_parsed(tmp_path: Path):
    """The Archive stores a VBR mp3's length as h:mm:ss, not as seconds."""
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: book
    title: "A book"
    content_type: narrated
    licence: pd
    selection:
      kind: item_files
      identifier: item-a
      max_episodes: 1
    min_duration_seconds: 300
    max_duration_seconds: 900
""",
    )
    client = FakeClient(
        {
            "item-a": _librivox_item(
                "item-a", [{"name": "a.mp3", "length": "10:00", "format": "VBR MP3"}]
            )
        }
    )
    report = discover(load_corpus_plan(plan), client)
    assert report.episodes[0].duration_seconds == pytest.approx(600.0)


def test_discovery_is_deterministic_and_emits_a_loadable_registry(tmp_path: Path):
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: book-one
    title: "Book one"
    content_type: conversational
    licence: pd
    selection:
      kind: item_files
      identifier: item-a
      max_episodes: 2
  - series_id: nasa-show
    title: "NASA show"
    content_type: podcast
    licence: gov
    selection:
      kind: item_files
      identifier: nasa-item
      max_episodes: 1
""",
    )
    items = {
        "item-a": _librivox_item(
            "item-a", [_mp3("a_01.mp3", 400), _mp3("a_02.mp3", 500)]
        ),
        "nasa-item": {
            "metadata": {
                "identifier": "nasa-item",
                "collection": ["nasaaudiocollection"],
            },
            "files": [_mp3("Ep1.mp3", 1200)],
        },
    }
    loaded = load_corpus_plan(plan)
    first = render_sources_yaml(discover(loaded, FakeClient(items)), "plan.yaml")
    second = render_sources_yaml(discover(loaded, FakeClient(items)), "plan.yaml")
    assert first == second

    written = tmp_path / "sources.yaml"
    written.write_text(first, encoding="utf-8")
    registry = load_sources(written)
    assert len(registry) == 3
    assert {entry.series_id for entry in registry} == {"book-one", "nasa-show"}
    for entry in registry:
        assert entry.license_name and entry.license_url
        assert entry.provenance["internet_archive_identifier"]


def test_a_search_show_must_declare_an_exact_creator(tmp_path: Path):
    """The Archive tokenizes a quoted creator phrase; the query alone is a
    recall filter that returns a superset, and which page it returns is not
    stable."""
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: feed
    title: "A feed"
    content_type: podcast
    licence: pd
    selection:
      kind: search
      query: 'creator:"Some Show"'
      max_episodes: 2
""",
    )
    with pytest.raises(DiscoveryError, match="exact 'creator'"):
        load_corpus_plan(plan)


def test_search_filters_on_the_exact_creator(tmp_path: Path):
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: feed
    title: "A feed"
    content_type: podcast
    licence: pd
    selection:
      kind: search
      query: 'creator:"Some Show"'
      creator: "Some Show"
      max_episodes: 5
""",
    )
    client = FakeClient(
        items={
            "keep": _librivox_item("keep", [_mp3("k.mp3", 600)]),
            "drop": _librivox_item("drop", [_mp3("d.mp3", 600)]),
        },
        search={
            'creator:"Some Show"': [
                {"identifier": "keep", "creator": "Some Show"},
                {"identifier": "drop", "creator": "Some Other Show"},
            ]
        },
    )
    report = discover(load_corpus_plan(plan), client)
    assert [e.provenance["internet_archive_identifier"] for e in report.episodes] == [
        "keep"
    ]
    # The rejected item is never opened, so a wrong creator costs no API call.
    assert client.metadata_calls == ["keep"]


def test_a_plan_may_not_declare_an_out_of_domain_content_type(tmp_path: Path):
    plan = _plan(
        tmp_path,
        _HEADER
        + """
shows:
  - series_id: songs
    title: "Songs"
    content_type: music
    licence: pd
    selection:
      kind: item_files
      identifier: item-a
      max_episodes: 1
""",
    )
    with pytest.raises(DiscoveryError, match="outside"):
        discover(load_corpus_plan(plan), FakeClient({}))


def test_the_committed_resume_plan_parses():
    """The plan that actually builds the corpus, validated in CI with no network."""
    plan = load_corpus_plan(
        Path(__file__).resolve().parents[1] / "configs" / "corpus_resume_v1.yaml"
    )
    assert plan.corpus_version == "corpus-resume-v1"
    assert len({show.series_id for show in plan.shows}) == len(plan.shows)
    for show in plan.shows:
        assert show.max_episodes >= 1
        assert show.licence.basis in (
            "declared_public_domain",
            "us_government_work",
        )


def test_the_committed_resume_registry_loads_and_is_fully_licensed():
    registry = load_sources(
        Path(__file__).resolve().parents[1] / "configs" / "sources_resume_v1.yaml"
    )
    assert len(registry) > 0
    for entry in registry:
        assert entry.source_type == "direct_download"
        assert entry.license_name and entry.license_url
        assert entry.provenance["licence_verified_by"] in (
            "item_license_url",
            "agency_collection",
            "manual_attestation",
        )


def test_no_two_registry_entries_name_the_same_upstream_file():
    """A duplicate would be the same audio counted twice under two episode ids."""
    registry = load_sources(
        Path(__file__).resolve().parents[1] / "configs" / "sources_resume_v1.yaml"
    )
    seen = [
        (
            entry.provenance["internet_archive_identifier"],
            entry.provenance["internet_archive_file"],
        )
        for entry in registry
    ]
    assert len(seen) == len(set(seen))
