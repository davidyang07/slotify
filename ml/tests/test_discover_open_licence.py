"""The `declared_open_licence` basis and the `collection_items` selection kind.

The basis exists to let a self-published podcast into the corpus without also
letting in the roughly 35,000 Internet Archive items where somebody uploaded
somebody else's show and ticked "public domain". It does that by requiring two
things at once -- an open licence URL *and* membership of the show's own home
collection -- so the tests that matter most here are the ones proving that
either half alone is refused.

No network: a fake client serves fixture metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from slotify_rank.data.discover import (
    DiscoveryError,
    discover,
    is_open_license_url,
    load_corpus_plan,
)
from slotify_rank.data.sources import load_sources


def _config(name):
    """Repo-relative config path, so this works from either working directory."""
    return Path(__file__).resolve().parents[1] / "configs" / name



class FakeClient:
    def __init__(self, items, scrapes=None):
        self.items = items
        self.scrapes = scrapes or {}
        self.scrape_calls = []

    def metadata(self, identifier):
        return self.items.get(identifier, {"metadata": {"identifier": identifier}})

    def scrape(self, query, count, sorts=None):
        self.scrape_calls.append((query, count, sorts))
        return self.scrapes.get(query, [])


def _item(identifier, licenseurl, collections, *, length="600", restricted=False):
    metadata = {
        "identifier": identifier,
        "title": identifier,
        "collection": collections,
    }
    if licenseurl is not None:
        metadata["licenseurl"] = licenseurl
    if restricted:
        metadata["access-restricted-item"] = "true"
    return {
        "metadata": metadata,
        "files": [
            {
                "name": f"{identifier}.mp3",
                "format": "128Kbps MP3",
                "length": length,
                "size": "9000000",
                "md5": "0" * 32,
            }
        ],
    }


def _plan(tmp_path, *, collection="myshow", extra_show=None):
    shows = [
        {
            "series_id": "my-show",
            "title": "My Show",
            "content_type": "podcast",
            "licence": "open_self_published",
            "selection": {
                "kind": "collection_items",
                "collection": collection,
                "max_episodes": 3,
            },
            "min_duration_seconds": 300,
            "max_duration_seconds": 1800,
            "homepage": "https://myshow.example/",
        }
    ]
    if extra_show:
        shows.append(extra_show)
    document = {
        "plan_version": "corpus-plan-v1.0.0",
        "corpus_version": "test-corpus",
        "scrape_page_size": 100,
        "search_scan_limit": 50,
        "licences": {
            "open_self_published": {
                "basis": "declared_open_licence",
                "name": "Open Creative Commons licence declared by the publisher",
                "url": "https://creativecommons.org/",
            }
        },
        "shows": shows,
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_corpus_plan(path)


def _scrapes(collection, identifiers):
    return {
        f"collection:{collection} AND mediatype:audio": [
            {"identifier": i} for i in identifiers
        ]
    }


# ---------------------------------------------------------------------------
# The licence matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://creativecommons.org/publicdomain/zero/1.0/",
        "https://creativecommons.org/publicdomain/mark/1.0/",
        "http://creativecommons.org/licenses/by/4.0/",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "http://creativecommons.org/licenses/by-sa/3.0/us/",
    ],
)
def test_open_licences_are_recognised(url):
    assert is_open_license_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://creativecommons.org/licenses/by-nc/3.0/",
        "https://creativecommons.org/licenses/by-nd/4.0/",
        "http://creativecommons.org/licenses/by-nc-sa/4.0",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/",
        "https://example.org/some-eula",
        None,
        "",
    ],
)
def test_non_open_licences_are_refused(url):
    """NC and ND are Creative Commons but they are not open.

    `by-nc` in particular must not match the `by` pattern -- that is what the
    trailing slash in `licenses/by/` is for.
    """
    assert not is_open_license_url(url)


# ---------------------------------------------------------------------------
# Both halves are required
# ---------------------------------------------------------------------------


def test_an_open_licence_in_the_shows_own_collection_is_accepted(tmp_path):
    plan = _plan(tmp_path)
    client = FakeClient(
        items={
            "ep1": _item(
                "ep1", "https://creativecommons.org/licenses/by-sa/4.0/", ["myshow", "podcasts"]
            )
        },
        scrapes=_scrapes("myshow", ["ep1"]),
    )
    report = discover(plan, client)
    assert len(report.episodes) == 1
    episode = report.episodes[0]
    assert episode.provenance["licence_verified_by"] == "open_licence_in_home_collection"
    assert "myshow" in episode.provenance["licence_evidence"]


def test_an_open_licence_outside_the_home_collection_is_refused(tmp_path):
    """The 35,000-item case: a stranger's upload with a public-domain tag."""
    plan = _plan(tmp_path)
    client = FakeClient(
        items={
            "ep1": _item(
                "ep1",
                "https://creativecommons.org/publicdomain/mark/1.0/",
                ["podcasts_miscellaneous", "podcasts"],
            )
        },
        scrapes=_scrapes("myshow", ["ep1"]),
    )
    report = discover(plan, client)
    assert report.episodes == []
    assert [r["reason"] for r in report.rejections] == ["licence_not_established"]


def test_membership_of_the_home_collection_without_an_open_licence_is_refused(tmp_path):
    plan = _plan(tmp_path)
    client = FakeClient(
        items={
            "ep1": _item(
                "ep1", "https://creativecommons.org/licenses/by-nc-sa/4.0/", ["myshow"]
            )
        },
        scrapes=_scrapes("myshow", ["ep1"]),
    )
    report = discover(plan, client)
    assert report.episodes == []
    assert [r["reason"] for r in report.rejections] == ["licence_not_established"]


def test_an_item_with_no_licence_at_all_is_refused(tmp_path):
    plan = _plan(tmp_path)
    client = FakeClient(
        items={"ep1": _item("ep1", None, ["myshow"])},
        scrapes=_scrapes("myshow", ["ep1"]),
    )
    report = discover(plan, client)
    assert report.episodes == []


def test_a_mixed_catalogue_contributes_only_its_open_episodes(tmp_path):
    """Labor Express, The Cinematic Tangent and Building Bridges are all like this."""
    plan = _plan(tmp_path)
    client = FakeClient(
        items={
            "ep1": _item("ep1", "https://creativecommons.org/licenses/by/4.0/", ["myshow"]),
            "ep2": _item("ep2", "https://creativecommons.org/licenses/by-nd/4.0/", ["myshow"]),
            "ep3": _item("ep3", "https://creativecommons.org/licenses/by-sa/4.0/", ["myshow"]),
        },
        scrapes=_scrapes("myshow", ["ep1", "ep2", "ep3"]),
    )
    report = discover(plan, client)
    assert sorted(e.provenance["internet_archive_identifier"] for e in report.episodes) == [
        "ep1",
        "ep3",
    ]


def test_a_declared_open_licence_show_must_name_a_collection(tmp_path):
    """Without a home collection the basis collapses to 'somebody tagged it'."""
    document = {
        "plan_version": "corpus-plan-v1.0.0",
        "corpus_version": "test-corpus",
        "licences": {
            "open_self_published": {
                "basis": "declared_open_licence",
                "name": "Open CC",
                "url": "https://creativecommons.org/",
            }
        },
        "shows": [
            {
                "series_id": "my-show",
                "title": "My Show",
                "content_type": "podcast",
                "licence": "open_self_published",
                "selection": {"kind": "item_files", "identifier": "someitem", "max_episodes": 1},
            }
        ],
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(DiscoveryError, match="needs a 'collection'"):
        load_corpus_plan(path)


# ---------------------------------------------------------------------------
# collection_items selection
# ---------------------------------------------------------------------------


def test_access_restricted_items_are_rejected_before_any_download(tmp_path):
    """The whole `podcasts_mirror` tree fails here, and must."""
    plan = _plan(tmp_path)
    client = FakeClient(
        items={
            "ep1": _item(
                "ep1",
                "https://creativecommons.org/licenses/by-sa/4.0/",
                ["myshow"],
                restricted=True,
            )
        },
        scrapes=_scrapes("myshow", ["ep1"]),
    )
    report = discover(plan, client)
    assert report.episodes == []
    assert [r["reason"] for r in report.rejections] == ["access_restricted"]


def test_one_item_contributes_one_episode(tmp_path):
    """An item in a podcast collection is one episode, however many files it has."""
    plan = _plan(tmp_path)
    item = _item("ep1", "https://creativecommons.org/licenses/by/4.0/", ["myshow"])
    item["files"].append(
        {
            "name": "ep1_part2.mp3",
            "format": "128Kbps MP3",
            "length": "600",
            "size": "9000000",
            "md5": "1" * 32,
        }
    )
    client = FakeClient(items={"ep1": item}, scrapes=_scrapes("myshow", ["ep1"]))
    report = discover(plan, client)
    assert len(report.episodes) == 1


def test_the_episode_cap_is_honoured(tmp_path):
    plan = _plan(tmp_path)
    identifiers = [f"ep{i}" for i in range(10)]
    client = FakeClient(
        items={
            i: _item(i, "https://creativecommons.org/licenses/by/4.0/", ["myshow"])
            for i in identifiers
        },
        scrapes=_scrapes("myshow", identifiers),
    )
    report = discover(plan, client)
    assert len(report.episodes) == 3  # max_episodes


def test_selection_is_deterministic_and_sorted(tmp_path):
    """Two runs must pick the same episodes regardless of API result order."""
    plan = _plan(tmp_path)
    identifiers = [f"ep{i}" for i in range(10)]
    items = {
        i: _item(i, "https://creativecommons.org/licenses/by/4.0/", ["myshow"])
        for i in identifiers
    }
    forward = discover(plan, FakeClient(items, _scrapes("myshow", identifiers)))
    backward = discover(
        plan, FakeClient(items, _scrapes("myshow", list(reversed(identifiers))))
    )
    assert [e.source_id for e in forward.episodes] == [
        e.source_id for e in backward.episodes
    ]


def test_a_stable_sort_is_requested_from_the_archive(tmp_path):
    """For a collection larger than one page, which page you get must be fixed."""
    plan = _plan(tmp_path)
    client = FakeClient(items={}, scrapes=_scrapes("myshow", []))
    discover(plan, client)
    assert client.scrape_calls
    assert client.scrape_calls[0][2] == "identifier asc"


def test_an_empty_collection_is_reported_rather_than_silently_dropping_a_show(tmp_path):
    plan = _plan(tmp_path)
    client = FakeClient(items={}, scrapes=_scrapes("myshow", []))
    report = discover(plan, client)
    assert [r["reason"] for r in report.rejections] == ["empty_collection"]


def test_duration_bounds_still_apply(tmp_path):
    plan = _plan(tmp_path)
    client = FakeClient(
        items={
            "ep1": _item(
                "ep1", "https://creativecommons.org/licenses/by/4.0/", ["myshow"], length="60"
            )
        },
        scrapes=_scrapes("myshow", ["ep1"]),
    )
    report = discover(plan, client)
    assert report.episodes == []
    assert [r["reason"] for r in report.rejections] == ["duration_out_of_range"]


def test_the_emitted_registry_loads(tmp_path):
    plan = _plan(tmp_path)
    client = FakeClient(
        items={
            "ep1": _item("ep1", "https://creativecommons.org/licenses/by-sa/4.0/", ["myshow"])
        },
        scrapes=_scrapes("myshow", ["ep1"]),
    )
    report = discover(plan, client)
    from slotify_rank.data.discover import render_sources_yaml

    path = tmp_path / "sources.yaml"
    path.write_text(render_sources_yaml(report, "plan.yaml"), encoding="utf-8")
    registry = load_sources(path)
    assert len(registry.entries) == 1


# ---------------------------------------------------------------------------
# The committed plan
# ---------------------------------------------------------------------------


def test_the_committed_v2_plan_parses_and_is_podcast_led():
    plan = load_corpus_plan(_config("corpus_v2.yaml"))
    podcasts = [s for s in plan.shows if s.content_type == "podcast"]
    assert len(podcasts) >= 14, "the corpus must carry enough podcast series to split"
    # Every open-licence show names its own collection and its homepage, so a
    # reader can check the claim the basis rests on.
    for show in plan.shows:
        if show.licence.basis == "declared_open_licence":
            assert show.collection
            assert show.licence.home_collections == (show.collection,)
            assert show.homepage, f"{show.series_id} must record a homepage"


def test_the_committed_registry_matches_the_committed_plan():
    """Every registry entry belongs to a show the plan declares."""
    plan = load_corpus_plan(_config("corpus_v2.yaml"))
    registry = load_sources(_config("sources_v2.yaml"))
    declared = {show.series_id for show in plan.shows}
    found = {entry.series_id for entry in registry.entries}
    assert found <= declared, f"registry has undeclared series: {found - declared}"
