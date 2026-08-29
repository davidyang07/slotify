"""Reproducible corpus discovery against the Internet Archive.

``dataset fetch`` needs a source registry: a committed list of direct audio URLs
with a named licence. Writing three thousand candidates' worth of that list by
hand is not practical, and pasting URLs found in a browser is not auditable. So
this module builds the registry mechanically from a declarative *corpus plan*
(``ml/configs/corpus_resume_v1.yaml``) and the Internet Archive's public
metadata APIs.

What it is, and what it deliberately is not:

* It is a **metadata** client. It reads ``/metadata/<identifier>`` and the
  public scrape endpoint. It downloads no audio -- that stays the job of
  :mod:`slotify_rank.data.fetch`, which checksums what it gets.
* It is **not a crawler**. Every show in the plan names either one Internet
  Archive item or one search query, and nothing follows a link out of the
  results.
* Its output is a **committed artifact**, not a runtime dependency. Discovery
  runs when the corpus changes; every later stage reads the emitted YAML. So a
  clone with no network can still see exactly which files the corpus is made of.

Licensing is the part worth reading carefully. Every show declares a licence
*basis*, and discovery verifies what a machine can verify:

``declared_public_domain``
    The Internet Archive item itself carries a ``licenseurl`` that is a
    public-domain dedication or the Public Domain Mark. Discovery checks the
    URL against :data:`PUBLIC_DOMAIN_LICENSE_PATTERNS` and **drops** any item
    that does not match. This is the strongest basis available here.

``us_government_work``
    The recording is a work of the United States federal government and is
    therefore in the public domain in the US under 17 U.S.C. Sec. 105,
    regardless of what an uploader tagged the item with. Because that is a fact
    about the *producer*, not about the Internet Archive record, it cannot be
    read off the metadata in general. Discovery therefore records **how** the
    basis was established for each episode:

    * ``agency_collection`` -- the item is in an official agency collection
      named by the show (for example NASA's own ``nasaaudiocollection``), which
      the metadata does prove; or
    * ``manual_attestation`` -- the plan asserts it, naming the agency, the
      programme and its official URL. The assertion is written into every
      episode's provenance and into the dataset card, so a reader can check it
      rather than take it on trust.

An episode whose basis cannot be established at all is dropped and counted, so
"we could not license it" never silently becomes "it is fine".

Access is checked at the same time as licence, and for the same reason: a plan
that resolves to URLs nobody can fetch is a plan that fails hours later, in the
middle of a download, with the corpus half-built. The Internet Archive marks
restricted items with ``access-restricted-item`` on the item and ``private`` on
the file; both are rejected here with a recorded reason, which is how the
SoundCloud-mirrored podcast feeds -- publicly listed, metadata readable, and
``401 Unauthorized`` on download -- are caught before a byte is requested.

Determinism: shows are processed in plan order, items are sorted by
``(identifier, file name)``, and the first ``max_episodes`` that pass the
duration and format filters are kept. Nothing depends on result ranking, on
wall-clock time or on the order the API happened to return.

One subtlety worth stating because it is silent when wrong: an Archive item
usually stores every logical recording several times -- a VBR mp3, an Ogg, a
128 kbps mp3 and a 64 kbps mp3 of the same chapter. Taking "the first two audio
files" from such an item takes one chapter twice, and because the two encodings
have different checksums, everything downstream treats them as two independent
episodes: two sets of candidates over identical audio, in the same split, both
counted. :func:`_audio_files` therefore groups files by recording and keeps one
rendition of each.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from slotify_rank.config.versions import PACKAGE_VERSION, SOURCE_MANIFEST_VERSION
from slotify_rank.data.checksum import atomic_write_bytes, sha256_text
from slotify_rank.data.schema import TARGET_DOMAIN_CONTENT_TYPES, slugify

__all__ = [
    "CORPUS_PLAN_VERSION",
    "item_is_restricted",
    "PUBLIC_DOMAIN_LICENSE_PATTERNS",
    "DiscoveryError",
    "LicenceSpec",
    "ShowSpec",
    "CorpusPlan",
    "DiscoveredEpisode",
    "DiscoveryReport",
    "InternetArchiveClient",
    "load_corpus_plan",
    "discover",
    "render_sources_yaml",
    "write_sources_yaml",
]

#: Bumped when the plan schema changes in a way an older build cannot read.
CORPUS_PLAN_VERSION = "corpus-plan-v1.0.0"

#: A ``licenseurl`` matching one of these is a public-domain dedication or the
#: Public Domain Mark. Matched case-insensitively against the URL path, with the
#: scheme and any ``www.`` ignored, because the Archive stores both ``http`` and
#: ``https`` forms of the same Creative Commons URL.
PUBLIC_DOMAIN_LICENSE_PATTERNS: tuple[str, ...] = (
    "creativecommons.org/publicdomain/zero/1.0",
    "creativecommons.org/publicdomain/mark/1.0",
    "creativecommons.org/licenses/publicdomain",
)

_AUDIO_SUFFIXES = (".mp3", ".m4a", ".ogg", ".flac", ".wav", ".opus")
_USER_AGENT = "slotify-rank/0.2 (corpus discovery; metadata only)"
_SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
_METADATA_URL = "https://archive.org/metadata/"
_DOWNLOAD_URL = "https://archive.org/download/"


class DiscoveryError(RuntimeError):
    """A corpus plan is malformed, or discovery could not satisfy it."""


# ---------------------------------------------------------------------------
# Plan schema
# ---------------------------------------------------------------------------


def _normalise_license_url(url: str) -> str:
    text = str(url).strip().lower()
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    return text.rstrip("/")


def is_public_domain_license_url(url: str | None) -> bool:
    """True when ``url`` is a recognised public-domain dedication or mark."""
    if not url:
        return False
    normalised = _normalise_license_url(url)
    return any(
        normalised.startswith(pattern) for pattern in PUBLIC_DOMAIN_LICENSE_PATTERNS
    )


@dataclass(frozen=True)
class LicenceSpec:
    """How a show's episodes may be redistributed, and how that was established."""

    basis: str
    name: str
    url: str
    #: Producing federal agency. Required for ``us_government_work``.
    agency: str | None = None
    #: The programme's own page, so an attestation can be checked.
    official_url: str | None = None
    #: Internet Archive collections that, by themselves, prove agency origin.
    agency_collections: tuple[str, ...] = ()
    attribution: str | None = None
    #: Free text recorded on every episode. Required when the basis rests on an
    #: attestation rather than on machine-checkable metadata.
    attestation: str | None = None

    def __post_init__(self) -> None:
        if self.basis not in ("declared_public_domain", "us_government_work"):
            raise DiscoveryError(
                f"unknown licence basis {self.basis!r}; expected "
                "'declared_public_domain' or 'us_government_work'"
            )
        if not self.name or not self.url:
            raise DiscoveryError("a licence needs both a name and a url")
        if self.basis == "us_government_work":
            if not self.agency:
                raise DiscoveryError(
                    "us_government_work requires 'agency': the claim is that a "
                    "named federal agency produced the recording"
                )
            if not self.agency_collections and not self.attestation:
                raise DiscoveryError(
                    "us_government_work requires either 'agency_collections' "
                    "(machine-checkable) or an explicit 'attestation' string "
                    "explaining who produced the work and how that was checked"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "name": self.name,
            "url": self.url,
            "agency": self.agency,
            "official_url": self.official_url,
            "agency_collections": list(self.agency_collections),
            "attribution": self.attribution,
            "attestation": self.attestation,
        }


@dataclass(frozen=True)
class ShowSpec:
    """One series in the corpus plan.

    ``selection.kind`` is either ``item_files`` (one Internet Archive item whose
    audio files are the episodes -- how a podcast feed is usually archived) or
    ``search`` (one item per episode, found by a query).
    """

    series_id: str
    title: str
    content_type: str
    licence: LicenceSpec
    kind: str
    max_episodes: int
    identifier: str | None = None
    query: str | None = None
    #: Exact ``creator`` an item must declare to belong to this show. Required
    #: for ``search``: the Archive's query parser tokenizes a quoted
    #: ``creator:"..."`` phrase, so the query is a recall filter that returns a
    #: superset, and this is the precision filter applied to what comes back.
    creator: str | None = None
    source_name: str = "Internet Archive"
    language: str = "en"
    min_duration_seconds: float = 240.0
    max_duration_seconds: float = 3600.0
    #: Substring every accepted file name must contain. Lets one archived feed
    #: with several formats resolve to exactly one file per episode.
    file_name_contains: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if slugify(self.series_id, max_length=128) != self.series_id:
            raise DiscoveryError(
                f"series_id {self.series_id!r} must be a lowercase hyphenated slug"
            )
        if self.kind not in ("item_files", "search"):
            raise DiscoveryError(
                f"{self.series_id}: selection kind must be 'item_files' or 'search', "
                f"got {self.kind!r}"
            )
        if self.kind == "item_files" and not self.identifier:
            raise DiscoveryError(f"{self.series_id}: item_files needs an 'identifier'")
        if self.kind == "search":
            if not self.query:
                raise DiscoveryError(f"{self.series_id}: search needs a 'query'")
            if not self.creator:
                raise DiscoveryError(
                    f"{self.series_id}: search needs an exact 'creator'. Without it "
                    "the Archive's tokenized phrase match returns every item that "
                    "shares a word with the show name, and which of them the first "
                    "page holds is not stable."
                )
        if self.max_episodes < 1:
            raise DiscoveryError(f"{self.series_id}: max_episodes must be positive")
        if self.min_duration_seconds <= 0 or (
            self.max_duration_seconds <= self.min_duration_seconds
        ):
            raise DiscoveryError(
                f"{self.series_id}: need 0 < min_duration_seconds < max_duration_seconds"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "title": self.title,
            "content_type": self.content_type,
            "kind": self.kind,
            "identifier": self.identifier,
            "query": self.query,
            "creator": self.creator,
            "max_episodes": self.max_episodes,
            "source_name": self.source_name,
            "language": self.language,
            "min_duration_seconds": self.min_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "file_name_contains": self.file_name_contains,
            "notes": self.notes,
            "licence": self.licence.to_dict(),
        }


@dataclass(frozen=True)
class CorpusPlan:
    """The whole declarative plan, as loaded from YAML."""

    plan_version: str
    corpus_version: str
    shows: tuple[ShowSpec, ...]
    #: How many items a ``search`` show may examine before giving up on filling
    #: its quota. A bound, not a target: it caps metadata API calls.
    search_scan_limit: int = 120
    #: Items requested from the scrape endpoint per show. Large enough that the
    #: exact-creator filter runs over the whole result set rather than over an
    #: arbitrary first page.
    scrape_page_size: int = 2000
    notes: str | None = None

    def digest(self) -> str:
        payload = {
            "plan_version": self.plan_version,
            "corpus_version": self.corpus_version,
            "search_scan_limit": self.search_scan_limit,
            "scrape_page_size": self.scrape_page_size,
            "shows": [show.to_dict() for show in self.shows],
        }
        return sha256_text(json.dumps(payload, sort_keys=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "corpus_version": self.corpus_version,
            "search_scan_limit": self.search_scan_limit,
            "scrape_page_size": self.scrape_page_size,
            "notes": self.notes,
            "shows": [show.to_dict() for show in self.shows],
        }


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiscoveryError(f"{where} must be a mapping")
    return value


def load_corpus_plan(path: Path | str) -> CorpusPlan:
    """Load and fully validate ``ml/configs/corpus_*.yaml``."""
    import yaml

    plan_path = Path(path)
    try:
        raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DiscoveryError(f"Corpus plan not found: {plan_path}") from error
    loaded = _require_mapping(raw, str(plan_path))

    plan_version = str(loaded.get("plan_version") or "")
    if plan_version != CORPUS_PLAN_VERSION:
        raise DiscoveryError(
            f"{plan_path}: plan_version is {plan_version!r}, but this build reads "
            f"{CORPUS_PLAN_VERSION!r}"
        )
    corpus_version = str(loaded.get("corpus_version") or "")
    if not corpus_version:
        raise DiscoveryError(f"{plan_path}: 'corpus_version' is required")

    defaults = dict(_require_mapping(loaded.get("defaults") or {}, "defaults"))
    licences_raw = _require_mapping(loaded.get("licences") or {}, "licences")
    licences: dict[str, LicenceSpec] = {}
    for name, entry in licences_raw.items():
        mapping = dict(_require_mapping(entry, f"licences.{name}"))
        collections = mapping.pop("agency_collections", ()) or ()
        licences[str(name)] = LicenceSpec(
            agency_collections=tuple(str(item) for item in collections), **mapping
        )

    shows_raw = loaded.get("shows")
    if not isinstance(shows_raw, Sequence) or isinstance(shows_raw, (str, bytes)):
        raise DiscoveryError(f"{plan_path}: 'shows' must be a list")

    shows: list[ShowSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(shows_raw):
        mapping = dict(_require_mapping(entry, f"shows[{index}]"))
        merged: dict[str, Any] = {**defaults, **mapping}
        licence_key = merged.pop("licence", None)
        if not licence_key:
            raise DiscoveryError(f"shows[{index}]: 'licence' is required")
        if str(licence_key) not in licences:
            raise DiscoveryError(
                f"shows[{index}]: unknown licence {licence_key!r}; declared: "
                f"{sorted(licences)}"
            )
        selection = dict(
            _require_mapping(merged.pop("selection", {}), f"shows[{index}].selection")
        )
        merged.pop("licences", None)
        known = set(ShowSpec.__dataclass_fields__) - {"licence", "kind"}
        unknown = sorted((set(merged) | set(selection)) - known - {"kind"})
        if unknown:
            raise DiscoveryError(f"shows[{index}]: unknown setting(s) {unknown}")
        show = ShowSpec(
            licence=licences[str(licence_key)],
            kind=str(selection.pop("kind", "")),
            **{**merged, **selection},
        )
        if show.series_id in seen:
            raise DiscoveryError(f"{plan_path}: duplicate series_id {show.series_id!r}")
        seen.add(show.series_id)
        shows.append(show)

    if not shows:
        raise DiscoveryError(f"{plan_path}: the plan declares no shows")

    return CorpusPlan(
        plan_version=plan_version,
        corpus_version=corpus_version,
        shows=tuple(shows),
        search_scan_limit=int(loaded.get("search_scan_limit", 120)),
        scrape_page_size=int(loaded.get("scrape_page_size", 2000)),
        notes=_optional_text(loaded.get("notes")),
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# Internet Archive client
# ---------------------------------------------------------------------------


class InternetArchiveClient:
    """The two public read-only endpoints this module needs.

    Injected into :func:`discover` so tests exercise selection, filtering and
    licensing against fixtures without a network.
    """

    def __init__(self, timeout: float = 60.0, user_agent: str = _USER_AGENT):
        self.timeout = timeout
        self.user_agent = user_agent

    def _get_json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise DiscoveryError(f"Could not read {url}: {error}") from error
        except json.JSONDecodeError as error:
            raise DiscoveryError(f"{url} did not return JSON: {error}") from error
        if not isinstance(payload, dict):
            raise DiscoveryError(f"{url} returned {type(payload).__name__}, not an object")
        return payload

    def metadata(self, identifier: str) -> dict[str, Any]:
        """Item metadata, including its file table."""
        return self._get_json(_METADATA_URL + urllib.parse.quote(identifier))

    def scrape(self, query: str, count: int) -> list[dict[str, Any]]:
        """Identifiers matching ``query``, capped at ``count``."""
        params = urllib.parse.urlencode(
            {
                "q": query,
                "fields": "identifier,title,creator,licenseurl,date",
                "count": max(100, min(int(count), 10_000)),
            }
        )
        payload = self._get_json(f"{_SCRAPE_URL}?{params}")
        items = payload.get("items")
        if not isinstance(items, list):
            raise DiscoveryError(f"scrape for {query!r} returned no 'items' list")
        return [item for item in items if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredEpisode:
    """One selected audio file, ready to become a source-registry entry."""

    source_id: str
    series_id: str
    title: str
    url: str
    content_type: str
    source_name: str
    language: str
    license_name: str
    license_url: str
    attribution: str | None
    provenance: Mapping[str, Any]
    duration_seconds: float
    size_bytes: int

    def to_source_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": self.source_id,
            "source_type": "direct_download",
            "url": self.url,
            "title": self.title,
            "series_id": self.series_id,
            "content_type": self.content_type,
            "source_name": self.source_name,
            "language": self.language,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "provenance": dict(self.provenance),
        }
        if self.attribution:
            entry["attribution"] = self.attribution
        return entry


@dataclass
class DiscoveryReport:
    """What discovery selected, and everything it rejected and why."""

    plan_version: str
    corpus_version: str
    plan_digest: str
    episodes: list[DiscoveredEpisode] = field(default_factory=list)
    per_show: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_duration_seconds(self) -> float:
        return sum(episode.duration_seconds for episode in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "corpus_version": self.corpus_version,
            "plan_digest": self.plan_digest,
            "package_version": PACKAGE_VERSION,
            "episode_count": len(self.episodes),
            "series_count": len({e.series_id for e in self.episodes}),
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "total_duration_hours": round(self.total_duration_seconds / 3600.0, 4),
            "per_show": list(self.per_show),
            "rejection_count": len(self.rejections),
            "rejections": list(self.rejections),
        }


def _is_truthy(value: Any) -> bool:
    """Internet Archive booleans arrive as the strings 'true'/'false'."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def item_is_restricted(metadata: Mapping[str, Any]) -> bool:
    """True when the item cannot be downloaded even though it can be listed."""
    item = metadata.get("metadata")
    item_metadata: Mapping[str, Any] = item if isinstance(item, Mapping) else {}
    return _is_truthy(item_metadata.get("access-restricted-item")) or _is_truthy(
        metadata.get("is_dark")
    )


#: Renditions of one recording, best first. An Internet Archive item usually
#: holds every logical track four times -- a VBR mp3, an Ogg, a 128 kbps mp3 and
#: a 64 kbps mp3 -- and taking "the first two files" from such an item means
#: taking one chapter twice at two bitrates, not two chapters. Everything
#: downstream would then treat the same recording as two independent episodes.
_FORMAT_PREFERENCE: tuple[str, ...] = (
    "128kbps mp3",
    "vbr mp3",
    "64kbps mp3",
    "ogg vorbis",
)


def _rendition_rank(entry: Mapping[str, Any]) -> int:
    fmt = str(entry.get("format") or "").strip().lower()
    return (
        _FORMAT_PREFERENCE.index(fmt)
        if fmt in _FORMAT_PREFERENCE
        else len(_FORMAT_PREFERENCE)
    )


def _track_key(entry: Mapping[str, Any]) -> str:
    """A stable identity for the *recording*, independent of its encoding.

    The file name with its bitrate suffix and extension removed. The Archive's
    own ``track`` number looks like the better key but is not: it is present on
    the mp3 renditions and absent on the Ogg, so keying on it leaves the Ogg in
    a group of its own and the same chapter is selected twice.
    """
    stem = Path(str(entry.get("name") or "")).stem.lower()
    for suffix in ("_128kb", "_64kb", "_32kb", "_96kb", "_vbr"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _audio_files(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """One entry per logical recording, deduplicated across encodings.

    Sorted by file name so selection is deterministic, and grouped by
    :func:`_track_key` so a chapter contributes exactly one episode however many
    formats the item stores it in.
    """
    files = metadata.get("files")
    if not isinstance(files, list):
        return []
    candidates: list[Mapping[str, Any]] = []
    for entry in files:
        if not isinstance(entry, Mapping):
            continue
        if _is_truthy(entry.get("private")):
            continue
        name = str(entry.get("name") or "")
        if Path(name).suffix.lower() in _AUDIO_SUFFIXES:
            candidates.append(entry)

    best: dict[str, Mapping[str, Any]] = {}
    for entry in candidates:
        key = _track_key(entry)
        current = best.get(key)
        if current is None:
            best[key] = entry
            continue
        # Prefer the better-known format; break ties on the name so the choice
        # never depends on the order the API listed the files.
        if (_rendition_rank(entry), str(entry.get("name"))) < (
            _rendition_rank(current),
            str(current.get("name")),
        ):
            best[key] = entry

    return sorted(best.values(), key=lambda entry: str(entry.get("name")))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration_seconds(entry: Mapping[str, Any]) -> float | None:
    """Internet Archive stores mp3 duration as ``length``, sometimes as h:mm:ss."""
    raw = entry.get("length")
    direct = _float_or_none(raw)
    if direct is not None:
        return direct
    if isinstance(raw, str) and ":" in raw:
        parts = raw.split(":")
        try:
            numbers = [float(part) for part in parts]
        except ValueError:
            return None
        seconds = 0.0
        for number in numbers:
            seconds = seconds * 60.0 + number
        return seconds
    return None


def _download_url(identifier: str, file_name: str) -> str:
    # Only the file name is percent-encoded; the identifier is already an
    # Archive slug. ``safe`` keeps path separators inside a nested file name.
    return (
        _DOWNLOAD_URL
        + urllib.parse.quote(identifier)
        + "/"
        + urllib.parse.quote(file_name, safe="/")
    )


def _licence_verification(
    licence: LicenceSpec, metadata: Mapping[str, Any]
) -> tuple[str, str] | None:
    """``(verified_by, evidence)`` for this item, or ``None`` when unlicensed."""
    item = metadata.get("metadata")
    item_metadata: Mapping[str, Any] = item if isinstance(item, Mapping) else {}
    license_url = item_metadata.get("licenseurl")

    if licence.basis == "declared_public_domain":
        if is_public_domain_license_url(license_url):
            return "item_license_url", str(license_url)
        return None

    collections = item_metadata.get("collection")
    if isinstance(collections, str):
        collections = [collections]
    present = {str(name) for name in (collections or [])}
    overlap = sorted(present & set(licence.agency_collections))
    if overlap:
        return "agency_collection", ",".join(overlap)
    if licence.attestation:
        return "manual_attestation", licence.attestation
    return None


def _source_id(series_id: str, identifier: str, file_name: str, taken: set[str]) -> str:
    """A stable, slug-shaped, unique id for one selected file."""
    stem = Path(file_name).stem
    base = slugify(f"{series_id}-{stem}", max_length=110) or slugify(
        f"{series_id}-{identifier}", max_length=110
    )
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base[:104]}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _episode_title(
    show: ShowSpec, metadata: Mapping[str, Any], file_entry: Mapping[str, Any]
) -> str:
    item = metadata.get("metadata")
    item_metadata: Mapping[str, Any] = item if isinstance(item, Mapping) else {}
    for key in ("title",):
        value = file_entry.get(key)
        if value:
            return f"{show.title} - {value}"
    item_title = item_metadata.get("title")
    if item_title and show.kind == "search":
        return f"{show.title} - {item_title}"
    stem = Path(str(file_entry.get("name") or "episode")).stem
    return f"{show.title} - {stem}"


def _select_from_item(
    show: ShowSpec,
    identifier: str,
    metadata: Mapping[str, Any],
    limit: int,
    taken: set[str],
    rejections: list[dict[str, Any]],
) -> list[DiscoveredEpisode]:
    """Pick up to ``limit`` audio files from one item, deterministically."""
    if item_is_restricted(metadata):
        rejections.append(
            {
                "series_id": show.series_id,
                "identifier": identifier,
                "reason": "access_restricted",
                "detail": (
                    "the item is marked access-restricted, so its files return 401 "
                    "on download even though the metadata is public"
                ),
            }
        )
        return []

    verification = _licence_verification(show.licence, metadata)
    if verification is None:
        rejections.append(
            {
                "series_id": show.series_id,
                "identifier": identifier,
                "reason": "licence_not_established",
                "detail": (
                    f"basis={show.licence.basis}; the item declares "
                    f"licenseurl={_item_field(metadata, 'licenseurl')!r} and is in "
                    f"collections {_item_field(metadata, 'collection')!r}"
                ),
            }
        )
        return []
    verified_by, evidence = verification

    selected: list[DiscoveredEpisode] = []
    for entry in _audio_files(metadata):
        if len(selected) >= limit:
            break
        name = str(entry.get("name"))
        if show.file_name_contains and show.file_name_contains not in name:
            continue
        duration = _duration_seconds(entry)
        if duration is None:
            rejections.append(
                {
                    "series_id": show.series_id,
                    "identifier": identifier,
                    "file": name,
                    "reason": "no_duration_metadata",
                }
            )
            continue
        if not (show.min_duration_seconds <= duration <= show.max_duration_seconds):
            rejections.append(
                {
                    "series_id": show.series_id,
                    "identifier": identifier,
                    "file": name,
                    "reason": "duration_out_of_range",
                    "detail": f"{duration:.1f}s outside "
                    f"[{show.min_duration_seconds:.0f}, {show.max_duration_seconds:.0f}]",
                }
            )
            continue

        provenance = {
            "internet_archive_identifier": identifier,
            "internet_archive_file": name,
            "internet_archive_md5": _optional_text(entry.get("md5")),
            "internet_archive_size_bytes": int(_float_or_none(entry.get("size")) or 0),
            "internet_archive_length_seconds": round(duration, 3),
            "internet_archive_uploader": _item_field(metadata, "uploader"),
            "internet_archive_collections": _collections(metadata),
            "internet_archive_license_url": _item_field(metadata, "licenseurl"),
            "licence_basis": show.licence.basis,
            "licence_verified_by": verified_by,
            "licence_evidence": evidence,
            "producing_agency": show.licence.agency,
            "official_programme_url": show.licence.official_url,
        }
        selected.append(
            DiscoveredEpisode(
                source_id=_source_id(show.series_id, identifier, name, taken),
                series_id=show.series_id,
                title=_episode_title(show, metadata, entry),
                url=_download_url(identifier, name),
                content_type=show.content_type,
                source_name=show.source_name,
                language=show.language,
                license_name=show.licence.name,
                license_url=show.licence.url,
                attribution=show.licence.attribution,
                provenance=provenance,
                duration_seconds=duration,
                size_bytes=int(_float_or_none(entry.get("size")) or 0),
            )
        )
    return selected


def _item_field(metadata: Mapping[str, Any], key: str) -> Any:
    item = metadata.get("metadata")
    if isinstance(item, Mapping):
        value = item.get(key)
        if isinstance(value, list):
            return [str(entry) for entry in value]
        return None if value is None else str(value)
    return None


def _collections(metadata: Mapping[str, Any]) -> list[str]:
    value = _item_field(metadata, "collection")
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [str(value)]


def discover(plan: CorpusPlan, client: InternetArchiveClient) -> DiscoveryReport:
    """Resolve every show in ``plan`` into concrete, licensed audio URLs."""
    report = DiscoveryReport(
        plan_version=plan.plan_version,
        corpus_version=plan.corpus_version,
        plan_digest=plan.digest(),
    )
    taken: set[str] = set()

    for show in plan.shows:
        if show.content_type not in TARGET_DOMAIN_CONTENT_TYPES:
            raise DiscoveryError(
                f"{show.series_id}: content_type {show.content_type!r} is outside "
                f"{sorted(TARGET_DOMAIN_CONTENT_TYPES)}; this plan builds the "
                "target-domain corpus only"
            )
        before = len(report.episodes)
        if show.kind == "item_files":
            assert show.identifier is not None
            metadata = client.metadata(show.identifier)
            report.episodes.extend(
                _select_from_item(
                    show,
                    show.identifier,
                    metadata,
                    show.max_episodes,
                    taken,
                    report.rejections,
                )
            )
        else:
            assert show.query is not None
            items = client.scrape(show.query, plan.scrape_page_size)
            matched = [
                item
                for item in items
                if item.get("identifier")
                and str(item.get("creator") or "") == show.creator
            ]
            if not matched:
                report.rejections.append(
                    {
                        "series_id": show.series_id,
                        "reason": "no_item_matched_creator",
                        "detail": (
                            f"{len(items)} item(s) came back for the query but none "
                            f"declares creator == {show.creator!r}"
                        ),
                    }
                )
            identifiers = sorted(
                {str(item["identifier"]) for item in matched}
            )
            scanned = identifiers[: plan.search_scan_limit]
            for identifier in scanned:
                remaining = show.max_episodes - (len(report.episodes) - before)
                if remaining <= 0:
                    break
                metadata = client.metadata(identifier)
                # One episode per item under `search`: an archived feed entry is
                # one recording, and taking several files from it would silently
                # duplicate the same show segment.
                report.episodes.extend(
                    _select_from_item(
                        show, identifier, metadata, 1, taken, report.rejections
                    )
                )

        found = len(report.episodes) - before
        report.per_show.append(
            {
                "series_id": show.series_id,
                "title": show.title,
                "content_type": show.content_type,
                "kind": show.kind,
                "requested_episodes": show.max_episodes,
                "selected_episodes": found,
                "selected_duration_seconds": round(
                    sum(e.duration_seconds for e in report.episodes[before:]), 3
                ),
                "licence_basis": show.licence.basis,
            }
        )

    return report


# ---------------------------------------------------------------------------
# Source-registry emission
# ---------------------------------------------------------------------------


_HEADER = """\
# GENERATED FILE -- do not edit by hand.
#
# Produced by `slotify-rank dataset discover --plan {plan_path}` from corpus
# plan {corpus_version} (plan digest {plan_digest_short}). Regenerate it with
# that command; every URL, duration and licence field below was read from the
# Internet Archive's public metadata API, not typed.
#
# It holds no audio and no credentials. `slotify_rank.data.sources.load_sources`
# revalidates every entry -- licence present, no secrets, real audio file URL --
# before a byte is fetched, and `slotify_rank.data.discover` recorded, per
# episode, how the licence was established (`provenance.licence_verified_by`):
#
#   item_license_url    the Archive item itself declares a public-domain
#                       dedication or the Public Domain Mark;
#   agency_collection   the item is in an official federal agency collection,
#                       which the item metadata proves;
#   manual_attestation  the corpus plan asserts, naming the agency and the
#                       programme's official URL, that the recording is a work
#                       of the United States federal government and therefore in
#                       the public domain in the US under 17 U.S.C. Sec. 105.
#
# Episodes: {episode_count}   Series: {series_count}   Audio: {hours:.2f} h
"""


def _yaml_scalar(value: Any) -> str:
    import yaml

    return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()


def render_sources_yaml(report: DiscoveryReport, plan_path: str) -> str:
    """Render a source registry that :func:`load_sources` accepts."""
    import yaml

    body: dict[str, Any] = {
        "source_manifest_version": SOURCE_MANIFEST_VERSION,
        "sources": [episode.to_source_entry() for episode in report.episodes],
    }
    header = _HEADER.format(
        plan_path=plan_path,
        corpus_version=report.corpus_version,
        plan_digest_short=report.plan_digest[:16],
        episode_count=len(report.episodes),
        series_count=len({e.series_id for e in report.episodes}),
        hours=report.total_duration_seconds / 3600.0,
    )
    rendered = yaml.safe_dump(
        body, sort_keys=False, allow_unicode=True, width=100, default_flow_style=False
    )
    return header + "\n" + rendered


def write_sources_yaml(path: Path, report: DiscoveryReport, plan_path: str) -> None:
    atomic_write_bytes(
        Path(path), render_sources_yaml(report, plan_path).encode("utf-8")
    )


def summarise(report: DiscoveryReport) -> Iterable[str]:
    """Human-readable lines for the CLI."""
    yield (
        f"Discovered {len(report.episodes)} episode(s) across "
        f"{len({e.series_id for e in report.episodes})} series, "
        f"{report.total_duration_seconds / 3600.0:.2f} h of audio."
    )
    for entry in report.per_show:
        yield (
            f"  {entry['series_id']:<34} {entry['selected_episodes']:>3}/"
            f"{entry['requested_episodes']:<3} episodes  "
            f"{entry['selected_duration_seconds'] / 3600.0:>5.2f} h  "
            f"({entry['licence_basis']})"
        )
    if report.rejections:
        reasons: dict[str, int] = {}
        for rejection in report.rejections:
            reasons[rejection["reason"]] = reasons.get(rejection["reason"], 0) + 1
        yield f"  rejected {len(report.rejections)}: {dict(sorted(reasons.items()))}"
