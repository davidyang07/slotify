"""Source registry: parsing and validating ``ml/configs/sources.yaml``.

A *source* is a declaration of where audio comes from and under what terms. It
is not audio, and registering one performs no I/O -- that is deliberate, so the
registry can be validated in CI without touching the network or the disk.

Licensing is enforced at this layer, not left to the operator's memory:

* ``direct_download`` **requires** ``license_name`` and ``license_url``. If you
  cannot name the licence, you may not put the URL in this file.
* ``local_file`` and ``existing_repository_fixture`` default to *no* declared
  licence and are marked private. They are usable locally and are never
  redistributed, and the code never assumes a public licence on your behalf.

The registry holds no secrets: any URL requiring a credential, token or signed
query string is rejected outright, because this file is committed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from slotify_rank.config.versions import (
    SOURCE_MANIFEST_VERSION,
    SUPPORTED_SOURCE_MANIFEST_VERSIONS,
)
from slotify_rank.data.schema import (
    TARGET_DOMAIN_CONTENT_TYPES,
    ContentType,
    SourceType,
    slugify,
)

__all__ = [
    "SourceEntry",
    "SourceRegistry",
    "load_sources",
    "SourceConfigError",
]

_SUPPORTED_SCHEMES = ("http", "https")
_SECRET_QUERY_KEYS = (
    "token",
    "access_token",
    "api_key",
    "apikey",
    "key",
    "signature",
    "sig",
    "x-amz-signature",
    "password",
)
_SECRET_TEXT_RE = re.compile(
    r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})"
)
_AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4")


class SourceConfigError(ValueError):
    """A source declaration is malformed, unlicensed, or unsafe to commit."""


@dataclass(frozen=True)
class SourceEntry:
    """One declared source. Contains no audio and performs no I/O."""

    id: str
    source_type: SourceType
    title: str
    series_id: str
    source_name: str
    content_type: ContentType
    language: str = "en"
    path: str | None = None
    url: str | None = None
    expected_sha256: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    transcript_path: str | None = None
    notes: str | None = None
    #: Free-form provenance recorded by the generator that produced this entry
    #: (see :mod:`slotify_rank.data.discover`): the upstream item identifier, the
    #: file it came from, and how its licence was established. Flat, scalar-only
    #: and secret-checked, so it stays reviewable in a diff and safe to commit.
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_target_domain(self) -> bool:
        return self.content_type in TARGET_DOMAIN_CONTENT_TYPES

    @property
    def source_uri(self) -> str:
        """Provenance string recorded on the episode.

        For remote sources this is the URL. For local and fixture sources it is
        the declared path, kept verbatim so the operator can see where a private
        file originally came from; it is provenance, not a resolvable manifest
        path, and the resolvable ones live in ``original_path``.
        """
        if self.source_type == "direct_download":
            assert self.url is not None  # guaranteed by validation
            return self.url
        assert self.path is not None
        return self.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "title": self.title,
            "series_id": self.series_id,
            "source_name": self.source_name,
            "content_type": self.content_type,
            "language": self.language,
            "path": self.path,
            "url": self.url,
            "expected_sha256": self.expected_sha256,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "attribution": self.attribution,
            "transcript_path": self.transcript_path,
            "notes": self.notes,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class SourceRegistry:
    path: Path
    manifest_version: str
    entries: tuple[SourceEntry, ...] = field(default_factory=tuple)

    def get(self, source_id: str) -> SourceEntry:
        for entry in self.entries:
            if entry.id == source_id:
                return entry
        available = ", ".join(sorted(entry.id for entry in self.entries)) or "(none)"
        raise KeyError(f"Unknown source id {source_id!r}. Available: {available}")

    def select(self, ids: Sequence[str] | None) -> list[SourceEntry]:
        if not ids:
            return list(self.entries)
        return [self.get(source_id) for source_id in ids]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


def _check_no_secret(value: str | None, where: str) -> None:
    if not value:
        return
    if _SECRET_TEXT_RE.search(value):
        raise SourceConfigError(
            f"{where} looks like it contains a credential. The source registry is "
            "committed to Git and must never hold secrets."
        )


def _validate_url(url: str, where: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _SUPPORTED_SCHEMES:
        raise SourceConfigError(
            f"{where}: url must use http or https (got {parsed.scheme or 'no'} scheme). "
            "Only direct downloadable URLs are supported; there is no scraping path."
        )
    if not parsed.netloc:
        raise SourceConfigError(f"{where}: url has no host")
    if "@" in parsed.netloc:
        raise SourceConfigError(
            f"{where}: url embeds credentials in the host, which must never be committed"
        )
    query_keys = {
        pair.split("=", 1)[0].lower() for pair in parsed.query.split("&") if pair
    }
    leaked = sorted(query_keys & set(_SECRET_QUERY_KEYS))
    if leaked:
        raise SourceConfigError(
            f"{where}: url carries credential-like query parameter(s) {leaked}. "
            "Use an unauthenticated direct link."
        )
    # A direct file URL is required, so the path must actually end in an audio
    # extension. An extensionless URL is almost always a landing page or a feed,
    # and accepting one is how a scraper gets built by accident.
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in _AUDIO_SUFFIXES:
        raise SourceConfigError(
            f"{where}: url path ends in {suffix or 'no extension'!r}, which is not a "
            f"recognised audio extension {list(_AUDIO_SUFFIXES)}. A direct file URL "
            "is required -- landing pages and feeds are not supported."
        )


def _entry_from_mapping(
    raw: Mapping[str, Any], defaults: Mapping[str, Any], index: int
) -> SourceEntry:
    if not isinstance(raw, Mapping):
        raise SourceConfigError(f"sources[{index}] must be a mapping")
    merged: dict[str, Any] = {**defaults, **raw}
    where = f"sources[{index}]"

    source_id = str(merged.get("id") or "").strip()
    if not source_id:
        raise SourceConfigError(f"{where}: 'id' is required")
    where = f"source {source_id!r}"
    if slugify(source_id, max_length=128) != source_id:
        raise SourceConfigError(
            f"{where}: id must be a lowercase hyphenated slug "
            f"(suggested: {slugify(source_id, max_length=128)!r})"
        )

    source_type = str(merged.get("source_type") or "")
    if source_type not in ("local_file", "direct_download", "existing_repository_fixture"):
        raise SourceConfigError(
            f"{where}: source_type must be one of local_file, direct_download, "
            f"existing_repository_fixture (got {source_type!r})"
        )

    for required in ("title", "source_name", "content_type"):
        if not merged.get(required):
            raise SourceConfigError(f"{where}: {required!r} is required")

    content_type = str(merged["content_type"])
    if content_type not in (
        "podcast",
        "interview",
        "conversational",
        "narrated",
        "meeting",
        "music",
        "other",
    ):
        raise SourceConfigError(f"{where}: unknown content_type {content_type!r}")

    path = merged.get("path")
    url = merged.get("url")
    if source_type == "direct_download":
        if not url:
            raise SourceConfigError(f"{where}: 'url' is required for direct_download")
        if path:
            raise SourceConfigError(f"{where}: 'path' is not allowed for direct_download")
        _validate_url(str(url), where)
        if not merged.get("license_name") or not merged.get("license_url"):
            raise SourceConfigError(
                f"{where}: direct_download requires both license_name and license_url. "
                "If the licence is unknown, download the file yourself and register it "
                "as a local_file (private, not redistributed)."
            )
    else:
        if not path:
            raise SourceConfigError(f"{where}: 'path' is required for {source_type}")
        if url:
            raise SourceConfigError(f"{where}: 'url' is not allowed for {source_type}")

    expected = merged.get("expected_sha256")
    if expected is not None:
        expected = str(expected).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise SourceConfigError(
                f"{where}: expected_sha256 must be 64 hex characters, got {expected!r}"
            )

    for key in ("url", "path", "attribution", "notes", "license_url"):
        _check_no_secret(
            None if merged.get(key) is None else str(merged[key]), f"{where}.{key}"
        )

    provenance = _validate_provenance(merged.get("provenance"), where)

    series_id = str(merged.get("series_id") or "").strip()
    if series_id and slugify(series_id, max_length=128) != series_id:
        raise SourceConfigError(
            f"{where}: series_id must be a lowercase hyphenated slug "
            f"(suggested: {slugify(series_id, max_length=128)!r})"
        )

    return SourceEntry(
        id=source_id,
        source_type=source_type,  # type: ignore[arg-type]
        title=str(merged["title"]),
        series_id=series_id or source_id,
        source_name=str(merged["source_name"]),
        content_type=content_type,  # type: ignore[arg-type]
        language=str(merged.get("language") or "en"),
        path=None if path is None else str(path).replace("\\", "/"),
        url=None if url is None else str(url),
        expected_sha256=expected,
        license_name=_optional_str(merged.get("license_name")),
        license_url=_optional_str(merged.get("license_url")),
        attribution=_optional_str(merged.get("attribution")),
        transcript_path=(
            None
            if merged.get("transcript_path") is None
            else str(merged["transcript_path"]).replace("\\", "/")
        ),
        notes=_optional_str(merged.get("notes")),
        provenance=provenance,
    )


def _validate_provenance(value: Any, where: str) -> dict[str, Any]:
    """Accept a flat mapping of scalars (and lists of scalars), or nothing.

    Nesting is refused rather than flattened: provenance is written into every
    episode's audit trail and read back by the dataset card, and a nested blob
    would turn a reviewable field into an opaque one.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SourceConfigError(f"{where}.provenance must be a mapping")
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            _check_no_secret(None if item is None else str(item), f"{where}.provenance.{name}")
            cleaned[name] = item
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            entries = list(item)
            if any(not isinstance(entry, (str, int, float, bool)) for entry in entries):
                raise SourceConfigError(
                    f"{where}.provenance.{name} may only contain scalars"
                )
            for entry in entries:
                _check_no_secret(str(entry), f"{where}.provenance.{name}")
            cleaned[name] = entries
            continue
        raise SourceConfigError(
            f"{where}.provenance.{name} must be a scalar or a list of scalars, "
            f"got {type(item).__name__}"
        )
    return cleaned


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_sources(path: Path | str) -> SourceRegistry:
    """Load and fully validate a source registry.

    Every problem is a hard failure. A source file that half-parses would let an
    unlicensed or credential-bearing entry into the corpus, and there is no
    development-time convenience worth that.
    """
    import yaml  # lazy: the registry is only needed by dataset commands

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Source registry not found: {config_path}") from error
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise SourceConfigError(f"{config_path} must contain a YAML mapping")

    manifest_version = str(loaded.get("source_manifest_version", ""))
    if manifest_version not in SUPPORTED_SOURCE_MANIFEST_VERSIONS:
        raise SourceConfigError(
            f"{config_path}: unsupported source_manifest_version "
            f"{manifest_version!r}; this build reads "
            f"{sorted(SUPPORTED_SOURCE_MANIFEST_VERSIONS)}"
        )

    defaults = loaded.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise SourceConfigError(f"{config_path}: 'defaults' must be a mapping")

    raw_entries = loaded.get("sources")
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, Iterable) or isinstance(raw_entries, (str, bytes)):
        raise SourceConfigError(f"{config_path}: 'sources' must be a list")

    entries: list[SourceEntry] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(raw_entries):
        entry = _entry_from_mapping(raw, defaults, index)
        if entry.id in seen:
            raise SourceConfigError(
                f"{config_path}: duplicate source id {entry.id!r} "
                f"(sources[{seen[entry.id]}] and sources[{index}])"
            )
        seen[entry.id] = index
        entries.append(entry)

    return SourceRegistry(
        path=config_path,
        manifest_version=manifest_version or SOURCE_MANIFEST_VERSION,
        entries=tuple(entries),
    )
