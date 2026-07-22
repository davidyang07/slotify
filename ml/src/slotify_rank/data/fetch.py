"""Reproducible download of remote audio declared in the source registry.

This is the only module in the package that touches the network, and it is
deliberately narrow: one direct file URL per source, no redirects to other
hosts' credentialed endpoints, no feed parsing, no crawling.

Three guarantees:

1. **Nothing is silently replaced.** A download lands on a temporary file and is
   moved into place only after its checksum is known. If a file already exists
   with different bytes, the fetch fails rather than overwriting it.
2. **Checksum mismatches are fatal.** If a source declares ``expected_sha256``
   and the bytes disagree, the partial file is deleted and the fetch raises.
3. **Re-running is free.** An existing file whose checksum already matches is
   left alone, so ``dataset fetch`` is resumable over a large corpus.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from slotify_rank.data.checksum import ChecksumMismatch, sha256_file
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import EpisodeRecord, make_episode_id
from slotify_rank.data.sources import SourceEntry

__all__ = ["FetchResult", "fetch_source", "download_to", "FetchError"]

_USER_AGENT = "slotify-rank/0.2 (dataset fetch; +https://github.com/)"
_CHUNK = 1024 * 256
#: Guard against a mis-declared URL pulling down something enormous.
_DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024


class FetchError(RuntimeError):
    """The remote file could not be retrieved."""


@dataclass(frozen=True)
class FetchResult:
    episode: EpisodeRecord
    downloaded: bool
    bytes_written: int

    @property
    def action(self) -> str:
        return "downloaded" if self.downloaded else "cached"


def download_to(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    timeout: float = 120.0,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> tuple[str, int, bool]:
    """Download ``url`` to ``destination``. Returns ``(sha256, bytes, downloaded)``.

    If ``destination`` already exists, its checksum decides the outcome: a match
    (or a match against ``expected_sha256``) is a no-op; a mismatch is an error.
    """
    destination = Path(destination)
    if destination.exists():
        actual = sha256_file(destination)
        if expected_sha256 and actual != expected_sha256.strip().lower():
            raise ChecksumMismatch(destination, expected_sha256.strip().lower(), actual)
        return actual, destination.stat().st_size, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".part"
    )
    temp_path = Path(temp_name)
    written = 0
    try:
        # The descriptor is adopted immediately. Opening it only after the request
        # succeeds would leak it when the request fails, and on Windows an open
        # handle makes the cleanup unlink fail -- leaving exactly the partial file
        # this function promises never to leave.
        with os.fdopen(handle, "wb") as stream:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise FetchError(
                            f"{url} exceeded the {max_bytes} byte download limit"
                        )
                    stream.write(chunk)
    except urllib.error.URLError as error:
        temp_path.unlink(missing_ok=True)
        raise FetchError(f"Could not download {url}: {error}") from error
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    if written == 0:
        temp_path.unlink(missing_ok=True)
        raise FetchError(f"{url} returned zero bytes")

    actual = sha256_file(temp_path)
    if expected_sha256 and actual != expected_sha256.strip().lower():
        temp_path.unlink(missing_ok=True)
        raise ChecksumMismatch(destination, expected_sha256.strip().lower(), actual)

    shutil.move(str(temp_path), str(destination))
    return actual, written, True


def fetch_source(
    entry: SourceEntry,
    paths: DataPaths,
    existing: dict[str, EpisodeRecord] | None = None,
    timeout: float = 120.0,
) -> FetchResult:
    """Download one ``direct_download`` source and build its episode record."""
    if entry.source_type != "direct_download":
        raise ValueError(
            f"{entry.id}: fetch_source handles direct_download only; use import for "
            f"{entry.source_type}"
        )
    assert entry.url is not None

    suffix = Path(entry.url.split("?", 1)[0]).suffix.lower() or ".audio"
    # The episode id depends on the content hash, which is unknown before the
    # download, so the file lands under a stable source-derived name first.
    destination = paths.raw_dir / f"{entry.id}{suffix}"
    digest, written, downloaded = download_to(
        entry.url, destination, entry.expected_sha256, timeout=timeout
    )

    episode_id = make_episode_id(entry.title, digest)
    episode = EpisodeRecord(
        episode_id=episode_id,
        series_id=entry.series_id,
        title=entry.title,
        source_type=entry.source_type,
        source_uri=entry.url,
        source_name=entry.source_name,
        license_name=entry.license_name,
        license_url=entry.license_url,
        attribution=entry.attribution,
        language=entry.language,
        content_type=entry.content_type,
        original_path=paths.relative(destination),
        sha256=digest,
        status="fetched",
        is_target_domain=entry.is_target_domain,
        notes=entry.notes,
    )
    previous = (existing or {}).get(episode_id)
    if previous is not None:
        episode = previous.replace(
            source_uri=episode.source_uri,
            license_name=episode.license_name,
            license_url=episode.license_url,
            attribution=episode.attribution,
            original_path=episode.original_path,
        )
    return FetchResult(episode=episode, downloaded=downloaded, bytes_written=written)
