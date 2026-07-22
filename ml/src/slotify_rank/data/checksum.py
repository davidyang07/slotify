"""SHA-256 helpers and atomic file writes.

Checksums are the dataset's integrity backbone: they give episodes deterministic
identities, let ``normalize`` skip work safely, let ``fetch`` refuse a corrupted
or substituted download, and let ``validate`` prove that a manifest still
describes the bytes on disk.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterator

__all__ = [
    "sha256_file",
    "sha256_text",
    "verify_sha256",
    "ChecksumMismatch",
    "atomic_write_bytes",
    "atomic_replace",
]

_CHUNK_SIZE = 1024 * 1024


class ChecksumMismatch(ValueError):
    """Raised when a file's SHA-256 does not match the expected value."""

    def __init__(self, path: Path, expected: str, actual: str):
        self.path = Path(path)
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def sha256_file(path: Path | str) -> str:
    """Streaming SHA-256 of a file. Rejects zero-length files."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Not a readable file: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Refusing to hash a zero-length file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_sha256(path: Path | str, expected: str) -> str:
    """Hash ``path`` and raise :class:`ChecksumMismatch` unless it matches."""
    normalized_expected = expected.strip().lower()
    if len(normalized_expected) != 64 or not all(
        character in "0123456789abcdef" for character in normalized_expected
    ):
        raise ValueError(
            f"Expected SHA-256 must be 64 hex characters, got {expected!r}"
        )
    actual = sha256_file(path)
    if actual != normalized_expected:
        raise ChecksumMismatch(Path(path), normalized_expected, actual)
    return actual


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a temporary file in the same directory.

    A half-written manifest or audio file that still carries a plausible name is
    far worse than a missing one, so nothing is ever written in place.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def atomic_replace(temp_path: Path, destination: Path) -> None:
    """Move a fully written temporary file into place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, destination)


def iter_files(root: Path, suffixes: tuple[str, ...]) -> Iterator[Path]:
    """Deterministically ordered recursive file listing filtered by suffix."""
    lowered = tuple(suffix.lower() for suffix in suffixes)
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in lowered:
            yield path
