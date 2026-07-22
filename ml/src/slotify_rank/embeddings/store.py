"""On-disk embedding arrays: ``.npy`` plus a JSON sidecar.

**Why not pickle.** ``numpy.save`` with ``allow_pickle=False`` is a documented,
stable binary format that loads without executing anything from the file. Pickle
would be shorter to write and is a remote-code-execution primitive; an artifact
format that has to be trusted before it can be read is not a long-term artifact
format. Every load in this module passes ``allow_pickle=False`` explicitly.

**Why a sidecar.** A bare ``.npy`` records a shape and a dtype and nothing about
where the numbers came from. The sidecar carries the cache identity, the model
id and revision, the row index and the semantic dimension -- so a stale array is
detectable *before* it is used rather than after it has quietly trained a model
on the wrong episode.

**Why row ids.** Embeddings are stored as one ``(n_rows, dim)`` matrix per
episode per kind, with the row order recorded explicitly in the sidecar rather
than implied by sort order. Implied order is the classic way these files go
wrong: the array and the manifest drift by one row and every candidate is
described by its neighbour's audio.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from slotify_rank.config.versions import (
    EMBEDDING_STORE_VERSION,
    SUPPORTED_EMBEDDING_STORE_VERSIONS,
)
from slotify_rank.data.checksum import _replace_with_retry, atomic_write_bytes
from slotify_rank.pipeline.identity import CacheIdentity

__all__ = [
    "EmbeddingMetadata",
    "StaleEmbeddingCache",
    "write_embeddings",
    "read_embeddings",
    "load_embeddings",
    "sidecar_path",
]


class StaleEmbeddingCache(ValueError):
    """A stored array exists but does not describe the requested computation."""


def sidecar_path(array_path: Path) -> Path:
    return Path(str(array_path) + ".json")


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Everything needed to decide whether an array may be reused."""

    kind: str
    episode_id: str
    model_id: str
    model_revision: str
    #: Semantic dimension of one row, e.g. 384 for whisper-tiny.en's encoder.
    dimension: int
    #: Row identities, in array order. Length must equal the array's first axis.
    row_ids: tuple[str, ...]
    identity_digest: str
    dtype: str = "float32"
    store_version: str = EMBEDDING_STORE_VERSION
    extra: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(f"{self.kind}: dimension must be positive")
        if not self.identity_digest:
            raise ValueError(
                f"{self.kind}: an embedding array must record the cache identity "
                "it was produced under, otherwise it can never be invalidated"
            )
        if self.extra is None:
            object.__setattr__(self, "extra", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "episode_id": self.episode_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dimension": self.dimension,
            "row_count": len(self.row_ids),
            "row_ids": list(self.row_ids),
            "identity_digest": self.identity_digest,
            "dtype": self.dtype,
            "store_version": self.store_version,
            "extra": dict(self.extra or {}),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EmbeddingMetadata":
        version = str(raw.get("store_version", ""))
        if version not in SUPPORTED_EMBEDDING_STORE_VERSIONS:
            raise StaleEmbeddingCache(
                f"Unsupported embedding store_version {version!r}; this build "
                f"reads {sorted(SUPPORTED_EMBEDDING_STORE_VERSIONS)}"
            )
        return cls(
            kind=str(raw["kind"]),
            episode_id=str(raw["episode_id"]),
            model_id=str(raw["model_id"]),
            model_revision=str(raw["model_revision"]),
            dimension=int(raw["dimension"]),
            row_ids=tuple(str(value) for value in raw.get("row_ids", ())),
            identity_digest=str(raw["identity_digest"]),
            dtype=str(raw.get("dtype", "float32")),
            store_version=version,
            extra=dict(raw.get("extra") or {}),
        )


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """``numpy.save`` through a temporary file in the same directory.

    ``np.save`` writes incrementally, so a crash mid-write leaves a file with a
    valid header and truncated data -- which loads without error and yields
    silently wrong numbers. Writing to a temp file and renaming means a reader
    can only ever see the complete array.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp_name, destination)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_embeddings(
    array_path: Path, array: np.ndarray, metadata: EmbeddingMetadata
) -> None:
    """Write an embedding matrix and its sidecar.

    The array goes first and the sidecar second, on purpose: the sidecar is what
    makes an array *usable*, so a crash between the two writes leaves an
    unreadable-but-harmless orphan rather than a trusted array with no
    provenance.
    """
    matrix = np.asarray(array)
    if matrix.ndim != 2:
        raise ValueError(
            f"{metadata.kind}: expected a 2-D (rows, dim) array, got shape "
            f"{matrix.shape}"
        )
    if matrix.shape[0] != len(metadata.row_ids):
        raise ValueError(
            f"{metadata.kind}: array has {matrix.shape[0]} rows but metadata lists "
            f"{len(metadata.row_ids)} row ids. Refusing to write a file whose rows "
            "cannot be attributed."
        )
    if matrix.shape[1] != metadata.dimension:
        raise ValueError(
            f"{metadata.kind}: array dimension {matrix.shape[1]} does not match the "
            f"declared dimension {metadata.dimension}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{metadata.kind} ({metadata.episode_id}): array contains NaN or "
            "infinity. Refusing to cache a corrupt embedding -- it would poison "
            "every model trained on it and is nearly invisible downstream."
        )

    target = Path(array_path)
    contiguous = np.ascontiguousarray(matrix, dtype=np.dtype(metadata.dtype))
    _atomic_write_npy(target, contiguous)
    payload = json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False) + "\n"
    atomic_write_bytes(sidecar_path(target), payload.encode("utf-8"))


def read_embeddings(array_path: Path) -> tuple[np.ndarray, EmbeddingMetadata]:
    """Load an array and its sidecar, with no cache-validity judgement."""
    target = Path(array_path)
    sidecar = sidecar_path(target)
    if not target.is_file():
        raise FileNotFoundError(f"Embedding array not found: {target}")
    if not sidecar.is_file():
        raise StaleEmbeddingCache(
            f"{target} has no metadata sidecar ({sidecar.name}). An array without "
            "provenance cannot be verified and is treated as unusable."
        )
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StaleEmbeddingCache(f"{sidecar} is not valid JSON: {error}") from error

    metadata = EmbeddingMetadata.from_mapping(raw)
    matrix = np.load(target, allow_pickle=False)
    if matrix.ndim != 2:
        raise StaleEmbeddingCache(
            f"{target} holds a {matrix.ndim}-D array; expected (rows, dim)"
        )
    if matrix.shape[1] != metadata.dimension:
        raise StaleEmbeddingCache(
            f"{target} has dimension {matrix.shape[1]} but its sidecar declares "
            f"{metadata.dimension}. The array and its metadata disagree."
        )
    if matrix.shape[0] != len(metadata.row_ids):
        raise StaleEmbeddingCache(
            f"{target} has {matrix.shape[0]} rows but its sidecar lists "
            f"{len(metadata.row_ids)} row ids."
        )
    return matrix, metadata


def load_embeddings(
    array_path: Path,
    expected: CacheIdentity | None = None,
    expected_dimension: int | None = None,
) -> tuple[np.ndarray, EmbeddingMetadata] | None:
    """Return a cached array, or ``None`` when it is absent or stale.

    Absence returns ``None`` (nothing has been computed). A *present but
    unusable* artifact -- bad dimension, missing sidecar, wrong store version --
    also returns ``None`` so the caller simply recomputes, except that the
    reason is worth surfacing, which :func:`read_embeddings` does by raising.
    """
    target = Path(array_path)
    if not target.is_file():
        return None
    try:
        matrix, metadata = read_embeddings(target)
    except StaleEmbeddingCache:
        return None
    if expected is not None and metadata.identity_digest != expected.digest:
        return None
    if expected_dimension is not None and metadata.dimension != expected_dimension:
        return None
    return matrix, metadata


def row_lookup(metadata: EmbeddingMetadata) -> dict[str, int]:
    """Map row id to row index, rejecting duplicates.

    A duplicated row id means two candidates would resolve to the same vector,
    which is undetectable downstream.
    """
    lookup: dict[str, int] = {}
    for index, row_id in enumerate(metadata.row_ids):
        if row_id in lookup:
            raise ValueError(
                f"{metadata.kind} ({metadata.episode_id}): duplicate row id "
                f"{row_id!r} at rows {lookup[row_id]} and {index}"
            )
        lookup[row_id] = index
    return lookup


def select_rows(
    matrix: np.ndarray, metadata: EmbeddingMetadata, row_ids: Sequence[str]
) -> np.ndarray:
    """Gather rows by id, raising on anything missing."""
    lookup = row_lookup(metadata)
    missing = [row_id for row_id in row_ids if row_id not in lookup]
    if missing:
        raise KeyError(
            f"{metadata.kind} ({metadata.episode_id}): no embedding row for "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return matrix[[lookup[row_id] for row_id in row_ids]]
