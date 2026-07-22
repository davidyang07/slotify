"""Repository-relative path handling.

Every path that reaches a manifest is stored **relative to the repository root,
with forward slashes**. That is the whole reason this module exists: a manifest
containing ``C:\\Users\\danda\\...`` is unusable on any other machine and cannot
be reviewed in a diff, so :func:`to_repo_relative` is the only sanctioned way to
put a path into a record and :func:`resolve` is the only sanctioned way to get
one back out.

Layout under the (git-ignored) data root::

    data/raw/          originals: copied local files and downloaded files
    data/normalized/   16 kHz mono PCM WAV renders, keyed by episode id
    data/transcripts/  optional timestamped transcripts
    data/manifests/    episodes.jsonl, candidates.jsonl, splits_*.json
    data/labels/       labels.sqlite3 and exported JSONL
    data/cache/clips/  context-window clips served by the labelling UI

``artifacts/dataset/`` holds the generated statistics, which *are* committed --
they are the evidence trail for ``docs/resume-claim-matrix.md``.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from slotify_rank.config.settings import find_repo_root

__all__ = [
    "DataPaths",
    "default_paths",
    "to_repo_relative",
    "resolve",
    "is_within",
]

_DATA_ROOT_ENV_VAR = "SLOTIFY_DATA_ROOT"


def to_repo_relative(path: Path | str, repo_root: Path | None = None) -> str:
    """Render ``path`` as a POSIX path relative to the repository root.

    Raises if the path escapes the repository: an episode that points outside
    the checkout cannot be reconstructed from the manifest, and silently
    absolutising it is exactly the failure this module exists to prevent.
    """
    root = (repo_root or find_repo_root()).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{resolved} is outside the repository root {root}; manifests may only "
            "reference paths inside the checkout. Import the file first so it is "
            "copied under the data root."
        ) from error
    return PurePosixPath(relative.as_posix()).as_posix()


def resolve(relative: str, repo_root: Path | None = None) -> Path:
    """Inverse of :func:`to_repo_relative`."""
    if not relative:
        raise ValueError("Cannot resolve an empty path")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            f"Manifest paths must be repository-relative and must not traverse "
            f"upward; got {relative!r}"
        )
    root = (repo_root or find_repo_root()).resolve()
    return root.joinpath(*candidate.parts)


def is_within(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or lives underneath it."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


class DataPaths:
    """Resolved locations for every dataset artifact.

    ``SLOTIFY_DATA_ROOT`` moves the data root off the repository (useful when
    the checkout is on a synced drive and the corpus is large). The default is
    ``<repo>/data``, which ``.gitignore`` excludes.
    """

    def __init__(self, repo_root: Path | None = None, data_root: Path | None = None):
        self.repo_root = (repo_root or find_repo_root()).resolve()
        if data_root is not None:
            self.data_root = Path(data_root).resolve()
        else:
            override = os.environ.get(_DATA_ROOT_ENV_VAR)
            self.data_root = (
                Path(override).expanduser().resolve()
                if override
                else self.repo_root / "data"
            )

    # -- directories -------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def normalized_dir(self) -> Path:
        return self.data_root / "normalized"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_root / "transcripts"

    @property
    def manifests_dir(self) -> Path:
        return self.data_root / "manifests"

    @property
    def labels_dir(self) -> Path:
        return self.data_root / "labels"

    @property
    def clips_dir(self) -> Path:
        return self.data_root / "cache" / "clips"

    @property
    def artifacts_dir(self) -> Path:
        # Derived from the data root, not pinned to the repository: statistics
        # describe a specific corpus, so pointing --data-root elsewhere must move
        # the artifacts with it. Pinning this to the repo root meant a test run
        # against a scratch corpus overwrote the committed statistics.
        return self.data_root.parent / "artifacts" / "dataset"

    # -- files -------------------------------------------------------------
    @property
    def episodes_manifest(self) -> Path:
        return self.manifests_dir / "episodes.jsonl"

    @property
    def candidates_manifest(self) -> Path:
        return self.manifests_dir / "candidates.jsonl"

    def split_manifest(self, version: str) -> Path:
        return self.manifests_dir / f"splits_{version}.json"

    @property
    def label_database(self) -> Path:
        return self.labels_dir / "labels.sqlite3"

    def mkdirs(self) -> None:
        for directory in (
            self.raw_dir,
            self.normalized_dir,
            self.transcripts_dir,
            self.manifests_dir,
            self.labels_dir,
            self.clips_dir,
            self.artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path | str) -> str:
        return to_repo_relative(path, self.repo_root)

    def absolute(self, relative_path: str) -> Path:
        return resolve(relative_path, self.repo_root)


def default_paths() -> DataPaths:
    return DataPaths()
