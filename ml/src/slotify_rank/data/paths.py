"""Machine-independent path handling for manifests.

Every path that reaches a manifest is stored **relative, with forward slashes**.
That is the whole reason this module exists: a manifest containing
``C:\\Users\\danda\\...`` is unusable on any other machine and cannot be
reviewed in a diff, so :meth:`DataPaths.relative` (or :func:`to_manifest_path`)
is the only sanctioned way to put a path into a record and
:meth:`DataPaths.absolute` (or :func:`resolve`) the only sanctioned way to get
one back out.

Relative to *what* depends on where the file lives:

* **Inside the repository** -- the default layout, with the data root at
  ``<repo>/data`` -- paths are repository-relative (``data/raw/x.mp3``),
  exactly as they always have been.
* **Inside a data root that is outside the repository** (``SLOTIFY_DATA_ROOT``
  or ``--data-root`` pointing at another drive) -- paths are stored relative to
  the data root behind the :data:`DATA_ROOT_PREFIX` marker
  (``@data-root/raw/x.mp3``) and resolved against whichever data root is
  configured when they are read. Such a path cannot be expressed relative to the
  repository at all, and an absolute one would tie the manifest to one machine;
  the marker keeps it relative and says unambiguously what it is relative to.

Anything outside both roots is refused: it could not be reconstructed from the
manifest on another machine.

Layout under the (git-ignored) data root::

    data/raw/          originals: copied local files and downloaded files
    data/normalized/   16 kHz mono PCM WAV renders, keyed by episode id
    data/transcripts/  optional timestamped transcripts
    data/manifests/    episodes.jsonl, candidates.jsonl, splits_*.json
    data/labels/       labels.sqlite3 and exported JSONL
    data/cache/clips/  context-window clips served by the labelling UI

Phase 3 adds the derived feature artifacts, all git-ignored::

    data/features/handcrafted/  per-episode acoustic + structural scalars
    data/features/audio/        per-episode Whisper encoder embeddings (.npy)
    data/features/text/         per-episode MiniLM embeddings (.npy)
    data/features/state/        per-stage resumability ledgers
    data/manifests/features.jsonl

``artifacts/dataset/`` holds the generated statistics, which *are* committed --
they are the evidence trail behind ``docs/dataset-card.md``.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from slotify_rank.config.settings import find_repo_root

__all__ = [
    "DATA_ROOT_PREFIX",
    "DataPaths",
    "default_paths",
    "to_manifest_path",
    "to_repo_relative",
    "resolve",
    "is_within",
]

_DATA_ROOT_ENV_VAR = "SLOTIFY_DATA_ROOT"

#: Leading component of a manifest path stored relative to the *data root*
#: rather than the repository. Only ever written when the data root lies
#: outside the repository; no repository directory starts with ``@``, so a
#: repository-relative path can never be mistaken for one.
DATA_ROOT_PREFIX = "@data-root/"


def _default_data_root(repo_root: Path) -> Path:
    override = os.environ.get(_DATA_ROOT_ENV_VAR)
    return (
        Path(override).expanduser().resolve() if override else repo_root / "data"
    )


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


def to_manifest_path(
    path: Path | str, repo_root: Path | None = None, data_root: Path | None = None
) -> str:
    """Render ``path`` in the form a manifest stores.

    Repository-relative whenever the file is inside the repository -- which is
    every file when the data root is ``<repo>/data``, so that layout is stored
    exactly as before. Otherwise, when it is inside a data root that lies
    outside the repository, ``@data-root/<relative to the data root>``.
    Anything else raises, for the reason :func:`to_repo_relative` gives.
    """
    root = (repo_root or find_repo_root()).resolve()
    resolved = Path(path).resolve()
    try:
        return PurePosixPath(resolved.relative_to(root).as_posix()).as_posix()
    except ValueError:
        pass
    data = Path(data_root).resolve() if data_root is not None else None
    if data is not None and not is_within(data, root):
        try:
            relative = resolved.relative_to(data)
        except ValueError:
            pass
        else:
            return DATA_ROOT_PREFIX + PurePosixPath(relative.as_posix()).as_posix()
    raise ValueError(
        f"{resolved} is outside the repository root {root}"
        + (
            f" and the data root {data}"
            if data is not None and not is_within(data, root)
            else ""
        )
        + "; manifests may only reference paths under one of them. Import the "
        "file first so it is copied under the data root."
    )


def _checked_parts(relative: str, what: str) -> tuple[str, ...]:
    candidate = PurePosixPath(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            f"Manifest paths must be {what}-relative and must not traverse "
            f"upward; got {relative!r}"
        )
    return candidate.parts


def resolve(
    relative: str, repo_root: Path | None = None, data_root: Path | None = None
) -> Path:
    """Inverse of :func:`to_manifest_path` (and so of :func:`to_repo_relative`).

    A ``@data-root/`` path resolves against ``data_root`` -- by default the one
    ``SLOTIFY_DATA_ROOT`` names, else ``<repo>/data`` -- so a corpus kept off
    the repository keeps working when the drive it is on is mounted somewhere
    else. Every other path resolves against the repository root.
    """
    if not relative:
        raise ValueError("Cannot resolve an empty path")
    root = (repo_root or find_repo_root()).resolve()
    if relative.startswith(DATA_ROOT_PREFIX):
        parts = _checked_parts(relative[len(DATA_ROOT_PREFIX) :], "data-root")
        base = (
            Path(data_root).resolve()
            if data_root is not None
            else _default_data_root(root)
        )
        return base.joinpath(*parts)
    return root.joinpath(*_checked_parts(relative, "repository"))


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
    ``<repo>/data``, which ``.gitignore`` excludes. Manifests written against an
    external data root store ``@data-root/`` paths; see the module docstring.
    """

    def __init__(self, repo_root: Path | None = None, data_root: Path | None = None):
        self.repo_root = (repo_root or find_repo_root()).resolve()
        if data_root is not None:
            self.data_root = Path(data_root).resolve()
        else:
            self.data_root = _default_data_root(self.repo_root)

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

    # -- Phase 3 feature artifacts ----------------------------------------
    @property
    def features_dir(self) -> Path:
        return self.data_root / "features"

    @property
    def handcrafted_dir(self) -> Path:
        return self.features_dir / "handcrafted"

    @property
    def audio_embeddings_dir(self) -> Path:
        return self.features_dir / "audio"

    @property
    def text_embeddings_dir(self) -> Path:
        return self.features_dir / "text"

    @property
    def feature_state_dir(self) -> Path:
        return self.features_dir / "state"

    @property
    def features_manifest(self) -> Path:
        return self.manifests_dir / "features.jsonl"

    def stage_ledger(self, stage: str) -> Path:
        return self.feature_state_dir / f"{stage}.json"

    @property
    def feature_artifacts_dir(self) -> Path:
        """Where the Phase 3 statistics reports are written.

        Sibling of ``artifacts/dataset/`` and derived from the data root for the
        same reason: a run against a scratch corpus must not overwrite the
        committed reports that describe the real one.
        """
        return self.data_root.parent / "artifacts" / "features"

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
            self.handcrafted_dir,
            self.audio_embeddings_dir,
            self.text_embeddings_dir,
            self.feature_state_dir,
            self.feature_artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path | str) -> str:
        """The manifest form of ``path``: see :func:`to_manifest_path`."""
        return to_manifest_path(path, self.repo_root, self.data_root)

    def absolute(self, relative_path: str) -> Path:
        """Inverse of :meth:`relative`, against *this* data root."""
        return resolve(relative_path, self.repo_root, self.data_root)


def default_paths() -> DataPaths:
    return DataPaths()
