"""SQLite storage for human labels.

SQLite rather than JSONL because labels are the only artifact in this project
that cannot be regenerated. A JSONL file appended to from a web handler can be
truncated by a crash mid-write or interleaved by two open tabs; SQLite gives
transactions, a real uniqueness constraint and foreign keys for the price of a
stdlib import.

Schema decisions worth stating:

* **One active label per (annotator, candidate)**, enforced by a unique index.
  Re-rating a candidate updates the existing row and bumps ``updated_at`` rather
  than appending a second opinion, so a count of rows is a count of judgements.
* **Foreign keys are on.** A label can only point at a candidate the labelling
  session actually registered, which is what makes "labels referencing missing
  candidates" a schema error rather than a validation finding.
* **``annotator_id`` is free text and expected to be a pseudonym.** No personal
  information is required, requested or stored anywhere in this schema.
* **``is_acceptable`` is derived, not entered.** It is computed from
  ``quality_score`` against a configurable threshold and stored so that changing
  the threshold later cannot silently rewrite history -- the stored value
  records what the rule said at the time.

The database file lives under ``data/labels/`` and is git-ignored.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from slotify_rank.config.versions import LABEL_RUBRIC_VERSION

__all__ = [
    "LabelRecord",
    "LabelDatabase",
    "DEFAULT_ACCEPTABLE_THRESHOLD",
    "MIN_QUALITY_SCORE",
    "MAX_QUALITY_SCORE",
]

MIN_QUALITY_SCORE = 1
MAX_QUALITY_SCORE = 5
#: ``is_acceptable = quality_score >= 3`` (docs/labelling-guide.md).
DEFAULT_ACCEPTABLE_THRESHOLD = 3

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id   TEXT PRIMARY KEY,
    episode_id     TEXT NOT NULL,
    timestamp_ms   INTEGER NOT NULL,
    dataset_split  TEXT NOT NULL DEFAULT 'unassigned',
    registered_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    label_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id   TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    episode_id     TEXT NOT NULL,
    annotator_id   TEXT NOT NULL,
    quality_score  INTEGER NOT NULL CHECK (quality_score BETWEEN 1 AND 5),
    is_acceptable  INTEGER NOT NULL CHECK (is_acceptable IN (0, 1)),
    is_unusable    INTEGER NOT NULL DEFAULT 0 CHECK (is_unusable IN (0, 1)),
    notes          TEXT,
    rubric_version TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS labels_unique_active
    ON labels (annotator_id, candidate_id);

CREATE INDEX IF NOT EXISTS labels_by_episode ON labels (episode_id);
CREATE INDEX IF NOT EXISTS candidates_by_episode ON candidates (episode_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class LabelRecord:
    label_id: int
    candidate_id: str
    episode_id: str
    annotator_id: str
    quality_score: int
    is_acceptable: bool
    is_unusable: bool
    notes: str | None
    rubric_version: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "annotator_id": self.annotator_id,
            "quality_score": self.quality_score,
            "is_acceptable": self.is_acceptable,
            "is_unusable": self.is_unusable,
            "notes": self.notes,
            "rubric_version": self.rubric_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LabelRecord":
        return cls(
            label_id=int(row["label_id"]),
            candidate_id=row["candidate_id"],
            episode_id=row["episode_id"],
            annotator_id=row["annotator_id"],
            quality_score=int(row["quality_score"]),
            is_acceptable=bool(row["is_acceptable"]),
            is_unusable=bool(row["is_unusable"]),
            notes=row["notes"],
            rubric_version=row["rubric_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class LabelDatabase:
    """Thin, explicit wrapper over the label store. No ORM, no abstraction layer."""

    def __init__(
        self,
        path: Path | str,
        acceptable_threshold: int = DEFAULT_ACCEPTABLE_THRESHOLD,
    ):
        if not MIN_QUALITY_SCORE <= acceptable_threshold <= MAX_QUALITY_SCORE:
            raise ValueError(
                f"acceptable_threshold must be between {MIN_QUALITY_SCORE} and "
                f"{MAX_QUALITY_SCORE}, got {acceptable_threshold}"
            )
        self.path = Path(path)
        self.acceptable_threshold = acceptable_threshold
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
        finally:
            connection.close()

    # -- candidate registration -------------------------------------------
    def register_candidates(
        self, candidates: Sequence[Any]
    ) -> int:
        """Register (or refresh) the candidates available for labelling.

        Accepts :class:`~slotify_rank.data.schema.DatasetCandidate` values.
        Synthetic product padding is rejected outright: it must never be shown to
        an annotator, and refusing it here means the UI cannot serve it even by
        mistake.
        """
        rows = []
        now = _now()
        for candidate in candidates:
            if getattr(candidate, "is_synthetic", False):
                raise ValueError(
                    f"{candidate.candidate_id}: synthetic product padding is not "
                    "eligible for labelling and must not be registered"
                )
            if not getattr(candidate, "eligible_for_labelling", True):
                continue
            rows.append(
                (
                    candidate.candidate_id,
                    candidate.episode_id,
                    int(candidate.timestamp_ms),
                    candidate.dataset_split,
                    now,
                )
            )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.executemany(
                """
                INSERT INTO candidates
                    (candidate_id, episode_id, timestamp_ms, dataset_split, registered_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    timestamp_ms = excluded.timestamp_ms,
                    dataset_split = excluded.dataset_split
                """,
                rows,
            )
            connection.execute("COMMIT")
        return len(rows)

    def registered_candidate_ids(self) -> set[str]:
        with self._connect() as connection:
            return {
                row["candidate_id"]
                for row in connection.execute("SELECT candidate_id FROM candidates")
            }

    # -- labels ------------------------------------------------------------
    def upsert_label(
        self,
        candidate_id: str,
        annotator_id: str,
        quality_score: int,
        is_unusable: bool = False,
        notes: str | None = None,
        rubric_version: str = LABEL_RUBRIC_VERSION,
    ) -> LabelRecord:
        """Insert or update this annotator's judgement of this candidate.

        The whole operation is one statement inside one transaction, so a label
        is either fully persisted or not persisted at all -- a partially written
        judgement is never visible to a resumed session.
        """
        if not annotator_id or not annotator_id.strip():
            raise ValueError("annotator_id is required (a pseudonym is fine)")
        if not isinstance(quality_score, int) or isinstance(quality_score, bool):
            raise TypeError(f"quality_score must be an int, got {quality_score!r}")
        if not MIN_QUALITY_SCORE <= quality_score <= MAX_QUALITY_SCORE:
            raise ValueError(
                f"quality_score must be between {MIN_QUALITY_SCORE} and "
                f"{MAX_QUALITY_SCORE}, got {quality_score}"
            )
        annotator = annotator_id.strip()
        now = _now()
        acceptable = int(quality_score >= self.acceptable_threshold)

        with self._connect() as connection:
            candidate = connection.execute(
                "SELECT episode_id FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise KeyError(
                    f"Unknown candidate_id {candidate_id!r}. Register the candidate "
                    "manifest with the labelling session before labelling."
                )
            connection.execute("BEGIN")
            connection.execute(
                """
                INSERT INTO labels (
                    candidate_id, episode_id, annotator_id, quality_score,
                    is_acceptable, is_unusable, notes, rubric_version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(annotator_id, candidate_id) DO UPDATE SET
                    quality_score = excluded.quality_score,
                    is_acceptable = excluded.is_acceptable,
                    is_unusable   = excluded.is_unusable,
                    notes         = excluded.notes,
                    rubric_version= excluded.rubric_version,
                    updated_at    = excluded.updated_at
                """,
                (
                    candidate_id,
                    candidate["episode_id"],
                    annotator,
                    quality_score,
                    acceptable,
                    int(bool(is_unusable)),
                    notes,
                    rubric_version,
                    now,
                    now,
                ),
            )
            connection.execute("COMMIT")
            row = connection.execute(
                "SELECT * FROM labels WHERE annotator_id = ? AND candidate_id = ?",
                (annotator, candidate_id),
            ).fetchone()
        return LabelRecord.from_row(row)

    def get_label(self, candidate_id: str, annotator_id: str) -> LabelRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM labels WHERE annotator_id = ? AND candidate_id = ?",
                (annotator_id.strip(), candidate_id),
            ).fetchone()
        return None if row is None else LabelRecord.from_row(row)

    def all_labels(self) -> list[LabelRecord]:
        with self._connect() as connection:
            return [
                LabelRecord.from_row(row)
                for row in connection.execute(
                    "SELECT * FROM labels ORDER BY episode_id, candidate_id, annotator_id"
                )
            ]

    def labelled_candidate_ids(self, annotator_id: str | None = None) -> set[str]:
        query = "SELECT candidate_id FROM labels"
        params: tuple[Any, ...] = ()
        if annotator_id:
            query += " WHERE annotator_id = ?"
            params = (annotator_id.strip(),)
        with self._connect() as connection:
            return {row["candidate_id"] for row in connection.execute(query, params)}

    def progress(self, annotator_id: str) -> dict[str, int]:
        """Counts backing the UI's progress display and session resumption."""
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS n FROM candidates"
            ).fetchone()["n"]
            done = connection.execute(
                "SELECT COUNT(*) AS n FROM labels WHERE annotator_id = ?",
                (annotator_id.strip(),),
            ).fetchone()["n"]
            unusable = connection.execute(
                "SELECT COUNT(*) AS n FROM labels WHERE annotator_id = ? AND is_unusable = 1",
                (annotator_id.strip(),),
            ).fetchone()["n"]
        return {
            "total_candidates": int(total),
            "labelled": int(done),
            "remaining": int(total) - int(done),
            "marked_unusable": int(unusable),
        }

    def annotators(self) -> list[str]:
        with self._connect() as connection:
            return [
                row["annotator_id"]
                for row in connection.execute(
                    "SELECT DISTINCT annotator_id FROM labels ORDER BY annotator_id"
                )
            ]
