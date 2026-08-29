"""SQLite storage for human labels.

SQLite rather than JSONL because labels are the only artifact in this project
that cannot be regenerated. A JSONL file appended to from a web handler can be
truncated by a crash mid-write or interleaved by two open tabs; SQLite gives
transactions, a real uniqueness constraint and foreign keys for the price of a
stdlib import.

Schema decisions worth stating:

* **A judgement is keyed on a *presentation*, not on a candidate.** The
  labelling queue can show one candidate twice under two presentation ids to
  measure whether an annotator agrees with themselves. Keying on the candidate
  would make the second showing silently overwrite the first and destroy exactly
  the measurement it exists to produce. The unique index is therefore
  ``(annotator_id, presentation_id)``; a candidate's *unique* label count counts
  only non-repeat presentations, and :meth:`repeat_pairs` returns the repeats for
  the consistency report.
* **``annotator_id`` is free text and expected to be a pseudonym.** No personal
  information is required, requested or stored anywhere in this schema.
* **``is_acceptable`` is derived, not entered.** It is computed from
  ``quality_score`` against a configurable threshold and stored so that changing
  the threshold later cannot silently rewrite history -- the stored value
  records what the rule said at the time.
* **Skipping is not labelling.** A deferred item goes in ``skips`` and comes back
  only when the annotator asks for it; it never becomes a judgement. An item that
  is *broken* is a label with ``is_unusable`` set, which is a judgement and is
  recorded as one.
* **Every row carries its own provenance**: annotator, candidate, episode,
  series, split, queue version, rubric version, label schema version, the wall
  clock, and how long the annotator spent on the item.

The 1-5 rubric is a graded relevance scale, not a binary one. NDCG consumes
``quality_score`` directly; the equivalent 0-4 grade is ``quality_score - 1``
and is exposed as :func:`graded_relevance` so no call site re-derives it.

The database file lives under ``data/labels/`` and is git-ignored.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from slotify_rank.config.versions import LABEL_RUBRIC_VERSION, LABEL_SCHEMA_VERSION

__all__ = [
    "LabelRecord",
    "SkipRecord",
    "LabelDatabase",
    "DEFAULT_ACCEPTABLE_THRESHOLD",
    "MIN_QUALITY_SCORE",
    "MAX_QUALITY_SCORE",
    "MIN_GRADED_RELEVANCE",
    "MAX_GRADED_RELEVANCE",
    "graded_relevance",
]

MIN_QUALITY_SCORE = 1
MAX_QUALITY_SCORE = 5
#: The same judgement expressed as a 0-based graded relevance grade, which is the
#: form NDCG literature uses. Derived, never entered, never stored.
MIN_GRADED_RELEVANCE = MIN_QUALITY_SCORE - 1
MAX_GRADED_RELEVANCE = MAX_QUALITY_SCORE - 1
#: ``is_acceptable = quality_score >= 3`` (docs/labelling-guide.md).
DEFAULT_ACCEPTABLE_THRESHOLD = 3


def graded_relevance(quality_score: float) -> float:
    """The 1-5 rubric score as a 0-4 graded relevance grade."""
    return float(quality_score) - 1.0


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id   TEXT PRIMARY KEY,
    episode_id     TEXT NOT NULL,
    series_id      TEXT NOT NULL DEFAULT '',
    timestamp_ms   INTEGER NOT NULL,
    dataset_split  TEXT NOT NULL DEFAULT 'unassigned',
    registered_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS presentations (
    presentation_id TEXT PRIMARY KEY,
    candidate_id    TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    queue_version   TEXT NOT NULL DEFAULT '',
    stage           TEXT NOT NULL DEFAULT 'primary',
    is_repeat       INTEGER NOT NULL DEFAULT 0 CHECK (is_repeat IN (0, 1)),
    position        INTEGER NOT NULL DEFAULT 0,
    registered_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    label_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    presentation_id TEXT NOT NULL,
    candidate_id   TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    episode_id     TEXT NOT NULL,
    series_id      TEXT NOT NULL DEFAULT '',
    dataset_split  TEXT NOT NULL DEFAULT 'unassigned',
    annotator_id   TEXT NOT NULL,
    quality_score  INTEGER NOT NULL CHECK (quality_score BETWEEN 1 AND 5),
    is_acceptable  INTEGER NOT NULL CHECK (is_acceptable IN (0, 1)),
    is_unusable    INTEGER NOT NULL DEFAULT 0 CHECK (is_unusable IN (0, 1)),
    is_repeat      INTEGER NOT NULL DEFAULT 0 CHECK (is_repeat IN (0, 1)),
    stage          TEXT NOT NULL DEFAULT 'primary',
    queue_version  TEXT NOT NULL DEFAULT '',
    elapsed_ms     INTEGER,
    notes          TEXT,
    rubric_version TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT 'label-schema-v1.0.0',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS labels_unique_active
    ON labels (annotator_id, presentation_id);

CREATE INDEX IF NOT EXISTS labels_by_episode ON labels (episode_id);
CREATE INDEX IF NOT EXISTS labels_by_candidate ON labels (candidate_id);
CREATE INDEX IF NOT EXISTS candidates_by_episode ON candidates (episode_id);

CREATE TABLE IF NOT EXISTS skips (
    skip_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    presentation_id TEXT NOT NULL,
    candidate_id    TEXT NOT NULL,
    annotator_id    TEXT NOT NULL,
    reason          TEXT,
    created_at      TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS skips_unique
    ON skips (annotator_id, presentation_id);
"""

#: Columns added after the first release. Applied with ``ALTER TABLE`` on open so
#: an existing database keeps its labels instead of being recreated -- labels are
#: irreplaceable, so a migration must never be a drop.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("candidates", "series_id", "TEXT NOT NULL DEFAULT ''"),
    ("labels", "presentation_id", "TEXT NOT NULL DEFAULT ''"),
    ("labels", "series_id", "TEXT NOT NULL DEFAULT ''"),
    ("labels", "dataset_split", "TEXT NOT NULL DEFAULT 'unassigned'"),
    ("labels", "is_repeat", "INTEGER NOT NULL DEFAULT 0"),
    ("labels", "stage", "TEXT NOT NULL DEFAULT 'primary'"),
    ("labels", "queue_version", "TEXT NOT NULL DEFAULT ''"),
    ("labels", "elapsed_ms", "INTEGER"),
    ("labels", "schema_version", "TEXT NOT NULL DEFAULT 'label-schema-v1.0.0'"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class LabelRecord:
    """One stored judgement.

    The fields added in ``label-schema-v1.1.0`` carry defaults so a caller that
    only knows the judgement itself -- a test fixture, a hand-built record --
    still constructs a valid one. ``presentation_id`` defaults to the candidate
    id, which is exactly what the queue uses for a non-repeat presentation, so
    the default is the correct value rather than a placeholder.
    """

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
    presentation_id: str = ""
    series_id: str = ""
    dataset_split: str = "unassigned"
    is_repeat: bool = False
    stage: str = "primary"
    queue_version: str = ""
    elapsed_ms: int | None = None
    schema_version: str = LABEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.presentation_id:
            object.__setattr__(self, "presentation_id", self.candidate_id)

    @property
    def graded_relevance(self) -> float:
        return graded_relevance(self.quality_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "presentation_id": self.presentation_id,
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "series_id": self.series_id,
            "dataset_split": self.dataset_split,
            "annotator_id": self.annotator_id,
            "quality_score": self.quality_score,
            "graded_relevance": self.graded_relevance,
            "is_acceptable": self.is_acceptable,
            "is_unusable": self.is_unusable,
            "is_repeat": self.is_repeat,
            "stage": self.stage,
            "queue_version": self.queue_version,
            "elapsed_ms": self.elapsed_ms,
            "notes": self.notes,
            "rubric_version": self.rubric_version,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LabelRecord":
        keys = set(row.keys())

        def value(name: str, default: Any = None) -> Any:
            return row[name] if name in keys else default

        return cls(
            label_id=int(row["label_id"]),
            presentation_id=str(value("presentation_id") or row["candidate_id"]),
            candidate_id=row["candidate_id"],
            episode_id=row["episode_id"],
            series_id=str(value("series_id") or ""),
            dataset_split=str(value("dataset_split") or "unassigned"),
            annotator_id=row["annotator_id"],
            quality_score=int(row["quality_score"]),
            is_acceptable=bool(row["is_acceptable"]),
            is_unusable=bool(row["is_unusable"]),
            is_repeat=bool(value("is_repeat", 0)),
            stage=str(value("stage") or "primary"),
            queue_version=str(value("queue_version") or ""),
            elapsed_ms=(
                None if value("elapsed_ms") is None else int(value("elapsed_ms"))
            ),
            notes=row["notes"],
            rubric_version=row["rubric_version"],
            schema_version=str(value("schema_version") or LABEL_SCHEMA_VERSION),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class SkipRecord:
    presentation_id: str
    candidate_id: str
    annotator_id: str
    reason: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "presentation_id": self.presentation_id,
            "candidate_id": self.candidate_id,
            "annotator_id": self.annotator_id,
            "reason": self.reason,
            "created_at": self.created_at,
        }


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
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Add columns a previous release did not have, keeping every row."""
        for table, column, definition in _MIGRATIONS:
            existing = {
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if not existing:  # table absent; the schema script will have made it
                continue
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                )
        # Pre-migration rows keyed a judgement on the candidate. Backfill the
        # presentation id with the candidate id, which is exactly what the queue
        # uses for a non-repeat presentation, so the unique index stays true.
        connection.execute(
            "UPDATE labels SET presentation_id = candidate_id "
            "WHERE presentation_id IS NULL OR presentation_id = ''"
        )
        connection.execute(
            "DROP INDEX IF EXISTS labels_unique_active"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS labels_unique_active "
            "ON labels (annotator_id, presentation_id)"
        )

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
        self,
        candidates: Sequence[Any],
        series_by_episode: Mapping[str, str] | None = None,
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
                    str((series_by_episode or {}).get(candidate.episode_id, "")),
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
                    (candidate_id, episode_id, series_id, timestamp_ms,
                     dataset_split, registered_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    series_id = CASE
                        WHEN excluded.series_id = '' THEN candidates.series_id
                        ELSE excluded.series_id END,
                    timestamp_ms = excluded.timestamp_ms,
                    dataset_split = excluded.dataset_split
                """,
                rows,
            )
            connection.execute("COMMIT")
        return len(rows)

    def register_presentations(
        self, presentations: Sequence[Any], queue_version: str = ""
    ) -> int:
        """Register the queue's presentation order.

        Accepts :class:`~slotify_rank.labelling.queue.QueuePresentation` values.
        A presentation whose candidate is not registered is skipped rather than
        raising: the queue may name candidates an ineligible episode dropped, and
        that is a corpus question, not a storage one.
        """
        known = self.registered_candidate_ids()
        now = _now()
        rows = [
            (
                presentation.presentation_id,
                presentation.candidate_id,
                queue_version,
                presentation.stage,
                int(bool(presentation.is_repeat)),
                position,
                now,
            )
            for position, presentation in enumerate(presentations)
            if presentation.candidate_id in known
        ]
        if not rows:
            return 0
        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.executemany(
                """
                INSERT INTO presentations
                    (presentation_id, candidate_id, queue_version, stage,
                     is_repeat, position, registered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(presentation_id) DO UPDATE SET
                    candidate_id = excluded.candidate_id,
                    queue_version = excluded.queue_version,
                    stage = excluded.stage,
                    is_repeat = excluded.is_repeat,
                    position = excluded.position
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
        presentation_id: str | None = None,
        stage: str = "primary",
        is_repeat: bool = False,
        queue_version: str = "",
        elapsed_ms: int | None = None,
    ) -> LabelRecord:
        """Insert or update this annotator's judgement of this presentation.

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
        presentation = presentation_id or candidate_id
        now = _now()
        acceptable = int(quality_score >= self.acceptable_threshold)

        with self._connect() as connection:
            candidate = connection.execute(
                "SELECT episode_id, series_id, dataset_split FROM candidates "
                "WHERE candidate_id = ?",
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
                    presentation_id, candidate_id, episode_id, series_id,
                    dataset_split, annotator_id, quality_score, is_acceptable,
                    is_unusable, is_repeat, stage, queue_version, elapsed_ms,
                    notes, rubric_version, schema_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(annotator_id, presentation_id) DO UPDATE SET
                    quality_score = excluded.quality_score,
                    is_acceptable = excluded.is_acceptable,
                    is_unusable   = excluded.is_unusable,
                    is_repeat     = excluded.is_repeat,
                    stage         = excluded.stage,
                    queue_version = excluded.queue_version,
                    elapsed_ms    = excluded.elapsed_ms,
                    notes         = excluded.notes,
                    rubric_version= excluded.rubric_version,
                    schema_version= excluded.schema_version,
                    updated_at    = excluded.updated_at
                """,
                (
                    presentation,
                    candidate_id,
                    candidate["episode_id"],
                    candidate["series_id"],
                    candidate["dataset_split"],
                    annotator,
                    quality_score,
                    acceptable,
                    int(bool(is_unusable)),
                    int(bool(is_repeat)),
                    stage,
                    queue_version,
                    None if elapsed_ms is None else int(elapsed_ms),
                    notes,
                    rubric_version,
                    LABEL_SCHEMA_VERSION,
                    now,
                    now,
                ),
            )
            # Labelling an item resolves any deferral of it.
            connection.execute(
                "DELETE FROM skips WHERE annotator_id = ? AND presentation_id = ?",
                (annotator, presentation),
            )
            connection.execute("COMMIT")
            row = connection.execute(
                "SELECT * FROM labels WHERE annotator_id = ? AND presentation_id = ?",
                (annotator, presentation),
            ).fetchone()
        return LabelRecord.from_row(row)

    def skip(
        self,
        candidate_id: str,
        annotator_id: str,
        presentation_id: str | None = None,
        reason: str | None = None,
    ) -> SkipRecord:
        """Defer an item without judging it.

        A skip is explicitly *not* a label: it never reaches the export, the
        readiness gate or a model. It only removes the item from this annotator's
        queue until :meth:`clear_skips` puts it back.
        """
        if not annotator_id or not annotator_id.strip():
            raise ValueError("annotator_id is required")
        annotator = annotator_id.strip()
        presentation = presentation_id or candidate_id
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.execute(
                """
                INSERT INTO skips
                    (presentation_id, candidate_id, annotator_id, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(annotator_id, presentation_id) DO UPDATE SET
                    reason = excluded.reason,
                    created_at = excluded.created_at
                """,
                (presentation, candidate_id, annotator, reason, now),
            )
            connection.execute("COMMIT")
        return SkipRecord(
            presentation_id=presentation,
            candidate_id=candidate_id,
            annotator_id=annotator,
            reason=reason,
            created_at=now,
        )

    def skipped_presentation_ids(self, annotator_id: str) -> set[str]:
        with self._connect() as connection:
            return {
                row["presentation_id"]
                for row in connection.execute(
                    "SELECT presentation_id FROM skips WHERE annotator_id = ?",
                    (annotator_id.strip(),),
                )
            }

    def all_skips(self) -> list[SkipRecord]:
        with self._connect() as connection:
            return [
                SkipRecord(
                    presentation_id=row["presentation_id"],
                    candidate_id=row["candidate_id"],
                    annotator_id=row["annotator_id"],
                    reason=row["reason"],
                    created_at=row["created_at"],
                )
                for row in connection.execute(
                    "SELECT * FROM skips ORDER BY annotator_id, presentation_id"
                )
            ]

    def clear_skips(self, annotator_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM skips WHERE annotator_id = ?", (annotator_id.strip(),)
            )
            return int(cursor.rowcount or 0)

    def get_label(
        self,
        candidate_id: str,
        annotator_id: str,
        presentation_id: str | None = None,
    ) -> LabelRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM labels WHERE annotator_id = ? AND presentation_id = ?",
                (annotator_id.strip(), presentation_id or candidate_id),
            ).fetchone()
        return None if row is None else LabelRecord.from_row(row)

    def all_labels(self) -> list[LabelRecord]:
        with self._connect() as connection:
            return [
                LabelRecord.from_row(row)
                for row in connection.execute(
                    "SELECT * FROM labels "
                    "ORDER BY episode_id, candidate_id, annotator_id, presentation_id"
                )
            ]

    def labelled_presentation_ids(self, annotator_id: str | None = None) -> set[str]:
        query = "SELECT presentation_id FROM labels"
        params: tuple[Any, ...] = ()
        if annotator_id:
            query += " WHERE annotator_id = ?"
            params = (annotator_id.strip(),)
        with self._connect() as connection:
            return {row["presentation_id"] for row in connection.execute(query, params)}

    def labelled_candidate_ids(self, annotator_id: str | None = None) -> set[str]:
        """Distinct candidates with at least one non-repeat judgement."""
        query = "SELECT candidate_id FROM labels WHERE is_repeat = 0"
        params: tuple[Any, ...] = ()
        if annotator_id:
            query += " AND annotator_id = ?"
            params = (annotator_id.strip(),)
        with self._connect() as connection:
            return {row["candidate_id"] for row in connection.execute(query, params)}

    def repeat_pairs(self) -> list[tuple[LabelRecord, LabelRecord]]:
        """``(first, repeat)`` label pairs for the same annotator and candidate.

        The raw material for intra-annotator consistency. Pairs are formed only
        where the same annotator judged one candidate under both a non-repeat and
        a repeat presentation.
        """
        by_key: dict[tuple[str, str], dict[bool, LabelRecord]] = {}
        for label in self.all_labels():
            slot = by_key.setdefault((label.annotator_id, label.candidate_id), {})
            # Keep the earliest judgement on each side; a re-rating updates in
            # place, so there is at most one row per (annotator, presentation).
            slot.setdefault(label.is_repeat, label)
        pairs: list[tuple[LabelRecord, LabelRecord]] = []
        for slot in by_key.values():
            if True in slot and False in slot:
                pairs.append((slot[False], slot[True]))
        pairs.sort(key=lambda pair: (pair[0].annotator_id, pair[0].candidate_id))
        return pairs

    def progress(self, annotator_id: str, target: int | None = None) -> dict[str, int]:
        """Counts backing the UI's progress display and session resumption."""
        annotator = annotator_id.strip()
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS n FROM candidates"
            ).fetchone()["n"]
            unique = connection.execute(
                "SELECT COUNT(DISTINCT candidate_id) AS n FROM labels "
                "WHERE annotator_id = ? AND is_repeat = 0",
                (annotator,),
            ).fetchone()["n"]
            judgements = connection.execute(
                "SELECT COUNT(*) AS n FROM labels WHERE annotator_id = ?",
                (annotator,),
            ).fetchone()["n"]
            repeats = connection.execute(
                "SELECT COUNT(*) AS n FROM labels WHERE annotator_id = ? AND is_repeat = 1",
                (annotator,),
            ).fetchone()["n"]
            unusable = connection.execute(
                "SELECT COUNT(*) AS n FROM labels WHERE annotator_id = ? AND is_unusable = 1",
                (annotator,),
            ).fetchone()["n"]
            skipped = connection.execute(
                "SELECT COUNT(*) AS n FROM skips WHERE annotator_id = ?",
                (annotator,),
            ).fetchone()["n"]
        goal = int(target) if target else int(total)
        return {
            "total_candidates": int(total),
            "target": goal,
            "labelled": int(unique),
            "judgements": int(judgements),
            "repeat_judgements": int(repeats),
            "remaining": max(0, goal - int(unique)),
            "marked_unusable": int(unusable),
            "skipped": int(skipped),
        }

    def annotators(self) -> list[str]:
        with self._connect() as connection:
            return [
                row["annotator_id"]
                for row in connection.execute(
                    "SELECT DISTINCT annotator_id FROM labels ORDER BY annotator_id"
                )
            ]
