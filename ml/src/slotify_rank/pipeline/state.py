"""Per-episode, per-stage processing state.

Resumability needs one thing the filesystem alone cannot give: the difference
between *not attempted yet* and *attempted and failed*. Without it, a retry loop
either recomputes successful work or silently skips broken work, and a corpus
that is 90% done looks identical to one that is 90% broken.

Five states, deliberately distinct:

``missing``
    Never attempted. The default for anything not in the ledger.
``partial``
    Started and interrupted. Written before the work begins and cleared after;
    a ``partial`` entry found at startup means the process died mid-stage, so
    the outputs are untrusted and the stage reruns.
``complete``
    Finished, with a cache identity recorded. Skipped on the next run.
``failed``
    Finished with an error, which is recorded. Skipped by default and retried
    only under ``--retry-failed``, so one unreadable episode does not block a
    corpus and does not quietly vanish either.
``stale``
    Complete, but under a cache identity that no longer matches. Recomputed.
    Reported separately from ``missing`` because a corpus that suddenly goes
    stale usually means a config change nobody intended.

The ledger is a single JSON file per stage, written atomically. It is a cache of
*decisions*, never the source of truth for the data itself: every ``complete``
entry is re-verified against the artifact's own sidecar identity before it is
trusted, so deleting the ledger costs a rescan, never correctness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from slotify_rank.config.versions import PACKAGE_VERSION
from slotify_rank.data.checksum import atomic_write_bytes

__all__ = [
    "StageStatus",
    "StageEntry",
    "StageLedger",
    "STAGE_NAMES",
]

StageStatus = Literal["missing", "partial", "complete", "failed", "stale"]

#: Canonical stage order. The pipeline runs them in this sequence and the status
#: command reports them in it.
STAGE_NAMES: tuple[str, ...] = (
    "transcribe",
    "acoustic",
    "audio_embedding",
    "text_embedding",
    "assemble",
)


@dataclass(frozen=True)
class StageEntry:
    """One (stage, episode) outcome."""

    episode_id: str
    status: StageStatus
    identity_digest: str | None = None
    failure_reason: str | None = None
    #: Free-form stage-specific counters (segments written, candidates pooled).
    #: Reported in statistics; never used to decide cache validity.
    details: dict[str, Any] = field(default_factory=dict)
    package_version: str = PACKAGE_VERSION

    def __post_init__(self) -> None:
        if self.status not in ("missing", "partial", "complete", "failed", "stale"):
            raise ValueError(f"Unknown stage status {self.status!r}")
        if self.status == "complete" and not self.identity_digest:
            raise ValueError(
                f"{self.episode_id}: a 'complete' entry must record the cache "
                "identity it completed under, otherwise it can never be "
                "invalidated"
            )
        if self.status == "failed" and not self.failure_reason:
            raise ValueError(
                f"{self.episode_id}: a 'failed' entry must record why -- an "
                "unexplained failure cannot be triaged or retried deliberately"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "status": self.status,
            "identity_digest": self.identity_digest,
            "failure_reason": self.failure_reason,
            "details": dict(self.details),
            "package_version": self.package_version,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StageEntry":
        return cls(
            episode_id=str(raw["episode_id"]),
            status=str(raw.get("status", "missing")),  # type: ignore[arg-type]
            identity_digest=raw.get("identity_digest"),
            failure_reason=raw.get("failure_reason"),
            details=dict(raw.get("details") or {}),
            package_version=str(raw.get("package_version", "")),
        )


class StageLedger:
    """The recorded state of one stage across every episode."""

    def __init__(self, stage: str, entries: Iterable[StageEntry] = ()):
        if stage not in STAGE_NAMES:
            raise ValueError(
                f"Unknown stage {stage!r}; known stages are {list(STAGE_NAMES)}"
            )
        self.stage = stage
        self._entries: dict[str, StageEntry] = {
            entry.episode_id: entry for entry in entries
        }

    # -- querying ----------------------------------------------------------
    def get(self, episode_id: str) -> StageEntry:
        return self._entries.get(
            episode_id, StageEntry(episode_id=episode_id, status="missing")
        )

    def status_for(self, episode_id: str, expected_digest: str) -> StageStatus:
        """Effective status, accounting for cache invalidation.

        A ``complete`` entry recorded under a different identity is reported as
        ``stale``, not ``complete``: the work was done, but not for the
        configuration being asked about now.
        """
        entry = self.get(episode_id)
        if entry.status == "complete" and entry.identity_digest != expected_digest:
            return "stale"
        return entry.status

    def needs_work(
        self, episode_id: str, expected_digest: str, retry_failed: bool = False
    ) -> bool:
        status = self.status_for(episode_id, expected_digest)
        if status == "failed":
            return retry_failed
        return status != "complete"

    def counts(self) -> dict[str, int]:
        tally = {status: 0 for status in ("complete", "failed", "partial", "stale")}
        for entry in self._entries.values():
            tally[entry.status] = tally.get(entry.status, 0) + 1
        return tally

    def entries(self) -> list[StageEntry]:
        return [self._entries[key] for key in sorted(self._entries)]

    # -- mutation ----------------------------------------------------------
    def record(self, entry: StageEntry) -> None:
        self._entries[entry.episode_id] = entry

    def mark_partial(self, episode_id: str) -> None:
        self.record(StageEntry(episode_id=episode_id, status="partial"))

    def mark_complete(
        self, episode_id: str, identity_digest: str, **details: Any
    ) -> None:
        self.record(
            StageEntry(
                episode_id=episode_id,
                status="complete",
                identity_digest=identity_digest,
                details=details,
            )
        )

    def mark_failed(self, episode_id: str, reason: str, **details: Any) -> None:
        self.record(
            StageEntry(
                episode_id=episode_id,
                status="failed",
                failure_reason=reason,
                details=details,
            )
        )

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "package_version": PACKAGE_VERSION,
            "entries": [entry.to_dict() for entry in self.entries()],
        }

    def write(self, path: Path) -> None:
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        atomic_write_bytes(Path(path), payload.encode("utf-8"))

    @classmethod
    def read(cls, path: Path, stage: str) -> "StageLedger":
        """Load a ledger, tolerating absence but not corruption.

        A missing file is an empty ledger (nothing has run yet). A *malformed*
        file raises: silently discarding a ledger we failed to parse would
        recompute a whole corpus and look like a performance mystery.
        """
        file_path = Path(path)
        if not file_path.is_file():
            return cls(stage)
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{file_path} is not valid JSON: {error}. Delete it to rescan "
                "from the artifacts on disk."
            ) from error
        if not isinstance(raw, Mapping):
            raise ValueError(f"{file_path} must contain a JSON object")
        recorded_stage = str(raw.get("stage", ""))
        if recorded_stage != stage:
            raise ValueError(
                f"{file_path} is a ledger for stage {recorded_stage!r}, not {stage!r}"
            )
        return cls(
            stage,
            [StageEntry.from_mapping(item) for item in raw.get("entries", [])],
        )
