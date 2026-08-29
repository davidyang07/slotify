"""One command that takes a corpus plan all the way to "ready to label".

Everything between a committed source registry and an annotator's first
keystroke is mechanical: fetch, probe, normalize, generate candidates, split,
transcribe, extract features, embed, assemble, validate, count, build the
queue, cut the clips. This module runs that sequence in order, in one process,
and reports what each step did.

Three properties it is built for:

**Resumability.** Every underlying stage is already idempotent and cache-aware,
so re-running after an interruption costs only the work that was actually lost.
Nothing here adds state of its own beyond the report it writes.

**Honesty about readiness.** The last step is not "done", it is a *measurement*:
how many candidates exist, how many carry a complete multimodal feature record,
how many the queue selected, and whether that is enough to reach the labelling
target. A shortfall is reported as a shortfall with the number, never rounded up
to success.

**No hidden preprocessing.** After this command finishes, the only remaining
work is human judgement. If a clip has not been cut or a transcript is missing,
that is a failure of this command, not something for the annotator to discover
one item at a time.

The steps are the existing CLI command functions, called with the same argument
namespaces the parser would build. They are not reimplemented here: a second
implementation of "normalize the corpus" is a second thing to keep correct.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from slotify_rank.config.versions import PACKAGE_VERSION
from slotify_rank.data.checksum import atomic_write_bytes

__all__ = [
    "PreparationStep",
    "StepResult",
    "PreparationReport",
    "ReadinessSummary",
    "build_steps",
    "run_preparation",
    "summarise_readiness",
]


@dataclass(frozen=True)
class PreparationStep:
    """One named stage, and the argument namespace it runs under."""

    name: str
    description: str
    run: Callable[[argparse.Namespace], int]
    args: argparse.Namespace
    #: A step that may fail without stopping the run, and why that is acceptable.
    optional_reason: str | None = None


@dataclass
class StepResult:
    name: str
    description: str
    exit_code: int
    seconds: float
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipped or self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "exit_code": self.exit_code,
            "seconds": round(self.seconds, 2),
            "skipped": self.skipped,
            "error": self.error,
        }


@dataclass
class ReadinessSummary:
    """What the corpus can support, measured rather than assumed."""

    target_labels: int
    generated_candidate_count: int
    labellable_candidate_count: int
    complete_multimodal_count: int
    queued_unique_count: int
    episode_count: int
    series_count: int
    processed_audio_hours: float
    episodes_by_split: Mapping[str, int]
    series_by_split: Mapping[str, list[str]]
    blocking_reasons: tuple[str, ...] = ()

    @property
    def ready_to_label(self) -> bool:
        return not self.blocking_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_to_label": self.ready_to_label,
            "target_labels": self.target_labels,
            "generated_candidate_count": self.generated_candidate_count,
            "labellable_candidate_count": self.labellable_candidate_count,
            "complete_multimodal_count": self.complete_multimodal_count,
            "queued_unique_count": self.queued_unique_count,
            "episode_count": self.episode_count,
            "series_count": self.series_count,
            "processed_audio_hours": self.processed_audio_hours,
            "episodes_by_split": dict(self.episodes_by_split),
            "series_by_split": {k: list(v) for k, v in self.series_by_split.items()},
            "blocking_reasons": list(self.blocking_reasons),
        }


@dataclass
class PreparationReport:
    corpus_version: str
    steps: list[StepResult] = field(default_factory=list)
    readiness: ReadinessSummary | None = None
    started_at: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_version": PACKAGE_VERSION,
            "corpus_version": self.corpus_version,
            "started_at": self.started_at,
            "seconds": round(self.seconds, 2),
            "all_steps_ok": self.ok,
            "steps": [step.to_dict() for step in self.steps],
            "readiness": self.readiness.to_dict() if self.readiness else None,
        }

    def write(self, path: Path) -> None:
        atomic_write_bytes(
            Path(path),
            (json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n").encode(
                "utf-8"
            ),
        )


def _namespace(**values: Any) -> argparse.Namespace:
    return argparse.Namespace(**values)


def build_steps(
    *,
    data_root: str | None,
    source_registries: Sequence[str],
    dataset_config: str | None,
    split_config: str | None,
    split_version: str,
    queue_config: str | None,
    queue_output: str | None,
    fixtures_registry: str | None,
    fetch_timeout: float,
    skip_fetch: bool,
    skip_features: bool,
    force_split: bool,
    limit: int | None,
) -> list[PreparationStep]:
    """The ordered stages, each bound to the namespace it runs under."""
    from slotify_rank import dataset_cli, features_cli

    common = {"data_root": data_root}
    steps: list[PreparationStep] = []

    if fixtures_registry:
        steps.append(
            PreparationStep(
                name="import-local",
                description=f"Register local and fixture sources from {fixtures_registry}",
                run=dataset_cli._cmd_import_local,
                args=_namespace(
                    **common, sources=fixtures_registry, source_id=None
                ),
            )
        )

    if not skip_fetch:
        for registry in source_registries:
            steps.append(
                PreparationStep(
                    name=f"fetch:{Path(registry).stem}",
                    description=f"Download the audio declared in {registry}",
                    run=dataset_cli._cmd_fetch,
                    args=_namespace(
                        **common,
                        sources=registry,
                        source_id=None,
                        timeout=fetch_timeout,
                    ),
                )
            )

    steps += [
        PreparationStep(
            name="probe",
            description="Measure duration, sample rate, channels and format",
            run=dataset_cli._cmd_probe,
            args=_namespace(**common, episode_id=None, force=False),
        ),
        PreparationStep(
            name="normalize",
            description="Render every episode to 16 kHz mono PCM WAV",
            run=dataset_cli._cmd_normalize,
            args=_namespace(**common, episode_id=None, force=False),
        ),
        PreparationStep(
            name="candidates",
            description="Generate the deterministic candidate pool",
            run=dataset_cli._cmd_generate,
            args=_namespace(
                **common,
                config=dataset_config,
                episode_id=None,
                include_product_padding=False,
            ),
        ),
        PreparationStep(
            name="split",
            description=f"Assign grouped train/validation/test splits ({split_version})",
            run=dataset_cli._cmd_split,
            args=_namespace(
                **common, config=split_config, seed=None, force=force_split
            ),
        ),
    ]

    if not skip_features:
        steps.append(
            PreparationStep(
                name="features",
                description=(
                    "Transcribe, extract acoustic features, embed audio and text, "
                    "assemble and validate"
                ),
                run=features_cli._cmd_pipeline_features,
                args=_namespace(
                    **common,
                    episode_id=None,
                    content_type=None,
                    split=None,
                    split_version=split_version,
                    limit=limit,
                    transcription_config=None,
                    features_config=None,
                    embeddings_config=None,
                    pipeline_config=None,
                    force=False,
                    retry_failed=True,
                    stage=None,
                    deep=False,
                ),
            )
        )

    steps += [
        PreparationStep(
            name="validate",
            description="Check dataset integrity",
            run=dataset_cli._cmd_validate,
            args=_namespace(
                **common, deep=False, split_version=split_version, output=None
            ),
        ),
        PreparationStep(
            name="stats",
            description="Recompute dataset, candidate, label and split statistics",
            run=dataset_cli._cmd_stats,
            args=_namespace(
                **common,
                split_version=split_version,
                output_dir=None,
                acceptable_threshold=3,
            ),
        ),
        PreparationStep(
            name="queue",
            description="Build the stratified labelling queue",
            run=dataset_cli._cmd_label_queue,
            args=_namespace(
                **common,
                config=queue_config,
                split_version=split_version,
                output=queue_output,
                force=True,
            ),
        ),
    ]
    return steps


def run_preparation(
    steps: Sequence[PreparationStep],
    corpus_version: str,
    log: Callable[[str], None] = print,
    stop_on_failure: bool = True,
) -> PreparationReport:
    """Run every step in order, timing each and recording its exit code."""
    from datetime import datetime, timezone

    report = PreparationReport(
        corpus_version=corpus_version,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    run_started = time.perf_counter()
    for index, step in enumerate(steps, start=1):
        log("")
        log(f"[{index}/{len(steps)}] {step.name} -- {step.description}")
        log("-" * 72)
        started = time.perf_counter()
        error: str | None = None
        try:
            code = int(step.run(step.args))
        except Exception as exception:  # noqa: BLE001 - reported, not swallowed
            code = 1
            error = f"{type(exception).__name__}: {exception}"
            log(f"  step raised: {error}")
        elapsed = time.perf_counter() - started
        result = StepResult(
            name=step.name,
            description=step.description,
            exit_code=code,
            seconds=elapsed,
            error=error,
        )
        report.steps.append(result)
        log(f"  -> exit {code} in {elapsed:.1f}s")
        if code != 0 and stop_on_failure and step.optional_reason is None:
            log(f"  stopping: {step.name} failed and later steps depend on it")
            break
    report.seconds = time.perf_counter() - run_started
    return report


def summarise_readiness(
    paths: Any,
    queue_path: Path,
    target_labels: int,
    split_version: str,
) -> ReadinessSummary:
    """Measure what the prepared corpus can actually support.

    Reads the manifests rather than the step log: a step that exited zero is not
    evidence that the corpus is big enough, and this is the number the operator
    needs before spending hours labelling.
    """
    from slotify_rank.data import manifests
    from slotify_rank.data.schema import TARGET_DOMAIN_CONTENT_TYPES
    from slotify_rank.features.assemble import read_feature_manifest
    from slotify_rank.labelling.queue import read_queue

    episodes = manifests.read_episodes(paths.episodes_manifest)
    candidates = manifests.read_candidates(paths.candidates_manifest)
    episode_by_id = {episode.episode_id: episode for episode in episodes}

    labellable = [
        candidate
        for candidate in candidates
        if candidate.eligible_for_labelling
        and not candidate.is_synthetic
        and episode_by_id.get(candidate.episode_id) is not None
        and episode_by_id[candidate.episode_id].content_type
        in TARGET_DOMAIN_CONTENT_TYPES
    ]

    complete = 0
    if paths.features_manifest.is_file():
        _, records = read_feature_manifest(paths.features_manifest)
        complete_ids = {
            record.candidate_id
            for record in records
            if record.feature_status == "complete"
        }
        complete = len({c.candidate_id for c in labellable} & complete_ids)

    queued = 0
    if Path(queue_path).is_file():
        queued = len(read_queue(Path(queue_path)).unique_candidate_ids)

    # One pass: an episode's split is whichever split its candidates carry, and
    # a candidate's split is stamped onto it by `dataset split`.
    episodes_in_split: dict[str, set[str]] = {}
    series_by_split: dict[str, set[str]] = {}
    for candidate in labellable:
        episode = episode_by_id[candidate.episode_id]
        series_by_split.setdefault(candidate.dataset_split, set()).add(
            episode.series_id
        )
        episodes_in_split.setdefault(candidate.dataset_split, set()).add(
            candidate.episode_id
        )
    episodes_by_split = {
        split: len(members) for split, members in sorted(episodes_in_split.items())
    }

    hours = sum(
        (episode.normalized_duration_ms or episode.duration_ms or 0)
        for episode in episodes
        if episode.content_type in TARGET_DOMAIN_CONTENT_TYPES
    ) / 3_600_000.0

    reasons: list[str] = []
    if queued < target_labels:
        reasons.append(
            f"the labelling queue holds {queued} unique candidate(s), fewer than the "
            f"{target_labels} the experiment targets. Raise queue.target_unique, or "
            "add episodes to the corpus plan."
        )
    if complete < target_labels:
        reasons.append(
            f"{complete} labellable candidate(s) carry a complete multimodal feature "
            f"record, fewer than the {target_labels} targeted. Run the feature "
            "pipeline to completion before labelling."
        )
    for split in ("train", "validation", "test"):
        if not series_by_split.get(split):
            reasons.append(
                f"no labellable candidate is assigned to the {split!r} split under "
                f"split manifest {split_version}; a held-out result needs all three."
            )

    return ReadinessSummary(
        target_labels=target_labels,
        generated_candidate_count=len(candidates),
        labellable_candidate_count=len(labellable),
        complete_multimodal_count=complete,
        queued_unique_count=queued,
        episode_count=len(episodes),
        series_count=len({episode.series_id for episode in episodes}),
        processed_audio_hours=round(hours, 4),
        episodes_by_split=episodes_by_split,
        series_by_split={
            split: sorted(names) for split, names in sorted(series_by_split.items())
        },
        blocking_reasons=tuple(reasons),
    )
