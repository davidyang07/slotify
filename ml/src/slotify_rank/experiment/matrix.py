"""The experiment's training matrix: five variants, several seeds, one choice.

Running the canonical experiment means training every declared ablation at every
declared seed and then picking *one* run to quote. This module does the picking,
and the rule is the whole reason it exists.

**The reported seed is the median by validation NDCG@3, not the best.** Training
the same configuration under three seeds and reporting the best of them is seed
cherry-picking with extra steps: it reports the upper tail of a distribution as
if it were the centre, and it does so entirely within the letter of "we used a
held-out test set". The median is the honest summary of a seed distribution, and
recording every run's score next to it lets a reader see the spread they would
otherwise have to take on trust.

**Selection reads validation, never test.** That is what a validation split is
for, and it is the last decision made before the test split is read at all.

**The ablations are the point, not decoration.** `handcrafted` against `gated`
answers whether the learned modalities earned their inference cost; `audio_only`
and `text_only` say which one carries the signal; `concat` against `gated` says
whether the gating mechanism does anything a plain concatenation does not. A
"multimodal" claim with no ablation is an architecture description, not a
finding.

Nothing here trains: :func:`plan_runs` says what to run, and
:func:`summarise_matrix` reads back what the runs recorded. The training itself
goes through the existing `training run` path, unchanged.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import PACKAGE_VERSION
from slotify_rank.data.checksum import atomic_write_bytes

__all__ = [
    "PlannedRun",
    "RunOutcome",
    "MatrixSummary",
    "plan_runs",
    "read_outcome",
    "select_reported_seed",
    "summarise_matrix",
    "write_matrix_summary",
]


@dataclass(frozen=True)
class PlannedRun:
    """One (variant, seed) cell of the matrix."""

    variant: str
    seed: int
    run_dir: Path

    @property
    def name(self) -> str:
        return f"{self.variant}-seed{self.seed}"


def plan_runs(
    variants: Sequence[str], seeds: Sequence[int], output_root: Path
) -> list[PlannedRun]:
    """Every cell, in a deterministic order.

    Variants in the order the experiment declares them and seeds ascending, so a
    partially completed matrix resumes at the same place and two operators
    produce the same directory names.
    """
    root = Path(output_root)
    return [
        PlannedRun(variant, seed, root / f"{variant}-seed{seed}")
        for variant in variants
        for seed in sorted(int(s) for s in seeds)
    ]


@dataclass
class RunOutcome:
    """What one cell produced, read back from its own summary."""

    variant: str
    seed: int
    run_dir: str
    exit_code: int
    label_source: str | None = None
    validation_ndcg_at_3: float | None = None
    validation_pairwise_accuracy: float | None = None
    parameter_count: int | None = None
    run_id: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.validation_ndcg_at_3 is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "seed": self.seed,
            "run_dir": self.run_dir,
            "exit_code": self.exit_code,
            "label_source": self.label_source,
            "validation_ndcg_at_3": self.validation_ndcg_at_3,
            "validation_pairwise_accuracy": self.validation_pairwise_accuracy,
            "model_parameter_count": self.parameter_count,
            "run_id": self.run_id,
            "error": self.error,
        }


def read_outcome(run: PlannedRun, exit_code: int) -> RunOutcome:
    """Read one run's own summary. A missing field stays ``None``."""
    summary_path = run.run_dir / "training_summary.json"
    outcome = RunOutcome(
        variant=run.variant,
        seed=run.seed,
        run_dir=str(run.run_dir),
        exit_code=exit_code,
    )
    if not summary_path.is_file():
        outcome.error = "the run wrote no training_summary.json"
        return outcome
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        outcome.error = f"training_summary.json unreadable: {error}"
        return outcome
    outcome.label_source = summary.get("label_source")
    outcome.run_id = summary.get("run_id")
    outcome.parameter_count = summary.get("model_parameter_count")
    ndcg = summary.get("best_validation_ndcg_at_3")
    outcome.validation_ndcg_at_3 = None if ndcg is None else float(ndcg)
    accuracy = summary.get("best_validation_pairwise_accuracy")
    outcome.validation_pairwise_accuracy = (
        None if accuracy is None else float(accuracy)
    )
    return outcome


def select_reported_seed(outcomes: Sequence[RunOutcome]) -> RunOutcome | None:
    """The median run by validation NDCG@3, breaking ties on the lower seed.

    With an even number of runs there is no single median element, so the lower
    of the two central runs is taken -- the conservative direction, and a fixed
    rule rather than a choice made after seeing the numbers.
    """
    usable = [outcome for outcome in outcomes if outcome.ok]
    if not usable:
        return None
    ordered = sorted(
        usable, key=lambda outcome: (outcome.validation_ndcg_at_3, outcome.seed)
    )
    return ordered[(len(ordered) - 1) // 2]


@dataclass
class MatrixSummary:
    experiment_version: str
    headline_variant: str
    seeds: tuple[int, ...]
    outcomes: list[RunOutcome]
    reported: RunOutcome | None
    generated_at: str = ""
    blocking_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.reported is not None and not self.blocking_reasons

    def by_variant(self) -> dict[str, dict[str, Any]]:
        """Per-variant validation spread -- the ablation comparison."""
        grouped: dict[str, list[RunOutcome]] = {}
        for outcome in self.outcomes:
            if outcome.ok:
                grouped.setdefault(outcome.variant, []).append(outcome)
        summary: dict[str, dict[str, Any]] = {}
        for variant, runs in sorted(grouped.items()):
            scores = [run.validation_ndcg_at_3 for run in runs]
            summary[variant] = {
                "seed_count": len(scores),
                "validation_ndcg_at_3_median": statistics.median(scores),
                "validation_ndcg_at_3_min": min(scores),
                "validation_ndcg_at_3_max": max(scores),
                "model_parameter_count": runs[0].parameter_count,
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_version": PACKAGE_VERSION,
            "experiment_version": self.experiment_version,
            "generated_at": self.generated_at,
            "headline_variant": self.headline_variant,
            "seeds": list(self.seeds),
            "seed_selection": "median_validation_ndcg_at_3",
            "seed_selection_note": (
                "The reported run is the MEDIAN seed by validation NDCG@3, not the "
                "best. Reporting the best of several seeds reports the upper tail "
                "of a distribution as if it were its centre."
            ),
            "ok": self.ok,
            "blocking_reasons": list(self.blocking_reasons),
            "reported_run": self.reported.to_dict() if self.reported else None,
            "by_variant": self.by_variant(),
            "runs": [outcome.to_dict() for outcome in self.outcomes],
        }


def summarise_matrix(
    experiment_version: str,
    headline_variant: str,
    seeds: Sequence[int],
    outcomes: Sequence[RunOutcome],
    allowed_label_sources: Sequence[str] = ("human",),
) -> MatrixSummary:
    """Assemble the matrix report and decide which run is quotable."""
    from datetime import datetime, timezone

    blocking: list[str] = []
    failed = [outcome for outcome in outcomes if outcome.exit_code != 0]
    if failed:
        blocking.append(
            f"{len(failed)} run(s) failed: "
            f"{[f'{o.variant}-seed{o.seed}' for o in failed][:5]}"
        )

    wrong_source = sorted(
        {
            str(outcome.label_source)
            for outcome in outcomes
            if outcome.label_source is not None
            and outcome.label_source not in allowed_label_sources
        }
    )
    if wrong_source:
        blocking.append(
            f"run(s) trained on label_source={wrong_source}, but this experiment "
            f"allows only {list(allowed_label_sources)}"
        )

    headline_runs = [
        outcome for outcome in outcomes if outcome.variant == headline_variant
    ]
    if not headline_runs:
        blocking.append(
            f"no run of the headline variant {headline_variant!r} is present"
        )

    reported = select_reported_seed(headline_runs)
    if reported is None and not blocking:
        blocking.append(
            f"no usable run of {headline_variant!r} recorded a validation NDCG@3"
        )

    return MatrixSummary(
        experiment_version=experiment_version,
        headline_variant=headline_variant,
        seeds=tuple(int(seed) for seed in seeds),
        outcomes=list(outcomes),
        reported=reported,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        blocking_reasons=tuple(blocking),
    )


def write_matrix_summary(path: Path, summary: MatrixSummary) -> None:
    atomic_write_bytes(
        Path(path),
        (json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )
