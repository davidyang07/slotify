"""Held-out comparison of the learned ranker against `heuristic_offline_v1`.

This module exists to produce exactly one number honestly:

    relative_improvement_percent
        = 100 * (model_ndcg_at_3 - baseline_ndcg_at_3) / baseline_ndcg_at_3

and to refuse to produce it when the conditions that would make it meaningful
are not met. The refusals are the point. Any pipeline can divide two floats.

Four gates, each of which independently marks the headline **not publishable**:

**Ground truth must be human.** A comparison against `heuristic_offline_v1`
scored with weak, heuristic-derived labels is circular: the baseline is the
teacher, so it scores near-perfectly by construction and the model can only
lose. `publishable` is false for any non-human label source, and the reason is
written into the artifact.

**The evaluation split must be held out.** The training run's episode set is
read back from its own `dataset_summary.json` and intersected with the
evaluation episodes. Any overlap is leakage, and leakage is fatal rather than
noted.

**The split must be a real split.** A degraded split manifest that assigns
everything to `development` cannot support a held-out claim, so `test` must
actually contain episodes that no other split does.

**Metrics must be defined.** An episode with no relevant candidate has an
undefined NDCG. Those episodes are excluded and counted, never scored as zero
(which would deflate) or as one (which would inflate).

**The inputs must be real.** The synthetic fixture corpus carries
``label_source: human`` on purpose, so it exercises the whole training and
evaluation path. That makes it exactly the thing that could produce a
publishable-looking result from generated numbers, so a synthetic label export
or feature manifest blocks the headline outright.

**Uncertainty is reported, not implied.** NDCG@3 is macro-averaged over
episodes, and a test split has a few dozen of them, so the point estimate has
real spread. A percentile bootstrap over episodes -- resampling the unit the
average is taken over -- gives an interval for the baseline, for the model and
for the relative improvement itself. The improvement's interval is the one that
matters: a headline of "+21%" whose interval spans zero is not a result.

**A second opinion on the metric.** ``sklearn.metrics.ndcg_score`` recomputes
the per-episode NDCG independently (:mod:`slotify_rank.evaluation.crosscheck`).
The headline is a ratio of two NDCG values from the same implementation, so a
bug in it would move numerator and denominator together and stay invisible to
any test that checked that implementation against itself.

**A second system to size the result against.** A scikit-learn gradient-boosted
model on the handcrafted scalars alone
(:mod:`slotify_rank.baselines.classical`) is scored on the same candidates and
reported alongside. It is not the denominator; it answers "would a good tabular
model have done just as well?"

Every number in the output is computed here. Nothing is copied from a previous
run, and the summary markdown is generated from the same dict the JSON is.
"""

from __future__ import annotations

import json
import math
import platform
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import PACKAGE_VERSION
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.datasets.schema import TrainingExample
from slotify_rank.evaluation.metrics import (
    EpisodeJudgements,
    EpisodePrediction,
    EvaluationReport,
    MetricConfig,
    evaluate_rankings,
)

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "CANONICAL_BASELINE",
    "BootstrapConfig",
    "ComparisonInputs",
    "ComparisonResult",
    "CohortCounts",
    "relative_improvement_percent",
    "bootstrap_interval",
    "compare",
    "write_comparison",
    "render_summary",
]

EVALUATION_SCHEMA_VERSION = "evaluation-v1.0.0"

#: The only baseline a headline improvement may be quoted against. Named here so
#: a different denominator cannot be substituted without editing this constant.
CANONICAL_BASELINE = "heuristic_offline_v1"

#: Cutoffs reported. 3 is the headline; the others exist so a reader can see
#: whether an improvement at 3 is an artifact of the cutoff.
DEFAULT_CUTOFFS: tuple[int, ...] = (1, 3, 5)


def relative_improvement_percent(
    model: float | None, baseline: float | None
) -> float | None:
    """The headline formula, and the only place it is written.

    ``None`` when either side is undefined or the baseline is zero -- a
    percentage improvement over zero is not a large number, it is undefined, and
    reporting it as one would be the single most misleading thing this file
    could do.
    """
    if model is None or baseline is None:
        return None
    if not math.isfinite(model) or not math.isfinite(baseline):
        return None
    if baseline == 0.0:
        return None
    return 100.0 * (model - baseline) / baseline


@dataclass(frozen=True)
class BootstrapConfig:
    """Percentile bootstrap over episodes.

    The episode is the resampling unit because the metric is macro-averaged over
    episodes; resampling candidates instead would treat one episode's fifty
    candidates as fifty independent observations and produce an interval several
    times too narrow.
    """

    resamples: int = 2000
    confidence: float = 0.95
    seed: int = 20260829

    def __post_init__(self) -> None:
        if self.resamples < 100:
            raise ValueError("a bootstrap with fewer than 100 resamples is noise")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0.5, 1.0)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "resamples": self.resamples,
            "confidence": self.confidence,
            "seed": self.seed,
            "unit": "episode",
            "method": "percentile",
        }


def bootstrap_interval(
    per_episode_baseline: Mapping[str, float],
    per_episode_model: Mapping[str, float],
    config: BootstrapConfig,
) -> dict[str, Any]:
    """Bootstrap the two macro-averages and their relative improvement.

    Both systems are resampled on the *same* episode draw, which is what makes
    the improvement's interval meaningful: the two scores are paired
    observations of one episode, and resampling them independently would throw
    that pairing away and widen the interval for no reason.

    ``None`` everywhere when there are fewer than two scored episodes -- an
    interval from one observation is not an interval.
    """
    episodes = sorted(set(per_episode_baseline) & set(per_episode_model))
    if len(episodes) < 2:
        return {
            "measured": False,
            "reason": (
                f"{len(episodes)} episode(s) have both scores; a bootstrap needs at "
                "least two"
            ),
            "config": config.to_dict(),
        }

    generator = random.Random(config.seed)
    baseline_draws: list[float] = []
    model_draws: list[float] = []
    improvement_draws: list[float] = []
    count = len(episodes)
    for _ in range(config.resamples):
        sample = [episodes[generator.randrange(count)] for _ in range(count)]
        baseline_mean = sum(per_episode_baseline[e] for e in sample) / count
        model_mean = sum(per_episode_model[e] for e in sample) / count
        baseline_draws.append(baseline_mean)
        model_draws.append(model_mean)
        improvement = relative_improvement_percent(model_mean, baseline_mean)
        if improvement is not None:
            improvement_draws.append(improvement)

    def percentiles(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        tail = (1.0 - config.confidence) / 2.0
        low = ordered[min(len(ordered) - 1, int(tail * len(ordered)))]
        high = ordered[min(len(ordered) - 1, int((1.0 - tail) * len(ordered)))]
        return {"low": low, "high": high}

    return {
        "measured": True,
        "config": config.to_dict(),
        "episode_count": count,
        "baseline_ndcg_at_3": percentiles(baseline_draws),
        "model_ndcg_at_3": percentiles(model_draws),
        "relative_improvement_percent": percentiles(improvement_draws),
        "improvement_draws_defined": len(improvement_draws),
    }


@dataclass(frozen=True)
class CohortCounts:
    """How much data the number rests on, per split and per grouping level."""

    labelled_candidate_count: int = 0
    candidates_by_split: Mapping[str, int] = field(default_factory=dict)
    episodes_by_split: Mapping[str, int] = field(default_factory=dict)
    series_by_split: Mapping[str, int] = field(default_factory=dict)
    evaluated_series: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "labelled_candidate_count": self.labelled_candidate_count,
            "candidates_by_split": dict(self.candidates_by_split),
            "episodes_by_split": dict(self.episodes_by_split),
            "series_by_split": dict(self.series_by_split),
            "evaluated_series": list(self.evaluated_series),
        }


@dataclass(frozen=True)
class ComparisonInputs:
    """Everything the comparison consumed, recorded for reproducibility."""

    split: str
    label_source: str
    label_export: str
    features_manifest: str
    split_manifest: str
    split_version: str
    model_run_id: str
    model_variant: str
    model_checkpoint: str
    model_training_label_source: str
    baseline_version: str
    baseline_config_version: str
    git_sha: str
    dataset_version: str
    seeds: tuple[int, ...] = ()
    #: The canonical experiment this run belongs to, and the digest of its
    #: committed definition. Empty for an ad-hoc comparison.
    experiment_version: str = ""
    experiment_config_digest: str = ""
    #: SHA-256 of the artifacts the number rests on, so "which data, which
    #: weights" is answerable from the report alone.
    artifact_hashes: Mapping[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "label_source": self.label_source,
            "label_export": self.label_export,
            "features_manifest": self.features_manifest,
            "split_manifest": self.split_manifest,
            "split_version": self.split_version,
            "model_run_id": self.model_run_id,
            "model_variant": self.model_variant,
            "model_checkpoint": self.model_checkpoint,
            "model_training_label_source": self.model_training_label_source,
            "baseline_version": self.baseline_version,
            "baseline_config_version": self.baseline_config_version,
            "git_sha": self.git_sha,
            "dataset_version": self.dataset_version,
            "seeds": list(self.seeds),
            "experiment_version": self.experiment_version,
            "experiment_config_digest": self.experiment_config_digest,
            "artifact_hashes": dict(self.artifact_hashes),
        }


@dataclass
class ComparisonResult:
    """The comparison, its inputs, and every reason it may not be quoted."""

    evaluation_id: str
    inputs: ComparisonInputs
    baseline_reports: dict[int, EvaluationReport]
    model_reports: dict[int, EvaluationReport]
    episode_count: int
    candidate_count: int
    relevant_candidate_count: int
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ranked_rows: list[dict[str, Any]] = field(default_factory=list)
    cohort: CohortCounts = field(default_factory=CohortCounts)
    bootstrap: Mapping[str, Any] = field(default_factory=dict)
    metric_crosscheck: Mapping[str, Any] = field(default_factory=dict)
    classical_baseline: Mapping[str, Any] = field(default_factory=dict)

    @property
    def publishable(self) -> bool:
        """Whether the headline number may be published."""
        return not self.blocking_reasons

    def headline(self) -> dict[str, Any]:
        baseline = self.baseline_reports[3].aggregate["ndcg_at_k"]
        model = self.model_reports[3].aggregate["ndcg_at_k"]
        return {
            "metric": "ndcg_at_3",
            "baseline": {"name": self.inputs.baseline_version, "ndcg_at_3": baseline},
            "model": {
                "name": self.inputs.model_variant,
                "run_id": self.inputs.model_run_id,
                "ndcg_at_3": model,
            },
            "relative_improvement_percent": relative_improvement_percent(
                model, baseline
            ),
            "absolute_improvement": (
                None if model is None or baseline is None else model - baseline
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "package_version": PACKAGE_VERSION,
            "evaluation_id": self.evaluation_id,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "inputs": self.inputs.to_dict(),
            "episode_count": self.episode_count,
            "candidate_count": self.candidate_count,
            "relevant_candidate_count": self.relevant_candidate_count,
            "cohort": self.cohort.to_dict(),
            "bootstrap": dict(self.bootstrap),
            "metric_crosscheck": dict(self.metric_crosscheck),
            "classical_baseline": dict(self.classical_baseline),
            "headline_publishable": self.publishable,
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
            "headline": self.headline(),
            "metrics": {
                "baseline": {
                    f"at_{k}": report.aggregate | report.coverage
                    for k, report in sorted(self.baseline_reports.items())
                },
                "model": {
                    f"at_{k}": report.aggregate | report.coverage
                    for k, report in sorted(self.model_reports.items())
                },
            },
        }


def _group_by_episode(
    examples: Sequence[TrainingExample],
) -> dict[str, list[TrainingExample]]:
    grouped: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        grouped[example.episode_id].append(example)
    return dict(grouped)


def build_predictions(
    grouped: Mapping[str, Sequence[TrainingExample]],
    scores: Mapping[str, float],
) -> list[EpisodePrediction]:
    """One ranked list per episode, best first, ties broken on candidate id."""
    predictions: list[EpisodePrediction] = []
    for episode_id in sorted(grouped):
        examples = grouped[episode_id]
        ordered = sorted(
            examples,
            key=lambda example: (-scores[example.candidate_id], example.candidate_id),
        )
        predictions.append(
            EpisodePrediction(
                episode_id=episode_id,
                ranked_candidate_ids=[example.candidate_id for example in ordered],
                scores={
                    example.candidate_id: scores[example.candidate_id]
                    for example in examples
                },
            )
        )
    return predictions


def build_judgements(
    grouped: Mapping[str, Sequence[TrainingExample]],
) -> list[EpisodeJudgements]:
    return [
        EpisodeJudgements(
            episode_id=episode_id,
            relevance={
                example.candidate_id: float(example.quality_score)
                for example in grouped[episode_id]
            },
        )
        for episode_id in sorted(grouped)
    ]


def compare(
    examples: Sequence[TrainingExample],
    baseline_scores: Mapping[str, float],
    model_scores: Mapping[str, float],
    inputs: ComparisonInputs,
    evaluation_id: str,
    training_episode_ids: Sequence[str] = (),
    relevance_threshold: float = 4.0,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    bootstrap: BootstrapConfig | None = None,
    series_by_episode: Mapping[str, str] | None = None,
    cohort_examples: Mapping[str, Sequence[TrainingExample]] | None = None,
    classical_baseline: Mapping[str, Any] | None = None,
    require_metric_crosscheck: bool = True,
    synthetic_inputs: Sequence[str] = (),
) -> ComparisonResult:
    """Score both systems on the same candidates with the same labels."""
    if not examples:
        raise ValueError(
            "No labelled candidate in the evaluation split, so there is nothing "
            "to compare. This is a blocked evaluation, not a zero."
        )

    missing_baseline = [e.candidate_id for e in examples if e.candidate_id not in baseline_scores]
    missing_model = [e.candidate_id for e in examples if e.candidate_id not in model_scores]
    if missing_baseline or missing_model:
        raise ValueError(
            "Both systems must score every evaluated candidate, or the two are "
            f"not being compared on the same set. Missing baseline scores: "
            f"{missing_baseline[:3]}; missing model scores: {missing_model[:3]}."
        )

    grouped = _group_by_episode(examples)
    judgements = build_judgements(grouped)
    baseline_predictions = build_predictions(grouped, baseline_scores)
    model_predictions = build_predictions(grouped, model_scores)

    baseline_reports: dict[int, EvaluationReport] = {}
    model_reports: dict[int, EvaluationReport] = {}
    for k in cutoffs:
        config = MetricConfig(k=k, relevance_threshold=relevance_threshold)
        baseline_reports[k] = evaluate_rankings(baseline_predictions, judgements, config)
        model_reports[k] = evaluate_rankings(model_predictions, judgements, config)

    relevant = sum(
        1 for example in examples if float(example.quality_score) >= relevance_threshold
    )

    blocking: list[str] = []
    warnings: list[str] = []

    if inputs.label_source != "human":
        blocking.append(
            f"ground truth is label_source={inputs.label_source!r}, not human. A "
            f"comparison against {CANONICAL_BASELINE} scored with heuristic-derived "
            "labels is circular: the baseline is the teacher."
        )
    if inputs.model_training_label_source != "human":
        blocking.append(
            "the model was trained with label_source="
            f"{inputs.model_training_label_source!r}, so its ordering reflects its "
            "teacher rather than human judgement."
        )
    if inputs.baseline_version != CANONICAL_BASELINE:
        blocking.append(
            f"the baseline is {inputs.baseline_version!r}, not the canonical "
            f"{CANONICAL_BASELINE}."
        )
    if inputs.split != "test":
        blocking.append(
            f"the evaluation ran on the {inputs.split!r} split; a headline number "
            "requires the held-out test split."
        )
    if synthetic_inputs:
        blocking.append(
            f"{sorted(synthetic_inputs)} declare themselves synthetic. The fixture "
            "corpus carries label_source=human so it can exercise this path; a "
            "number measured on generated data is not a result."
        )

    leaked = sorted(set(training_episode_ids) & set(grouped))
    if leaked:
        blocking.append(
            f"{len(leaked)} evaluation episode(s) were also trained on "
            f"({leaked[:3]}). That is leakage."
        )

    if relevant == 0:
        blocking.append(
            f"no candidate in this split scored >= {relevance_threshold}, so NDCG "
            "is undefined for every episode."
        )

    if len(grouped) < 3:
        warnings.append(
            f"only {len(grouped)} evaluation episode(s); metrics are macro-averaged "
            "over episodes, so this estimate has very wide uncertainty."
        )
    evaluated_series = {
        (series_by_episode or {}).get(episode_id, episode_id) for episode_id in grouped
    }
    if len(evaluated_series) < 3:
        warnings.append(
            f"the test partition holds {len(evaluated_series)} series "
            f"({sorted(evaluated_series)}). Episodes of one show share hosts, room, "
            "mic chain and editing rhythm, so a macro-average over them measures "
            "one show's quirks as much as the model. Read the interval, not the "
            "point estimate."
        )
    if not inputs.seeds:
        warnings.append(
            "a single training run was evaluated; no seed-to-seed variance is "
            "reported."
        )

    skipped = baseline_reports[3].coverage["ndcg_at_k_n_skipped"]
    if skipped:
        warnings.append(
            f"{skipped} episode(s) had an undefined NDCG@3 and were excluded from "
            "the average rather than scored as zero."
        )

    ranked_rows: list[dict[str, Any]] = []
    for prediction in model_predictions:
        for rank, candidate_id in enumerate(prediction.ranked_candidate_ids, start=1):
            ranked_rows.append(
                {
                    "episode_id": prediction.episode_id,
                    "candidate_id": candidate_id,
                    "model_rank": rank,
                    "model_score": model_scores[candidate_id],
                    "baseline_score": baseline_scores[candidate_id],
                    "relevance": next(
                        float(e.quality_score)
                        for e in grouped[prediction.episode_id]
                        if e.candidate_id == candidate_id
                    ),
                }
            )

    # -- uncertainty -------------------------------------------------------
    headline_k = 3 if 3 in cutoffs else sorted(cutoffs)[0]
    per_episode_baseline = {
        entry.episode_id: entry.ndcg_at_k
        for entry in baseline_reports[headline_k].per_episode
        if entry.ndcg_at_k is not None
    }
    per_episode_model = {
        entry.episode_id: entry.ndcg_at_k
        for entry in model_reports[headline_k].per_episode
        if entry.ndcg_at_k is not None
    }
    bootstrap_result = bootstrap_interval(
        per_episode_baseline, per_episode_model, bootstrap or BootstrapConfig()
    )
    interval = (
        bootstrap_result.get("relative_improvement_percent")
        if bootstrap_result.get("measured")
        else None
    )
    if interval is not None and interval["low"] <= 0.0 <= interval["high"]:
        warnings.append(
            f"the {bootstrap_result['config']['confidence']:.0%} bootstrap interval "
            f"for the relative improvement is [{interval['low']:.2f}, "
            f"{interval['high']:.2f}] %, which spans zero: this many episodes cannot "
            "distinguish the model from the baseline."
        )

    # -- an independent implementation of the metric ------------------------
    from slotify_rank.evaluation.crosscheck import cross_check_ndcg

    crosscheck = cross_check_ndcg(
        model_predictions, judgements, k=headline_k
    ).to_dict()
    if not crosscheck["available"]:
        message = (
            "the NDCG implementation was not independently verified: "
            f"{crosscheck['reason']}"
        )
        (blocking if require_metric_crosscheck else warnings).append(message)
    elif crosscheck["disagreements"]:
        blocking.append(
            f"this project's NDCG@{headline_k} disagrees with scikit-learn's on "
            f"{len(crosscheck['disagreements'])} episode(s); the metric itself is "
            "in question, so no number computed from it may be published."
        )

    # -- how much data the number rests on ----------------------------------
    series_lookup = dict(series_by_episode or {})
    cohorts = dict(cohort_examples or {})
    cohorts.setdefault(inputs.split, list(examples))
    cohort = CohortCounts(
        labelled_candidate_count=sum(len(rows) for rows in cohorts.values()),
        candidates_by_split={
            split: len(rows) for split, rows in sorted(cohorts.items())
        },
        episodes_by_split={
            split: len({row.episode_id for row in rows})
            for split, rows in sorted(cohorts.items())
        },
        series_by_split={
            split: len({series_lookup.get(row.episode_id, row.episode_id) for row in rows})
            for split, rows in sorted(cohorts.items())
        },
        evaluated_series=tuple(
            sorted({series_lookup.get(episode_id, episode_id) for episode_id in grouped})
        ),
    )

    classical = dict(classical_baseline or {})
    if classical.get("available") is False:
        warnings.append(
            "the classical scikit-learn baseline did not run: "
            f"{classical.get('reason')}"
        )

    return ComparisonResult(
        evaluation_id=evaluation_id,
        inputs=inputs,
        baseline_reports=baseline_reports,
        model_reports=model_reports,
        episode_count=len(grouped),
        candidate_count=len(examples),
        relevant_candidate_count=relevant,
        blocking_reasons=blocking,
        warnings=warnings,
        ranked_rows=ranked_rows,
        cohort=cohort,
        bootstrap=bootstrap_result,
        metric_crosscheck=crosscheck,
        classical_baseline=classical,
    )


def _format(value: Any) -> str:
    if value is None:
        return "NOT YET AVAILABLE"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _interval(block: Mapping[str, Any] | None, suffix: str = "") -> str:
    if not block:
        return "NOT YET AVAILABLE"
    return f"[{block['low']:.4f}, {block['high']:.4f}]{suffix}"


def _render_uncertainty(payload: Mapping[str, Any]) -> list[str]:
    bootstrap = payload.get("bootstrap") or {}
    lines = ["## Uncertainty", ""]
    if not bootstrap.get("measured"):
        lines += [
            f"Not measured: {bootstrap.get('reason', 'no bootstrap was run')}.",
            "",
        ]
        return lines
    config = bootstrap["config"]
    lines += [
        f"Percentile bootstrap, {config['resamples']} resamples over "
        f"{bootstrap['episode_count']} episode(s), seed {config['seed']}. The "
        "episode is the resampling unit because the metric is macro-averaged over "
        "episodes.",
        "",
        f"- Baseline NDCG@3 {int(config['confidence'] * 100)}% interval: "
        + _interval(bootstrap.get("baseline_ndcg_at_3")),
        f"- Model NDCG@3 {int(config['confidence'] * 100)}% interval: "
        + _interval(bootstrap.get("model_ndcg_at_3")),
        f"- Relative improvement {int(config['confidence'] * 100)}% interval: "
        + _interval(bootstrap.get("relative_improvement_percent"), " %"),
        "",
    ]
    return lines


def _render_cohort(payload: Mapping[str, Any]) -> list[str]:
    cohort = payload.get("cohort") or {}
    if not cohort:
        return []
    lines = [
        "## What the number rests on",
        "",
        "| Split | Labelled candidates | Episodes | Series |",
        "| --- | --- | --- | --- |",
    ]
    splits = sorted(cohort.get("candidates_by_split") or {})
    for split in splits:
        lines.append(
            f"| {split} | {cohort['candidates_by_split'][split]} "
            f"| {cohort['episodes_by_split'].get(split, 0)} "
            f"| {cohort['series_by_split'].get(split, 0)} |"
        )
    lines += [
        "",
        f"Total human-labelled candidates across the splits above: "
        f"{cohort.get('labelled_candidate_count')}.",
        "",
    ]
    return lines


def _render_crosscheck(payload: Mapping[str, Any]) -> list[str]:
    crosscheck = payload.get("metric_crosscheck") or {}
    lines = ["## Independent metric verification", ""]
    if not crosscheck.get("available"):
        lines += [f"Did not run: {crosscheck.get('reason', 'unavailable')}.", ""]
        return lines
    lines += [
        f"`sklearn.metrics.ndcg_score` ({crosscheck.get('sklearn_version')}) "
        f"recomputed NDCG@{crosscheck.get('k')} on "
        f"{crosscheck.get('compared_episode_count')} episode(s); "
        f"{crosscheck.get('skipped_episode_count')} were skipped as undefined for "
        "one or both implementations.",
        "",
        f"- Agreement: {'yes' if crosscheck.get('agrees') else 'NO'}",
        f"- Largest absolute difference: "
        f"{_format(crosscheck.get('max_absolute_difference'))}",
        "",
    ]
    for entry in crosscheck.get("disagreements", ()):
        lines.append(
            f"- **{entry['episode_id']}**: this project {entry['slotify_ndcg']:.6f} "
            f"vs scikit-learn {entry['sklearn_ndcg']:.6f}"
        )
    if crosscheck.get("disagreements"):
        lines.append("")
    return lines


def _render_classical(payload: Mapping[str, Any]) -> list[str]:
    classical = payload.get("classical_baseline") or {}
    if not classical:
        return []
    lines = ["## Classical comparison point", ""]
    if not classical.get("available"):
        lines += [f"Did not run: {classical.get('reason', 'unavailable')}.", ""]
        return lines
    lines += [
        "A scikit-learn gradient-boosted model over the handcrafted scalars alone, "
        "tuned by episode-grouped cross-validation inside the training split. It is "
        "**not** the denominator of the headline; it is here so a reader can see "
        "whether the learned modalities earned their inference cost.",
        "",
        f"- NDCG@3: {_format(classical.get('ndcg_at_3'))}",
        f"- Selected hyperparameters: `{classical.get('best_params')}`",
        f"- Trained on {classical.get('train_example_count')} example(s) across "
        f"{classical.get('train_episode_count')} episode(s), "
        f"{classical.get('feature_dimension')} features",
        f"- scikit-learn {classical.get('sklearn_version')}",
        "",
    ]
    return lines


def render_summary(payload: Mapping[str, Any]) -> str:
    """Markdown generated from the same dict the JSON is written from.

    Not a second source of truth: every number here is read out of ``payload``,
    so the two cannot disagree. A test asserts it.
    """
    headline = payload["headline"]
    lines = [
        f"# Evaluation {payload['evaluation_id']}",
        "",
        f"- **Generated**: {payload['generated_at']}",
        f"- **Git SHA**: {payload['inputs']['git_sha']}",
        f"- **Split**: {payload['inputs']['split']} "
        f"({payload['inputs']['split_version']})",
        f"- **Episodes**: {payload['episode_count']}  "
        f"**Candidates**: {payload['candidate_count']}  "
        f"**Relevant**: {payload['relevant_candidate_count']}",
        f"- **Ground truth**: label_source={payload['inputs']['label_source']}",
        f"- **Model**: {payload['inputs']['model_variant']} "
        f"({payload['inputs']['model_run_id']}), trained on "
        f"label_source={payload['inputs']['model_training_label_source']}",
        f"- **Baseline**: {payload['inputs']['baseline_version']}",
        "",
    ]

    if payload["headline_publishable"]:
        lines += ["## Headline", ""]
    else:
        lines += [
            "## Headline - NOT PUBLISHABLE",
            "",
            "This comparison ran, but the number below **must not be quoted**:",
            "",
        ]
        for reason in payload["blocking_reasons"]:
            lines.append(f"- {reason}")
        lines.append("")

    lines += [
        f"- Baseline NDCG@3: {_format(headline['baseline']['ndcg_at_3'])}",
        f"- Model NDCG@3: {_format(headline['model']['ndcg_at_3'])}",
        f"- Absolute improvement: {_format(headline['absolute_improvement'])}",
        "- Relative improvement: "
        + (
            "NOT YET AVAILABLE"
            if headline["relative_improvement_percent"] is None
            else f"{headline['relative_improvement_percent']:.2f} %"
        ),
        "",
        "Computed as `100 * (model - baseline) / baseline` in "
        "`slotify_rank.evaluation.compare.relative_improvement_percent`.",
        "",
    ]

    lines += _render_uncertainty(payload)
    lines += _render_cohort(payload)
    lines += _render_crosscheck(payload)
    lines += _render_classical(payload)

    lines += [
        "## All cutoffs",
        "",
        "| Metric | Baseline | Model |",
        "| --- | --- | --- |",
    ]
    for cutoff_key in sorted(payload["metrics"]["baseline"]):
        baseline_block = payload["metrics"]["baseline"][cutoff_key]
        model_block = payload["metrics"]["model"][cutoff_key]
        for metric in ("ndcg_at_k", "precision_at_k", "recall_at_k", "mrr"):
            lines.append(
                f"| {metric} ({cutoff_key}) | {_format(baseline_block.get(metric))} "
                f"| {_format(model_block.get(metric))} |"
            )

    if payload["warnings"]:
        lines += ["", "## Caveats", ""]
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def write_comparison(directory: Path, result: ComparisonResult) -> dict[str, Path]:
    """Write every artifact for one comparison. Returns the paths written."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()

    written: dict[str, Path] = {}

    def write(name: str, text: str) -> None:
        path = target / name
        atomic_write_bytes(path, text.encode("utf-8"))
        written[name] = path

    write("comparison.json", json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write(
        "metrics.json",
        json.dumps(
            {
                "evaluation_id": result.evaluation_id,
                "metrics": payload["metrics"],
                "episode_count": result.episode_count,
                "candidate_count": result.candidate_count,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    write(
        "per_episode_metrics.json",
        json.dumps(
            {
                "baseline": {
                    f"at_{k}": [e.to_dict() for e in report.per_episode]
                    for k, report in sorted(result.baseline_reports.items())
                },
                "model": {
                    f"at_{k}": [e.to_dict() for e in report.per_episode]
                    for k, report in sorted(result.model_reports.items())
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    write(
        "ranked_candidates.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result.ranked_rows
        ),
    )
    write("summary.md", render_summary(payload))
    return written
