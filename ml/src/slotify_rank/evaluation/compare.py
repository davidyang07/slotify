"""Held-out comparison of the learned ranker against `heuristic_offline_v1`.

This module exists to produce exactly one number honestly:

    relative_improvement_percent
        = 100 * (model_ndcg_at_3 - baseline_ndcg_at_3) / baseline_ndcg_at_3

and to refuse to produce it when the conditions that would make it meaningful
are not met. The refusals are the point. Any pipeline can divide two floats.

Each of these gates independently marks the headline **not publishable**:

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

**The inputs must be real.** The synthetic fixture corpus is stamped
``label_source: synthetic``, so the ground-truth gate above already refuses it;
fixture exports written before that stamp claimed ``human``, and the loader
treats any row marked ``synthetic: true`` as synthetic whatever it claims. A
synthetic label export or feature manifest still blocks the headline here as
well, by its own ``synthetic`` flag, so generated numbers are refused twice.

**The protocol must have been followed** (when the caller supplies one -- the
CLI always does). The evaluated checkpoint must be the run the training matrix
nominated, and the label export must match the frozen snapshot; see
``evaluation_cli``. A comparison run outside that protocol is recorded as such
and is not publishable.

**Uncertainty is reported, not implied.** NDCG@3 is macro-averaged over
episodes, and the test split is small: the committed one (split v4,
``artifacts/dataset/split_statistics.json``) holds 7 episodes from 3 series. A
percentile bootstrap gives an interval for the baseline, for the model and for
the relative improvement itself, and it resamples **series**, not episodes --
episodes of one show share hosts, room, mic chain and editing rhythm, so they
are correlated, and the split is grouped by series for exactly that reason.
Resampling episodes would treat 7 correlated observations as 7 independent ones
and produce an interval that is too narrow. A drawn series contributes all of
its episodes, and both systems are resampled on the same draw. With 3 series the
bootstrap can only produce a handful of distinct resamples, and the report says
so. The improvement's interval is the one that matters: a headline of "+21%"
whose interval spans zero is not a result.

**The claim is judged, not described.** When the pre-registered experiment is
supplied, the report carries a PASS/FAIL verdict against its
``minimum_relative_improvement_percent``. PASS requires a publishable headline
at or above the threshold; anything else is FAIL, with the reason.

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
    "MAX_SERIES_FOR_THIN_PARTITION_WARNING",
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

#: v1.1.0: the bootstrap resamples series (``bootstrap.config.unit``,
#: ``bootstrap.series_count``), and the report carries ``claim`` (the PASS/FAIL
#: verdict) and ``protocol`` (training-matrix nomination, label-snapshot check).
EVALUATION_SCHEMA_VERSION = "evaluation-v1.1.0"

#: A test partition with this many series or fewer gets the explicit caveat. The
#: committed split has exactly 3, which is the case the caveat exists for.
MAX_SERIES_FOR_THIN_PARTITION_WARNING = 3

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
    """Percentile cluster bootstrap: series are drawn, episodes ride along.

    The metric is macro-averaged over episodes, but episodes are not the
    independent unit: episodes of one series are correlated, and the split is
    grouped by series precisely because of it. Resampling episodes would treat
    a show's six episodes as six independent observations (and resampling
    candidates, fifty per episode, would be worse still); either produces an
    interval too narrow by an amount that depends on how alike a show's
    episodes are. So series are resampled with replacement, every episode of a
    drawn series is kept, and the macro-average is taken over the resulting
    episodes.
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
            "unit": "series",
            "method": "percentile",
        }


def bootstrap_interval(
    per_episode_baseline: Mapping[str, float],
    per_episode_model: Mapping[str, float],
    config: BootstrapConfig,
    series_by_episode: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bootstrap the two macro-averages and their relative improvement.

    Clusters are series (``series_by_episode``; an episode missing from it is
    its own series, and how many were is recorded). When the mapping covers none
    of the scored episodes every episode is its own cluster, which is an episode
    bootstrap, and the result's ``config.unit`` says ``episode`` so it is not
    mistaken for the series one.

    Both systems are resampled on the *same* draw, which is what makes the
    improvement's interval meaningful: the two scores are paired observations of
    one episode, and resampling them independently would throw that pairing
    away and widen the interval for no reason.

    Not measured when fewer than two clusters have both scores -- an interval
    from one series is not an interval.
    """
    lookup = dict(series_by_episode or {})
    episodes = sorted(set(per_episode_baseline) & set(per_episode_model))
    unmapped = [episode for episode in episodes if episode not in lookup]
    unit = "series" if len(unmapped) < len(episodes) else "episode"
    recorded_config = {**config.to_dict(), "unit": unit}
    clusters: dict[str, list[str]] = {}
    for episode_id in episodes:
        clusters.setdefault(lookup.get(episode_id, episode_id), []).append(episode_id)
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < 2:
        return {
            "measured": False,
            "reason": (
                f"{len(episodes)} episode(s) in {len(cluster_ids)} {unit}(s) have both "
                f"scores; a bootstrap over {unit} needs at least two"
            ),
            "config": recorded_config,
        }

    generator = random.Random(config.seed)
    baseline_draws: list[float] = []
    model_draws: list[float] = []
    improvement_draws: list[float] = []
    cluster_count = len(cluster_ids)
    for _ in range(config.resamples):
        drawn = [cluster_ids[generator.randrange(cluster_count)] for _ in range(cluster_count)]
        sample = [episode for cluster in drawn for episode in clusters[cluster]]
        baseline_mean = sum(per_episode_baseline[e] for e in sample) / len(sample)
        model_mean = sum(per_episode_model[e] for e in sample) / len(sample)
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
        "config": recorded_config,
        "episode_count": len(episodes),
        "series_count": cluster_count,
        # Episodes with no series in the mapping, each resampled as its own.
        "unmapped_episode_count": len(unmapped),
        # Multisets of k clusters drawn with replacement: C(2k-1, k). With 3
        # series that is 10, so a 95% interval is read off 10 possible values.
        "distinct_resamples_possible": math.comb(2 * cluster_count - 1, cluster_count),
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
    #: How the run related to the pre-registered protocol: was the checkpoint
    #: the nominated one, was the label export the frozen one. Empty when the
    #: caller applied no protocol (a library call); the CLI always fills it.
    protocol: Mapping[str, Any] = field(default_factory=dict)
    #: The pre-registered threshold, when the experiment was supplied.
    claim_threshold_percent: float | None = None
    claim_statement: str = ""

    @property
    def publishable(self) -> bool:
        """Whether the headline number may be published."""
        return not self.blocking_reasons

    def claim(self) -> dict[str, Any]:
        """PASS/FAIL against the pre-registered claim, computed on demand.

        On demand because blocking reasons can still be added after
        :func:`compare` returns (the CLI adds protocol findings), and the verdict
        must reflect every one of them.
        """
        if self.claim_threshold_percent is None:
            return {
                "evaluated": False,
                "verdict": None,
                "reason": (
                    "no pre-registered claim was supplied (pass --experiment-config), "
                    "so there is nothing to judge this comparison against"
                ),
            }
        threshold = float(self.claim_threshold_percent)
        improvement = self.headline()["relative_improvement_percent"]
        reasons: list[str] = []
        if not self.publishable:
            reasons.append(
                "the headline is not publishable "
                f"({len(self.blocking_reasons)} blocking reason(s)), so the claim "
                "is unsupported whatever the number says"
            )
        if improvement is None:
            reasons.append("the relative improvement is undefined")
        elif improvement < threshold:
            reasons.append(
                f"the measured relative improvement {improvement:.2f} % is below the "
                f"pre-registered {threshold:g} %"
            )
        interval = (
            self.bootstrap.get("relative_improvement_percent")
            if self.bootstrap.get("measured")
            else None
        )
        return {
            "evaluated": True,
            "verdict": "FAIL" if reasons else "PASS",
            "statement": self.claim_statement,
            "minimum_relative_improvement_percent": threshold,
            "measured_relative_improvement_percent": improvement,
            "relative_improvement_interval": interval,
            "reasons": reasons,
        }

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
            "protocol": dict(self.protocol),
            "headline_publishable": self.publishable,
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
            "headline": self.headline(),
            "claim": self.claim(),
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
    protocol: Mapping[str, Any] | None = None,
    claim_threshold_percent: float | None = None,
    claim_statement: str = "",
) -> ComparisonResult:
    """Score both systems on the same candidates with the same labels.

    ``protocol`` carries the pre-registered checks the caller made (see
    ``evaluation_cli``): a checkpoint the training matrix did not nominate, or a
    label export not verified against the frozen snapshot, blocks the headline.
    """
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

    protocol_block = dict(protocol or {})
    nomination = protocol_block.get("nomination")
    if isinstance(nomination, Mapping) and not nomination.get("nominated"):
        blocking.append(
            "the evaluated checkpoint is not the run the training matrix nominated "
            f"(median seed): {nomination.get('reason')}. It was evaluated only "
            "because --allow-unnominated was passed."
        )
    snapshot = protocol_block.get("label_snapshot")
    if isinstance(snapshot, Mapping) and not snapshot.get("verified"):
        blocking.append(
            "the label export was not verified against a frozen label snapshot: "
            f"{snapshot.get('reason')}"
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
    if len(evaluated_series) <= MAX_SERIES_FOR_THIN_PARTITION_WARNING:
        warnings.append(
            f"the test partition holds {len(evaluated_series)} series "
            f"({sorted(evaluated_series)}). Episodes of one show share hosts, room, "
            "mic chain and editing rhythm, so a macro-average over them measures "
            "one show's quirks as much as the model. The bootstrap resamples "
            "series, and with this few it has only a handful of distinct resamples "
            "to draw from, so even the interval is coarse. Read the interval, not "
            "the point estimate."
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
        per_episode_baseline,
        per_episode_model,
        bootstrap or BootstrapConfig(),
        series_by_episode=series_by_episode,
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
            f"{interval['high']:.2f}] %, which spans zero: this many series cannot "
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
        protocol=protocol_block,
        claim_threshold_percent=claim_threshold_percent,
        claim_statement=claim_statement,
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
    if config.get("unit") == "series":
        unit_sentence = (
            f"Percentile cluster bootstrap, {config['resamples']} resamples of "
            f"{bootstrap.get('series_count')} series covering "
            f"{bootstrap['episode_count']} episode(s), seed {config['seed']}. Series "
            "are the resampling unit because episodes of one show are correlated; a "
            "drawn series contributes all of its episodes. With "
            f"{bootstrap.get('series_count')} series only "
            f"{bootstrap.get('distinct_resamples_possible')} distinct resamples exist."
        )
    else:
        unit_sentence = (
            f"Percentile bootstrap, {config['resamples']} resamples over "
            f"{bootstrap['episode_count']} episode(s), seed {config['seed']}. No "
            "series mapping was supplied, so each episode was treated as "
            "independent -- the interval is narrower than a series bootstrap's."
        )
    lines += [
        unit_sentence,
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


def _render_claim(payload: Mapping[str, Any]) -> list[str]:
    claim = payload.get("claim") or {}
    if not claim.get("evaluated"):
        return [
            "## Claim verdict - NOT EVALUATED",
            "",
            f"{claim.get('reason', 'no pre-registered claim was supplied')}.",
            "",
        ]
    lines = [
        f"## Claim verdict: {claim['verdict']}",
        "",
        f"- Claim: {claim.get('statement') or '(no statement recorded)'}",
        f"- Pre-registered minimum relative improvement: "
        f"{claim['minimum_relative_improvement_percent']:g} %",
        "- Measured relative improvement: "
        + (
            "undefined"
            if claim.get("measured_relative_improvement_percent") is None
            else f"{claim['measured_relative_improvement_percent']:.2f} %"
        ),
        "- Interval: " + _interval(claim.get("relative_improvement_interval"), " %"),
        "",
    ]
    for reason in claim.get("reasons", ()):
        lines.append(f"- FAIL because {reason}.")
    if claim.get("reasons"):
        lines.append("")
    return lines


def _render_protocol(payload: Mapping[str, Any]) -> list[str]:
    protocol = payload.get("protocol") or {}
    if not protocol:
        return []
    nomination = protocol.get("nomination") or {}
    snapshot = protocol.get("label_snapshot") or {}
    return [
        "## Protocol",
        "",
        "- Nominated by the training matrix: "
        + ("yes" if nomination.get("nominated") else "NO")
        + (f" ({nomination.get('reason')})" if nomination.get("reason") else "")
        + (
            "; evaluated under --allow-unnominated"
            if nomination.get("allowed_unnominated")
            else ""
        ),
        "- Label export verified against the frozen snapshot: "
        + ("yes" if snapshot.get("verified") else "NO")
        + (f" ({snapshot.get('reason')})" if snapshot.get("reason") else ""),
        "",
    ]


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

    lines += _render_claim(payload)
    lines += _render_protocol(payload)
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
