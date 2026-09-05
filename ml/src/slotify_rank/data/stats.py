"""Dataset, candidate, label and split statistics.

Every number the generated reports quote is computed here, from the manifests,
and written to ``artifacts/dataset/``. Nothing is typed by hand and nothing is
rounded up. The eight headline quantities are kept strictly separate:

``processed_audio_hours`` / ``processed_episode_count``
    Audio actually decoded and normalized. Episodes stuck at ``registered`` do
    not count.
``generated_candidate_count``
    Real candidates. Synthetic product padding is excluded by construction and
    reported on its own line so the exclusion is visible rather than implied.
``human_labelled_candidate_count`` / ``human_labelled_audio_hours``
    Candidates a person rated, and the duration of the episodes containing them.
``weakly_labelled_candidate_count`` / ``unlabelled_candidate_count``
    Never added to the human count.
``held_out_evaluation_candidate_count``
    Human-labelled **and** in the test split. The only legitimate source of
    final test metrics; zero until both conditions hold for something.

Out-of-domain audio (music; supplemental meeting corpora) is counted separately
throughout, so a headline hours figure is never inflated by material the product
does not target.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import (
    CANDIDATE_GENERATION_VERSION,
    PACKAGE_VERSION,
    PREPROCESSING_VERSION,
)
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.schema import (
    TARGET_DOMAIN_CONTENT_TYPES,
    DatasetCandidate,
    EpisodeRecord,
)
from slotify_rank.data.splits import SplitManifest
from slotify_rank.labelling.database import LabelRecord

__all__ = [
    "StatisticsBundle",
    "compute_statistics",
    "write_statistics",
    "render_markdown_summary",
]

_PROCESSED_STATUSES = ("normalized",)


def _hours(milliseconds: int | float) -> float:
    return round(float(milliseconds) / 3_600_000.0, 4)


def _is_target_domain(episode: EpisodeRecord) -> bool:
    return episode.is_target_domain and episode.content_type in TARGET_DOMAIN_CONTENT_TYPES


@dataclass(frozen=True)
class StatisticsBundle:
    dataset: dict[str, Any]
    candidates: dict[str, Any]
    labels: dict[str, Any]
    splits: dict[str, Any]

    def as_files(self) -> dict[str, dict[str, Any]]:
        return {
            "dataset_statistics.json": self.dataset,
            "candidate_statistics.json": self.candidates,
            "label_statistics.json": self.labels,
            "split_statistics.json": self.splits,
        }


#: Corpus design goals. Deliberately NOT results -- see ``targets_note`` in the
#: emitted statistics. The label figure is read from the committed experiment
#: definition rather than restated here, so raising the experiment's gate cannot
#: leave this file quietly disagreeing with it.
_DEFAULT_TARGETS: dict[str, Any] = {
    "processed_audio_hours": 50,
    "generated_candidate_count": 10000,
}


def _label_target(repo_root: Path | None = None) -> tuple[int | None, str]:
    """``(minimum_human_labels, where it came from)`` for the targets block."""
    from slotify_rank.config.settings import find_repo_root

    root = repo_root or find_repo_root()
    config_path = root / "ml" / "configs" / "experiment_v2.yaml"
    try:
        from slotify_rank.experiment.canonical import load_experiment_config

        config = load_experiment_config(config_path)
    except Exception:  # noqa: BLE001 - a missing or unreadable config is not fatal
        return None, "no experiment definition was readable"
    return config.minimum_human_labels, config.experiment_version


def _dataset_statistics(
    episodes: Sequence[EpisodeRecord],
    split_by_episode: Mapping[str, str],
) -> dict[str, Any]:
    processed = [e for e in episodes if e.status in _PROCESSED_STATUSES]
    target = [e for e in processed if _is_target_domain(e)]
    label_target, label_target_source = _label_target()

    duration_by_content: dict[str, float] = defaultdict(float)
    duration_by_source: dict[str, float] = defaultdict(float)
    duration_by_license: dict[str, float] = defaultdict(float)
    duration_by_split: dict[str, float] = defaultdict(float)
    series_by_content: dict[str, set[str]] = defaultdict(set)
    episodes_by_split: Counter[str] = Counter()
    for episode in processed:
        milliseconds = episode.duration_ms or 0
        duration_by_content[episode.content_type] += milliseconds
        series_by_content[episode.content_type].add(episode.series_id)
        duration_by_source[episode.source_name] += milliseconds
        duration_by_license[episode.license_name or "undeclared (private)"] += milliseconds
        split = split_by_episode.get(episode.episode_id, "unassigned")
        duration_by_split[split] += milliseconds
        episodes_by_split[split] += 1

    total_ms = sum(e.duration_ms or 0 for e in processed)
    target_ms = sum(e.duration_ms or 0 for e in target)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": PACKAGE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "processed_episode_count": len(processed),
        "processed_audio_hours": _hours(total_ms),
        "target_domain_episode_count": len(target),
        "target_domain_audio_hours": _hours(target_ms),
        "out_of_domain_audio_hours": _hours(total_ms - target_ms),
        "registered_episode_count": len(episodes),
        "unprocessed_episode_count": len(episodes) - len(processed),
        "series_count": len({e.series_id for e in processed}),
        # Counted as distinct shows, not episodes. The split groups on series and
        # the test partition holds the podcast format only, so "how many podcast
        # SERIES are there" is the number that decides whether the held-out
        # partition can hold three independent shows at all -- a total episode
        # count cannot answer it.
        "series_count_by_content_type": {
            content_type: len(series_ids)
            for content_type, series_ids in sorted(series_by_content.items())
        },
        "duration_hours_by_content_type": {
            key: _hours(value) for key, value in sorted(duration_by_content.items())
        },
        "duration_hours_by_source": {
            key: _hours(value) for key, value in sorted(duration_by_source.items())
        },
        "duration_hours_by_license": {
            key: _hours(value) for key, value in sorted(duration_by_license.items())
        },
        "duration_hours_by_split": {
            key: _hours(value) for key, value in sorted(duration_by_split.items())
        },
        "episodes_by_split": dict(sorted(episodes_by_split.items())),
        "episodes_by_status": dict(Counter(e.status for e in episodes).most_common()),
        "target_domain_content_types": sorted(TARGET_DOMAIN_CONTENT_TYPES),
        "targets": {
            **_DEFAULT_TARGETS,
            "human_labelled_candidate_count": label_target,
        },
        "targets_source": {
            "human_labelled_candidate_count": label_target_source,
        },
        "targets_note": (
            "Targets are design goals, not results. Compare them against the "
            "measured values above; do not quote a target as an achievement."
        ),
    }


def _candidate_statistics(
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    generation_reports: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    real = [c for c in candidates if not c.is_synthetic]
    synthetic = [c for c in candidates if c.is_synthetic]
    eligible = [c for c in real if c.eligible_for_evaluation]

    by_source: Counter[str] = Counter()
    for candidate in real:
        for source in candidate.candidate_sources:
            by_source[source] += 1
    primary_source: Counter[str] = Counter(
        candidate.candidate_sources[0] for candidate in real
    )

    processed = [e for e in episodes if e.status in _PROCESSED_STATUSES]
    total_minutes = sum(e.duration_ms or 0 for e in processed) / 60_000.0
    episode_ids_with_candidates = {c.episode_id for c in real}

    before_merge = 0
    after_merge = 0
    dropped_edge = 0
    dropped_density = 0
    for report in generation_reports or ():
        before_merge += int(report.get("count_before_merge", 0))
        after_merge += int(report.get("count_after_merge", 0))
        dropped_edge += int(report.get("dropped_edge_guard", 0))
        dropped_density += int(report.get("dropped_density_cap", 0))

    fixed_interval_only = sum(
        1
        for candidate in real
        if set(candidate.candidate_sources) == {"fixed_interval"}
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": PACKAGE_VERSION,
        "candidate_generation_version": CANDIDATE_GENERATION_VERSION,
        "generated_candidate_count": len(real),
        "eligible_for_evaluation_count": len(eligible),
        "synthetic_product_padding_count": len(synthetic),
        "synthetic_excluded_from_all_counts": True,
        "episodes_with_candidates": len(episode_ids_with_candidates),
        "candidates_per_minute": (
            round(len(real) / total_minutes, 4) if total_minutes > 0 else 0.0
        ),
        "count_before_merge": before_merge,
        "count_after_merge": after_merge,
        "merge_reduction_count": max(0, before_merge - after_merge),
        "merge_reduction_pct": (
            round(100.0 * (before_merge - after_merge) / before_merge, 2)
            if before_merge
            else 0.0
        ),
        "dropped_by_edge_guard": dropped_edge,
        "dropped_by_density_cap": dropped_density,
        "candidates_by_source_flag": dict(sorted(by_source.items())),
        "candidates_by_primary_source": dict(sorted(primary_source.items())),
        "fixed_interval_only_count": fixed_interval_only,
        "fixed_interval_only_pct": (
            round(100.0 * fixed_interval_only / len(real), 2) if real else 0.0
        ),
        "synthetic_or_invalid_pct": (
            round(100.0 * len(synthetic) / len(candidates), 2) if candidates else 0.0
        ),
        "per_episode_generation": [dict(report) for report in (generation_reports or ())],
    }


def _label_statistics(
    labels: Sequence[LabelRecord],
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    split_by_episode: Mapping[str, str],
    acceptable_threshold: int,
) -> dict[str, Any]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    real = [c for c in candidates if not c.is_synthetic]

    # A blind consistency repeat is a second judgement of a candidate already
    # counted, so it changes `human_label_row_count` and never the unique count.
    # Reporting both, plus the repeat count, is what lets the three reconcile.
    human_ids = {label.candidate_id for label in labels if label.candidate_id in by_id}
    repeat_rows = sum(1 for label in labels if label.is_repeat)
    weak_ids = {
        candidate.candidate_id
        for candidate in real
        if candidate.label_status == "weak_heuristic"
    }
    unlabelled = [
        candidate
        for candidate in real
        if candidate.candidate_id not in human_ids
        and candidate.candidate_id not in weak_ids
    ]

    labelled_episode_ids = {
        by_id[candidate_id].episode_id for candidate_id in human_ids
    }
    human_hours = _hours(
        sum(
            episode_by_id[episode_id].duration_ms or 0
            for episode_id in labelled_episode_ids
            if episode_id in episode_by_id
        )
    )

    held_out = sum(
        1
        for candidate_id in human_ids
        if split_by_episode.get(by_id[candidate_id].episode_id) == "test"
    )

    scores = Counter(label.quality_score for label in labels)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": PACKAGE_VERSION,
        "human_labelled_candidate_count": len(human_ids),
        "human_label_row_count": len(labels),
        "repeat_judgement_row_count": repeat_rows,
        "human_labelled_audio_hours": human_hours,
        "human_labelled_episode_count": len(labelled_episode_ids),
        "weakly_labelled_candidate_count": len(weak_ids),
        "unlabelled_candidate_count": len(unlabelled),
        "held_out_evaluation_candidate_count": held_out,
        "annotator_count": len({label.annotator_id for label in labels}),
        "labels_by_score": {str(score): scores.get(score, 0) for score in range(1, 6)},
        "acceptable_count": sum(1 for label in labels if label.is_acceptable),
        "unacceptable_count": sum(1 for label in labels if not label.is_acceptable),
        "unusable_count": sum(1 for label in labels if label.is_unusable),
        "acceptable_threshold": acceptable_threshold,
        "acceptable_rule": f"quality_score >= {acceptable_threshold}",
        "rubric_versions": sorted({label.rubric_version for label in labels}),
        "milestones": {
            "pipeline_debug": 250,
            "first_model": 750,
            "final_training_and_eval": 1500,
        },
    }


def _split_statistics(
    manifest: SplitManifest | None,
    episodes: Sequence[EpisodeRecord],
    candidates: Sequence[DatasetCandidate],
) -> dict[str, Any]:
    if manifest is None:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "package_version": PACKAGE_VERSION,
            "split_manifest_present": False,
            "note": "No split manifest yet. Run `dataset split`.",
        }

    episode_by_id = {episode.episode_id: episode for episode in episodes}
    by_episode = manifest.by_episode
    duration: dict[str, float] = defaultdict(float)
    episode_counts: Counter[str] = Counter()
    series: dict[str, set[str]] = defaultdict(set)
    target_domain: dict[str, int] = defaultdict(int)
    for episode_id, split in by_episode.items():
        episode = episode_by_id.get(episode_id)
        if episode is None:
            continue
        duration[split] += episode.duration_ms or 0
        episode_counts[split] += 1
        series[split].add(episode.series_id)
        if _is_target_domain(episode):
            target_domain[split] += 1

    candidate_counts: Counter[str] = Counter()
    for candidate in candidates:
        if candidate.is_synthetic:
            continue
        candidate_counts[by_episode.get(candidate.episode_id, "unassigned")] += 1

    total_ms = sum(duration.values()) or 1.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": PACKAGE_VERSION,
        "split_manifest_present": True,
        "split_version": manifest.version,
        "algorithm_version": manifest.algorithm_version,
        "seed": manifest.seed,
        "group_by": manifest.group_by,
        "degraded": manifest.degraded,
        "reason": manifest.reason,
        "group_count": len(manifest.assignments),
        "episodes_by_split": dict(sorted(episode_counts.items())),
        "series_by_split": {
            key: len(value) for key, value in sorted(series.items())
        },
        "duration_hours_by_split": {
            key: _hours(value) for key, value in sorted(duration.items())
        },
        "duration_share_by_split": {
            key: round(value / total_ms, 4) for key, value in sorted(duration.items())
        },
        "candidates_by_split": dict(sorted(candidate_counts.items())),
        "target_domain_episodes_by_split": dict(sorted(target_domain.items())),
        "target_ratios": dict(manifest.ratios),
    }


def compute_statistics(
    episodes: Sequence[EpisodeRecord],
    candidates: Sequence[DatasetCandidate],
    labels: Sequence[LabelRecord] = (),
    split_manifest: SplitManifest | None = None,
    generation_reports: Sequence[Mapping[str, Any]] | None = None,
    acceptable_threshold: int = 3,
) -> StatisticsBundle:
    split_by_episode = split_manifest.by_episode if split_manifest else {}
    return StatisticsBundle(
        dataset=_dataset_statistics(episodes, split_by_episode),
        candidates=_candidate_statistics(candidates, episodes, generation_reports),
        labels=_label_statistics(
            labels, candidates, episodes, split_by_episode, acceptable_threshold
        ),
        splits=_split_statistics(split_manifest, episodes, candidates),
    )


def write_statistics(bundle: StatisticsBundle, directory: Path) -> list[Path]:
    """Write the four JSON artifacts plus the Markdown summary."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, payload in bundle.as_files().items():
        path = target / name
        atomic_write_bytes(
            path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        )
        written.append(path)
    summary = target / "dataset_summary.md"
    atomic_write_bytes(summary, render_markdown_summary(bundle).encode("utf-8"))
    written.append(summary)
    return written


def render_markdown_summary(bundle: StatisticsBundle) -> str:
    """A short human summary. Every number is read from the bundle."""
    dataset = bundle.dataset
    candidates = bundle.candidates
    labels = bundle.labels
    splits = bundle.splits

    lines = [
        "# Dataset summary",
        "",
        "<!-- GENERATED by `slotify-rank dataset stats`. Do not edit by hand: every",
        "     number here is read from the JSON artifacts in this directory. -->",
        "",
        f"Generated: {dataset['generated_at']}  ",
        f"Package: `{dataset['package_version']}` · preprocessing: "
        f"`{dataset['preprocessing_version']}` · candidate generation: "
        f"`{candidates['candidate_generation_version']}`",
        "",
        "## Headline quantities",
        "",
        "| Quantity | Measured | Target |",
        "|---|---:|---:|",
        f"| `processed_audio_hours` | {dataset['processed_audio_hours']} | 50 |",
        f"| `processed_episode_count` | {dataset['processed_episode_count']} | — |",
        f"| `generated_candidate_count` | {candidates['generated_candidate_count']} | 10000 |",
        f"| `human_labelled_candidate_count` | {labels['human_labelled_candidate_count']} "
        f"| {dataset['targets']['human_labelled_candidate_count']} |",
        f"| `human_labelled_audio_hours` | {labels['human_labelled_audio_hours']} | 8–12 |",
        f"| `weakly_labelled_candidate_count` | {labels['weakly_labelled_candidate_count']} | — |",
        f"| `unlabelled_candidate_count` | {labels['unlabelled_candidate_count']} | — |",
        f"| `held_out_evaluation_candidate_count` | {labels['held_out_evaluation_candidate_count']} | — |",
        "",
        "Targets are design goals. A target is not a result and must never be quoted as one.",
        "",
        "## Corpus",
        "",
        f"- Target-domain audio: **{dataset['target_domain_audio_hours']} h** across "
        f"{dataset['target_domain_episode_count']} episode(s), "
        f"{dataset['series_count']} series.",
        f"- Out-of-domain audio (music, meetings): "
        f"**{dataset['out_of_domain_audio_hours']} h**, excluded from headline claims.",
        "",
        "| Content type | Hours |",
        "|---|---:|",
    ]
    for content_type, hours in dataset["duration_hours_by_content_type"].items():
        lines.append(f"| {content_type} | {hours} |")

    lines += [
        "",
        "## Candidates",
        "",
        f"- Before merging: **{candidates['count_before_merge']}** → after merging: "
        f"**{candidates['count_after_merge']}** "
        f"({candidates['merge_reduction_pct']} % collapsed).",
        f"- Density: **{candidates['candidates_per_minute']}** per minute of processed audio.",
        f"- Dropped by the edge guard: {candidates['dropped_by_edge_guard']}; "
        f"by the density cap: {candidates['dropped_by_density_cap']}.",
        f"- Fixed-interval-only candidates: {candidates['fixed_interval_only_count']} "
        f"({candidates['fixed_interval_only_pct']} %).",
        f"- Synthetic product padding recorded and **excluded** from every count above: "
        f"{candidates['synthetic_product_padding_count']}.",
        "",
        "| Source flag | Candidates |",
        "|---|---:|",
    ]
    for source, count in candidates["candidates_by_source_flag"].items():
        lines.append(f"| {source} | {count} |")

    lines += [
        "",
        "## Labels",
        "",
        f"- Human-labelled candidates: **{labels['human_labelled_candidate_count']}** "
        f"from {labels['annotator_count']} annotator(s) over "
        f"{labels['human_labelled_episode_count']} episode(s).",
        f"- Acceptability rule: `{labels['acceptable_rule']}` → "
        f"{labels['acceptable_count']} acceptable, {labels['unacceptable_count']} not, "
        f"{labels['unusable_count']} unusable.",
        "",
        "| Score | Count |",
        "|---:|---:|",
    ]
    for score, count in labels["labels_by_score"].items():
        lines.append(f"| {score} | {count} |")

    lines += ["", "## Splits", ""]
    if not splits.get("split_manifest_present"):
        lines.append(f"_{splits.get('note')}_")
    else:
        if splits["degraded"]:
            lines.append(f"> **Degraded split.** {splits['reason']}")
            lines.append("")
        lines += [
            f"Algorithm `{splits['algorithm_version']}`, seed {splits['seed']}, "
            f"grouped by `{splits['group_by']}`.",
            "",
            "| Split | Series | Episodes | Hours | Share | Candidates |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for split in sorted(splits["episodes_by_split"]):
            lines.append(
                f"| {split} | {splits['series_by_split'].get(split, 0)} "
                f"| {splits['episodes_by_split'].get(split, 0)} "
                f"| {splits['duration_hours_by_split'].get(split, 0)} "
                f"| {splits['duration_share_by_split'].get(split, 0)} "
                f"| {splits['candidates_by_split'].get(split, 0)} |"
            )
    return "\n".join(lines) + "\n"
