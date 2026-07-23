"""Builders for the Phase 5 queue, quality and readiness tests.

Synthetic manifests and synthetic labels, never real audio and never real human
judgements. The point of these tests is the *machinery* -- stratification,
determinism, the gate arithmetic -- so the data is fabricated and labelled as
such. A readiness test that passed on fabricated labels would be a bug if those
numbers ever reached a report; they never leave the test process.
"""

from __future__ import annotations

from dataclasses import dataclass

from slotify_rank.datasets.labels import AggregatedLabel
from slotify_rank.labelling.database import LabelRecord
from tests.dataset_fixtures import make_candidate, make_episode


@dataclass
class Corpus:
    episodes: list
    candidates: list
    aggregated: dict
    raw_labels: list
    feature_status: dict


_SPLIT_LETTER = {"train": "t", "validation": "v", "test": "x"}


def build_corpus(
    series_splits: dict[str, str],
    episodes_per_series: int = 1,
    candidates_per_episode: int = 30,
    labelled: bool = True,
    content_type: str = "podcast",
) -> Corpus:
    """Build a multi-series corpus with optional synthetic labels.

    ``series_splits`` maps series_id -> split. Each candidate gets a spread of
    heuristic scores and, when ``labelled``, a quality score that varies within
    every episode so within-episode pairs exist.
    """
    episodes: list = []
    candidates: list = []
    aggregated: dict = {}
    raw_labels: list = []
    feature_status: dict = {}
    label_id = 0

    for series_index, (series_id, split) in enumerate(sorted(series_splits.items())):
        for ep in range(episodes_per_series):
            letter = _SPLIT_LETTER[split]
            sha_seed = f"{letter}{series_index}{ep}"[:1]
            # Distinct sha per episode via a unique letter run.
            sha_char = chr(ord("a") + (series_index * episodes_per_series + ep) % 26)
            episode = make_episode(
                title=f"{series_id} ep {ep}",
                series_id=series_id,
                sha_seed=sha_char,
                content_type=content_type,
                duration_ms=600_000,
            )
            episodes.append(episode)
            for c in range(candidates_per_episode):
                timestamp = 10_000 + c * 8_000
                score = 0.4 + (c % 6) * 0.1  # 0.4 .. 0.9
                candidate = make_candidate(
                    episode_id=episode.episode_id,
                    timestamp_ms=timestamp,
                    sources=(("silence", "pause", "rms_minimum", "fixed_interval")[c % 4],),
                    heuristic_score=round(score, 4),
                    silence_duration_ms=200 + (c % 5) * 500,
                    normalized_episode_position=round((c + 1) / (candidates_per_episode + 1), 4),
                    dataset_split=split,
                )
                candidates.append(candidate)
                feature_status[candidate.candidate_id] = "complete"
                if labelled:
                    quality = 1 + (c % 5)  # 1..5, varies within an episode
                    acceptable = quality >= 3
                    aggregated[candidate.candidate_id] = AggregatedLabel(
                        candidate_id=candidate.candidate_id,
                        episode_id=episode.episode_id,
                        quality_score=float(quality),
                        is_acceptable=acceptable,
                        annotator_count=1,
                        annotator_ids=("annot-1",),
                        rubric_version="rubric-v1.0.0",
                    )
                    label_id += 1
                    raw_labels.append(
                        LabelRecord(
                            label_id=label_id,
                            candidate_id=candidate.candidate_id,
                            episode_id=episode.episode_id,
                            annotator_id="annot-1",
                            quality_score=quality,
                            is_acceptable=acceptable,
                            is_unusable=False,
                            notes=None,
                            rubric_version="rubric-v1.0.0",
                            created_at="2026-07-23T00:00:00+00:00",
                            updated_at="2026-07-23T00:00:00+00:00",
                        )
                    )

    return Corpus(
        episodes=episodes,
        candidates=candidates,
        aggregated=aggregated,
        raw_labels=raw_labels,
        feature_status=feature_status,
    )


def ready_corpus() -> Corpus:
    """A corpus that clears every readiness threshold."""
    series_splits = {
        "series-train-a": "train",
        "series-train-b": "train",
        "series-train-c": "train",
        "series-train-d": "train",
        "series-val-a": "validation",
        "series-test-a": "test",
        "series-train-e": "train",
        "series-val-b": "validation",
    }
    # 8 series, 1 episode each, 30 candidates each = 240 labelled candidates.
    return build_corpus(series_splits, episodes_per_series=1, candidates_per_episode=30)
