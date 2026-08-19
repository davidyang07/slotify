"""Machine-readable and Markdown statistics for the feature pipeline.

These reports are the evidence trail behind ``docs/evaluation-evidence.md``.
The rule they follow is the same one Phase 2 established: quantities that mean
different things are never summed. ``complete`` multimodal records are reported
separately from ``audio_only`` and ``text_only`` ones; transcribed audio
duration is reported separately from episode duration, because the gap between
them is silence and counting it would overstate the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from slotify_rank.config.versions import (
    FEATURE_PIPELINE_VERSION,
    FEATURE_SPEC_VERSION,
    PACKAGE_VERSION,
    TRANSCRIPTION_VERSION,
)
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import EpisodeRecord
from slotify_rank.features.schema import CandidateFeatureRecord
from slotify_rank.pipeline.state import STAGE_NAMES, StageLedger

__all__ = ["compute_feature_statistics", "write_feature_statistics"]


def _directory_bytes(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def compute_feature_statistics(
    paths: DataPaths,
    header: Mapping[str, Any],
    records: Sequence[CandidateFeatureRecord],
    episodes: Sequence[EpisodeRecord],
    stage_outcomes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Everything the reports need, computed once."""
    names: list[str] = list(header.get("handcrafted_feature_names", []))
    by_status: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_content_type: dict[str, int] = {}
    content_lookup = {episode.episode_id: episode.content_type for episode in episodes}

    for record in records:
        by_status[record.feature_status] = by_status.get(record.feature_status, 0) + 1
        by_split[record.dataset_split] = by_split.get(record.dataset_split, 0) + 1
        content = content_lookup.get(record.episode_id, "unknown")
        by_content_type[content] = by_content_type.get(content, 0) + 1

    # Missing-value rate per feature, which is what tells a modeller whether a
    # column is usable at all before they train on it.
    missing_rate: dict[str, float] = {}
    if records and names:
        mask = np.asarray(
            [record.handcrafted_missing_mask for record in records], dtype=bool
        )
        if mask.shape[1] == len(names):
            rates = mask.mean(axis=0)
            missing_rate = {
                name: round(float(rate), 6) for name, rate in zip(names, rates)
            }

    # -- transcription ------------------------------------------------------
    transcribe_ledger = StageLedger.read(
        paths.stage_ledger("transcribe"), "transcribe"
    )
    transcript_segments = 0
    transcribed_ms = 0
    transcript_failures = 0
    for entry in transcribe_ledger.entries():
        if entry.status == "complete":
            transcript_segments += int(entry.details.get("segments", 0))
            transcribed_ms += int(entry.details.get("transcribed_ms", 0))
        elif entry.status == "failed":
            transcript_failures += 1

    ledger_counts = {
        stage: StageLedger.read(paths.stage_ledger(stage), stage).counts()
        for stage in STAGE_NAMES
    }

    audio_dimensions = sorted(
        {
            record.audio_embedding_dimension
            for record in records
            if record.audio_embedding_dimension
        }
    )
    text_dimensions = sorted(
        {
            record.text_embedding_dimension
            for record in records
            if record.text_embedding_dimension
        }
    )

    cache_hits = sum(int(outcome.get("cache_hits", 0)) for outcome in stage_outcomes)
    cache_misses = sum(
        int(outcome.get("cache_misses", 0)) for outcome in stage_outcomes
    )
    total_seconds = sum(float(outcome.get("seconds", 0.0)) for outcome in stage_outcomes)

    return {
        "package_version": PACKAGE_VERSION,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "transcription_version": TRANSCRIPTION_VERSION,
        "episodes": {
            "selected": len(episodes),
            "with_feature_records": len({record.episode_id for record in records}),
        },
        "transcription": {
            "episodes_transcribed": ledger_counts["transcribe"]["complete"],
            "failures": transcript_failures,
            "segment_count": transcript_segments,
            "transcribed_audio_ms": transcribed_ms,
            "transcribed_audio_hours": round(transcribed_ms / 3_600_000.0, 4),
        },
        "candidates": {
            "processed": len(records),
            "complete_multimodal": by_status.get("complete", 0),
            "audio_only": by_status.get("audio_only", 0),
            "text_only": by_status.get("text_only", 0),
            "handcrafted_only": by_status.get("handcrafted_only", 0),
            "failed": by_status.get("failed", 0),
            "missing_both_learned_modalities": by_status.get("handcrafted_only", 0),
            "by_split": by_split,
            "by_content_type": by_content_type,
        },
        "features": {
            "handcrafted_feature_count": len(names),
            "handcrafted_feature_names": names,
            "missing_value_rate": missing_rate,
        },
        "embeddings": {
            "audio_embedding_model": header.get(
                "audio_embedding_model", "openai/whisper-tiny.en"
            ),
            "audio_embedding_dimensions": audio_dimensions,
            "text_embedding_model": header.get(
                "text_embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "text_embedding_native_dimensions": text_dimensions,
            # Stated explicitly so the constructed width is never mistaken for
            # a model output dimension.
            "constructed_text_vector_dimension": (
                text_dimensions[0] * 4 if text_dimensions else None
            ),
            "constructed_text_vector_note": (
                "4 blocks of the native dimension: "
                "[before | after | |before-after| | before*after]"
            ),
        },
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": (
                round(cache_hits / (cache_hits + cache_misses), 4)
                if (cache_hits + cache_misses)
                else None
            ),
            "stage_counts": ledger_counts,
        },
        "runtime": {
            "total_seconds": round(total_seconds, 3),
            "stages": list(stage_outcomes),
            "candidates_per_second": (
                round(len(records) / total_seconds, 3) if total_seconds > 0 else None
            ),
        },
        "disk_usage_bytes": {
            "transcripts": _directory_bytes(paths.transcripts_dir),
            "handcrafted": _directory_bytes(paths.handcrafted_dir),
            "audio_embeddings": _directory_bytes(paths.audio_embeddings_dir),
            "text_embeddings": _directory_bytes(paths.text_embeddings_dir),
            "feature_manifest": (
                paths.features_manifest.stat().st_size
                if paths.features_manifest.is_file()
                else 0
            ),
        },
    }


def _markdown(statistics: Mapping[str, Any]) -> str:
    candidates = statistics["candidates"]
    embeddings = statistics["embeddings"]
    transcription = statistics["transcription"]
    cache = statistics["cache"]

    lines = [
        "# Feature pipeline statistics",
        "",
        f"Pipeline version `{statistics['feature_pipeline_version']}`, "
        f"feature spec `{statistics['feature_spec_version']}`.",
        "",
        "## Coverage",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| Episodes selected | {statistics['episodes']['selected']} |",
        f"| Episodes with feature records | "
        f"{statistics['episodes']['with_feature_records']} |",
        f"| Episodes transcribed | {transcription['episodes_transcribed']} |",
        f"| Transcription failures | {transcription['failures']} |",
        f"| Transcript segments | {transcription['segment_count']} |",
        f"| Transcribed audio (hours) | "
        f"{transcription['transcribed_audio_hours']} |",
        "",
        "## Candidates",
        "",
        "Counts are kept separate on purpose -- a partial-modality record is a",
        "usable training example with a mask, but it is not a complete one.",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        f"| Processed | {candidates['processed']} |",
        f"| Complete multimodal | {candidates['complete_multimodal']} |",
        f"| Audio only | {candidates['audio_only']} |",
        f"| Text only | {candidates['text_only']} |",
        f"| Handcrafted only (no learned modality) | "
        f"{candidates['handcrafted_only']} |",
        f"| Failed | {candidates['failed']} |",
        "",
        "### By split",
        "",
        "| Split | Count |",
        "| --- | ---: |",
    ]
    for split, count in sorted(candidates["by_split"].items()):
        lines.append(f"| {split} | {count} |")

    lines += [
        "",
        "### By content type",
        "",
        "| Content type | Count |",
        "| --- | ---: |",
    ]
    for content, count in sorted(candidates["by_content_type"].items()):
        lines.append(f"| {content} | {count} |")

    lines += [
        "",
        "## Dimensions",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Audio embedding model | `{embeddings['audio_embedding_model']}` |",
        f"| Audio embedding dimension (raw encoder) | "
        f"{embeddings['audio_embedding_dimensions'] or 'n/a'} |",
        f"| Text embedding model | `{embeddings['text_embedding_model']}` |",
        f"| Text embedding dimension (native) | "
        f"{embeddings['text_embedding_native_dimensions'] or 'n/a'} |",
        f"| Constructed text vector (4 blocks) | "
        f"{embeddings['constructed_text_vector_dimension'] or 'n/a'} |",
        f"| Handcrafted feature count | "
        f"{statistics['features']['handcrafted_feature_count']} |",
        "",
        "The constructed text width is arithmetic, not a model property: it is",
        "four concatenated blocks of MiniLM's native output.",
        "",
        "## Cache",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| Hits | {cache['hits']} |",
        f"| Misses | {cache['misses']} |",
        f"| Hit rate | {cache['hit_rate'] if cache['hit_rate'] is not None else 'n/a'} |",
        f"| Total runtime (s) | {statistics['runtime']['total_seconds']} |",
        "",
    ]

    missing = statistics["features"]["missing_value_rate"]
    incomplete = {name: rate for name, rate in missing.items() if rate > 0}
    if incomplete:
        lines += [
            "## Features with missing values",
            "",
            "| Feature | Missing rate |",
            "| --- | ---: |",
        ]
        for name, rate in sorted(incomplete.items(), key=lambda kv: -kv[1])[:25]:
            lines.append(f"| `{name}` | {rate:.3f} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_feature_statistics(
    paths: DataPaths,
    statistics: Mapping[str, Any],
    output_dir: Path | None = None,
) -> list[Path]:
    """Write the JSON reports and the Markdown summary."""
    directory = Path(output_dir) if output_dir else paths.feature_artifacts_dir
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    documents = {
        "feature_statistics.json": statistics,
        "transcription_statistics.json": statistics["transcription"],
        "embedding_statistics.json": statistics["embeddings"],
        "cache_statistics.json": {
            **statistics["cache"],
            "runtime": statistics["runtime"],
            "disk_usage_bytes": statistics["disk_usage_bytes"],
        },
    }
    for name, payload in documents.items():
        path = directory / name
        atomic_write_bytes(
            path,
            (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        )
        written.append(path)

    summary = directory / "feature_summary.md"
    atomic_write_bytes(summary, _markdown(statistics).encode("utf-8"))
    written.append(summary)
    return written
