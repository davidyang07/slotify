"""Structural features: position, provenance and the baseline's own opinion.

These cost nothing to compute -- they come straight from the candidate record
written in Phase 2 -- and they carry a surprising amount of signal. Where a
candidate sits in an episode matters (nobody wants an ad 30 seconds from the
end), which generators proposed it matters (a timestamp found by three
independent generators is more likely a real boundary than one found by the
fixed-interval grid alone), and the heuristic's score is a strong prior that a
learned model should be able to beat rather than rediscover.

**What is deliberately excluded.** No label-derived quantity appears here: not
the human rating, not a rank derived from it, not the acceptability flag. Those
are targets. Including any of them would leak the label into the features and
produce a model that looks excellent and predicts nothing.

The heuristic's *component* scores are not leakage -- they are computed from the
audio by ``heuristic_offline_v1``, with no access to a label.
"""

from __future__ import annotations

from typing import Any, Mapping

from slotify_rank.data.schema import DatasetCandidate

__all__ = [
    "CANDIDATE_SOURCE_FLAGS",
    "HEURISTIC_COMPONENTS",
    "extract_structural_features",
]

#: Generators whose participation is recorded as a 0/1 flag. ``product_padding``
#: is excluded on purpose: synthetic candidates never reach feature extraction,
#: so a flag for them would be constant-zero and misleading.
CANDIDATE_SOURCE_FLAGS: tuple[str, ...] = (
    "silence",
    "pause",
    "rms_minimum",
    "transcript_segment_end",
    "fixed_interval",
)

#: Additive components emitted by the Phase 1 scorer, in its accumulation order
#: (:class:`slotify_rank.candidates.schema.ComponentScores`). Named explicitly
#: so a component appearing or disappearing is a loud failure rather than a
#: silently shorter feature vector.
#:
#: ``raw_total`` is handled separately below -- it is the pre-clamp sum, not
#: another additive term, and adding it here would double-count.
HEURISTIC_COMPONENTS: tuple[str, ...] = (
    "base",
    "pause",
    "mode",
    "sentence",
    "position",
    "edge",
)


def extract_structural_features(
    candidate: DatasetCandidate,
    episode_duration_ms: int,
    edge_guard_ms: int,
) -> tuple[dict[str, float], dict[str, bool]]:
    """Structural features for one candidate.

    Returns ``(values, missing)``. Missing is used only where a quantity is
    genuinely absent -- an episode with no heuristic score, a candidate with no
    sentence-end information -- never as a stand-in for zero.
    """
    if episode_duration_ms <= 0:
        raise ValueError(
            f"{candidate.candidate_id}: episode_duration_ms must be positive"
        )

    values: dict[str, float] = {}
    missing: dict[str, bool] = {}

    timestamp = int(candidate.timestamp_ms)
    values["normalized_episode_position"] = timestamp / episode_duration_ms
    missing["normalized_episode_position"] = False
    values["ms_from_episode_start"] = float(timestamp)
    missing["ms_from_episode_start"] = False
    values["ms_to_episode_end"] = float(max(0, episode_duration_ms - timestamp))
    missing["ms_to_episode_end"] = False

    # Inside the guard band at either end. The generator already excludes these,
    # so it is normally 0 -- it exists so a future change to the guard is
    # visible in the features rather than silently altering the pool.
    inside_guard = (
        timestamp < edge_guard_ms or timestamp > episode_duration_ms - edge_guard_ms
    )
    values["edge_guard_violation"] = 1.0 if inside_guard else 0.0
    missing["edge_guard_violation"] = False

    sources = set(candidate.candidate_sources)
    for source in CANDIDATE_SOURCE_FLAGS:
        values[f"source_{source}"] = 1.0 if source in sources else 0.0
        missing[f"source_{source}"] = False
    values["merged_source_count"] = float(len(sources))
    missing["merged_source_count"] = False
    values["merged_timestamp_count"] = float(len(candidate.merged_from_ms))
    missing["merged_timestamp_count"] = False

    values["pause_duration_ms"] = float(candidate.pause_duration_ms)
    missing["pause_duration_ms"] = False
    values["candidate_silence_duration_ms"] = float(candidate.silence_duration_ms)
    missing["candidate_silence_duration_ms"] = False

    if candidate.sentence_end is None:
        values["sentence_end"] = 0.0
        missing["sentence_end"] = True
    else:
        values["sentence_end"] = 1.0 if candidate.sentence_end else 0.0
        missing["sentence_end"] = False

    energy = candidate.local_energy_summary
    for name, attribute in (
        ("candidate_energy_mean_dbfs", "mean_dbfs"),
        ("candidate_energy_min_dbfs", "min_dbfs"),
        ("candidate_energy_max_dbfs", "max_dbfs"),
    ):
        if energy is None:
            values[name] = 0.0
            missing[name] = True
        else:
            values[name] = float(getattr(energy, attribute))
            missing[name] = False

    components: Mapping[str, Any] = candidate.raw_component_scores or {}
    for component in HEURISTIC_COMPONENTS:
        name = f"heuristic_component_{component}"
        raw = components.get(component)
        if raw is None:
            values[name] = 0.0
            missing[name] = True
        else:
            values[name] = float(raw)
            missing[name] = False

    # The pre-clamp sum and whether the clamp actually bit. A clamped score has
    # lost information -- two candidates can share a final score while their raw
    # totals differ -- so both are carried.
    raw_total = components.get("raw_total")
    if raw_total is None:
        values["heuristic_raw_total"] = 0.0
        missing["heuristic_raw_total"] = True
    else:
        values["heuristic_raw_total"] = float(raw_total)
        missing["heuristic_raw_total"] = False

    clamped = components.get("clamped")
    if clamped is None:
        values["heuristic_clamped"] = 0.0
        missing["heuristic_clamped"] = True
    else:
        values["heuristic_clamped"] = 1.0 if clamped else 0.0
        missing["heuristic_clamped"] = False

    if candidate.heuristic_score is None:
        values["heuristic_total_score"] = 0.0
        missing["heuristic_total_score"] = True
    else:
        values["heuristic_total_score"] = float(candidate.heuristic_score)
        missing["heuristic_total_score"] = False

    if set(values) != set(missing):
        raise AssertionError(
            "structural values and mask disagree on keys: "
            f"{sorted(set(values) ^ set(missing))}"
        )
    return values, missing
