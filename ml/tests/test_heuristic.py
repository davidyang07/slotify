"""Tests for the canonical ``heuristic_offline_v1`` baseline.

The parity tests compare the Python port against
``tests/fixtures/heuristic_golden.json``, which is generated **only** by the
TypeScript product code via ``backend/scripts/dump-heuristic-golden.ts``. The
fixture is never regenerated from the Python implementation under test: doing so
would make the test assert that Python agrees with Python.

No test here touches the network, a paid API, a downloaded model, or real audio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slotify_rank.candidates.heuristic import (
    ends_with_sentence_boundary,
    merge_candidates,
    rank_episode,
    rank_episodes,
    score_candidate,
    select_top_slots,
)
from slotify_rank.candidates.schema import (
    Candidate,
    EpisodeInput,
    make_candidate_id,
)
from slotify_rank.config.settings import load_heuristic_config
from slotify_rank.jsnum import js_round, js_to_fixed

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "heuristic_golden.json"

# Exact equality is the goal: both languages use IEEE-754 doubles and the port
# preserves operation order. This tolerance exists only to give a readable
# failure message rather than to license drift.
SCORE_TOLERANCE = 0.0


@pytest.fixture(scope="module")
def config():
    return load_heuristic_config()


@pytest.fixture(scope="module")
def profile(config):
    return config.canonical


@pytest.fixture(scope="module")
def golden():
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _episode_input(case: dict) -> EpisodeInput:
    payload = case["input"]
    return EpisodeInput(
        episode_id=case["episode_id"],
        duration_seconds=payload["duration_seconds"],
        mode=payload["mode"],
        silence_candidates=tuple(
            Candidate.from_mapping(entry, "silence")
            for entry in payload["silence_candidates"]
        ),
        transcript_candidates=tuple(
            Candidate.from_mapping(entry, "transcript_segment_end")
            for entry in payload["transcript_candidates"]
        ),
        count=payload["count"],
    )


# --------------------------------------------------------------------------
# Golden parity with the TypeScript implementation
# --------------------------------------------------------------------------


def test_fixture_is_generated_from_typescript(golden):
    assert golden["generator"] == "backend/scripts/dump-heuristic-golden.ts"
    assert golden["baseline_version"] == "heuristic_offline_v1"
    assert len(golden["episodes"]) >= 9


def test_fixture_covers_every_required_scenario(golden):
    covered = {tag for case in golden["episodes"] for tag in case["covers"]}
    required = {
        "long_silence_at_sentence_boundary",
        "silence_without_sentence_boundary",
        "sentence_boundary_without_long_silence",
        "incomplete_sentence",
        "early_episode_candidate",
        "late_episode_candidate",
        "near_duplicate_merge_keeps_incumbent",
        "near_duplicate_merge_replaces_incumbent",
        "candidates_closer_than_minimum_spacing",
        "tied_scores",
        "fewer_than_three_valid_candidates",
        "more_than_three_valid_candidates",
    }
    assert required <= covered, f"fixture is missing scenarios: {sorted(required - covered)}"


@pytest.mark.parametrize("index", range(9))
def test_merge_parity(golden, profile, index):
    case = golden["episodes"][index]
    episode = _episode_input(case)
    duration = episode.duration_seconds
    max_ms = duration * 1000 if duration else None
    bounded = (
        [c for c in episode.silence_candidates if c.ms <= max_ms]
        if max_ms is not None
        else list(episode.silence_candidates)
    )
    merged = merge_candidates(
        bounded, list(episode.transcript_candidates), int(profile.merge["min_gap_ms"])
    )
    expected = case["merged_candidates"]
    assert [c.ms for c in merged] == [e["ms"] for e in expected]
    assert [c.silence_ms for c in merged] == [e["silenceMs"] for e in expected]
    assert [c.snippet for c in merged] == [e["snippet"] for e in expected]


@pytest.mark.parametrize("index", range(9))
def test_score_parity(golden, profile, index):
    case = golden["episodes"][index]
    duration = case["input"]["duration_seconds"]
    mode = case["input"]["mode"]
    for expected in case["scored_candidates"]:
        candidate = Candidate(
            ms=expected["ms"],
            silence_ms=expected["silenceMs"],
            snippet=expected["snippet"],
        )
        scored = score_candidate(candidate, duration, mode, profile)
        assert scored.score == pytest.approx(expected["score"], abs=SCORE_TOLERANCE), (
            f"{case['episode_id']} @ {expected['ms']}ms: "
            f"python={scored.score!r} typescript={expected['score']!r}"
        )


@pytest.mark.parametrize("index", range(9))
def test_component_scores_reconstruct_the_total(golden, profile, index):
    """The recorded components must explain the score, not merely accompany it."""
    case = golden["episodes"][index]
    duration = case["input"]["duration_seconds"]
    mode = case["input"]["mode"]
    for expected in case["scored_candidates"]:
        candidate = Candidate(
            ms=expected["ms"],
            silence_ms=expected["silenceMs"],
            snippet=expected["snippet"],
        )
        scored = score_candidate(candidate, duration, mode, profile)
        components = scored.components
        rebuilt = components.base
        rebuilt += components.pause
        rebuilt += components.mode
        rebuilt += components.sentence
        rebuilt += components.position
        rebuilt += components.edge
        assert rebuilt == components.raw_total
        assert scored.score == min(1.0, max(0.0, components.raw_total))


@pytest.mark.parametrize("index", range(9))
def test_full_pipeline_parity(golden, index):
    """End-to-end parity: points, confidences and every finalised slot field."""
    case = golden["episodes"][index]
    ranking = rank_episode(_episode_input(case))

    assert list(ranking.points) == case["points"], case["episode_id"]
    assert list(ranking.confidences) == case["confidences"], case["episode_id"]
    assert ranking.used_route_fallback_candidates == case["used_route_fallback_candidates"]

    assert len(ranking.slots) == len(case["slots"])
    for produced, expected in zip(ranking.slots, case["slots"]):
        assert produced.insertion_ms == expected["insertion_ms"]
        assert produced.insertion_time_seconds == expected["insertion_time_seconds"]
        assert produced.confidence_percent == expected["confidence_percent"]
        assert produced.silence_ms == expected["silence_ms"]
        assert produced.snippet == expected["snippet"]
        assert produced.score == pytest.approx(expected["score"], abs=SCORE_TOLERANCE)


@pytest.mark.parametrize("index", range(9))
def test_selection_parity(golden, index):
    """The spacing-constrained selection must match, including padding entries."""
    case = golden["episodes"][index]
    ranking = rank_episode(_episode_input(case))
    selected = sorted(
        (c for c in ranking.candidates if c.selected), key=lambda c: c.selection_rank
    )
    expected = case["selected_candidates"]
    assert [c.timestamp_ms for c in selected] == [entry["ms"] for entry in expected]
    assert [c.total_score for c in selected] == [entry["score"] for entry in expected]


# --------------------------------------------------------------------------
# Individual scoring components
# --------------------------------------------------------------------------


def test_base_score_only(profile):
    """A candidate with no pause, no snippet and unknown duration scores the base."""
    scored = score_candidate(Candidate(ms=1000, silence_ms=0, snippet=""), None, "podcast", profile)
    assert scored.score == 0.4
    assert scored.components.pause == 0.0
    assert scored.components.sentence == 0.0
    assert scored.sentence_end is None


def test_pause_reward_scales_then_saturates(profile):
    half = score_candidate(Candidate(ms=100000, silence_ms=1000, snippet=""), None, "podcast", profile)
    full = score_candidate(Candidate(ms=100000, silence_ms=2000, snippet=""), None, "podcast", profile)
    beyond = score_candidate(Candidate(ms=100000, silence_ms=9000, snippet=""), None, "podcast", profile)
    assert half.components.pause == pytest.approx(0.2)
    assert full.components.pause == pytest.approx(0.4)
    assert beyond.components.pause == pytest.approx(0.4), "pause reward must saturate"


def test_zero_pause_contributes_nothing(profile):
    scored = score_candidate(Candidate(ms=100000, silence_ms=0, snippet=""), None, "podcast", profile)
    assert scored.components.pause == 0.0


def test_sentence_end_reward_and_penalty(profile):
    good = score_candidate(
        Candidate(ms=100000, silence_ms=0, snippet="That is settled."), None, "podcast", profile
    )
    bad = score_candidate(
        Candidate(ms=100000, silence_ms=0, snippet="and then we"), None, "podcast", profile
    )
    assert good.components.sentence == pytest.approx(0.3)
    assert good.sentence_end is True
    assert bad.components.sentence == pytest.approx(-0.2)
    assert bad.sentence_end is False


def test_unavailable_transcript_sentinel_is_neutral(profile):
    """The sentinel must not be treated as an incomplete sentence."""
    scored = score_candidate(
        Candidate(ms=100000, silence_ms=0, snippet="TRANSCRIPT_UNAVAILABLE"),
        None,
        "podcast",
        profile,
    )
    assert scored.components.sentence == 0.0
    assert scored.sentence_end is None


def test_song_mode_bonus(profile):
    podcast = score_candidate(Candidate(ms=100000, silence_ms=0, snippet=""), None, "podcast", profile)
    song = score_candidate(Candidate(ms=100000, silence_ms=0, snippet=""), None, "song", profile)
    assert song.score - podcast.score == pytest.approx(0.1)


def test_mid_episode_reward_and_edge_penalty(profile):
    duration = 100.0
    middle = score_candidate(Candidate(ms=50000, silence_ms=0, snippet=""), duration, "podcast", profile)
    early = score_candidate(Candidate(ms=2000, silence_ms=0, snippet=""), duration, "podcast", profile)
    late = score_candidate(Candidate(ms=98000, silence_ms=0, snippet=""), duration, "podcast", profile)
    assert middle.components.position == pytest.approx(0.1)
    assert middle.components.edge == 0.0
    assert early.components.position == 0.0
    assert early.components.edge == pytest.approx(-0.3)
    assert late.components.edge == pytest.approx(-0.3)


def test_position_window_is_inclusive_at_both_ends(profile):
    duration = 100.0
    at_min = score_candidate(Candidate(ms=20000, silence_ms=0, snippet=""), duration, "podcast", profile)
    at_max = score_candidate(Candidate(ms=80000, silence_ms=0, snippet=""), duration, "podcast", profile)
    assert at_min.components.position == pytest.approx(0.1)
    assert at_max.components.position == pytest.approx(0.1)


def test_score_is_clamped_to_unit_interval(profile):
    high = score_candidate(
        Candidate(ms=50000, silence_ms=5000, snippet="Done."), 100.0, "song", profile
    )
    assert high.score == 1.0
    assert high.components.raw_total > 1.0
    assert high.components.clamped is True


def test_unknown_duration_disables_position_and_edge_terms(profile):
    scored = score_candidate(Candidate(ms=1000, silence_ms=0, snippet=""), None, "podcast", profile)
    assert scored.components.position == 0.0
    assert scored.components.edge == 0.0
    assert scored.normalized_position is None


# --------------------------------------------------------------------------
# Sentence boundary detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Done.", True),
        ("Really?", True),
        ("Stop!", True),
        ('He said "go."', True),
        ("(that is all.)", True),
        ("Trailing space. ", True),
        ("and then", False),
        ("", False),
        ("Mid. sentence continues", False),
        ("no punctuation at all", False),
    ],
)
def test_ends_with_sentence_boundary(text, expected):
    assert ends_with_sentence_boundary(text) is expected


def test_ends_with_sentence_boundary_handles_none():
    assert ends_with_sentence_boundary(None) is False


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def test_merge_collapses_within_tolerance(profile):
    gap = int(profile.merge["min_gap_ms"])
    base = [Candidate(ms=10000, silence_ms=500, snippet="")]
    extra = [Candidate(ms=10000 + gap, silence_ms=500, snippet="")]
    assert len(merge_candidates(base, extra, gap)) == 1


def test_merge_keeps_candidates_beyond_tolerance(profile):
    gap = int(profile.merge["min_gap_ms"])
    base = [Candidate(ms=10000, silence_ms=500, snippet="")]
    extra = [Candidate(ms=10000 + gap + 1, silence_ms=500, snippet="")]
    assert len(merge_candidates(base, extra, gap)) == 2


def test_merge_prefers_longer_pause():
    survivor = merge_candidates(
        [Candidate(ms=10000, silence_ms=500, snippet="")],
        [Candidate(ms=10200, silence_ms=1500, snippet="")],
        400,
    )
    assert survivor[0].ms == 10200
    assert survivor[0].silence_ms == 1500


def test_merge_prefers_sentence_boundary_when_pause_is_not_longer():
    survivor = merge_candidates(
        [Candidate(ms=10000, silence_ms=1500, snippet="mid thought")],
        [Candidate(ms=10200, silence_ms=500, snippet="Complete thought.")],
        400,
    )
    assert survivor[0].ms == 10200


def test_merge_keeps_incumbent_when_neither_criterion_is_met():
    survivor = merge_candidates(
        [Candidate(ms=10000, silence_ms=1500, snippet="Complete thought.")],
        [Candidate(ms=10200, silence_ms=500, snippet="Also complete.")],
        400,
    )
    assert survivor[0].ms == 10000


def test_merge_preserves_all_source_flags():
    merged = merge_candidates(
        [Candidate(ms=10000, silence_ms=500, snippet="", sources=("silence",))],
        [
            Candidate(
                ms=10200,
                silence_ms=1500,
                snippet="",
                sources=("transcript_segment_end",),
            )
        ],
        400,
    )
    assert set(merged[0].sources) == {"silence", "transcript_segment_end"}


def test_merge_output_is_sorted_and_strictly_increasing():
    candidates = [Candidate(ms=ms, silence_ms=100, snippet="") for ms in (5000, 1000, 3000)]
    merged = merge_candidates(candidates, [], 400)
    times = [c.ms for c in merged]
    assert times == sorted(times)
    assert all(later > earlier for earlier, later in zip(times, times[1:]))


def test_merge_is_idempotent():
    candidates = [Candidate(ms=ms, silence_ms=100, snippet="") for ms in (1000, 1200, 5000)]
    once = merge_candidates(candidates, [], 400)
    twice = merge_candidates(once, [], 400)
    assert [c.ms for c in once] == [c.ms for c in twice]


def test_merge_of_empty_input_is_empty():
    assert merge_candidates([], [], 400) == []


# --------------------------------------------------------------------------
# Selection, spacing and tie-breaking
# --------------------------------------------------------------------------


def _scored(profile, times_ms, duration=1000.0, silence=1000):
    return [
        score_candidate(Candidate(ms=ms, silence_ms=silence, snippet=""), duration, "podcast", profile)
        for ms in times_ms
    ]


def test_spacing_constraint_rejects_close_neighbours(profile):
    scored = _scored(profile, [100000, 101000, 103000, 200000, 300000])
    selected = select_top_slots(scored, 1000.0, 6, 3, profile, "ep")
    times = sorted(entry.ms for entry in selected)
    assert all(
        later - earlier >= 6000 for earlier, later in zip(times, times[1:])
    ), times


def test_tie_break_prefers_the_earlier_timestamp(profile):
    scored = _scored(profile, [300000, 200000, 400000])
    assert all(item.score == scored[0].score for item in scored), "setup: scores must tie"
    selected = select_top_slots(scored, 1000.0, 6, 1, profile, "ep")
    assert selected[0].ms == 200000


def test_tie_break_is_deterministic_across_input_orders(profile):
    forward = select_top_slots(_scored(profile, [100000, 200000, 300000]), 1000.0, 6, 3, profile, "ep")
    reverse = select_top_slots(_scored(profile, [300000, 200000, 100000]), 1000.0, 6, 3, profile, "ep")
    assert [e.ms for e in forward] == [e.ms for e in reverse]


def test_higher_score_outranks_earlier_timestamp(profile):
    scored = [
        score_candidate(Candidate(ms=100000, silence_ms=0, snippet=""), 1000.0, "podcast", profile),
        score_candidate(Candidate(ms=500000, silence_ms=2000, snippet="Done."), 1000.0, "podcast", profile),
    ]
    selected = select_top_slots(scored, 1000.0, 6, 1, profile, "ep")
    assert selected[0].ms == 500000


def test_ratio_fallback_pads_an_underfilled_selection(profile):
    selected = select_top_slots(_scored(profile, [88000], duration=200.0), 200.0, 6, 3, profile, "ep")
    assert len(selected) == 3
    origins = [entry.origin for entry in selected]
    assert origins[0] == "candidate"
    assert "ratio_fallback" in origins


def test_spacing_fallback_used_when_duration_is_unknown(profile):
    scored = _scored(profile, [45000], duration=None)
    selected = select_top_slots(scored, None, 6, 3, profile, "ep")
    assert len(selected) == 3
    assert [entry.origin for entry in selected[1:]] == ["spacing_fallback"] * 2


def test_selection_never_returns_more_than_requested(profile):
    scored = _scored(profile, [100000 + i * 20000 for i in range(10)])
    assert len(select_top_slots(scored, 1000.0, 6, 3, profile, "ep")) == 3


# --------------------------------------------------------------------------
# Episode-level behaviour and the synthetic-padding contract
# --------------------------------------------------------------------------


def test_synthetic_padding_is_flagged_and_unranked():
    """Invented padding must never enter an evaluation ranking."""
    episode = EpisodeInput(
        episode_id="ep-pad",
        duration_seconds=200.0,
        silence_candidates=(Candidate(ms=88000, silence_ms=1600, snippet="Pause."),),
    )
    ranking = rank_episode(episode)
    synthetic = [c for c in ranking.candidates if c.is_synthetic]
    assert synthetic, "this episode is expected to require padding"
    assert all(c.rank is None for c in synthetic)
    assert all(c.candidate_id not in ranking.ranked_candidate_ids for c in synthetic)
    assert all(c.selection_origin != "candidate" for c in synthetic)


def test_route_fallback_candidates_used_when_there_are_no_candidates():
    ranking = rank_episode(EpisodeInput(episode_id="ep-empty", duration_seconds=120.0))
    assert ranking.used_route_fallback_candidates is True
    assert [c.timestamp_ms for c in ranking.candidates if not c.is_synthetic] == [
        30000,
        60000,
        90000,
    ]


def test_candidate_ids_are_deterministic_and_unique():
    episode = EpisodeInput(
        episode_id="ep-ids",
        duration_seconds=600.0,
        silence_candidates=tuple(
            Candidate(ms=ms, silence_ms=1000, snippet="Yes.") for ms in (60000, 120000, 180000)
        ),
    )
    first = rank_episode(episode)
    second = rank_episode(episode)
    ids = [c.candidate_id for c in first.candidates]
    assert ids == [c.candidate_id for c in second.candidates]
    assert len(ids) == len(set(ids)), "candidate ids must be unique within an episode"
    assert first.candidates[0].candidate_id == make_candidate_id("ep-ids", 60000)


def test_ranked_ids_are_ordered_by_descending_score():
    episode = EpisodeInput(
        episode_id="ep-order",
        duration_seconds=600.0,
        silence_candidates=(
            Candidate(ms=60000, silence_ms=0, snippet="and then"),
            Candidate(ms=180000, silence_ms=2000, snippet="Settled."),
            Candidate(ms=300000, silence_ms=800, snippet="Right."),
        ),
    )
    ranking = rank_episode(episode)
    by_id = {c.candidate_id: c.total_score for c in ranking.candidates}
    scores = [by_id[cid] for cid in ranking.ranked_candidate_ids]
    assert scores == sorted(scores, reverse=True)


def test_records_carry_baseline_and_config_versions():
    ranking = rank_episode(EpisodeInput(episode_id="ep-v", duration_seconds=120.0))
    config = load_heuristic_config()
    assert ranking.baseline_version == "heuristic_offline_v1"
    assert ranking.config_version == config.config_version
    assert all(c.baseline_version == "heuristic_offline_v1" for c in ranking.candidates)
    assert all(c.config_version == config.config_version for c in ranking.candidates)


def test_rank_episodes_rejects_duplicate_episode_ids():
    episode = EpisodeInput(episode_id="dupe", duration_seconds=120.0)
    with pytest.raises(ValueError, match="Duplicate episode_id"):
        rank_episodes([episode, episode])


# --------------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------------


def test_negative_timestamp_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        Candidate(ms=-1, silence_ms=0, snippet="")


def test_negative_silence_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        Candidate(ms=100, silence_ms=-5, snippet="")


def test_non_integer_milliseconds_are_rejected():
    with pytest.raises(ValueError, match="whole number"):
        Candidate.from_mapping({"ms": 100.5}, "silence")


def test_missing_ms_key_is_rejected():
    with pytest.raises(KeyError, match="ms"):
        Candidate.from_mapping({"silenceMs": 100}, "silence")


def test_non_numeric_ms_is_rejected():
    with pytest.raises(TypeError, match="numeric"):
        Candidate.from_mapping({"ms": "100"}, "silence")


def test_empty_episode_id_is_rejected():
    with pytest.raises(ValueError, match="episode_id"):
        EpisodeInput(episode_id="", duration_seconds=100.0)


def test_negative_duration_is_rejected():
    with pytest.raises(ValueError, match="duration_seconds"):
        EpisodeInput(episode_id="ep", duration_seconds=-1.0)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        EpisodeInput(episode_id="ep", duration_seconds=100.0, mode="video")


def test_episode_input_requires_episode_id_key():
    with pytest.raises(KeyError, match="episode_id"):
        EpisodeInput.from_mapping({"duration_seconds": 10.0})


# --------------------------------------------------------------------------
# JavaScript numeric semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.5, 1),      # Python's round() gives 0
        (1.5, 2),
        (2.5, 3),      # Python's round() gives 2
        (-0.5, 0),     # ties go toward +Infinity
        (-1.5, -1),
        (2.4999, 2),
        (0.0, 0),
    ],
)
def test_js_round_matches_ecmascript(value, expected):
    assert js_round(value) == expected


def test_js_round_rejects_non_finite():
    with pytest.raises(ValueError):
        js_round(float("nan"))


@pytest.mark.parametrize(
    "value,digits,expected",
    [
        # Expected values verified against Node: `Number(v.toFixed(d))`.
        # 1.0005 and 1.005 are *not* ties -- the nearest double is below the
        # midpoint -- so they round down. Rounding the decimal literal instead of
        # the exact binary value would get these wrong.
        (1.0005, 3, 1.0),
        (1.005, 2, 1.0),
        (0.0005, 3, 0.001),
        (2.3456, 3, 2.346),
        (12.0, 3, 12.0),
        (88.0, 3, 88.0),
    ],
)
def test_js_to_fixed(value, digits, expected):
    assert js_to_fixed(value, digits) == expected
