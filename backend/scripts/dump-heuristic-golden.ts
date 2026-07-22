/**
 * Export golden inputs/outputs for the canonical offline heuristic baseline
 * (`heuristic_offline_v1`) straight from the live TypeScript product scorer.
 *
 * The canonical baseline is the product path:
 *   1. candidate merging      -> src/lib/candidates.ts `mergeCandidates`
 *   2. scoring                -> src/lib/candidates.ts `scoreCandidate`
 *   3. spacing selection      -> src/lib/candidates.ts `selectTopSlots`
 *   4. slot finalisation      -> src/routes/insert-sections.ts:136-260
 *
 * Steps 1-3 are imported from the real module. Step 4 is *mirrored* here
 * because the route interleaves it with an OpenAI enrichment call that cannot
 * run offline; the mirrored block is annotated with the exact route lines it
 * reproduces. See docs/multimodal-ranking-mvp-plan.md §5 (drift risk R17).
 *
 * The Python port in ml/src/slotify_rank/candidates/heuristic.py must reproduce
 * this file byte-for-byte in value. Expected outputs are generated ONLY here,
 * never from the Python implementation under test.
 *
 * Usage:
 *   npx tsx scripts/dump-heuristic-golden.ts --out ../ml/tests/fixtures/heuristic_golden.json
 */

import fs from "node:fs";
import path from "node:path";
import {
  mergeCandidates,
  scoreCandidate,
  selectTopSlots,
} from "../src/lib/candidates";
import { clamp } from "../src/lib/text";
import type { Candidate, InsertionMode, ScoredCandidate } from "../src/types";

interface EpisodeCase {
  episode_id: string;
  description: string;
  covers: string[];
  mode: InsertionMode;
  duration_seconds: number | null;
  /** Silence candidates, as produced by ad_inserter/analyze_cli.py. */
  silence_candidates: Candidate[];
  /** Sentence-end candidates, as produced by insert-sections.ts:90-105. */
  transcript_candidates: Candidate[];
  /** `count` form field; the route uses max(3, count) then slices to 3. */
  count: number;
}

const CASES: EpisodeCase[] = [
  {
    episode_id: "ep-01-mixed-signals",
    description:
      "Long silence at a sentence boundary, silence without a boundary, boundary without silence, unavailable transcript, and two near-duplicate pairs that must merge in opposite directions.",
    covers: [
      "long_silence_at_sentence_boundary",
      "silence_without_sentence_boundary",
      "sentence_boundary_without_long_silence",
      "incomplete_sentence",
      "near_duplicate_merge_keeps_incumbent",
      "near_duplicate_merge_replaces_incumbent",
      "transcript_unavailable_snippet",
      "more_than_three_valid_candidates",
    ],
    mode: "podcast",
    duration_seconds: 600,
    silence_candidates: [
      { ms: 60000, silenceMs: 1800, snippet: "So that was the whole story." },
      { ms: 120000, silenceMs: 1500, snippet: "and then we were going to" },
      { ms: 180000, silenceMs: 300, snippet: "That's exactly right." },
      { ms: 240000, silenceMs: 0, snippet: "TRANSCRIPT_UNAVAILABLE" },
      { ms: 300000, silenceMs: 2400, snippet: "We'll come back to that later." },
    ],
    transcript_candidates: [
      // 300 ms from 60000 -> merges; incumbent kept (larger silence, both end on '.')
      { ms: 60300, silenceMs: 900, snippet: "So that was the whole story." },
      // 200 ms from 120000 -> merges; replaces incumbent (larger silenceMs)
      {
        ms: 120200,
        silenceMs: 2500,
        snippet: "and then we were going to the store.",
      },
      { ms: 420000, silenceMs: 700, snippet: "That is a fair point." },
    ],
    count: 3,
  },
  {
    episode_id: "ep-02-edges",
    description:
      "Early and late candidates that trigger the edge penalty, plus a mid-episode candidate that earns the position reward.",
    covers: [
      "early_episode_candidate",
      "late_episode_candidate",
      "mid_episode_position_reward",
      "edge_penalty",
    ],
    mode: "podcast",
    duration_seconds: 60,
    silence_candidates: [
      { ms: 2000, silenceMs: 1200, snippet: "Welcome back." },
      { ms: 4000, silenceMs: 800, snippet: "Let's get into it." },
      { ms: 30000, silenceMs: 1000, snippet: "That makes sense." },
      { ms: 57000, silenceMs: 1400, snippet: "Thanks for listening." },
    ],
    transcript_candidates: [],
    count: 3,
  },
  {
    episode_id: "ep-03-spacing",
    description:
      "High-scoring candidates packed closer than the 6 s minimum separation, forcing the spacing constraint to reject neighbours.",
    covers: ["candidates_closer_than_minimum_spacing"],
    mode: "podcast",
    duration_seconds: 300,
    silence_candidates: [
      { ms: 100000, silenceMs: 2000, snippet: "Right, exactly." },
      { ms: 101000, silenceMs: 2000, snippet: "Right, exactly." },
      { ms: 103000, silenceMs: 2000, snippet: "Right, exactly." },
      { ms: 105500, silenceMs: 2000, snippet: "Right, exactly." },
      { ms: 150000, silenceMs: 2000, snippet: "Sure thing." },
      { ms: 200000, silenceMs: 2000, snippet: "Of course." },
    ],
    transcript_candidates: [],
    count: 3,
  },
  {
    episode_id: "ep-04-ties",
    description:
      "Candidates with byte-identical scoring inputs at different timestamps; selection must break ties by earlier timestamp.",
    covers: ["tied_scores", "deterministic_tie_breaking"],
    mode: "podcast",
    duration_seconds: 400,
    silence_candidates: [
      { ms: 90000, silenceMs: 1000, snippet: "Understood." },
      { ms: 120000, silenceMs: 1000, snippet: "Understood." },
      { ms: 150000, silenceMs: 1000, snippet: "Understood." },
      { ms: 180000, silenceMs: 1000, snippet: "Understood." },
    ],
    transcript_candidates: [],
    count: 3,
  },
  {
    episode_id: "ep-05-insufficient",
    description:
      "A single valid candidate, so selection pads with the 0.22/0.50/0.78 ratio fallbacks.",
    covers: ["fewer_than_three_valid_candidates", "ratio_fallback_padding"],
    mode: "podcast",
    duration_seconds: 200,
    silence_candidates: [
      { ms: 88000, silenceMs: 1600, snippet: "Let's pause there." },
    ],
    transcript_candidates: [],
    count: 3,
  },
  {
    episode_id: "ep-06-no-duration",
    description:
      "Unknown duration: no position term, no ratio fallback, and padding falls through to the minimum-separation loop.",
    covers: [
      "unknown_duration",
      "spacing_fallback_padding",
      "fewer_than_three_valid_candidates",
    ],
    mode: "podcast",
    duration_seconds: null,
    silence_candidates: [
      { ms: 45000, silenceMs: 900, snippet: "Okay." },
      { ms: 46000, silenceMs: 1900, snippet: "Okay." },
    ],
    transcript_candidates: [],
    count: 3,
  },
  {
    episode_id: "ep-07-song-mode",
    description: "Song mode, which adds a flat mode bonus to every candidate.",
    covers: ["song_mode_bonus"],
    mode: "song",
    duration_seconds: 200,
    silence_candidates: [
      { ms: 50000, silenceMs: 0, snippet: "" },
      { ms: 100000, silenceMs: 0, snippet: "" },
      { ms: 150000, silenceMs: 0, snippet: "" },
    ],
    transcript_candidates: [],
    count: 3,
  },
  {
    episode_id: "ep-08-empty",
    description:
      "No candidates at all, so the route substitutes 0.25/0.50/0.75 duration fallbacks before scoring.",
    covers: ["no_candidates", "route_level_fallback_candidates"],
    mode: "podcast",
    duration_seconds: 120,
    silence_candidates: [],
    transcript_candidates: [],
    count: 3,
  },
  {
    episode_id: "ep-09-many",
    description:
      "A long episode with many well-separated candidates; more than three are valid and only the top three survive.",
    covers: ["more_than_three_valid_candidates"],
    mode: "podcast",
    duration_seconds: 1200,
    silence_candidates: [
      { ms: 60000, silenceMs: 700, snippet: "Sure." },
      { ms: 180000, silenceMs: 1100, snippet: "That tracks." },
      { ms: 300000, silenceMs: 2600, snippet: "Good point." },
      { ms: 420000, silenceMs: 400, snippet: "and so the thing is" },
      { ms: 540000, silenceMs: 1900, snippet: "Absolutely." },
      { ms: 660000, silenceMs: 2200, snippet: "No question." },
      { ms: 780000, silenceMs: 500, snippet: "Right?" },
      { ms: 900000, silenceMs: 1300, snippet: "I agree." },
      { ms: 1020000, silenceMs: 3000, snippet: "Let's move on." },
      { ms: 1140000, silenceMs: 800, snippet: "Fair enough." },
    ],
    transcript_candidates: [],
    count: 3,
  },
];

interface GoldenSlot {
  insertion_ms: number;
  insertion_time_seconds: number;
  confidence_percent: number;
  silence_ms: number;
  snippet: string;
  score: number;
}

interface GoldenEpisode {
  episode_id: string;
  description: string;
  covers: string[];
  input: {
    mode: InsertionMode;
    duration_seconds: number | null;
    count: number;
    silence_candidates: Candidate[];
    transcript_candidates: Candidate[];
  };
  merged_candidates: Candidate[];
  scored_candidates: ScoredCandidate[];
  used_route_fallback_candidates: boolean;
  selected_candidates: ScoredCandidate[];
  slots: GoldenSlot[];
  points: number[];
  confidences: number[];
}

/**
 * Reproduces the deterministic slot pipeline of routes/insert-sections.ts.
 * Line references are to that file as of the Phase 1 baseline commit.
 */
const runEpisode = (episodeCase: EpisodeCase): GoldenEpisode => {
  const { mode, duration_seconds: durationSeconds, count } = episodeCase;

  // insert-sections.ts:136-140 - bound raw candidates by duration.
  const maxMs = durationSeconds ? durationSeconds * 1000 : null;
  const boundedCandidates =
    maxMs !== null
      ? episodeCase.silence_candidates.filter((entry) => entry.ms <= maxMs)
      : episodeCase.silence_candidates;

  // insert-sections.ts:141-144
  const combinedCandidates = mergeCandidates(
    boundedCandidates,
    episodeCase.transcript_candidates,
  );

  // insert-sections.ts:146-161
  const fallbackCandidates: Candidate[] =
    durationSeconds && durationSeconds > 0
      ? [0.25, 0.5, 0.75].map((ratio) => ({
          ms: Math.round(durationSeconds * ratio * 1000),
          silenceMs: 0,
          snippet: "",
        }))
      : [
          { ms: 12000, silenceMs: 0, snippet: "" },
          { ms: 24000, silenceMs: 0, snippet: "" },
          { ms: 36000, silenceMs: 0, snippet: "" },
        ];

  const usedRouteFallback = combinedCandidates.length === 0;
  const usableCandidates = usedRouteFallback
    ? fallbackCandidates
    : combinedCandidates;

  // insert-sections.ts:162-167
  const scoredCandidates: ScoredCandidate[] = usableCandidates.map(
    (candidate) => ({
      ...candidate,
      score: scoreCandidate(candidate, durationSeconds, mode),
    }),
  );

  // insert-sections.ts:174-179
  const selected = selectTopSlots(
    scoredCandidates,
    durationSeconds,
    6,
    Number.isFinite(count) ? Math.max(3, count) : 3,
  ).slice(0, 3);

  // insert-sections.ts:181-198
  const maxSlotMs =
    durationSeconds !== null && durationSeconds !== undefined
      ? Math.max(0, Math.round(durationSeconds * 1000) - 200)
      : null;
  const normalizedSelected = selected.map((candidate) => {
    if (maxSlotMs === null) return candidate;
    return { ...candidate, ms: Math.min(candidate.ms, maxSlotMs) };
  });
  const dedupedSelected: ScoredCandidate[] = [];
  const seenMs = new Set<number>();
  for (const candidate of normalizedSelected) {
    if (seenMs.has(candidate.ms)) continue;
    seenMs.add(candidate.ms);
    dedupedSelected.push(candidate);
  }

  // insert-sections.ts:200-224 (pros/cons/rationale omitted: not deterministic
  // baseline state, and the OpenAI enrichment at :226-258 cannot run offline)
  const slots: GoldenSlot[] = dedupedSelected.map((candidate) => {
    const timeSeconds = candidate.ms / 1000;
    const clampedTimeSeconds =
      durationSeconds !== null && durationSeconds !== undefined
        ? Math.min(Math.max(0, timeSeconds), durationSeconds)
        : Math.max(0, timeSeconds);
    const clampedMs = Math.round(clampedTimeSeconds * 1000);
    const confidence = Math.round(clamp(70 + candidate.score * 25, 70, 95));
    return {
      insertion_ms: clampedMs,
      insertion_time_seconds: Number(clampedTimeSeconds.toFixed(3)),
      confidence_percent: confidence,
      silence_ms: candidate.silenceMs,
      snippet: candidate.snippet ?? "",
      score: candidate.score,
    };
  });

  // insert-sections.ts:260
  slots.sort((a, b) => b.confidence_percent - a.confidence_percent);

  return {
    episode_id: episodeCase.episode_id,
    description: episodeCase.description,
    covers: episodeCase.covers,
    input: {
      mode,
      duration_seconds: durationSeconds,
      count,
      silence_candidates: episodeCase.silence_candidates,
      transcript_candidates: episodeCase.transcript_candidates,
    },
    merged_candidates: combinedCandidates,
    scored_candidates: scoredCandidates,
    used_route_fallback_candidates: usedRouteFallback,
    selected_candidates: selected,
    slots,
    // insert-sections.ts:300-301
    points: slots.map((slot) => slot.insertion_time_seconds),
    confidences: slots.map((slot) => slot.confidence_percent),
  };
};

const parseOutPath = (): string => {
  const index = process.argv.indexOf("--out");
  if (index === -1 || !process.argv[index + 1]) {
    throw new Error("Usage: tsx scripts/dump-heuristic-golden.ts --out <path>");
  }
  return path.resolve(process.argv[index + 1]);
};

const main = (): void => {
  const outPath = parseOutPath();
  const payload = {
    _comment:
      "GENERATED FILE - do not edit by hand. Produced from the TypeScript product scorer by backend/scripts/dump-heuristic-golden.ts. Regenerate with: npx tsx scripts/dump-heuristic-golden.ts --out ../ml/tests/fixtures/heuristic_golden.json",
    baseline_version: "heuristic_offline_v1",
    generator: "backend/scripts/dump-heuristic-golden.ts",
    source_of_truth: [
      "backend/src/lib/candidates.ts::mergeCandidates",
      "backend/src/lib/candidates.ts::scoreCandidate",
      "backend/src/lib/candidates.ts::selectTopSlots",
      "backend/src/routes/insert-sections.ts:136-260 (mirrored in this script)",
    ],
    episodes: CASES.map(runEpisode),
  };
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  console.log(`Wrote ${payload.episodes.length} golden episodes to ${outPath}`);
};

main();
