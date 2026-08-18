import { clamp, endsWithSentenceBoundary } from "./text";
import { HEURISTIC_V1 } from "./heuristic-config";
import type {
  Candidate,
  InsertionMode,
  ProsCons,
  ScoredCandidate,
} from "../types";

// Baseline constants live in config/heuristic_offline_v1.json and are shared
// with the Python port in ml/src/slotify_rank/candidates/heuristic.py. Editing
// them changes the `heuristic_offline_v1` baseline; see
// docs/multimodal-ranking-mvp-plan.md §5.
const { merge: MERGE, scoring: SCORING } = HEURISTIC_V1;

export const mergeCandidates = (
  base: Candidate[],
  extra: Candidate[],
  minGapMs = MERGE.min_gap_ms,
): Candidate[] => {
  const combined = [...base, ...extra].filter(Boolean);
  combined.sort((a, b) => a.ms - b.ms);
  const merged: Candidate[] = [];
  for (const cand of combined) {
    const last = merged[merged.length - 1];
    if (!last || Math.abs(cand.ms - last.ms) > minGapMs) {
      merged.push(cand);
      continue;
    }
    const prefer =
      (cand.silenceMs ?? 0) > (last.silenceMs ?? 0) ||
      (endsWithSentenceBoundary(cand.snippet) &&
        !endsWithSentenceBoundary(last.snippet));
    if (prefer) {
      merged[merged.length - 1] = cand;
    }
  }
  return merged;
};

export const scoreCandidate = (
  candidate: Candidate,
  durationSeconds: number | null,
  mode: InsertionMode,
): number => {
  const timeSeconds = candidate.ms / 1000;
  let score = SCORING.base_score;
  if (candidate.silenceMs) {
    score += Math.min(
      SCORING.pause_reward_max,
      (candidate.silenceMs / SCORING.pause_reward_saturation_ms) *
        SCORING.pause_reward_max,
    );
  }
  if (mode === "song") {
    score += SCORING.song_mode_bonus;
  }
  if (
    candidate.snippet &&
    candidate.snippet !== SCORING.unavailable_snippet_sentinel
  ) {
    if (endsWithSentenceBoundary(candidate.snippet)) {
      score += SCORING.sentence_end_reward;
    } else {
      score -= SCORING.no_sentence_end_penalty;
    }
  }
  if (durationSeconds) {
    const ratio = timeSeconds / durationSeconds;
    if (
      ratio >= SCORING.mid_episode_min_ratio &&
      ratio <= SCORING.mid_episode_max_ratio
    ) {
      score += SCORING.mid_episode_reward;
    }
    if (
      timeSeconds < SCORING.edge_guard_seconds ||
      timeSeconds > durationSeconds - SCORING.edge_guard_seconds
    ) {
      score -= SCORING.edge_penalty;
    }
  }
  return clamp(score, SCORING.score_min, SCORING.score_max);
};

export const buildFallbackProsCons = ({
  mode,
  silenceMs,
  timeSeconds,
  durationSeconds,
}: {
  mode: InsertionMode;
  silenceMs: number;
  timeSeconds: number;
  durationSeconds: number | null;
}): ProsCons => {
  const pros: string[] = [];
  if (mode === "song") {
    pros.push("Beat-aligned low-energy valley");
  } else if (silenceMs >= 800) {
    pros.push(`Natural pause detected (~${Math.round(silenceMs)}ms)`);
  } else if (silenceMs >= 500) {
    pros.push("Clear pause boundary detected");
  }
  pros.push("Low background energy at cut");
  pros.push("Clean sentence boundary / transition");

  const cons: string[] = [];
  if (silenceMs > 0 && silenceMs < 600) {
    cons.push("Short pause may feel abrupt");
  }
  if (durationSeconds) {
    if (timeSeconds < 10) {
      cons.push("Early placement may feel disruptive");
    } else if (timeSeconds > durationSeconds - 10) {
      cons.push("Late placement may feel rushed");
    }
  }
  cons.push("Slight background noise present");

  const pickedPros = pros.slice(0, 3);
  while (pickedPros.length < 3) {
    pickedPros.push("Natural pacing supports insertion");
  }
  const pickedCons = cons.slice(0, 2);
  while (pickedCons.length < 2) {
    pickedCons.push("Minor tonal shift possible");
  }

  return {
    pros: pickedPros,
    cons: pickedCons,
    rationale: `Chosen for a clear pause near ${timeSeconds.toFixed(1)}s that minimizes disruption.`,
  };
};

/**
 * Choose up to `count` insertion points from the candidates that analysis
 * actually found, honouring a minimum spacing between them.
 *
 * This function NEVER invents a slot. The result length is bounded by the
 * number of real candidates that survive the spacing constraint, so a caller
 * that asks for three and receives one has genuinely been told "there is one
 * defensible insertion point in this audio". The padding the product used to
 * apply here (fixed ratio positions, then multiples of the minimum separation)
 * produced timestamps that no signal supported and were indistinguishable in
 * the response from real detections. It now lives in
 * `baseline-parity.ts` and is reachable only from the golden-fixture
 * generator, which must keep reproducing the frozen `heuristic_offline_v1`
 * record.
 */
export const selectTopSlots = (
  candidates: ScoredCandidate[],
  minSeparationSeconds: number,
  count: number,
): ScoredCandidate[] => {
  const minSeparationMs = minSeparationSeconds * 1000;
  const sorted = [...candidates].sort(
    (a, b) => b.score - a.score || a.ms - b.ms,
  );
  const selected: ScoredCandidate[] = [];
  for (const candidate of sorted) {
    const tooClose = selected.some(
      (entry) => Math.abs(entry.ms - candidate.ms) < minSeparationMs,
    );
    if (!tooClose) {
      selected.push(candidate);
    }
    if (selected.length >= count) break;
  }
  return selected;
};
