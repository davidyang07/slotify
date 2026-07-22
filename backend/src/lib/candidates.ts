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
const { merge: MERGE, scoring: SCORING, selection: SELECTION } = HEURISTIC_V1;

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

export const selectTopSlots = (
  candidates: ScoredCandidate[],
  durationSeconds: number | null,
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

  if (durationSeconds) {
    const fallbackTimes = SELECTION.ratio_fallback_positions.map(
      (ratio) => ratio * durationSeconds * 1000,
    );
    for (const fallback of fallbackTimes) {
      if (selected.length >= count) break;
      const tooClose = selected.some(
        (entry) => Math.abs(entry.ms - fallback) < minSeparationMs,
      );
      if (!tooClose && fallback >= 0 && fallback <= durationSeconds * 1000) {
        selected.push({
          ms: Math.round(fallback),
          silenceMs: 0,
          snippet: "",
          score: SELECTION.ratio_fallback_score,
        });
      }
    }
  }

  while (selected.length < count) {
    let base = minSeparationMs;
    if (selected.length) {
      const latest = [...selected].sort((a, b) => a.ms - b.ms).slice(-1)[0];
      base = latest.ms + minSeparationMs;
    }
    const tooClose = selected.some(
      (entry) => Math.abs(entry.ms - base) < minSeparationMs,
    );
    const candidateMs = tooClose ? base + minSeparationMs : base;
    selected.push({
      ms: Math.round(candidateMs),
      silenceMs: 0,
      snippet: "",
      score: SELECTION.spacing_fallback_score,
    });
  }

  return selected.slice(0, count);
};
