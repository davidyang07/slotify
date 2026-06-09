import { clamp, endsWithSentenceBoundary } from "./text";
import type {
  Candidate,
  InsertionMode,
  ProsCons,
  ScoredCandidate,
} from "../types";

export const mergeCandidates = (
  base: Candidate[],
  extra: Candidate[],
  minGapMs = 400,
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
  let score = 0.4;
  if (candidate.silenceMs) {
    score += Math.min(0.4, (candidate.silenceMs / 2000) * 0.4);
  }
  if (mode === "song") {
    score += 0.1;
  }
  if (candidate.snippet && candidate.snippet !== "TRANSCRIPT_UNAVAILABLE") {
    if (endsWithSentenceBoundary(candidate.snippet)) {
      score += 0.3;
    } else {
      score -= 0.2;
    }
  }
  if (durationSeconds) {
    const ratio = timeSeconds / durationSeconds;
    if (ratio >= 0.2 && ratio <= 0.8) {
      score += 0.1;
    }
    if (timeSeconds < 5 || timeSeconds > durationSeconds - 5) {
      score -= 0.3;
    }
  }
  return clamp(score, 0, 1);
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
    const fallbackTimes = [0.22, 0.5, 0.78].map(
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
          score: 0.5,
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
      score: 0.4,
    });
  }

  return selected.slice(0, count);
};
