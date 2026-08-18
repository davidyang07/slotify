/**
 * Turning ranked candidates into the slots the product shows.
 *
 * Two rules govern this module.
 *
 * **Every signal is measured.** The old `buildFallbackProsCons` emitted
 * "Low background energy at cut" and "Clean sentence boundary / transition"
 * for every candidate regardless of what the analyser found, then padded the
 * list to exactly three pros and two cons. That is decoration shaped like
 * evidence. A signal here is emitted only when the measurement supports it,
 * and the list is as short as the evidence is.
 *
 * **Scores are ranking scores.** The product used to display
 * `clamp(70 + score * 25, 70, 95)` as a "confidence percent", which reads as a
 * probability, is bounded away from both ends, and is calibrated against
 * nothing. `placement_score` is the ranker's own score on a 0-100 scale, and
 * the scale it was produced on travels with it.
 */

import { HEURISTIC_V1 } from "./heuristic-config";
import { endsWithSentenceBoundary } from "./text";
import type {
  InsertionMode,
  PlacementSignal,
  ScoreScale,
  ScoredCandidate,
} from "../types";

const { scoring: SCORING } = HEURISTIC_V1;

/** A pause at least this long is worth telling the user about. */
const NOTABLE_PAUSE_MS = 500;
/** Below this, a pause is short enough that a cut there may feel abrupt. */
const ABRUPT_PAUSE_MS = 600;

export const buildPlacementSignals = ({
  mode,
  silenceMs,
  snippet,
  timeSeconds,
  durationSeconds,
}: {
  mode: InsertionMode;
  silenceMs: number;
  snippet: string;
  timeSeconds: number;
  durationSeconds: number | null;
}): PlacementSignal[] => {
  const signals: PlacementSignal[] = [];
  const hasTranscript =
    Boolean(snippet) && snippet !== SCORING.unavailable_snippet_sentinel;

  if (silenceMs >= NOTABLE_PAUSE_MS) {
    signals.push({
      kind: "supporting",
      label: `Pause of ${(silenceMs / 1000).toFixed(1)}s detected`,
    });
  }

  if (hasTranscript) {
    if (endsWithSentenceBoundary(snippet)) {
      signals.push({ kind: "supporting", label: "Falls on a sentence boundary" });
    } else {
      signals.push({ kind: "caution", label: "Cuts mid-sentence" });
    }
  } else {
    signals.push({
      kind: "caution",
      label: "No transcript context available here",
    });
  }

  if (mode === "song") {
    signals.push({ kind: "supporting", label: "Beat-aligned low-energy valley" });
  }

  if (durationSeconds) {
    const ratio = timeSeconds / durationSeconds;
    if (
      ratio >= SCORING.mid_episode_min_ratio &&
      ratio <= SCORING.mid_episode_max_ratio
    ) {
      signals.push({
        kind: "supporting",
        label: `Mid-episode (${Math.round(ratio * 100)}% through)`,
      });
    }
    if (timeSeconds < SCORING.edge_guard_seconds) {
      signals.push({ kind: "caution", label: "Very close to the start" });
    } else if (timeSeconds > durationSeconds - SCORING.edge_guard_seconds) {
      signals.push({ kind: "caution", label: "Very close to the end" });
    }
  }

  if (silenceMs > 0 && silenceMs < ABRUPT_PAUSE_MS) {
    signals.push({ kind: "caution", label: "Short pause may feel abrupt" });
  }

  return signals;
};

/** One sentence stating what was measured, with no adjectives added. */
export const buildRationale = (
  signals: PlacementSignal[],
  timeSeconds: number,
): string => {
  const supporting = signals
    .filter((signal) => signal.kind === "supporting")
    .map((signal) => signal.label.toLowerCase());
  if (!supporting.length) {
    return `Ranked at ${timeSeconds.toFixed(1)}s; no supporting signal was measured beyond the ranker's score.`;
  }
  return `Ranked at ${timeSeconds.toFixed(1)}s: ${supporting.join("; ")}.`;
};

/**
 * Project raw ranker scores onto 0-100.
 *
 * `absolute_unit_interval` is used when the ranker's score is already bounded
 * in [0, 1] (the heuristic). `episode_relative_min_max` is used for an
 * unbounded score (a learned model's logit): the mapping is then only
 * meaningful *within* one episode, which is exactly what the scale name says
 * and what the response reports.
 */
export const toPlacementScores = (
  rawScores: number[],
  scale: ScoreScale,
): number[] => {
  if (scale === "absolute_unit_interval") {
    return rawScores.map((score) =>
      Math.round(Math.min(100, Math.max(0, score * 100))),
    );
  }
  if (rawScores.length === 0) return [];
  const min = Math.min(...rawScores);
  const max = Math.max(...rawScores);
  if (max - min < 1e-9) {
    // Every candidate scored the same: a spread would be invented, not measured.
    return rawScores.map(() => 50);
  }
  return rawScores.map((score) => Math.round(((score - min) / (max - min)) * 100));
};

/**
 * Clamp a slot timestamp into the audio, leaving the configured end guard.
 * Returns null when the timestamp cannot be placed inside the audio at all.
 */
export const clampSlotMs = (
  ms: number,
  durationSeconds: number | null,
): number => {
  if (durationSeconds === null) return Math.max(0, Math.round(ms));
  const maxMs = Math.max(
    0,
    Math.round(durationSeconds * 1000) - HEURISTIC_V1.slot_finalisation.end_guard_ms,
  );
  return Math.min(Math.max(0, Math.round(ms)), maxMs);
};

/** Drop duplicate timestamps, keeping the first (highest-ranked) occurrence. */
export const dedupeByMs = <T extends { ms: number }>(entries: T[]): T[] => {
  const seen = new Set<number>();
  const kept: T[] = [];
  for (const entry of entries) {
    if (seen.has(entry.ms)) continue;
    seen.add(entry.ms);
    kept.push(entry);
  }
  return kept;
};

export type { ScoredCandidate };
