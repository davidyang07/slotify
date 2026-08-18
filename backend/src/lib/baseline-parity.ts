/**
 * FIXTURE-ONLY code. Not reachable from any Express route.
 *
 * `heuristic_offline_v1` is the frozen denominator every NDCG@3 comparison in
 * this repository is measured against, and its record includes two padding
 * behaviours the *product* used to have: when fewer than `count` real
 * candidates survived spacing, the selector appended fixed ratio positions and
 * then multiples of the minimum separation. Those padded entries carry no
 * signal, so the product no longer emits them (see
 * `candidates.ts::selectTopSlots`).
 *
 * The baseline record still has to be reproducible, because
 * `ml/src/slotify_rank/candidates/heuristic.py` is a byte-exact port of it and
 * `ml/tests/fixtures/heuristic_golden.json` is generated from this TypeScript.
 * So the padding is preserved here, verbatim, behind a name nobody can call by
 * accident, and every entry it produces is tagged `origin` so a consumer can
 * never mistake it for a detection.
 *
 * Only `backend/scripts/dump-heuristic-golden.ts` imports this module. A test
 * asserts no file under `src/routes` does.
 */

import { HEURISTIC_V1 } from "./heuristic-config";
import type { ScoredCandidate } from "../types";

const { selection: SELECTION } = HEURISTIC_V1;

/** Where a padded selection entry came from. */
export type SelectionOrigin =
  | "candidate"
  | "ratio_fallback"
  | "spacing_fallback";

export interface PaddedSelection extends ScoredCandidate {
  origin: SelectionOrigin;
}

/**
 * Reproduce the `heuristic_offline_v1` padding on top of a real selection.
 *
 * `selected` must already be the output of `selectTopSlots`, i.e. only real
 * candidates. Everything this function appends is synthetic by construction.
 */
export const padSelectionForBaselineParity = (
  selected: ScoredCandidate[],
  durationSeconds: number | null,
  minSeparationSeconds: number,
  count: number,
): PaddedSelection[] => {
  const minSeparationMs = minSeparationSeconds * 1000;
  const padded: PaddedSelection[] = selected.map((entry) => ({
    ...entry,
    origin: "candidate" as const,
  }));

  if (durationSeconds) {
    const fallbackTimes = SELECTION.ratio_fallback_positions.map(
      (ratio) => ratio * durationSeconds * 1000,
    );
    for (const fallback of fallbackTimes) {
      if (padded.length >= count) break;
      const tooClose = padded.some(
        (entry) => Math.abs(entry.ms - fallback) < minSeparationMs,
      );
      if (!tooClose && fallback >= 0 && fallback <= durationSeconds * 1000) {
        padded.push({
          ms: Math.round(fallback),
          silenceMs: 0,
          snippet: "",
          score: SELECTION.ratio_fallback_score,
          origin: "ratio_fallback",
        });
      }
    }
  }

  while (padded.length < count) {
    let base = minSeparationMs;
    if (padded.length) {
      const latest = [...padded].sort((a, b) => a.ms - b.ms).slice(-1)[0];
      base = latest.ms + minSeparationMs;
    }
    const tooClose = padded.some(
      (entry) => Math.abs(entry.ms - base) < minSeparationMs,
    );
    const candidateMs = tooClose ? base + minSeparationMs : base;
    padded.push({
      ms: Math.round(candidateMs),
      silenceMs: 0,
      snippet: "",
      score: SELECTION.spacing_fallback_score,
      origin: "spacing_fallback",
    });
  }

  return padded.slice(0, count);
};
