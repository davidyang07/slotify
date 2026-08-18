/**
 * The one place that decides which ranker answers a request.
 *
 * Callers ask for a ranking and receive one that names its own source. They
 * never choose the ranker, so there is no route that can accidentally serve
 * heuristic output under a learned-model label -- the decision, the fallback and
 * the reporting all happen here.
 */

import { rankWithHeuristic, type RankingResult } from "../lib/ranking";
import { resolveRankerMode } from "../lib/ranker-mode";
import { runLearnedRanker, relativeToRepo } from "./learned-ranker";
import type { Candidate, InsertionMode } from "../types";

export interface RankRequest {
  candidates: Candidate[];
  episodeId: string;
  /** Path to the uploaded audio. The learned path re-derives features from it. */
  audioPath: string;
  durationSeconds: number | null;
  mode: InsertionMode;
  minSeparationSeconds: number;
  count: number;
}

/**
 * Enforce the product's minimum spacing over an already-ordered list.
 *
 * Applied identically to both rankers so the two paths differ only in the
 * ordering they produce, never in the constraint. Like `selectTopSlots`, it can
 * only ever shorten the list.
 */
const applySpacing = <T extends { ms: number }>(
  ordered: T[],
  minSeparationSeconds: number,
  count: number,
): T[] => {
  const minSeparationMs = minSeparationSeconds * 1000;
  const kept: T[] = [];
  for (const entry of ordered) {
    if (kept.some((chosen) => Math.abs(chosen.ms - entry.ms) < minSeparationMs)) {
      continue;
    }
    kept.push(entry);
    if (kept.length >= count) break;
  }
  return kept;
};

export const rankCandidates = async (
  request: RankRequest,
): Promise<RankingResult> => {
  const resolved = resolveRankerMode();

  if (resolved.mode === "heuristic" || !resolved.checkpointPath) {
    const result = rankWithHeuristic(request);
    // In auto mode the reason explains why the model did not run, so a demo
    // watcher can see whether they are looking at the model or the baseline.
    return resolved.requested === "auto"
      ? { ...result, warnings: [resolved.reason] }
      : result;
  }

  const payload = await runLearnedRanker({
    audioPath: request.audioPath,
    checkpointPath: resolved.checkpointPath,
    normalizerPath: resolved.normalizerPath,
    episodeId: request.episodeId,
    skipTranscription: process.env.SLOTIFY_RANKER_SKIP_TRANSCRIPTION === "1",
  });

  // Candidate generation happens twice under the learned path: once here, from
  // the analyser, and once inside the ML feature pipeline, which needs the
  // candidate's own acoustic context to build a feature vector. The learned
  // result is authoritative; the analyser's snippets are joined back on by
  // timestamp purely so the UI can quote the transcript around a slot.
  const snippetByMs = new Map<number, Candidate>();
  for (const candidate of request.candidates) {
    snippetByMs.set(candidate.ms, candidate);
  }
  const nearest = (ms: number): Candidate | undefined => {
    const exact = snippetByMs.get(ms);
    if (exact) return exact;
    let best: Candidate | undefined;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const candidate of request.candidates) {
      const distance = Math.abs(candidate.ms - ms);
      if (distance < bestDistance && distance <= 1500) {
        best = candidate;
        bestDistance = distance;
      }
    }
    return best;
  };

  const ordered = payload.ranked.map((entry) => {
    const matched = nearest(entry.insertion_ms);
    return {
      candidateId: entry.candidate_id,
      ms: entry.insertion_ms,
      silenceMs: matched?.silenceMs ?? 0,
      snippet: matched?.snippet ?? "",
      rawScore: entry.raw_score,
      // Already computed over every candidate the model scored.
      normalizedScore: Math.round(entry.normalized_score),
      textAvailable: entry.text_available,
    };
  });

  const warnings = [...payload.warnings];
  if (payload.model.training_label_source !== "human") {
    // The single most important thing a viewer of this demo needs to know.
    warnings.push(
      `The learned ranker was trained with label_source=` +
        `${payload.model.training_label_source} (${payload.model.training_data_provenance}); ` +
        "its ordering is not evidence of ranking quality.",
    );
  }
  if (payload.excluded.length) {
    warnings.push(
      `${payload.excluded.length} candidate(s) could not be featurised and were ` +
        "not scored.",
    );
  }

  return {
    ranked: applySpacing(ordered, request.minSeparationSeconds, request.count),
    source: "learned_ranker",
    mode: "learned",
    scoreScale: "episode_relative_min_max",
    modelVariant: payload.model.model_variant,
    modelRunId: payload.model.model_run_id,
    warnings,
  };
};

export { relativeToRepo };
