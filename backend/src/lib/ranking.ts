/**
 * The ranking step, isolated from transport.
 *
 * Candidate *generation* and candidate *ranking* are different problems and are
 * kept apart here. Generation is a deterministic signal pass over the audio
 * (silence detection, transcript sentence ends) and always runs. Ranking orders
 * what generation found, and can be done either by the offline heuristic or by
 * the learned PyTorch model -- which is why the result carries the identity of
 * whichever one spoke.
 */

import { BASELINE_VERSION } from "./heuristic-config";
import { scoreCandidate, selectTopSlots } from "./candidates";
import type {
  Candidate,
  InsertionMode,
  RankerMode,
  RecommendationSource,
  ScoreScale,
} from "../types";

export interface RankedCandidate {
  candidateId: string;
  ms: number;
  silenceMs: number;
  snippet: string;
  /** The ranker's own score, on whatever scale that ranker uses. */
  rawScore: number;
  /**
   * 0-100, computed by the ranker over EVERY candidate it scored -- not over
   * the shortlist that survives spacing. Normalizing after the shortlist would
   * stretch two adjacent scores to 100 and 0 and make a good second choice look
   * like a bad one.
   */
  normalizedScore: number;
  /**
   * Whether the ranker had transcript context here. Only the learned path
   * transcribes, so this is how the UI stops claiming "no transcript" for a
   * candidate the model read a transcript for.
   */
  textAvailable: boolean | null;
}

export interface RankingResult {
  /** Best first. Never longer than the candidate set it was given. */
  ranked: RankedCandidate[];
  source: RecommendationSource;
  mode: RankerMode;
  scoreScale: ScoreScale;
  modelVariant: string | null;
  modelRunId: string | null;
  warnings: string[];
}

/**
 * Deterministic candidate identity, matching
 * `ml/src/slotify_rank/candidates/schema.py::make_candidate_id` so a slot the
 * product showed can be joined against a labelled or evaluated candidate.
 */
export const makeCandidateId = (episodeId: string, ms: number): string =>
  `${episodeId}:${String(Math.max(0, Math.round(ms))).padStart(9, "0")}`;

export const rankWithHeuristic = ({
  candidates,
  episodeId,
  durationSeconds,
  mode,
  minSeparationSeconds,
  count,
}: {
  candidates: Candidate[];
  episodeId: string;
  durationSeconds: number | null;
  mode: InsertionMode;
  minSeparationSeconds: number;
  count: number;
}): RankingResult => {
  const scored = candidates.map((candidate) => ({
    ...candidate,
    score: scoreCandidate(candidate, durationSeconds, mode),
  }));
  const selected = selectTopSlots(scored, minSeparationSeconds, count);
  return {
    ranked: selected.map((entry) => ({
      candidateId: makeCandidateId(episodeId, entry.ms),
      ms: entry.ms,
      silenceMs: entry.silenceMs,
      snippet: entry.snippet,
      rawScore: entry.score,
      // scoreCandidate is bounded in [0, 1], so this is an absolute scale and
      // needs no reference to the other candidates.
      normalizedScore: Math.round(Math.min(100, Math.max(0, entry.score * 100))),
      textAvailable: null,
    })),
    source: BASELINE_VERSION as RecommendationSource,
    mode: "heuristic",
    // scoreCandidate clamps into [0, 1] by construction.
    scoreScale: "absolute_unit_interval",
    modelVariant: null,
    modelRunId: null,
    warnings: [],
  };
};
