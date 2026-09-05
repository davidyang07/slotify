// Shared domain types for the ad-insertion pipeline.

export type InsertionMode = "podcast" | "song";

/** A candidate insertion point discovered during analysis. */
export interface Candidate {
  ms: number;
  silenceMs: number;
  snippet: string;
}

/** A candidate with a computed desirability score in [0, 1]. */
export interface ScoredCandidate extends Candidate {
  score: number;
}

/**
 * Which system produced a recommendation.
 *
 * `heuristic_offline_v1` is the deterministic, credential-free signal baseline
 * in `lib/candidates.ts`. `learned_ranker` is the PyTorch model in
 * `ml/src/slotify_rank`. `manual` is a slot the user typed. The API must never
 * report one as another; a demo that cannot say which system spoke is a demo
 * that cannot be trusted about either.
 */
export type RecommendationSource =
  | "heuristic_offline_v1"
  | "learned_ranker"
  | "manual";

export type RankerMode = "learned" | "heuristic" | "auto";

/**
 * The outcome of a placement request, kept separate from HTTP status so the
 * three genuinely different situations stay distinguishable:
 *
 * - `ok`               analysis ran and found at least one eligible point
 * - `no_candidates`    analysis ran and found none (a real answer, not an error)
 * - `unavailable`      analysis itself failed; nothing is known about this audio
 */
export type PlacementStatus = "ok" | "no_candidates" | "unavailable";

/**
 * A measured observation about one insertion point.
 *
 * Every signal must be derivable from something the analyser actually
 * measured. Generic filler ("low background energy at cut") is not a signal,
 * it is decoration that reads like evidence.
 */
export interface PlacementSignal {
  kind: "supporting" | "caution";
  label: string;
}

/** How a 0-100 placement score was derived from the ranker's raw score. */
export type ScoreScale =
  /** raw score already lies in [0, 1]; presented as raw x 100 */
  | "absolute_unit_interval"
  /** unbounded raw scores mapped onto 0-100 relative to this episode only */
  | "episode_relative_min_max";

export interface RankerProvenance {
  /** Which system generated the candidate set. */
  candidateGeneration: string;
  /** Which system ordered them. */
  source: RecommendationSource;
  /** The configured mode that produced this choice. */
  mode: RankerMode;
  modelVariant: string | null;
  modelRunId: string | null;
  scoreScale: ScoreScale;
  /**
   * Always false today. `placement_score` is a ranking score, not a calibrated
   * probability, and nothing in this repository has been calibrated against
   * held-out human labels.
   */
  isCalibratedProbability: false;
}

/** A selected ad slot returned to the client. */
export interface Slot {
  candidate_id: string;
  insertion_ms: number;
  insertion_time_seconds: number;
  /**
   * 0-100 ranking score. NOT a probability and NOT a confidence: see
   * `RankerProvenance.scoreScale` for how it was derived.
   */
  placement_score: number;
  /** The ranker's own untransformed score, for auditability. */
  raw_score: number;
  rank: number;
  source: RecommendationSource;
  signals: PlacementSignal[];
  rationale: string;
  /** Where `rationale` came from: measured signals, or an OpenAI narrative. */
  rationale_source: "measured_signals" | "openai";
  silence_ms: number;
  snippet: string;
}

export interface SponsorStatement {
  id: string;
  name: string;
  statement: string;
  generated: boolean;
}
