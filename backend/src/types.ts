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

export interface ProsCons {
  pros: string[];
  cons: string[];
  rationale: string;
}

/** A selected ad slot returned to the client. */
export interface Slot {
  insertion_ms: number;
  insertion_time_seconds: number;
  confidence_percent: number;
  pros: string[];
  cons: string[];
  rationale: string;
  silence_ms: number;
  snippet: string;
}

export interface SponsorStatement {
  id: string;
  name: string;
  statement: string;
  generated: boolean;
}
