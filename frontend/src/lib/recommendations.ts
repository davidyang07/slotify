/**
 * Reading the placement API's answer, without improving on it.
 *
 * The UI used to top up anything shorter than three recommendations with slots
 * at 22 %, 48 % and 72 % of the duration carrying hard-coded scores of 92, 85
 * and 78. Those numbers were not measurements of anything; they made an
 * analysis that found nothing look identical to one that found three good
 * breaks. Nothing in this module invents a slot, a score, or a reason.
 *
 * Kept as plain functions with no React import so the guarantee is directly
 * testable (`recommendations.test.ts`) without a DOM.
 */

export type PlacementStatus = "ok" | "no_candidates" | "unavailable";

export type RecommendationSource =
  | "heuristic_offline_v1"
  | "learned_ranker"
  | "manual";

export interface PlacementSignal {
  kind: "supporting" | "caution";
  label: string;
}

export interface ApiSlot {
  candidate_id: string;
  insertion_ms: number;
  insertion_time_seconds: number;
  placement_score: number;
  raw_score: number;
  rank: number;
  source: RecommendationSource;
  signals: PlacementSignal[];
  rationale: string;
  rationale_source: "measured_signals" | "openai";
  silence_ms: number;
  snippet: string;
}

export interface Provenance {
  candidateGeneration: string;
  source: RecommendationSource | null;
  mode: string;
  modelVariant: string | null;
  modelRunId: string | null;
  scoreScale: string | null;
  isCalibratedProbability: boolean;
  /**
   * Where the learned ranker's training labels came from, or null when the
   * server does not say. `"weak_heuristic"` means the model was distilled from
   * the heuristic's own signals rather than trained on human labels.
   */
  trainingLabelSource: string | null;
}

export interface PlacementResult {
  status: PlacementStatus;
  slots: Slot[];
  duration: number | null;
  candidateCount: number | null;
  provenance: Provenance | null;
  /** Human-readable explanation for a non-ok status, or an advisory note. */
  message: string | null;
}

export interface Slot {
  id: string;
  candidateId: string;
  time: number;
  /** 0-100 ranking score. Never presented as a probability. */
  placementScore: number | null;
  rank: number | null;
  source: RecommendationSource;
  signals: PlacementSignal[];
  rationale: string;
}

const asFiniteNumber = (value: unknown): number | null => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

/**
 * Convert one API slot. Returns null when the payload does not describe a
 * placeable point -- a slot with no usable timestamp is dropped, never
 * defaulted to zero.
 */
const toSlot = (raw: unknown, index: number): Slot | null => {
  if (!raw || typeof raw !== "object") return null;
  const entry = raw as Partial<ApiSlot>;
  const time = asFiniteNumber(entry.insertion_time_seconds);
  if (time === null || time < 0) return null;
  const score = asFiniteNumber(entry.placement_score);
  return {
    id: `slot-${index + 1}`,
    candidateId: String(entry.candidate_id ?? `slot-${index + 1}`),
    time,
    placementScore: score,
    rank: asFiniteNumber(entry.rank),
    source: (entry.source as RecommendationSource) ?? "heuristic_offline_v1",
    signals: Array.isArray(entry.signals) ? entry.signals : [],
    rationale: typeof entry.rationale === "string" ? entry.rationale : "",
  };
};

/** Read the provenance block, defaulting fields an older server omits. */
const toProvenance = (raw: unknown): Provenance | null => {
  if (!raw || typeof raw !== "object") return null;
  const entry = raw as Partial<Provenance>;
  return {
    ...(entry as Provenance),
    trainingLabelSource:
      typeof entry.trainingLabelSource === "string" && entry.trainingLabelSource
        ? entry.trainingLabelSource
        : null,
  };
};

/**
 * Parse a `/api/insert-sections` response.
 *
 * `slots` is exactly as long as the server's list. Zero stays zero, one stays
 * one, two stay two.
 */
export const parsePlacementResponse = (data: unknown): PlacementResult => {
  const payload = (data ?? {}) as Record<string, unknown>;
  const status = (payload.placementStatus as PlacementStatus) ?? "unavailable";
  const rawSlots = Array.isArray(payload.slots) ? payload.slots : [];
  const slots = rawSlots
    .map((entry, index) => toSlot(entry, index))
    .filter((entry): entry is Slot => entry !== null);

  const message =
    typeof payload.warning === "string" && payload.warning
      ? payload.warning
      : typeof payload.error === "string" && payload.error
        ? payload.error
        : null;

  return {
    status,
    slots,
    duration: asFiniteNumber(payload.duration),
    candidateCount: asFiniteNumber(payload.candidateCount),
    provenance: toProvenance(payload.provenance),
    message,
  };
};

/** Clamp slot times into the audio the browser decoded, without moving ranks. */
export const clampSlotsToDuration = (
  slots: Slot[],
  audioDuration: number | null,
): Slot[] => {
  if (!audioDuration || audioDuration <= 0) return slots;
  return slots.map((slot) => ({
    ...slot,
    time: Math.min(Math.max(0, slot.time), audioDuration),
  }));
};

/** A short, accurate label for whichever system produced the ordering. */
export const describeSource = (provenance: Provenance | null): string => {
  if (!provenance || !provenance.source) return "No ranking";
  if (provenance.source === "learned_ranker") {
    const variant = provenance.modelVariant ? ` (${provenance.modelVariant})` : "";
    return `Learned ranker${variant}`;
  }
  if (provenance.source === "manual") return "Manual";
  return "Heuristic baseline";
};

/** What the placement score means, spelled out so nobody reads it as a probability. */
export const describeScoreScale = (provenance: Provenance | null): string => {
  if (!provenance?.scoreScale) return "";
  if (provenance.scoreScale === "episode_relative_min_max") {
    return "Ranking score, relative within this episode. Not a probability.";
  }
  return "Ranking score on a fixed 0-100 scale. Not a probability.";
};

/**
 * A caveat about the ranker's training labels, or null when there is nothing
 * to admit. A model distilled from the heuristic has learned the heuristic,
 * not what listeners prefer.
 */
export const describeTrainingLabels = (
  provenance: Provenance | null,
): string | null => {
  if (provenance?.trainingLabelSource === "weak_heuristic") {
    return "Ranked by a model distilled from the signal-based heuristic; no human labels yet.";
  }
  return null;
};

/** The honest empty state, differentiated by why there is nothing to show. */
export const emptyStateMessage = (result: PlacementResult | null): string => {
  if (!result) return "Upload audio and run analysis to see insertion points.";
  if (result.status === "unavailable") {
    return (
      result.message ??
      "Analysis could not run on this audio, so no insertion point can be reported."
    );
  }
  if (result.status === "no_candidates") {
    return (
      result.message ??
      "No reliable insertion point was detected for this audio. Try a longer clip, or review the audio manually."
    );
  }
  return "No insertion points were returned.";
};

/**
 * A user-typed slot. Marked `manual` so it is never mistaken for a detection.
 *
 * Its id is `manual-N`, a namespace the server's `slot-N` ids never enter, so
 * a manual slot keeps its id (and its selection) when a new placement arrives.
 */
export const makeManualSlot = (timeSeconds: number, existing: Slot[]): Slot => {
  const used = new Set(existing.map((slot) => slot.id));
  let index = existing.filter((slot) => slot.source === "manual").length + 1;
  while (used.has(`manual-${index}`)) index += 1;
  return {
    id: `manual-${index}`,
    candidateId: `manual:${Math.round(timeSeconds * 1000)}`,
    time: timeSeconds,
    placementScore: null,
    rank: null,
    source: "manual",
    signals: [],
    rationale: "Added manually; not produced by analysis.",
  };
};

/**
 * The slots the UI shows: the server's, in the server's order, followed by the
 * user's manual slots in time order, all clamped to the decoded duration.
 * Manual slots live apart from the placement, so neither a new placement nor
 * the duration becoming known can erase them.
 */
export const combineSlots = (
  placementSlots: Slot[],
  manualSlots: Slot[],
  audioDuration: number | null,
): Slot[] =>
  clampSlotsToDuration(
    [...placementSlots, ...[...manualSlots].sort((a, b) => a.time - b.time)],
    audioDuration,
  );

/**
 * Which slots stay selected when a new placement arrives: those that still
 * exist. When none do, the first slot is selected, as on a first analysis.
 */
export const selectionAfterPlacement = (
  selectedIds: string[],
  nextSlots: Slot[],
): string[] => {
  const ids = new Set(nextSlots.map((slot) => slot.id));
  const kept = selectedIds.filter((id) => ids.has(id));
  if (kept.length || !nextSlots.length) return kept;
  return [nextSlots[0].id];
};

/** A short name for a slot, e.g. "Slot 2" or "Custom slot 1". */
export const slotLabel = (slot: Slot): string =>
  slot.source === "manual"
    ? `Custom slot ${slot.id.replace("manual-", "")}`
    : `Slot ${slot.id.replace("slot-", "")}`;
