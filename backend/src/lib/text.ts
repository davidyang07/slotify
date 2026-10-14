// Pure text/value helpers shared across routes.

import { HEURISTIC_V1 } from "./heuristic-config";

// The scorer, the Python port and the ML baseline must agree on what a
// sentence end is, so the pattern lives in the canonical config.
const SENTENCE_BOUNDARY = new RegExp(HEURISTIC_V1.sentence_boundary_pattern.regex);

/**
 * A string or finite number as text; anything else as "". Request bodies are
 * attacker-shaped: `String(value)` throws on `{"toString": 1}`, and inside an
 * async handler that used to take the process down.
 */
export const asText = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return "";
};

export const parseJsonField = (value: unknown): any => {
  if (!value || typeof value !== "string") return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

const toStatements = (entries: unknown[]): string[] =>
  entries.map((entry) => asText(entry).trim()).filter(Boolean);

export const normalizeStatements = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return toStatements(value);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return [];
    const parsed = parseJsonField(trimmed);
    if (Array.isArray(parsed)) {
      return toStatements(parsed);
    }
    return [trimmed];
  }
  return [];
};

export const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

export const endsWithSentenceBoundary = (text: unknown): boolean =>
  SENTENCE_BOUNDARY.test(asText(text).trim());
