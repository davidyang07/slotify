// Pure text/value helpers shared across routes.

export const parseJsonField = (value: unknown): any => {
  if (!value || typeof value !== "string") return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

export const normalizeStatements = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value
      .map((entry) => (entry == null ? "" : String(entry).trim()))
      .filter(Boolean);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return [];
    const parsed = parseJsonField(trimmed);
    if (Array.isArray(parsed)) {
      return parsed
        .map((entry) => (entry == null ? "" : String(entry).trim()))
        .filter(Boolean);
    }
    return [trimmed];
  }
  return [];
};

export const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

export const endsWithSentenceBoundary = (text: unknown): boolean =>
  /[.!?]["')\]]?\s*$/.test(String(text ?? "").trim());
