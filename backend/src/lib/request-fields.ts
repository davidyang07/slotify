/**
 * Typed reads of request fields. Each returns undefined when the field is
 * absent and throws a 400 `HttpError` when it is present with the wrong shape.
 *
 * Multer turns `name[key]=v` into an object and repeated fields into arrays,
 * and JSON bodies can carry anything, so `req.body.x?.toString()` is not a
 * string check -- `{"toString": 1}` made it throw.
 */

import { HttpError } from "./http";
import { normalizeStatements } from "./text";

const DEFAULT_MAX_LENGTH = 5_000;

export const fieldOf = (body: unknown, key: string): unknown =>
  body !== null && typeof body === "object" && Object.prototype.hasOwnProperty.call(body, key)
    ? (body as Record<string, unknown>)[key]
    : undefined;

/** A trimmed string; blank counts as absent. */
export const optionalString = (
  body: unknown,
  key: string,
  maxLength = DEFAULT_MAX_LENGTH,
): string | undefined => {
  const value = fieldOf(body, key);
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string") throw new HttpError(400, `${key} must be a string.`);
  const trimmed = value.trim();
  if (trimmed.length > maxLength) {
    throw new HttpError(400, `${key} must be at most ${maxLength} characters.`);
  }
  return trimmed || undefined;
};

/** A finite number, from a JSON number or a numeric form string. */
export const optionalNumber = (body: unknown, key: string): number | undefined => {
  const value = fieldOf(body, key);
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string" && !value.trim()) return undefined;
  const parsed =
    typeof value === "number" ? value : typeof value === "string" ? Number(value.trim()) : NaN;
  if (!Number.isFinite(parsed)) throw new HttpError(400, `${key} must be a number.`);
  return parsed;
};

/** One of `allowed`, compared case-insensitively. */
export const optionalEnum = <T extends string>(
  body: unknown,
  key: string,
  allowed: readonly T[],
): T | undefined => {
  const value = optionalString(body, key, 100)?.toLowerCase();
  if (value === undefined) return undefined;
  if (!(allowed as readonly string[]).includes(value)) {
    throw new HttpError(400, `${key} must be one of: ${allowed.join(", ")}.`);
  }
  return value as T;
};

/** true/false as a JSON boolean or the form strings "true"/"false"/"1"/"0". */
export const optionalBoolean = (body: unknown, key: string): boolean | undefined => {
  const value = fieldOf(body, key);
  if (value === undefined || value === null) return undefined;
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (!normalized) return undefined;
    if (normalized === "true" || normalized === "1") return true;
    if (normalized === "false" || normalized === "0") return false;
  }
  throw new HttpError(400, `${key} must be true or false.`);
};

/**
 * Sponsor statements: a string, a JSON-array string (multipart), or an array of
 * strings (JSON). Empty entries are dropped.
 */
export const optionalStatements = (body: unknown, key: string): string[] | undefined => {
  const value = fieldOf(body, key);
  if (value === undefined || value === null) return undefined;
  const valid =
    typeof value === "string" ||
    (Array.isArray(value) && value.every((entry) => typeof entry === "string"));
  if (!valid) throw new HttpError(400, `${key} must be a string or an array of strings.`);
  const statements = normalizeStatements(value);
  if (statements.some((statement) => statement.length > DEFAULT_MAX_LENGTH)) {
    throw new HttpError(400, `Each ${key} entry must be at most ${DEFAULT_MAX_LENGTH} characters.`);
  }
  return statements;
};

/** A nested JSON object (e.g. `sponsor`), or undefined. */
export const optionalObject = (
  body: unknown,
  key: string,
): Record<string, unknown> | undefined => {
  const value = fieldOf(body, key);
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new HttpError(400, `${key} must be an object.`);
  }
  return value as Record<string, unknown>;
};
