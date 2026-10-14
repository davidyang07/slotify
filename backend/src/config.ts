import { randomBytes } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Backend project root (the directory that contains the `ad_inserter` Python
// package). All Python subprocesses are spawned with this as their cwd so that
// `python -m ad_inserter.*` resolves.
export const BACKEND_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);

export const port = Number.parseInt(process.env.PORT ?? "3001", 10);

export const allowedOrigins = (
  process.env.CORS_ORIGIN ?? "http://localhost:5173"
)
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

export const pythonBin = process.env.PYTHON_BIN ?? "python";

// Base URL the Python pipeline uses to call back into this API (for TTS/merge).
export const apiBaseUrl =
  process.env.API_BASE_URL ?? `http://localhost:${port}`;

/**
 * A non-negative number from the environment, or `fallback` when unset. A value
 * that is set but unparseable is a startup error, not a silent default: a typo
 * in a limit should not quietly remove the limit.
 */
const numberFromEnv = (name: string, fallback: number): number => {
  const raw = process.env[name]?.trim();
  if (!raw) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a non-negative number, got ${JSON.stringify(raw)}.`);
  }
  return value;
};

/** Optional shared secret for every route except health and capabilities. */
export const configuredApiToken = (): string | null =>
  process.env.SLOTIFY_API_TOKEN?.trim() || null;

/**
 * Per-process secret handed to every Python child as SLOTIFY_INTERNAL_TOKEN.
 * The pipelines call back into this API for TTS and merge; presenting it lets
 * those callbacks through auth and rate limiting without the operator's token
 * ever reaching a subprocess.
 */
export const internalToken = randomBytes(32).toString("hex");

export const maxUploadBytes =
  Math.round((numberFromEnv("SLOTIFY_MAX_UPLOAD_MB", 200) || 200) * 1024 * 1024);

export const ffmpegTimeoutMs = numberFromEnv("SLOTIFY_FFMPEG_TIMEOUT_MS", 120_000) || 120_000;
export const analyserTimeoutMs =
  numberFromEnv("SLOTIFY_ANALYSER_TIMEOUT_MS", 300_000) || 300_000;
export const pipelineTimeoutMs =
  numberFromEnv("SLOTIFY_PIPELINE_TIMEOUT_MS", 600_000) || 600_000;

/** Requests per client IP per minute; 0 disables that bucket. */
export const rateLimitsFromEnv = (): { paidPerMin: number; computePerMin: number } => ({
  paidPerMin: Math.floor(numberFromEnv("SLOTIFY_RATE_LIMIT_PAID_PER_MIN", 10)),
  computePerMin: Math.floor(numberFromEnv("SLOTIFY_RATE_LIMIT_COMPUTE_PER_MIN", 30)),
});
