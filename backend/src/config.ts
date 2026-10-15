import { randomBytes } from "node:crypto";
import os from "node:os";
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
 * Per-process secret handed as SLOTIFY_INTERNAL_TOKEN only to the Python
 * pipelines that call back into this API for TTS, merge and clone
 * (`ad_inserter.cli`, `ad_inserter.insert_ad`; see `childEnv` in
 * lib/subprocess.ts). Presenting it lets those callbacks through auth, rate
 * limiting and the job limiter. The analyser, ffmpeg and the learned ranker do
 * not get it, and no child ever gets the operator's SLOTIFY_API_TOKEN.
 */
export const internalToken = randomBytes(32).toString("hex");

export const maxUploadBytes =
  Math.round((numberFromEnv("SLOTIFY_MAX_UPLOAD_MB", 200) || 200) * 1024 * 1024);

export const ffmpegTimeoutMs = numberFromEnv("SLOTIFY_FFMPEG_TIMEOUT_MS", 120_000) || 120_000;
export const analyserTimeoutMs =
  numberFromEnv("SLOTIFY_ANALYSER_TIMEOUT_MS", 300_000) || 300_000;
export const pipelineTimeoutMs =
  numberFromEnv("SLOTIFY_PIPELINE_TIMEOUT_MS", 600_000) || 600_000;

/**
 * SIGTERM, then this long for the child to run its own cleanup (Python
 * `finally` blocks remove their temp files), then SIGKILL. 0 kills at once.
 */
export const killGraceMs = numberFromEnv("SLOTIFY_KILL_GRACE_MS", 5_000);

/**
 * Heavy jobs (analysis, ranking, pipelines, merges) running at once. Each holds
 * a decoded copy of an upload in a Python or ffmpeg process, so this, not the
 * request rate, is what bounds memory.
 */
export const maxConcurrentJobs =
  Math.floor(numberFromEnv("SLOTIFY_MAX_CONCURRENT_JOBS", 0)) ||
  Math.max(1, Math.floor(os.availableParallelism() / 2));
/** Heavy jobs allowed to wait for a slot; beyond this a request gets a 503. */
export const maxQueuedJobs = Math.floor(numberFromEnv("SLOTIFY_MAX_QUEUED_JOBS", 8));

/** Longest upload, by decoded duration, any route will process. */
export const maxAudioSeconds =
  (numberFromEnv("SLOTIFY_MAX_AUDIO_MINUTES", 180) || 180) * 60;

/**
 * Song analysis is far hungrier: librosa's beat tracking holds about 180 MB
 * per minute of audio (a 15-minute song peaked at 2.9 GB), against ~143 MB for
 * an hour of podcast. So song mode has its own, much lower, cap.
 */
export const maxSongSeconds = (numberFromEnv("SLOTIFY_MAX_SONG_MINUTES", 10) || 10) * 60;

/**
 * Voice-clone samples. The UI sends the whole episode, which streams to disk
 * like any upload, so a sample may be as large as any upload; what reaches
 * ElevenLabs is a trimmed excerpt of SLOTIFY_CLONE_SAMPLE_SECONDS (IVC needs a
 * minute or two of speech, and more does not help).
 */
export const maxCloneSamples = Math.floor(numberFromEnv("SLOTIFY_MAX_CLONE_SAMPLES", 3)) || 3;
export const maxCloneSampleBytes =
  Math.round(numberFromEnv("SLOTIFY_MAX_CLONE_SAMPLE_MB", 0) * 1024 * 1024) || maxUploadBytes;
export const cloneSampleSeconds = numberFromEnv("SLOTIFY_CLONE_SAMPLE_SECONDS", 180) || 180;

/** Cloned voices idle this long are deleted from the ElevenLabs account. */
export const voiceTtlMs = (numberFromEnv("SLOTIFY_VOICE_TTL_MIN", 30) || 30) * 60_000;
/** Cloned voices kept at once; the least recently used is deleted first. */
export const maxLiveVoices = Math.floor(numberFromEnv("SLOTIFY_MAX_LIVE_VOICES", 3)) || 3;

/**
 * /api/tts spend per request. Every statement is a separate paid ElevenLabs
 * call, so both the count and the total text are capped before the first one.
 */
export const ttsMaxStatements = Math.floor(numberFromEnv("SLOTIFY_TTS_MAX_STATEMENTS", 8)) || 8;
export const ttsMaxChars = Math.floor(numberFromEnv("SLOTIFY_TTS_MAX_CHARS", 1_500)) || 1_500;

/**
 * OpenAI models a client may ask the /ad/insert pipeline for. Read per call so
 * tests can change it; `ad_inserter` enforces the same list.
 */
export const allowedOpenAiModels = (): string[] => {
  const models = (process.env.SLOTIFY_OPENAI_MODELS ?? "")
    .split(",")
    .map((model) => model.trim())
    .filter(Boolean);
  return models.length ? models : ["gpt-4o-mini"];
};

/** Requests per client IP per minute; 0 disables that bucket. */
export const rateLimitsFromEnv = (): { paidPerMin: number; computePerMin: number } => ({
  paidPerMin: Math.floor(numberFromEnv("SLOTIFY_RATE_LIMIT_PAID_PER_MIN", 10)),
  computePerMin: Math.floor(numberFromEnv("SLOTIFY_RATE_LIMIT_COMPUTE_PER_MIN", 30)),
});
