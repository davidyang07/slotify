import type { Request, RequestHandler } from "express";
import { isInternalRequest, routeKey } from "./auth";

export type RateLimitBucket = "paid" | "compute";

// Each of these can spend OpenAI or ElevenLabs credit on every call.
const PAID_ROUTES = new Set([
  "POST /api/tts",
  "POST /api/clone",
  "POST /api/generate",
  "POST /ad/insert",
  "POST /api/related-products",
]);

/** Which budget a request draws on, or null for unlimited (reads). */
export const rateLimitBucket = (req: Request): RateLimitBucket | null => {
  if (req.method !== "POST") return null;
  const key = routeKey(req);
  if (PAID_ROUTES.has(key)) return "paid";
  // Placement is credential-free, but with an OpenAI key it also transcribes,
  // writes up to three sponsor reads and narrates the slots.
  if (key === "POST /api/insert-sections" && process.env.OPENAI_API_KEY) return "paid";
  return "compute";
};

export interface RateLimitOptions {
  /** Requests per client IP per window; 0 disables the bucket. */
  paidPerMin: number;
  computePerMin: number;
  windowMs?: number;
  now?: () => number;
}

/**
 * Per-IP fixed-window limits. In memory, so per process: enough to stop one
 * client from spending the operator's API credit in a loop, which is the job.
 * Callbacks from our own Python children are exempt.
 */
export const createRateLimiter = ({
  paidPerMin,
  computePerMin,
  windowMs = 60_000,
  now = Date.now,
}: RateLimitOptions): RequestHandler => {
  const limits: Record<RateLimitBucket, number> = { paid: paidPerMin, compute: computePerMin };
  const windows = new Map<string, { start: number; count: number }>();
  let lastSweep = now();

  return (req, res, next) => {
    const bucket = rateLimitBucket(req);
    if (!bucket || limits[bucket] <= 0 || isInternalRequest(req)) {
      next();
      return;
    }

    const current = now();
    if (current - lastSweep >= windowMs) {
      // Expired windows would otherwise accumulate one entry per client forever.
      for (const [key, window] of windows) {
        if (current - window.start >= windowMs) windows.delete(key);
      }
      lastSweep = current;
    }

    const key = `${bucket}:${req.ip ?? req.socket.remoteAddress ?? "unknown"}`;
    let window = windows.get(key);
    if (!window || current - window.start >= windowMs) {
      window = { start: current, count: 0 };
      windows.set(key, window);
    }
    window.count += 1;
    if (window.count <= limits[bucket]) {
      next();
      return;
    }

    const retryAfter = Math.max(1, Math.ceil((window.start + windowMs - current) / 1000));
    res.setHeader("Retry-After", String(retryAfter));
    res.status(429).json({
      error:
        `Rate limit exceeded: at most ${limits[bucket]} ${bucket === "paid" ? "paid-API" : "processing"} ` +
        `requests per minute per client. Try again in ${retryAfter} s.`,
    });
  };
};
