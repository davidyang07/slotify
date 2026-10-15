/**
 * A global cap on heavy work: analysis, ranking, the ad pipelines and merges.
 *
 * Each of those decodes an upload into a Python or ffmpeg process -- an hour
 * of stereo audio is over a gigabyte as float samples -- and nothing bounded
 * how many ran at once, so a handful of concurrent uploads could exhaust
 * memory. Jobs beyond `maxConcurrent` wait in a short queue; beyond that the
 * request is refused with 503 and Retry-After, which a client can act on,
 * instead of every job slowing down until they all time out.
 */

import type { RequestHandler } from "express";
import { maxConcurrentJobs, maxQueuedJobs } from "../config";
import { isInternalRequest } from "../middleware/auth";
import { ClientDisconnectedError, HttpError } from "./http";
import { isShuttingDown } from "./lifecycle";

/** Seconds a refused client is told to wait: about one short analysis. */
export const BUSY_RETRY_AFTER_SECONDS = 30;
const SHUTDOWN_RETRY_AFTER_SECONDS = 10;

export const serverBusyError = () =>
  new HttpError(
    503,
    "The server is busy with other audio jobs. Try again shortly.",
    { "Retry-After": String(BUSY_RETRY_AFTER_SECONDS) },
  );

export const shuttingDownError = () =>
  new HttpError(503, "The server is restarting. Try again shortly.", {
    "Retry-After": String(SHUTDOWN_RETRY_AFTER_SECONDS),
  });

export interface JobLimiterOptions {
  maxConcurrent: number;
  maxQueued: number;
}

interface Waiter {
  start: () => void;
  fail: (error: Error) => void;
}

export class JobLimiter {
  private active = 0;
  private readonly queue: Waiter[] = [];
  readonly maxConcurrent: number;
  readonly maxQueued: number;

  constructor({ maxConcurrent, maxQueued }: JobLimiterOptions) {
    this.maxConcurrent = Math.max(1, Math.floor(maxConcurrent));
    this.maxQueued = Math.max(0, Math.floor(maxQueued));
  }

  get running(): number {
    return this.active;
  }

  get waiting(): number {
    return this.queue.length;
  }

  /** True when a new job would be refused rather than run or queued. */
  isFull(): boolean {
    return this.active >= this.maxConcurrent && this.queue.length >= this.maxQueued;
  }

  /**
   * Run `job` when a slot is free. Rejects with a 503 `HttpError` when the
   * queue is full or the server is shutting down, and with
   * `ClientDisconnectedError` if `signal` aborts while the job is still
   * waiting -- a queued job for a client that has gone gives up its place.
   */
  async run<T>(job: () => Promise<T>, { signal }: { signal?: AbortSignal } = {}): Promise<T> {
    const release = await this.lease({ signal });
    try {
      return await job();
    } finally {
      release();
    }
  }

  /**
   * Take a slot (same rules as `run`) and get back the function that frees
   * it, for work that does not fit one callback. Calling it twice is harmless.
   */
  async lease({ signal }: { signal?: AbortSignal } = {}): Promise<() => void> {
    await this.acquire(signal);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.release();
    };
  }

  private acquire(signal?: AbortSignal): Promise<void> {
    if (isShuttingDown()) return Promise.reject(shuttingDownError());
    if (signal?.aborted) return Promise.reject(new ClientDisconnectedError("A queued job"));
    if (this.active < this.maxConcurrent) {
      this.active += 1;
      return Promise.resolve();
    }
    if (this.queue.length >= this.maxQueued) return Promise.reject(serverBusyError());

    return new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        const index = this.queue.indexOf(waiter);
        if (index !== -1) this.queue.splice(index, 1);
        reject(new ClientDisconnectedError("A queued job"));
      };
      const waiter: Waiter = {
        start: () => {
          signal?.removeEventListener("abort", onAbort);
          resolve();
        },
        fail: (error) => {
          signal?.removeEventListener("abort", onAbort);
          reject(error);
        },
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      this.queue.push(waiter);
    });
  }

  private release(): void {
    // Shutdown refuses whatever is still waiting rather than starting it.
    if (isShuttingDown()) {
      for (const waiter of this.queue.splice(0)) waiter.fail(shuttingDownError());
    }
    const next = this.queue.shift();
    if (next) {
      // The slot passes straight to the next job; `active` is unchanged.
      next.start();
      return;
    }
    this.active -= 1;
  }
}

/** The process-wide limiter (SLOTIFY_MAX_CONCURRENT_JOBS / SLOTIFY_MAX_QUEUED_JOBS). */
export const heavyJobs = new JobLimiter({
  maxConcurrent: maxConcurrentJobs,
  maxQueued: maxQueuedJobs,
});

/**
 * Callbacks from our own pipelines (their TTS and merge requests) are part of
 * a job that already holds a slot. Making them queue too would deadlock a
 * full limiter: the pipeline would wait on a merge that waits on the pipeline.
 */
export const runHeavyJob = <T>(
  req: Parameters<typeof isInternalRequest>[0],
  job: () => Promise<T>,
  { signal, limiter = heavyJobs }: { signal?: AbortSignal; limiter?: JobLimiter } = {},
): Promise<T> => (isInternalRequest(req) ? job() : limiter.run(job, { signal }));

/** `runHeavyJob` as a lease: resolves to the function that frees the slot. */
export const leaseHeavyJob = async (
  req: Parameters<typeof isInternalRequest>[0],
  { signal, limiter = heavyJobs }: { signal?: AbortSignal; limiter?: JobLimiter } = {},
): Promise<() => void> => (isInternalRequest(req) ? () => undefined : limiter.lease({ signal }));

/**
 * Refuse a heavy request before its upload is read when it would only be
 * refused afterwards: a full queue or a shutdown costs the client a 503, not
 * a 200 MB upload and then a 503.
 */
export const refuseWhenBusy =
  ({
    limiter = heavyJobs,
    extra = {},
  }: {
    limiter?: JobLimiter;
    /** Fields the route's own error bodies carry (insert-sections' placementStatus). */
    extra?: Record<string, unknown>;
  } = {}): RequestHandler =>
  (req, res, next) => {
    const refusal = isInternalRequest(req)
      ? null
      : isShuttingDown()
        ? shuttingDownError()
        : limiter.isFull()
          ? serverBusyError()
          : null;
    if (!refusal) {
      next();
      return;
    }
    for (const [name, value] of Object.entries(refusal.headers)) res.setHeader(name, value);
    // The upload is never read; closing the connection stops the client
    // sending the rest of it.
    res.setHeader("Connection", "close");
    res.status(refusal.status).json({ ...extra, error: refusal.message });
  };
