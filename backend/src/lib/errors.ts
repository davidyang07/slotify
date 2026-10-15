import type { Response } from "express";
import { ElevenLabsError, ElevenLabsTimeoutError } from "@elevenlabs/elevenlabs-js";
import { ClientDisconnectedError, HttpError } from "./http";
import { SubprocessError } from "./subprocess";

const MAX_DETAIL_LENGTH = 300;

// Traceback scaffolding is never the error itself.
const NOISE = [
  /^Traceback \(most recent call last\):$/,
  /^File ".*", line \d+/,
  /^During handling of the above exception/,
  /^The above exception was the direct cause/,
  /^[\^~\s]+$/,
];

// Absolute POSIX or Windows paths: temp dirs, the checkout, site-packages.
// This API's own routes (/api/..., /ad/...) are not filesystem paths.
const PATH_PATTERN = /(?:[A-Za-z]:)?(?!\/(?:api|ad)\/)(?:[\\/][^\s\\/'"():,;]+){2,}[\\/]?/g;

/**
 * The last meaningful line of a subprocess's output (or every line, for a
 * short message), with filesystem paths replaced, trimmed to one short
 * string. A Python traceback or ffmpeg log is mostly the repository's
 * internal layout; the client gets the actual error and the server log keeps
 * the whole thing.
 */
export const clientSafeDetail = (text: string, { lastLineOnly = true } = {}): string => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !NOISE.some((pattern) => pattern.test(line)));
  const kept = (lastLineOnly ? lines.slice(-1) : lines).join(" ").replace(PATH_PATTERN, "<path>");
  return kept.length > MAX_DETAIL_LENGTH
    ? `${kept.slice(0, MAX_DETAIL_LENGTH - 1)}…`
    : kept;
};

export type Provider = "elevenlabs" | "openai" | "serpapi";

/** What a client is told when a provider fails. Never the provider's own words. */
export const PROVIDER_MESSAGES: Record<Provider, string> = {
  elevenlabs: "The voice provider rejected the request.",
  openai: "The language-model provider rejected the request.",
  serpapi: "The product-search provider rejected the request.",
};

/**
 * A third-party API answered with an error. `detail` (its response body) is for
 * the server log only: provider bodies carry request ids, account state and
 * messages such as "Invalid API key" that are not the client's business.
 */
export class ProviderError extends Error {
  constructor(
    readonly provider: Provider,
    readonly status: number | null,
    readonly detail = "",
  ) {
    super(`${provider} request failed${status !== null ? ` with HTTP ${status}` : ""}.`);
    this.name = "ProviderError";
  }
}

/** The provider an error came from, or null when it is not a provider error. */
export const providerOf = (error: unknown): Provider | null => {
  if (error instanceof ProviderError) return error.provider;
  if (error instanceof ElevenLabsError || error instanceof ElevenLabsTimeoutError) return "elevenlabs";
  return null;
};

// A pipeline's last stderr line is often a provider SDK's exception, e.g.
// "openai.AuthenticationError: Error code: 401 - {'error': {'message':
// 'Incorrect API key provided: sk-...'". Anything that looks like one is
// replaced, not trimmed: the safe part of such a line is the part we can say.
const PROVIDER_LEAK =
  /\bopenai\.\w+|\belevenlabs\b|api[ _-]?key|request[ _-]?id|\bquota\b|insufficient_quota|billing|error code: \d{3}|status(?:[ _]code)?[ :=]+[45]\d\d|\b[45]\d\d\b[^{]*\{|\bsk-[A-Za-z0-9]/i;
const UPSTREAM_MESSAGE = "An upstream provider rejected the request.";

/**
 * Log a failure in full and return the part of it the client may see. A
 * subprocess that exited non-zero is described by its stderr; one that timed
 * out or was stopped is described by our own message, which says why.
 */
export const logFailure = (context: string, error: unknown): string => {
  if (error instanceof SubprocessError) {
    console.error(context, error.message, error.stderr);
    const detail = error.reason === "exit" ? clientSafeDetail(error.stderr) : "";
    if (detail && PROVIDER_LEAK.test(detail)) return UPSTREAM_MESSAGE;
    return detail || clientSafeDetail(error.message);
  }
  const provider = providerOf(error);
  if (provider) {
    console.error(
      context,
      error instanceof ProviderError ? `${error.message} ${error.detail}` : error,
    );
    return PROVIDER_MESSAGES[provider];
  }
  console.error(context, error);
  // Our own and SDK errors are short; keep all of it, minus any paths.
  const message =
    error instanceof Error ? error.message : typeof error === "string" ? error : "Unknown error.";
  return clientSafeDetail(message, { lastLineOnly: false });
};

/**
 * A JSON error body for a failed request: one sentence the UI shows as is,
 * with the client-safe detail also on its own for programmatic callers.
 */
export const failureBody = (
  summary: string,
  error: unknown,
): { error: string; detail: string } => {
  const detail = logFailure(summary, error);
  return { error: detail ? `${summary} ${detail}` : summary, detail };
};

export interface Failure {
  status: number;
  body: { error: string; detail?: string };
  headers: Readonly<Record<string, string>>;
}

/**
 * The status, body and headers for a failed request. A request-shaped problem
 * keeps its own status (4xx, or a 503 with Retry-After); a provider failure is
 * a 502 with a generic sentence; a server shutting down under the request is a
 * 503; anything else is a 500 with the client-safe detail.
 */
export const describeFailure = (summary: string, error: unknown): Failure => {
  if (error instanceof HttpError) {
    return { status: error.status, body: { error: error.message }, headers: error.headers };
  }
  const body = failureBody(summary, error);
  if (providerOf(error)) return { status: 502, body, headers: {} };
  if (error instanceof SubprocessError && error.reason === "shutdown") {
    return { status: 503, body, headers: { "Retry-After": "10" } };
  }
  return { status: 500, body, headers: {} };
};

/**
 * Answer a failed request. Nothing is sent to a client that has gone; a
 * response already streaming can only be cut off.
 */
export const sendFailure = (
  res: Response,
  summary: string,
  error: unknown,
  { signal, extra = {} }: { signal?: AbortSignal; extra?: Record<string, unknown> } = {},
): void => {
  if (error instanceof ClientDisconnectedError || signal?.aborted) {
    logFailure(summary, error);
    if (!res.writableEnded) res.destroy();
    return;
  }
  if (res.headersSent) {
    logFailure(summary, error);
    res.destroy();
    return;
  }
  const failure = describeFailure(summary, error);
  for (const [name, value] of Object.entries(failure.headers)) res.setHeader(name, value);
  res.status(failure.status).json({ ...extra, ...failure.body });
};
