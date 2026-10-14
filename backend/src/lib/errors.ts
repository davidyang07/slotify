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

/**
 * Log a failure in full and return the part of it the client may see. A
 * subprocess that exited non-zero is described by its stderr; one that timed
 * out or was stopped is described by our own message, which says why.
 */
export const logFailure = (context: string, error: unknown): string => {
  if (error instanceof SubprocessError) {
    console.error(context, error.message, error.stderr);
    const detail = error.reason === "exit" ? clientSafeDetail(error.stderr) : "";
    return detail || clientSafeDetail(error.message);
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
