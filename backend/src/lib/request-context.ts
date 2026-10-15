/**
 * Request ids, so one request can be followed through the log.
 *
 * Each request gets an id -- the caller's `X-Request-Id` when it is a sane
 * token, otherwise a fresh UUID -- which is echoed in the response header, put
 * on every error body as `requestId`, and prefixed to every console line logged
 * while the request is being handled. The prefix comes from AsyncLocalStorage,
 * so code that logs (including services this module never sees) needs no
 * changes: a client quoting the id from an error is enough to find the full,
 * unredacted failure in the server log. (The body parsers keep the context
 * across their stream events: raw-body and multer's on-finished bind their
 * callbacks; request-context.test.ts checks it.)
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import type { RequestHandler, Response } from "express";

interface RequestContext {
  requestId: string;
}

const storage = new AsyncLocalStorage<RequestContext>();

// Long enough for a UUID or a trace id; nothing that could smuggle a newline
// or a fake log prefix into the log.
const CLIENT_ID_PATTERN = /^[A-Za-z0-9._:-]{1,128}$/;

/** The id of the request being handled, or undefined outside one. */
export const currentRequestId = (): string | undefined => storage.getStore()?.requestId;

/** The id assigned to `res`'s request (set by `requestIdMiddleware`). */
export const requestIdOf = (res: Response): string | undefined =>
  typeof res.locals.requestId === "string" ? res.locals.requestId : undefined;

export const requestIdMiddleware: RequestHandler = (req, res, next) => {
  const presented = req.get("x-request-id")?.trim() ?? "";
  const requestId = CLIENT_ID_PATTERN.test(presented) ? presented : randomUUID();
  res.locals.requestId = requestId;
  res.setHeader("X-Request-Id", requestId);

  // Every error body carries the id, whichever route or middleware sent it.
  const json = res.json.bind(res);
  res.json = (body?: unknown) => {
    if (
      res.statusCode >= 400 &&
      body !== null &&
      typeof body === "object" &&
      !Array.isArray(body) &&
      !("requestId" in body)
    ) {
      return json({ ...body, requestId });
    }
    return json(body);
  };

  storage.run({ requestId }, () => next());
};

let consolePatched = false;

/** Prefix console output made during a request with `[req <id>]`. Idempotent. */
export const installRequestLogPrefix = (): void => {
  if (consolePatched) return;
  consolePatched = true;
  for (const method of ["log", "info", "warn", "error", "debug"] as const) {
    const original = console[method].bind(console);
    console[method] = (...args: unknown[]) => {
      const requestId = currentRequestId();
      if (requestId) original(`[req ${requestId}]`, ...args);
      else original(...args);
    };
  }
};
