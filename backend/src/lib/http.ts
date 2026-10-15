import type { NextFunction, Request, RequestHandler, Response } from "express";

/**
 * An error whose message is safe to show the client, with the status to send
 * and any headers that go with it (a 503's Retry-After).
 */
export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly headers: Readonly<Record<string, string>> = {},
  ) {
    super(message);
    this.name = "HttpError";
  }
}

/** Work stopped because the client went away; there is no one to answer. */
export class ClientDisconnectedError extends Error {
  constructor(what = "The request") {
    super(`${what} was stopped because the client disconnected.`);
    this.name = "ClientDisconnectedError";
  }
}

/** Throw before starting paid or heavy work for a client that has gone. */
export const throwIfClientGone = (signal: AbortSignal | undefined, what?: string): void => {
  if (signal?.aborted) throw new ClientDisconnectedError(what);
};

/**
 * Express 4 ignores a rejected promise from a handler, and Node then exits on
 * the unhandled rejection: one malformed request took the whole API down.
 * Every async handler goes through this so a rejection reaches the JSON error
 * middleware instead.
 */
export const asyncHandler =
  (
    handler: (req: Request, res: Response, next: NextFunction) => Promise<unknown>,
  ): RequestHandler =>
  (req, res, next) => {
    handler(req, res, next).catch(next);
  };

/**
 * Aborts when the client disconnects before the response has been sent, so a
 * subprocess working for nobody is killed instead of running to its timeout.
 */
export const clientAbortSignal = (req: Request, res: Response): AbortSignal => {
  const controller = new AbortController();
  if (req.socket.destroyed) {
    controller.abort();
    return controller.signal;
  }
  res.once("close", () => {
    if (!res.writableFinished) controller.abort();
  });
  return controller.signal;
};

/**
 * Send a JSON body and resolve true once it has been handed to the socket, or
 * false if the client disconnected first. A route that created something for
 * this client (a cloned voice) uses this to know whether to delete it again.
 */
export const sendJsonConfirmed = (res: Response, body: unknown): Promise<boolean> =>
  new Promise((resolve) => {
    if (res.destroyed || res.socket?.destroyed) {
      resolve(false);
      return;
    }
    res.once("finish", () => resolve(true));
    res.once("close", () => resolve(res.writableFinished));
    res.json(body);
  });
