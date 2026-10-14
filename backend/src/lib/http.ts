import type { NextFunction, Request, RequestHandler, Response } from "express";

/** An error whose message is safe to show the client, with the status to send. */
export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

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
