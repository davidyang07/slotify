import type { ErrorRequestHandler } from "express";
import multer from "multer";
import { describeFailure, providerOf } from "../lib/errors";
import { HttpError } from "../lib/http";

/** Raised by the CORS origin check; becomes a 403 rather than a 500. */
export class CorsRejectedError extends Error {
  constructor() {
    super("Origin not allowed by CORS.");
    this.name = "CorsRejectedError";
  }
}

const MULTER_MESSAGES: Partial<Record<multer.ErrorCode, string>> = {
  LIMIT_FILE_SIZE: "Uploaded file is too large (see SLOTIFY_MAX_UPLOAD_MB).",
  LIMIT_FIELD_VALUE: "A form field is too large.",
  LIMIT_FILE_COUNT: "Too many files in one upload.",
  LIMIT_FIELD_COUNT: "Too many form fields.",
  LIMIT_PART_COUNT: "Too many parts in the multipart upload.",
  LIMIT_FIELD_KEY: "A form field name is too long.",
  LIMIT_UNEXPECTED_FILE: "Unexpected file field (or too many files for it).",
};

/**
 * The last middleware. Every error becomes JSON `{ error, requestId }`
 * (requestId is added by lib/request-context.ts); nothing reaches the client
 * as an HTML page, a stack trace, a server path or a provider's response.
 */
export const errorHandler: ErrorRequestHandler = (error, req, res, _next) => {
  if (res.headersSent) {
    // Mid-stream: there is no status left to send, only a connection to end.
    console.error(`Error after response started (${req.method} ${req.path}):`, error);
    res.destroy();
    return;
  }
  if (error instanceof CorsRejectedError) {
    res.status(403).json({ error: error.message });
    return;
  }
  if (error instanceof HttpError) {
    for (const [name, value] of Object.entries(error.headers)) res.setHeader(name, value);
    res.status(error.status).json({ error: error.message });
    return;
  }
  if (providerOf(error)) {
    const failure = describeFailure("Request failed.", error);
    res.status(failure.status).json(failure.body);
    return;
  }
  if (error instanceof multer.MulterError) {
    const status =
      error.code === "LIMIT_FILE_SIZE" || error.code === "LIMIT_FIELD_VALUE" ? 413 : 400;
    res.status(status).json({ error: MULTER_MESSAGES[error.code] ?? "Invalid upload." });
    return;
  }
  // body-parser marks its own errors with a 4xx status and a type.
  const status = typeof error?.status === "number" ? error.status : 500;
  if (status >= 400 && status < 500) {
    const message =
      error?.type === "entity.parse.failed"
        ? "Request body is not valid JSON."
        : error?.type === "entity.too.large"
          ? "Request body is too large."
          : "Bad request.";
    res.status(status).json({ error: message });
    return;
  }
  console.error(`Unhandled error (${req.method} ${req.path}):`, error);
  res.status(500).json({ error: "Internal server error." });
};
