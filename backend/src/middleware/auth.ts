import { createHash, timingSafeEqual } from "node:crypto";
import type { Request, RequestHandler } from "express";
import { configuredApiToken, internalToken } from "../config";

// Readable without a token so the UI can say what is wrong before it fails.
const PUBLIC_ROUTES = new Set(["GET /api/health", "GET /api/capabilities"]);

/** Express matches routes case-insensitively and ignores a trailing slash. */
export const routeKey = (req: Request): string =>
  `${req.method} ${req.path.toLowerCase().replace(/\/+$/, "") || "/"}`;

const bearerToken = (req: Request): string | null => {
  const match = /^Bearer\s+(\S+)\s*$/i.exec(req.headers.authorization ?? "");
  return match ? match[1] : null;
};

// Hashing first makes the comparison length-independent as well as
// constant-time.
const digest = (value: string): Buffer => createHash("sha256").update(value).digest();
export const tokensMatch = (presented: string, expected: string): boolean =>
  timingSafeEqual(digest(presented), digest(expected));

/** A callback from one of our own Python children (see lib/subprocess.ts). */
export const isInternalRequest = (req: Request): boolean => {
  const presented = bearerToken(req);
  return presented !== null && tokensMatch(presented, internalToken);
};

/**
 * SLOTIFY_API_TOKEN, when set, is required on every route but health and
 * capabilities. Unset, the API is open, as the local demo always was.
 */
export const requireApiToken: RequestHandler = (req, res, next) => {
  const expected = configuredApiToken();
  if (!expected || PUBLIC_ROUTES.has(routeKey(req)) || isInternalRequest(req)) {
    next();
    return;
  }
  const presented = bearerToken(req);
  if (presented && tokensMatch(presented, expected)) {
    next();
    return;
  }
  res.setHeader("WWW-Authenticate", 'Bearer realm="slotify"');
  res.status(401).json({
    error: presented
      ? "Invalid API token."
      : "This API requires a token: send Authorization: Bearer <SLOTIFY_API_TOKEN>.",
  });
};
