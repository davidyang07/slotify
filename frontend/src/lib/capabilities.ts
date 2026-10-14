/**
 * What the backend can do, fetched once so the UI can disable rather than fail.
 *
 * The important consequence: with no ElevenLabs key the app still uploads,
 * analyses and ranks. Only ad generation is switched off, and it says why.
 * When the capabilities cannot be read at all, that is reported as what it is
 * (the backend is unreachable, or rejected the request), not as a missing key.
 */

import { readApiError, type ApiFetch } from "./api";

export interface Capabilities {
  placement: boolean;
  voiceCloning: boolean;
  tts: boolean;
  openaiEnhancement: boolean;
  learnedRanker: boolean;
  /** The API wants `Authorization: Bearer <token>`. Older servers omit it. */
  authRequired: boolean;
  ranker: {
    requested: string;
    active: "learned" | "heuristic";
    reason: string;
    baselineVersion: string;
    heuristicConfigVersion: string;
  };
  /**
   * Client-side only: why the capabilities could not be read (backend down,
   * CORS, a rejected token), or null when they were.
   */
  loadError: string | null;
}

/**
 * The assumption to hold before /api/capabilities answers: placement may work,
 * generation may not. Optimism about a paid API is what produces a button that
 * fails on click.
 */
export const UNKNOWN_CAPABILITIES: Capabilities = {
  placement: true,
  voiceCloning: false,
  tts: false,
  openaiEnhancement: false,
  learnedRanker: false,
  authRequired: false,
  ranker: {
    requested: "auto",
    active: "heuristic",
    reason: "Capabilities have not been fetched yet.",
    baselineVersion: "heuristic_offline_v1",
    heuristicConfigVersion: "",
  },
  loadError: null,
};

export const BACKEND_UNREACHABLE_MESSAGE =
  "Can't reach the Slotify backend, so analysis and ad generation are unavailable. Start the API server (or check VITE_API_BASE_URL) and reload.";

/** The capabilities to hold when they could not be read, and why. */
export const unreadableCapabilities = (loadError: string): Capabilities => ({
  ...UNKNOWN_CAPABILITIES,
  loadError,
});

/** Read a capabilities body, tolerating fields an older server omits. */
export const parseCapabilities = (data: unknown): Capabilities => {
  const payload = (data ?? {}) as Partial<Capabilities>;
  return {
    ...UNKNOWN_CAPABILITIES,
    ...payload,
    ranker: { ...UNKNOWN_CAPABILITIES.ranker, ...payload.ranker },
    authRequired: payload.authRequired === true,
    loadError: null,
  };
};

/**
 * Never throws for an unreachable or rejecting backend; the answer carries a
 * `loadError` instead, so the UI can say which of the two happened.
 */
export const fetchCapabilities = async (
  apiFetch: ApiFetch,
): Promise<Capabilities> => {
  let response: Response;
  try {
    response = await apiFetch("/api/capabilities");
  } catch {
    // A network failure or a CORS rejection: the browser cannot tell them apart.
    return unreadableCapabilities(BACKEND_UNREACHABLE_MESSAGE);
  }
  if (!response.ok) {
    // 502-504 come from a proxy in front of a backend that is not answering;
    // a 404 means whatever answered has no Slotify API behind it.
    const fallback =
      response.status === 404 ||
      (response.status >= 502 && response.status <= 504)
        ? BACKEND_UNREACHABLE_MESSAGE
        : `The backend refused the capabilities check (HTTP ${response.status}).`;
    return unreadableCapabilities(await readApiError(response, fallback));
  }
  const data: unknown = await response.json().catch(() => null);
  if (!data || typeof data !== "object") {
    // Something answered, but not the Slotify API (e.g. a static host's
    // index.html fallback).
    return unreadableCapabilities(BACKEND_UNREACHABLE_MESSAGE);
  }
  return parseCapabilities(data);
};

/** Why ad generation is unavailable, or null when it is available. */
export const generationBlockedReason = (
  capabilities: Capabilities,
): string | null => {
  if (capabilities.loadError) return capabilities.loadError;
  if (capabilities.tts && capabilities.voiceCloning) return null;
  return "Ad generation needs an ElevenLabs API key on the server. Placement analysis works without one.";
};
