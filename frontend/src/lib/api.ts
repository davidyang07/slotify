/**
 * The one way the UI talks to the API.
 *
 * Every request goes through `createApiFetch`, so the base URL and the bearer
 * token are applied in one place instead of at each call site, and every
 * failed response is turned into a sentence by `readApiError`.
 *
 * No React and no `import.meta.env` here: App.tsx passes the env values in,
 * which keeps this module runnable under `node --test`.
 */

export interface ApiConfig {
  /** Prefix for `/api/...` paths. Empty means same-origin. */
  base: string;
  /** Sent as `Authorization: Bearer <token>` when non-empty. */
  token: string;
}

export type ApiFetch = (path: string, init?: RequestInit) => Promise<Response>;

const envString = (value: unknown): string =>
  typeof value === "string" ? value.trim() : "";

/**
 * Same-origin unless VITE_API_BASE_URL says otherwise. The old default,
 * http://localhost:3001, was baked into production builds that set nothing.
 */
export const resolveApiConfig = (env: {
  VITE_API_BASE_URL?: unknown;
  VITE_API_TOKEN?: unknown;
}): ApiConfig => ({
  base: envString(env.VITE_API_BASE_URL).replace(/\/+$/, ""),
  token: envString(env.VITE_API_TOKEN),
});

export const createApiFetch =
  (
    config: ApiConfig,
    fetchImpl: typeof fetch = (input, init) => fetch(input, init),
  ): ApiFetch =>
  (path, init = {}) => {
    const headers = new Headers(init.headers);
    if (config.token && !headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${config.token}`);
    }
    return fetchImpl(`${config.base}${path}`, { ...init, headers });
  };

/** Used only when a 401 or 429 arrives without the `{error}` body it should carry. */
const STATUS_FALLBACKS: Record<number, string> = {
  401: "The API rejected this request: it needs a valid API token (VITE_API_TOKEN).",
  429: "Too many requests to the API. Wait a moment and try again.",
};

/**
 * The sentence to show for a failed response. The server's JSON `{error}`
 * wins; a short plain-text body is next; an HTML error page is never shown.
 */
export const apiErrorMessage = (
  status: number,
  body: string,
  fallback: string,
): string => {
  const text = body.trim();
  if (text) {
    try {
      const parsed: unknown = JSON.parse(text);
      const error = (parsed as { error?: unknown } | null)?.error;
      if (typeof error === "string" && error.trim()) return error.trim();
    } catch {
      if (!text.startsWith("<")) return text.slice(0, 300);
    }
  }
  return STATUS_FALLBACKS[status] ?? fallback;
};

export const readApiError = async (
  response: Response,
  fallback: string,
): Promise<string> => {
  const body = await response.text().catch(() => "");
  return apiErrorMessage(response.status, body, fallback);
};

/** True for a request that was aborted (cancelled, timed out, or superseded). */
export const isAbortError = (error: unknown): boolean =>
  typeof error === "object" &&
  error !== null &&
  (error as { name?: unknown }).name === "AbortError";
