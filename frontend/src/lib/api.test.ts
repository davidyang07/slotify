import assert from "node:assert/strict";
import { test, describe } from "node:test";

import {
  apiErrorMessage,
  createApiFetch,
  isAbortError,
  readApiError,
  resolveApiConfig,
} from "./api";

const recordingFetch = () => {
  const calls: { url: string; headers: Headers }[] = [];
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(input), headers: new Headers(init?.headers) });
    return new Response("{}", { status: 200 });
  }) as typeof fetch;
  return { calls, fetchImpl };
};

describe("API base", () => {
  test("defaults to same-origin, not localhost", () => {
    assert.equal(resolveApiConfig({}).base, "");
    assert.equal(resolveApiConfig({ VITE_API_BASE_URL: "" }).base, "");
  });

  test("VITE_API_BASE_URL overrides it, without a trailing slash", () => {
    const config = resolveApiConfig({ VITE_API_BASE_URL: "https://api.example.com/" });
    assert.equal(config.base, "https://api.example.com");
  });
});

describe("auth header", () => {
  test("is sent on every request when VITE_API_TOKEN is set", async () => {
    const { calls, fetchImpl } = recordingFetch();
    const apiFetch = createApiFetch(resolveApiConfig({ VITE_API_TOKEN: "s3cret" }), fetchImpl);
    await apiFetch("/api/capabilities");
    await apiFetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    assert.deepEqual(
      calls.map((call) => call.url),
      ["/api/capabilities", "/api/tts"],
    );
    for (const call of calls) {
      assert.equal(call.headers.get("Authorization"), "Bearer s3cret");
    }
    assert.equal(calls[1].headers.get("Content-Type"), "application/json");
  });

  test("is not sent when no token is configured", async () => {
    const { calls, fetchImpl } = recordingFetch();
    const apiFetch = createApiFetch(resolveApiConfig({}), fetchImpl);
    await apiFetch("/api/capabilities");
    assert.equal(calls[0].headers.has("Authorization"), false);
  });
});

describe("error messages", () => {
  test("a 401 or 429 shows the server's {error}", async () => {
    const unauthorized = new Response(JSON.stringify({ error: "Missing or invalid API token." }), {
      status: 401,
    });
    assert.equal(await readApiError(unauthorized, "TTS failed."), "Missing or invalid API token.");
    const limited = new Response(JSON.stringify({ error: "Rate limit exceeded; retry in 30s." }), {
      status: 429,
    });
    assert.equal(await readApiError(limited, "TTS failed."), "Rate limit exceeded; retry in 30s.");
  });

  test("a 401 or 429 without a body still says what happened", () => {
    assert.match(apiErrorMessage(401, "", "TTS failed."), /API token/);
    assert.match(apiErrorMessage(429, "", "TTS failed."), /Too many requests/);
  });

  test("raw JSON is never shown as the message", () => {
    assert.equal(apiErrorMessage(500, '{"error":"ffmpeg exited 1"}', "Merge failed."), "ffmpeg exited 1");
    assert.equal(apiErrorMessage(500, '{"detail":"x"}', "Merge failed."), "Merge failed.");
  });

  test("an HTML error page falls back; short text is shown", () => {
    assert.equal(apiErrorMessage(502, "<html>Bad gateway</html>", "Merge failed."), "Merge failed.");
    assert.equal(apiErrorMessage(413, "Payload Too Large", "Merge failed."), "Payload Too Large");
  });

  test("aborts are recognised", () => {
    const controller = new AbortController();
    controller.abort();
    assert.equal(isAbortError(controller.signal.reason), true);
    assert.equal(isAbortError(new TypeError("Failed to fetch")), false);
  });
});
