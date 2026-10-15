import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { createApiFetch, resolveApiConfig } from "./api";
import { releaseVoice, voiceReleasePath } from "./voice";

const recordingFetch = (outcome: () => Promise<Response>) => {
  const calls: { url: string; init: RequestInit; headers: Headers }[] = [];
  const fetchImpl = ((input: string | URL | Request, init: RequestInit = {}) => {
    calls.push({ url: String(input), init, headers: new Headers(init.headers) });
    return outcome();
  }) as typeof fetch;
  return { calls, fetchImpl };
};

describe("releasing a cloned voice", () => {
  test("POSTs to the release route with keepalive and the bearer token", async () => {
    const { calls, fetchImpl } = recordingFetch(async () => new Response("{}"));
    const apiFetch = createApiFetch(
      resolveApiConfig({ VITE_API_BASE_URL: "https://api.example.com", VITE_API_TOKEN: "s3cret" }),
      fetchImpl,
    );
    await releaseVoice(apiFetch, "abc123");
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "https://api.example.com/api/clone/abc123/release");
    assert.equal(calls[0].init.method, "POST");
    assert.equal(calls[0].init.keepalive, true);
    assert.equal(calls[0].headers.get("Authorization"), "Bearer s3cret");
  });

  test("never rejects, whatever the network does", async () => {
    const { fetchImpl } = recordingFetch(async () => {
      throw new TypeError("Failed to fetch");
    });
    await releaseVoice(createApiFetch(resolveApiConfig({}), fetchImpl), "v");
  });

  test("the voice id is path-encoded", () => {
    assert.equal(voiceReleasePath("a/b?c"), "/api/clone/a%2Fb%3Fc/release");
  });
});
