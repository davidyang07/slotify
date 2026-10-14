import assert from "node:assert/strict";
import { test, describe } from "node:test";

import type { ApiFetch } from "./api";
import {
  BACKEND_UNREACHABLE_MESSAGE,
  UNKNOWN_CAPABILITIES,
  fetchCapabilities,
  generationBlockedReason,
  parseCapabilities,
} from "./capabilities";

const serverBody = {
  placement: true,
  voiceCloning: true,
  tts: true,
  openaiEnhancement: false,
  learnedRanker: false,
  ranker: {
    requested: "auto",
    active: "heuristic",
    reason: "No model artifact configured.",
    baselineVersion: "heuristic_offline_v1",
    heuristicConfigVersion: "v1",
  },
};

const answering =
  (status: number, body: unknown): ApiFetch =>
  async () =>
    new Response(typeof body === "string" ? body : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

describe("an unreachable backend is not reported as a missing key", () => {
  test("a network or CORS failure says the backend is unreachable", async () => {
    const failing: ApiFetch = async () => {
      throw new TypeError("Failed to fetch");
    };
    const capabilities = await fetchCapabilities(failing);
    const reason = generationBlockedReason(capabilities);
    assert.equal(reason, BACKEND_UNREACHABLE_MESSAGE);
    assert.match(reason ?? "", /reach the Slotify backend/);
    assert.doesNotMatch(reason ?? "", /ElevenLabs/);
  });

  test("a dead backend behind a proxy (502) says the same", async () => {
    const capabilities = await fetchCapabilities(answering(502, ""));
    assert.equal(generationBlockedReason(capabilities), BACKEND_UNREACHABLE_MESSAGE);
  });

  test("a proxy's own explanation is shown when it gives one", async () => {
    const capabilities = await fetchCapabilities(
      answering(502, { error: "The Slotify API is not reachable at http://localhost:3001." }),
    );
    assert.match(generationBlockedReason(capabilities) ?? "", /not reachable/);
  });

  test("a non-JSON answer (a static host's index.html) is not the API", async () => {
    const capabilities = await fetchCapabilities(
      answering(200, "<!doctype html><html></html>"),
    );
    assert.equal(generationBlockedReason(capabilities), BACKEND_UNREACHABLE_MESSAGE);
  });

  test("a rejected token shows the server's message", async () => {
    const capabilities = await fetchCapabilities(
      answering(401, { error: "Missing or invalid API token." }),
    );
    assert.equal(generationBlockedReason(capabilities), "Missing or invalid API token.");
  });
});

describe("a reachable backend", () => {
  test("without ElevenLabs still names the missing key", async () => {
    const capabilities = await fetchCapabilities(
      answering(200, { ...serverBody, voiceCloning: false, tts: false }),
    );
    assert.equal(capabilities.loadError, null);
    assert.match(generationBlockedReason(capabilities) ?? "", /ElevenLabs API key/);
  });

  test("with ElevenLabs does not block generation", async () => {
    const capabilities = await fetchCapabilities(answering(200, serverBody));
    assert.equal(generationBlockedReason(capabilities), null);
  });

  test("authRequired is read when present and false when an older server omits it", () => {
    assert.equal(parseCapabilities({ ...serverBody, authRequired: true }).authRequired, true);
    assert.equal(parseCapabilities(serverBody).authRequired, false);
  });

  test("before any answer, generation is assumed unavailable but not unreachable", () => {
    assert.equal(UNKNOWN_CAPABILITIES.loadError, null);
    assert.match(generationBlockedReason(UNKNOWN_CAPABILITIES) ?? "", /ElevenLabs/);
  });
});
