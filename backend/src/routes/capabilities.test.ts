import assert from "node:assert/strict";
import { test, describe, afterEach } from "node:test";

import { readCapabilities } from "./capabilities";

const saved = {
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  RANKER_MODE: process.env.RANKER_MODE,
  SLOTIFY_RANKER_CHECKPOINT: process.env.SLOTIFY_RANKER_CHECKPOINT,
};

const restore = () => {
  for (const [key, value] of Object.entries(saved)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
};

describe("capabilities", () => {
  afterEach(restore);

  test("placement works with no credentials at all", () => {
    delete process.env.ELEVENLABS_API_KEY;
    delete process.env.OPENAI_API_KEY;
    delete process.env.SLOTIFY_RANKER_CHECKPOINT;
    process.env.RANKER_MODE = "auto";

    const capabilities = readCapabilities();
    assert.equal(capabilities.placement, true);
    assert.equal(capabilities.voiceCloning, false);
    assert.equal(capabilities.tts, false);
    assert.equal(capabilities.openaiEnhancement, false);
    assert.equal(capabilities.learnedRanker, false);
    assert.equal(capabilities.ranker.active, "heuristic");
  });

  test("an ElevenLabs key enables generation without touching placement", () => {
    process.env.ELEVENLABS_API_KEY = "test-key";
    const capabilities = readCapabilities();
    assert.equal(capabilities.voiceCloning, true);
    assert.equal(capabilities.tts, true);
    assert.equal(capabilities.placement, true);
  });

  test("a misconfigured learned mode is reported, not hidden", () => {
    process.env.RANKER_MODE = "learned";
    delete process.env.SLOTIFY_RANKER_CHECKPOINT;
    const capabilities = readCapabilities();
    assert.equal(capabilities.learnedRanker, false);
    assert.match(capabilities.ranker.reason, /unusable/);
  });

  test("the baseline version is named so a client can record it", () => {
    const capabilities = readCapabilities();
    assert.equal(capabilities.ranker.baselineVersion, "heuristic_offline_v1");
    assert.ok(capabilities.ranker.heuristicConfigVersion.startsWith("heuristic-config-"));
  });
});
