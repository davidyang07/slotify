import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import { afterEach, describe, test } from "node:test";

import { buildSponsorBlock, ttsModelFor } from "./elevenlabs";
import { buildStatementPrompt, generateBrandStatement } from "./openai";

const saved = {
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
};
const restore = () => {
  for (const [key, value] of Object.entries(saved)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
};

describe("sponsor copy follows the requested tone and language", () => {
  afterEach(restore);

  test("the prompt carries the tone and the language", () => {
    const prompt = buildStatementPrompt({
      name: "Acme",
      productDesc: "coffee",
      tone: "energetic",
      language: "es",
    });
    assert.match(prompt, /tone is energetic and upbeat/);
    assert.match(prompt, /entirely in Spanish/);
    assert.match(prompt, /Brand name: Acme\./);
  });

  test("with neither, the prompt is the original one", () => {
    const prompt = buildStatementPrompt({ name: "Acme", productDesc: "" });
    assert.match(prompt, /calm and conversational/);
    assert.ok(!/entirely in/.test(prompt));
  });

  test("without OpenAI the fallback is still written in the requested language", async () => {
    delete process.env.OPENAI_API_KEY;
    const english = await generateBrandStatement({ name: "Acme", productDesc: "" });
    assert.match(english, /^Acme supports this episode/);
    const japanese = await generateBrandStatement({ name: "Acme", productDesc: "", language: "ja" });
    assert.match(japanese, /Acmeの提供でお送りします/);
    const german = await generateBrandStatement({ name: "", productDesc: "", language: "de" });
    assert.match(german, /^Diese Folge wird unterstützt von unserem Sponsor/);
  });
});

describe("TTS model for a language", () => {
  test("defaults to the multilingual model with no language code", () => {
    assert.deepEqual(ttsModelFor(undefined, undefined), { modelId: "eleven_multilingual_v2" });
    assert.deepEqual(ttsModelFor(undefined, "fr"), { modelId: "eleven_multilingual_v2" });
  });

  test("an English-only model is swapped for a non-English read", () => {
    assert.deepEqual(ttsModelFor("eleven_turbo_v2", "ko"), { modelId: "eleven_multilingual_v2" });
    assert.deepEqual(ttsModelFor("eleven_turbo_v2", "en"), { modelId: "eleven_turbo_v2" });
  });

  test("languageCode is sent only to models that accept it", () => {
    assert.deepEqual(ttsModelFor("eleven_flash_v2_5", "de"), {
      modelId: "eleven_flash_v2_5",
      languageCode: "de",
    });
    assert.equal(ttsModelFor("eleven_multilingual_v2", "de").languageCode, undefined);
  });
});

describe("buildSponsorBlock", () => {
  afterEach(restore);

  test("a failure removes its whole temp dir", async () => {
    delete process.env.ELEVENLABS_API_KEY;
    const count = () =>
      fs.readdirSync(os.tmpdir()).filter((entry) => entry.startsWith("sponsor-block-")).length;
    const before = count();
    await assert.rejects(
      buildSponsorBlock({ voiceId: "v", statements: ["One.", "Two."] }),
      /ELEVENLABS_API_KEY is not set/,
    );
    assert.equal(count(), before);
  });
});
