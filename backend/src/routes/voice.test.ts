/**
 * The paid voice routes over the real app, with a fake ElevenLabs client: what
 * is cloned, reused and deleted, and what is refused before anything is paid
 * for.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { after, before, beforeEach, describe, test } from "node:test";
import type { Server } from "node:http";
import { ElevenLabsError } from "@elevenlabs/elevenlabs-js";

// Read at config load, so set before the app is imported.
process.env.SLOTIFY_RATE_LIMIT_PAID_PER_MIN = "0";
process.env.SLOTIFY_RATE_LIMIT_COMPUTE_PER_MIN = "0";
process.env.SLOTIFY_MAX_CLONE_SAMPLE_MB = "2";
process.env.SLOTIFY_CLONE_SAMPLE_SECONDS = "5";
delete process.env.SLOTIFY_API_TOKEN;
delete process.env.ELEVENLABS_API_KEY;
delete process.env.OPENAI_API_KEY;
delete process.env.SLOTIFY_OPENAI_MODELS;

/**
 * A mono 16-bit PCM WAV of a tone; `seed` varies the samples. The first
 * `introSeconds` are silent, or a different tone (`introSeed`).
 */
const wav = (
  seed: number,
  { seconds = 0.25, rate = 16_000, introSeconds = 0, introSeed = null as number | null } = {},
): Buffer => {
  const samples = Math.round(seconds * rate);
  const introSamples = Math.round(introSeconds * rate);
  const data = Buffer.alloc(samples * 2);
  for (let index = 0; index < samples; index += 1) {
    const inIntro = index < introSamples;
    const value =
      inIntro && introSeed === null
        ? 0
        : Math.sin((index + (inIntro ? (introSeed as number) : seed)) / 8) * 8_000;
    data.writeInt16LE(Math.round(value), index * 2);
  }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
};

/** What the clone call was given: each file as ffprobe and volumedetect see it. */
interface SentSample {
  path: string;
  seconds: number;
  channels: number;
  sampleRate: number;
  codec: string;
  maxVolumeDb: number;
  sha256: string;
}

const describeSample = (filePath: string): SentSample => {
  const probe = spawnSync(
    "ffprobe",
    ["-v", "error", "-show_entries", "format=duration:stream=channels,sample_rate,codec_name", "-of", "json", filePath],
    { encoding: "utf8" },
  );
  const parsed = JSON.parse(probe.stdout);
  const volume = spawnSync("ffmpeg", ["-v", "info", "-i", filePath, "-af", "volumedetect", "-f", "null", "-"], {
    encoding: "utf8",
  });
  const maxVolume = /max_volume: (-?[\d.]+|-inf) dB/.exec(volume.stderr)?.[1] ?? "-inf";
  return {
    path: filePath,
    seconds: Number(parsed.format.duration),
    channels: parsed.streams[0].channels,
    sampleRate: Number(parsed.streams[0].sample_rate),
    codec: parsed.streams[0].codec_name,
    maxVolumeDb: maxVolume === "-inf" ? Number.NEGATIVE_INFINITY : Number(maxVolume),
    sha256: createHash("sha256").update(fs.readFileSync(filePath)).digest("hex"),
  };
};

interface Fake {
  created: number;
  sent: SentSample[][];
  deleted: string[];
  converted: string[];
  /** When set, ivc.create waits on it. */
  createGate: Promise<void> | null;
  convertError: Error | null;
}

const fake: Fake = { created: 0, sent: [], deleted: [], converted: [], createGate: null, convertError: null };

const fakeClient = {
  voices: {
    ivc: {
      create: async ({ files }: { files: Array<{ path: string }> }) => {
        // Read now: the work dir is removed once the request is done.
        fake.sent.push(files.map((file) => describeSample(String(file.path))));
        if (fake.createGate) await fake.createGate;
        fake.created += 1;
        return { voiceId: `fakevoice${fake.created}` };
      },
    },
    delete: async (voiceId: string) => {
      fake.deleted.push(voiceId);
      return {};
    },
  },
  textToSpeech: {
    convert: async (_voiceId: string, request: { text: string }) => {
      fake.converted.push(request.text);
      if (fake.convertError) throw fake.convertError;
      return new Blob([Buffer.from("ID3fake-mp3")]).stream();
    },
  },
};

let server: Server;
let baseUrl: string;
let registry: typeof import("../services/elevenlabs").voiceRegistry;

const clone = (sample: Buffer, init: RequestInit = {}) => {
  const form = new FormData();
  form.append("files", new Blob([sample], { type: "audio/wav" }), "sample.wav");
  return fetch(`${baseUrl}/api/clone`, { method: "POST", body: form, ...init });
};

const release = (voiceId: string) =>
  fetch(`${baseUrl}/api/clone/${encodeURIComponent(voiceId)}/release`, { method: "POST" });

const tts = (body: unknown) =>
  fetch(`${baseUrl}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

const waitFor = async (condition: () => boolean, timeoutMs = 5_000) => {
  const deadline = Date.now() + timeoutMs;
  while (!condition() && Date.now() < deadline) await new Promise((r) => setTimeout(r, 20));
  return condition();
};

before(async () => {
  const elevenlabs = await import("../services/elevenlabs");
  elevenlabs.setElevenLabsClientForTests(fakeClient);
  registry = elevenlabs.voiceRegistry;
  const { createApp } = await import("../app");
  await new Promise<void>((resolve) => {
    server = createApp().listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  baseUrl = typeof address === "object" && address ? `http://127.0.0.1:${address.port}` : "";
});

after(async () => {
  await registry.releaseAll();
  server?.closeAllConnections();
  server?.close();
});

beforeEach(async () => {
  await registry.releaseAll();
  Object.assign(fake, { created: 0, sent: [], deleted: [], converted: [], createGate: null, convertError: null });
});

describe("POST /api/clone", () => {
  test("the same audio is cloned once and then reused", async () => {
    const first = await clone(wav(1));
    assert.equal(first.status, 200);
    const one = (await first.json()) as { voiceId: string; reused: boolean };
    const two = (await (await clone(wav(1))).json()) as { voiceId: string; reused: boolean };
    assert.equal(two.voiceId, one.voiceId);
    assert.deepEqual([one.reused, two.reused], [false, true]);
    assert.equal(fake.created, 1);

    const other = (await (await clone(wav(2))).json()) as { voiceId: string };
    assert.notEqual(other.voiceId, one.voiceId);
    assert.equal(fake.created, 2);
  });

  test("a voice created after its client disconnected is deleted", async () => {
    let open!: () => void;
    fake.createGate = new Promise((resolve) => {
      open = resolve;
    });
    const controller = new AbortController();
    const request = clone(wav(3), { signal: controller.signal }).catch(() => undefined);
    await waitFor(() => fake.createGate !== null);
    await new Promise((resolve) => setTimeout(resolve, 300));
    controller.abort();
    await request;
    await new Promise((resolve) => setTimeout(resolve, 100));
    open();
    assert.ok(await waitFor(() => fake.deleted.length === 1), "the orphaned clone was not deleted");
    assert.deepEqual(fake.deleted, ["fakevoice1"]);
    assert.equal(registry.size, 0);
  });

  test("more samples than SLOTIFY_MAX_CLONE_SAMPLES is a 400; a large one a 413", async () => {
    const form = new FormData();
    for (let index = 0; index < 4; index += 1) {
      form.append("files", new Blob([wav(index)], { type: "audio/wav" }), `s${index}.wav`);
    }
    assert.equal((await fetch(`${baseUrl}/api/clone`, { method: "POST", body: form })).status, 400);

    const big = await clone(wav(1, { seconds: 70 }));
    assert.equal(big.status, 413);
    assert.match(((await big.json()) as { error: string }).error, /SLOTIFY_MAX_CLONE_SAMPLE_MB/);
    assert.equal(fake.created, 0);
  });

  test("a whole episode is uploaded, but only a trimmed excerpt past its intro is sent", async () => {
    // 100 s whose first 30 s are silent: an intro the excerpt must skip.
    const episode = wav(11, { seconds: 100, rate: 8_000, introSeconds: 30 });
    const response = await clone(episode);
    assert.equal(response.status, 200);
    assert.equal(fake.sent.length, 1);
    const [sample] = fake.sent[0];
    assert.match(sample.path, /clone-sample-0\.mp3$/);
    assert.equal(sample.codec, "mp3");
    assert.equal(sample.channels, 1);
    assert.equal(sample.sampleRate, 44_100);
    assert.ok(sample.seconds <= 5.1 && sample.seconds >= 4.5, `excerpt is ${sample.seconds} s`);
    assert.ok(sample.maxVolumeDb > -30, `excerpt is silent (${sample.maxVolumeDb} dB): the intro was not skipped`);
  });

  test("a short sample is sent from its start", async () => {
    await clone(wav(12, { seconds: 3 }));
    const [sample] = fake.sent[0];
    assert.ok(sample.seconds > 2.5 && sample.seconds <= 3.2, `excerpt is ${sample.seconds} s`);
  });

  test("dedupe hashes the upload, not the excerpt", async () => {
    // Same audio after 0:30, different intros: identical excerpts, different files.
    const a = wav(13, { seconds: 100, rate: 8_000, introSeconds: 30 });
    const b = wav(13, { seconds: 100, rate: 8_000, introSeconds: 30, introSeed: 99 });
    const first = (await (await clone(a)).json()) as { voiceId: string };
    const second = (await (await clone(b)).json()) as { voiceId: string; reused: boolean };
    assert.equal(fake.sent[0][0].sha256, fake.sent[1][0].sha256, "excerpts should match");
    assert.notEqual(second.voiceId, first.voiceId);
    assert.equal(second.reused, false);
    // The same episode again reuses its voice without cutting a new excerpt.
    const again = (await (await clone(a)).json()) as { voiceId: string; reused: boolean };
    assert.equal(again.voiceId, first.voiceId);
    assert.equal(again.reused, true);
    assert.equal(fake.sent.length, 2);
  });

  test("a long filename-derived name is shortened, not refused", async () => {
    const form = new FormData();
    form.append("files", new Blob([wav(4)], { type: "audio/wav" }), `${"n".repeat(300)}.wav`);
    form.append("name", `${"n".repeat(300)}.wav Clone`);
    const response = await fetch(`${baseUrl}/api/clone`, { method: "POST", body: form });
    assert.equal(response.status, 200);
  });
});

describe("POST /api/clone/:voiceId/release", () => {
  test("deletes a voice this server cloned, once; repeating is harmless", async () => {
    const { voiceId } = (await (await clone(wav(5))).json()) as { voiceId: string };
    const first = await release(voiceId);
    assert.equal(first.status, 200);
    assert.deepEqual(await first.json(), { voiceId, released: true });
    const second = await release(voiceId);
    assert.equal(second.status, 200);
    assert.deepEqual(await second.json(), { voiceId, released: false });
    assert.deepEqual(fake.deleted, [voiceId]);
  });

  test("never deletes a voice it did not create", async () => {
    const response = await release("SomeoneElsesVoice1");
    assert.equal(response.status, 404);
    assert.equal((await release("bad id!")).status, 400);
    assert.deepEqual(fake.deleted, []);
  });
});

describe("POST /api/tts", () => {
  test("400 statements are refused before any provider call", async () => {
    const statements = Array.from({ length: 400 }, (_, index) => `Line number ${index}.`);
    const response = await tts({ voiceId: "v", statements });
    assert.equal(response.status, 400);
    assert.match(((await response.json()) as { error: string }).error, /At most 8 statements/);
    assert.deepEqual(fake.converted, []);
  });

  test("too much text, or an unbounded pause, is refused before any provider call", async () => {
    assert.equal((await tts({ voiceId: "v", text: "z".repeat(1_501) })).status, 400);
    assert.equal((await tts({ voiceId: "v", statements: ["a.", "b."], pauseMs: 600_000 })).status, 400);
    assert.equal((await tts({ voiceId: "v", text: "hi", pauseMs: -1 })).status, 400);
    assert.deepEqual(fake.converted, []);
  });

  test("a released clone is a 410, not a provider error", async () => {
    const { voiceId } = (await (await clone(wav(6))).json()) as { voiceId: string };
    await release(voiceId);
    const response = await tts({ voiceId, text: "Hello." });
    assert.equal(response.status, 410);
    assert.deepEqual(fake.converted, []);
  });

  test("a provider rejection is a 502 that keeps the provider's body to the log", async () => {
    fake.convertError = new ElevenLabsError({
      message: "Unauthorized",
      statusCode: 401,
      body: { detail: { status: "invalid_api_key", message: "Invalid API key", request_id: "req_9" } },
    });
    const original = console.error;
    console.error = () => undefined;
    let response: Response;
    try {
      response = await tts({ voiceId: "v", text: "Hello." });
    } finally {
      console.error = original;
    }
    assert.equal(response.status, 502);
    const body = (await response.json()) as { error: string; requestId: string };
    assert.equal(body.error, "TTS failed. The voice provider rejected the request.");
    assert.ok(body.requestId);
    for (const leaked of ["Invalid API key", "req_9", "invalid_api_key"]) {
      assert.ok(!JSON.stringify(body).includes(leaked), leaked);
    }
  });

  test("a registry voice in use is kept alive", async () => {
    const { voiceId } = (await (await clone(wav(7))).json()) as { voiceId: string };
    const response = await tts({ voiceId, text: "Hello." });
    assert.equal(response.status, 200);
    assert.deepEqual(fake.converted, ["Hello."]);
    assert.equal(registry.has(voiceId), true);
  });
});

describe("POST /ad/insert", () => {
  const insert = (llmModel: string) => {
    const form = new FormData();
    form.append("audio", new Blob([Buffer.from("not audio")], { type: "audio/mpeg" }), "a.mp3");
    for (const [key, value] of Object.entries({
      productName: "Acme",
      productBlurb: "Coffee.",
      adStyle: "casual",
      adMode: "A_ONLY",
      llmModel,
    })) {
      form.append(key, value);
    }
    return fetch(`${baseUrl}/ad/insert`, { method: "POST", body: form });
  };

  test("an OpenAI model outside SLOTIFY_OPENAI_MODELS is a 400", async () => {
    const response = await insert("gpt-4.5-preview");
    assert.equal(response.status, 400);
    assert.match(((await response.json()) as { error: string }).error, /gpt-4o-mini/);
  });

  test("an allowed model gets past validation (and this upload is then refused as not audio)", async () => {
    assert.equal((await insert("gpt-4o-mini")).status, 415);
  });
});
