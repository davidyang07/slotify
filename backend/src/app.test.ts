/**
 * The whole middleware stack over a real socket: request-shape validation,
 * the JSON error handler, auth and upload limits.
 *
 * The crafted bodies here each used to throw inside an async handler. Express 4
 * does not catch a rejected handler, so Node exited on the unhandled rejection
 * and one request took the API down; the health check after them is the point.
 */

import assert from "node:assert/strict";
import { after, before, describe, test } from "node:test";
import type { Server } from "node:http";

// Read at config load, so set before the app is imported. 1 KB keeps the
// oversized-upload case cheap.
process.env.SLOTIFY_MAX_UPLOAD_MB = "0.001";
process.env.SLOTIFY_RATE_LIMIT_PAID_PER_MIN = "0";
process.env.SLOTIFY_RATE_LIMIT_COMPUTE_PER_MIN = "0";
delete process.env.SLOTIFY_API_TOKEN;
delete process.env.ELEVENLABS_API_KEY;
delete process.env.OPENAI_API_KEY;

let server: Server;
let baseUrl: string;
let internalToken: string;

const audio = () => new Blob([Buffer.alloc(64, 1)]);

const postJson = (path: string, body: unknown, headers: Record<string, string> = {}) =>
  fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });

const postForm = (path: string, fields: Record<string, string>, files: Record<string, Blob> = {}) => {
  const form = new FormData();
  for (const [name, value] of Object.entries(fields)) form.append(name, value);
  for (const [name, blob] of Object.entries(files)) form.append(name, blob, `${name}.mp3`);
  return fetch(`${baseUrl}${path}`, { method: "POST", body: form });
};

const expectJsonError = async (response: Response, status: number) => {
  assert.equal(response.status, status, `expected ${status}, got ${response.status}`);
  assert.match(response.headers.get("content-type") ?? "", /application\/json/);
  const body = (await response.json()) as { error?: unknown };
  assert.equal(typeof body.error, "string");
  return body as { error: string };
};

const assertStillServing = async () => {
  const response = await fetch(`${baseUrl}/api/health`);
  assert.equal(response.status, 200);
};

before(async () => {
  const { createApp } = await import("./app");
  ({ internalToken } = await import("./config"));
  await new Promise<void>((resolve) => {
    server = createApp().listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  baseUrl = typeof address === "object" && address ? `http://127.0.0.1:${address.port}` : "";
});

after(() => {
  server?.close();
});

describe("attacker-shaped request bodies are 4xx, and the server keeps serving", () => {
  test("/api/tts with a toString-shaped text entry", async () => {
    await expectJsonError(await postJson("/api/tts", { voiceId: "v", text: [{ toString: 1 }] }), 400);
    await assertStillServing();
  });

  test("/api/tts with non-string voiceId, sponsor name or statements", async () => {
    for (const body of [
      { voiceId: { toString: 1 }, text: "hello" },
      { voiceId: "v", sponsor: { name: { toString: 1 } } },
      { voiceId: "v", statements: 5 },
      { voiceId: "v", sponsor: "acme" },
      { voiceId: "v", text: "hi", pauseMs: { valueOf: 1 } },
    ]) {
      await expectJsonError(await postJson("/api/tts", body), 400);
    }
    await assertStillServing();
  });

  test("/ad/insert with productName[toString]=1", async () => {
    const response = await postForm(
      "/ad/insert",
      { "productName[toString]": "1", productBlurb: "b", adStyle: "casual", adMode: "DUO" },
      { audio: audio() },
    );
    await expectJsonError(response, 400);
    await assertStillServing();
  });

  test("/api/generate, /api/merge, /api/clone and /api/insert-sections with object fields", async () => {
    const cases: Array<[string, Record<string, string>, Record<string, Blob>]> = [
      ["/api/generate", { "brand[toString]": "1", voiceId: "v" }, { audio: audio() }],
      ["/api/merge", { "insertAt[toString]": "1" }, { audio: audio(), insert: audio() }],
      ["/api/clone", { "name[toString]": "1" }, { files: audio() }],
      ["/api/insert-sections", { "count[valueOf]": "1" }, { audio: audio() }],
      ["/api/insert-sections", { "mode[toString]": "1" }, { audio: audio() }],
    ];
    for (const [path, fields, files] of cases) {
      await expectJsonError(await postForm(path, fields, files), 400);
    }
    await assertStillServing();
  });

  test("repeated form fields are a 400, not an array smuggled into argv", async () => {
    const form = new FormData();
    form.append("audio", audio(), "a.mp3");
    form.append("brand", "a");
    form.append("brand", "b");
    form.append("voiceId", "v");
    const response = await fetch(`${baseUrl}/api/generate`, { method: "POST", body: form });
    await expectJsonError(response, 400);
  });

  test("malformed JSON is a JSON 400", async () => {
    const response = await fetch(`${baseUrl}/api/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: '{"voiceId":',
    });
    await expectJsonError(response, 400);
  });

  test("a failure inside a paid route is a JSON 500 without paths", async () => {
    // No ELEVENLABS_API_KEY: the handler gets as far as the SDK and fails.
    const response = await postJson("/api/tts", { voiceId: "v", text: "Hello there." });
    const body = await expectJsonError(response, 500);
    assert.match(body.error, /^TTS failed\./);
    assert.ok(!body.error.includes("/home/"), body.error);
    await assertStillServing();
  });
});

describe("tone and language", () => {
  test("an unknown tone or language is a 400 on /api/tts", async () => {
    await expectJsonError(await postJson("/api/tts", { voiceId: "v", text: "hi", tone: "angry" }), 400);
    await expectJsonError(await postJson("/api/tts", { voiceId: "v", text: "hi", language: "xx" }), 400);
  });

  test("an unknown tone or language is a 400 on /api/insert-sections", async () => {
    await expectJsonError(await postForm("/api/insert-sections", { tone: "angry" }, { audio: audio() }), 400);
    await expectJsonError(await postForm("/api/insert-sections", { language: "klingon" }, { audio: audio() }), 400);
  });
});

describe("sponsor statements are capped before any OpenAI call", () => {
  test("more than three sponsors is a 400", async () => {
    const sponsors = JSON.stringify([1, 2, 3, 4].map((n) => ({ name: `Brand ${n}` })));
    const body = await expectJsonError(
      await postForm("/api/insert-sections", { sponsors }, { audio: audio() }),
      400,
    );
    assert.match(body.error, /At most 3/);
  });

  test("a sponsor with a non-string field is a 400", async () => {
    const sponsors = JSON.stringify([{ name: { toString: 1 } }]);
    await expectJsonError(await postForm("/api/insert-sections", { sponsors }, { audio: audio() }), 400);
  });
});

describe("uploads", () => {
  test("a file over SLOTIFY_MAX_UPLOAD_MB is a JSON 413", async () => {
    const big = new Blob([Buffer.alloc(4096, 1)]);
    await expectJsonError(await postForm("/api/insert-sections", {}, { audio: big }), 413);
    await assertStillServing();
  });

  test("a non-audio file is rejected", async () => {
    const form = new FormData();
    form.append("audio", new Blob(["hello"], { type: "text/plain" }), "notes.txt");
    const response = await fetch(`${baseUrl}/api/insert-sections`, { method: "POST", body: form });
    await expectJsonError(response, 415);
  });

  test("more clone samples than allowed is a 400", async () => {
    const form = new FormData();
    for (let index = 0; index < 11; index += 1) form.append("files", audio(), `s${index}.wav`);
    const response = await fetch(`${baseUrl}/api/clone`, { method: "POST", body: form });
    await expectJsonError(response, 400);
  });

  test("a truncated multipart body is a JSON 400", async () => {
    const response = await fetch(`${baseUrl}/api/merge`, {
      method: "POST",
      headers: { "Content-Type": "multipart/form-data; boundary=xyz" },
      body: '--xyz\r\nContent-Disposition: form-data; name="insertAt"\r\n\r\n1',
    });
    await expectJsonError(response, 400);
    await assertStillServing();
  });
});

describe("CORS", () => {
  test("a disallowed origin is a JSON 403", async () => {
    const response = await fetch(`${baseUrl}/api/capabilities`, {
      headers: { Origin: "http://evil.example" },
    });
    await expectJsonError(response, 403);
  });
});

describe("SLOTIFY_API_TOKEN", () => {
  after(() => {
    delete process.env.SLOTIFY_API_TOKEN;
  });

  test("unset: no auth, and capabilities says so", async () => {
    delete process.env.SLOTIFY_API_TOKEN;
    const capabilities = (await (await fetch(`${baseUrl}/api/capabilities`)).json()) as any;
    assert.equal(capabilities.authRequired, false);
    await expectJsonError(await postJson("/api/tts", {}), 400);
  });

  test("set: 401 without the token, through with it", async () => {
    process.env.SLOTIFY_API_TOKEN = "s3cret";
    const capabilities = (await (await fetch(`${baseUrl}/api/capabilities`)).json()) as any;
    assert.equal(capabilities.authRequired, true);
    assert.equal((await fetch(`${baseUrl}/api/health`)).status, 200);

    const missing = await expectJsonError(await postJson("/api/tts", {}), 401);
    assert.match(missing.error, /Bearer/);
    await expectJsonError(await postJson("/api/tts", {}, { Authorization: "Bearer wrong" }), 401);
    await expectJsonError(
      await fetch(`${baseUrl}/api/related-products`, { method: "POST" }),
      401,
    );
    // Past auth, the empty body is the next problem.
    await expectJsonError(await postJson("/api/tts", {}, { Authorization: "Bearer s3cret" }), 400);
  });

  test("set: the per-process internal token passes too", async () => {
    process.env.SLOTIFY_API_TOKEN = "s3cret";
    await expectJsonError(
      await postJson("/api/tts", {}, { Authorization: `Bearer ${internalToken}` }),
      400,
    );
  });
});

describe("asyncHandler", () => {
  test("an unexpected rejection becomes a JSON 500 with no detail", async () => {
    const express = (await import("express")).default;
    const { asyncHandler } = await import("./lib/http");
    const { errorHandler } = await import("./middleware/error-handler");
    const app = express();
    app.get(
      "/boom",
      asyncHandler(async () => {
        throw new Error("secret at /srv/app/internal.ts");
      }),
    );
    app.use(errorHandler);
    const local = await new Promise<Server>((resolve) => {
      const instance = app.listen(0, "127.0.0.1", () => resolve(instance));
    });
    try {
      const address = local.address();
      const port = typeof address === "object" && address ? address.port : 0;
      const body = await expectJsonError(await fetch(`http://127.0.0.1:${port}/boom`), 500);
      assert.equal(body.error, "Internal server error.");
    } finally {
      local.close();
    }
  });
});
