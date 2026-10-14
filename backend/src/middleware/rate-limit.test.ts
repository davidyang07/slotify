import assert from "node:assert/strict";
import { after, afterEach, before, describe, test } from "node:test";
import type { Server } from "node:http";
import express from "express";

import { internalToken } from "../config";
import { createRateLimiter, type RateLimitOptions } from "./rate-limit";

let clock = 0;
const servers: Server[] = [];

const startApp = async (options: Omit<RateLimitOptions, "now">): Promise<string> => {
  const app = express();
  app.use(createRateLimiter({ ...options, now: () => clock }));
  app.all("*", (_req, res) => {
    res.json({ ok: true });
  });
  const server = await new Promise<Server>((resolve) => {
    const instance = app.listen(0, "127.0.0.1", () => resolve(instance));
  });
  servers.push(server);
  const address = server.address();
  return typeof address === "object" && address ? `http://127.0.0.1:${address.port}` : "";
};

const post = (url: string, headers: Record<string, string> = {}) =>
  fetch(url, { method: "POST", headers });

describe("per-IP rate limiting", () => {
  const savedOpenAi = process.env.OPENAI_API_KEY;

  before(() => {
    clock = 1_000_000;
  });
  afterEach(() => {
    if (savedOpenAi === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = savedOpenAi;
  });
  after(() => {
    for (const server of servers) server.close();
  });

  test("the paid bucket answers 429 with Retry-After once spent, then resets", async () => {
    const base = await startApp({ paidPerMin: 2, computePerMin: 5 });
    assert.equal((await post(`${base}/api/tts`)).status, 200);
    assert.equal((await post(`${base}/api/clone`)).status, 200);

    clock += 15_000;
    const limited = await post(`${base}/api/generate`);
    assert.equal(limited.status, 429);
    assert.equal(limited.headers.get("retry-after"), "45");
    const body = (await limited.json()) as { error: string };
    assert.match(body.error, /at most 2 paid-API requests per minute/);

    // Compute work has its own budget.
    assert.equal((await post(`${base}/api/merge`)).status, 200);

    clock += 45_000;
    assert.equal((await post(`${base}/ad/insert`)).status, 200);
  });

  test("route matching follows Express: case and trailing slash do not dodge a bucket", async () => {
    const base = await startApp({ paidPerMin: 1, computePerMin: 0 });
    assert.equal((await post(`${base}/API/TTS/`)).status, 200);
    assert.equal((await post(`${base}/api/tts`)).status, 429);
  });

  test("reads are never limited, and 0 disables a bucket", async () => {
    const base = await startApp({ paidPerMin: 1, computePerMin: 0 });
    for (let index = 0; index < 5; index += 1) {
      assert.equal((await fetch(`${base}/api/capabilities`)).status, 200);
      assert.equal((await post(`${base}/api/merge`)).status, 200);
    }
  });

  test("callbacks from our own Python children are exempt", async () => {
    const base = await startApp({ paidPerMin: 1, computePerMin: 1 });
    const internal = { Authorization: `Bearer ${internalToken}` };
    for (let index = 0; index < 4; index += 1) {
      assert.equal((await post(`${base}/api/tts`, internal)).status, 200);
      assert.equal((await post(`${base}/api/merge`, internal)).status, 200);
    }
  });

  test("insert-sections draws on the paid bucket only when it can call OpenAI", async () => {
    delete process.env.OPENAI_API_KEY;
    const offline = await startApp({ paidPerMin: 1, computePerMin: 3 });
    for (let index = 0; index < 3; index += 1) {
      assert.equal((await post(`${offline}/api/insert-sections`)).status, 200);
    }
    assert.equal((await post(`${offline}/api/insert-sections`)).status, 429);

    process.env.OPENAI_API_KEY = "test-key";
    const online = await startApp({ paidPerMin: 1, computePerMin: 3 });
    assert.equal((await post(`${online}/api/insert-sections`)).status, 200);
    assert.equal((await post(`${online}/api/insert-sections`)).status, 429);
  });
});
