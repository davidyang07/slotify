import assert from "node:assert/strict";
import { after, before, describe, test } from "node:test";
import type { Server } from "node:http";
import express from "express";

import { upload } from "../middleware/upload";
import { asyncHandler, HttpError } from "./http";
import { errorHandler } from "../middleware/error-handler";
import { currentRequestId, installRequestLogPrefix, requestIdMiddleware } from "./request-context";

let server: Server;
let baseUrl: string;
const lines: unknown[][] = [];
const originalLog = console.log;

before(async () => {
  // Captures what reaches the real console.log, after the prefix is added.
  console.log = (...args: unknown[]) => {
    lines.push(args);
  };
  installRequestLogPrefix();

  const app = express();
  app.use(requestIdMiddleware);
  app.use(express.json({ limit: "4mb" }));
  app.post("/json", (req, res) => {
    console.log("handled json");
    res.json({ id: currentRequestId(), echoed: req.body?.a });
  });
  app.post("/upload", upload.single("audio"), (_req, res) => {
    console.log("handled upload");
    res.json({ id: currentRequestId() });
  });
  app.get(
    "/fail",
    asyncHandler(async () => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      throw new HttpError(409, "Conflict.");
    }),
  );
  app.get("/ok", (_req, res) => {
    res.json({ fine: true });
  });
  app.use(errorHandler);
  await new Promise<void>((resolve) => {
    server = app.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  baseUrl = typeof address === "object" && address ? `http://127.0.0.1:${address.port}` : "";
});

after(() => {
  console.log = originalLog;
  server?.close();
});

describe("request ids", () => {
  test("a fresh id is assigned and echoed when the client sends none", async () => {
    const response = await fetch(`${baseUrl}/ok`);
    assert.match(response.headers.get("x-request-id") ?? "", /^[0-9a-f-]{36}$/);
    assert.deepEqual(await response.json(), { fine: true }, "success bodies are unchanged");
  });

  test("a sane client id is kept; anything else is replaced", async () => {
    const kept = await fetch(`${baseUrl}/ok`, { headers: { "X-Request-Id": "trace-42.a:b" } });
    assert.equal(kept.headers.get("x-request-id"), "trace-42.a:b");
    for (const bad of ["has space", "x".repeat(200), "semi;colon"]) {
      const replaced = await fetch(`${baseUrl}/ok`, { headers: { "X-Request-Id": bad } });
      assert.notEqual(replaced.headers.get("x-request-id"), bad);
    }
  });

  test("error bodies carry the id", async () => {
    const response = await fetch(`${baseUrl}/fail`, { headers: { "X-Request-Id": "req-1" } });
    assert.equal(response.status, 409);
    assert.deepEqual(await response.json(), { error: "Conflict.", requestId: "req-1" });
  });

  test("log lines carry the id, after JSON parsing and after a multipart upload", async () => {
    lines.length = 0;
    // Large enough to arrive over many socket reads, so the parsers finish on
    // a socket event: the context must survive that.
    const json = await fetch(`${baseUrl}/json`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Request-Id": "req-json" },
      body: JSON.stringify({ a: 1, padding: "p".repeat(2 * 1024 * 1024) }),
    });
    assert.deepEqual(await json.json(), { id: "req-json", echoed: 1 });

    const form = new FormData();
    form.append("audio", new Blob([Buffer.alloc(4 * 1024 * 1024, 1)], { type: "audio/mpeg" }), "a.mp3");
    const uploaded = await fetch(`${baseUrl}/upload`, {
      method: "POST",
      body: form,
      headers: { "X-Request-Id": "req-upload" },
    });
    assert.deepEqual(await uploaded.json(), { id: "req-upload" });

    assert.deepEqual(lines, [
      ["[req req-json]", "handled json"],
      ["[req req-upload]", "handled upload"],
    ]);
  });

  test("outside a request, lines are unprefixed", () => {
    lines.length = 0;
    console.log("boot");
    assert.deepEqual(lines, [["boot"]]);
  });
});
