import assert from "node:assert/strict";
import { after, describe, test } from "node:test";
import type { Server } from "node:http";
import express from "express";

import { createReadiness, defaultChecks, type ReadinessCheck } from "../services/readiness";
import { createHealthRouter } from "./health";

const servers: Server[] = [];
after(() => {
  for (const server of servers) server.close();
});

const serve = async (checks: ReadinessCheck[]) => {
  const app = express();
  app.use(createHealthRouter(createReadiness({ checks })));
  const server = await new Promise<Server>((resolve) => {
    const instance = app.listen(0, "127.0.0.1", () => resolve(instance));
  });
  servers.push(server);
  const address = server.address();
  return typeof address === "object" && address ? `http://127.0.0.1:${address.port}` : "";
};

const passing = (name: string): ReadinessCheck => ({ name, run: async () => undefined });
const failing = (name: string, message: string): ReadinessCheck => ({
  name,
  run: async () => {
    throw new Error(message);
  },
});

describe("GET /api/health is a readiness check", () => {
  test("200 when every dependency works", async () => {
    const base = await serve([passing("ffmpeg"), passing("analyser")]);
    const response = await fetch(`${base}/api/health`);
    assert.equal(response.status, 200);
    const body = (await response.json()) as any;
    assert.equal(body.status, "ok");
    assert.deepEqual(body.failed, []);
    assert.equal(body.capabilities.placement, true);
  });

  test("503 naming the check that failed, without server paths", async () => {
    const base = await serve([
      passing("ffmpeg"),
      failing("analyser", "ModuleNotFoundError: No module named 'librosa' (/srv/venv/lib/python3.12)"),
    ]);
    const response = await fetch(`${base}/api/health`);
    assert.equal(response.status, 503);
    const body = (await response.json()) as any;
    assert.equal(body.status, "unavailable");
    assert.deepEqual(body.failed, ["analyser"]);
    assert.match(body.error, /analyser/);
    assert.match(body.checks.analyser.error, /No module named 'librosa'/);
    assert.ok(!JSON.stringify(body).includes("/srv/venv"));
  });

  test("HEAD answers with the same status and no body", async () => {
    const base = await serve([failing("ffmpeg", "spawn ffmpeg ENOENT")]);
    const response = await fetch(`${base}/api/health`, { method: "HEAD" });
    assert.equal(response.status, 503);
    assert.equal(await response.text(), "");
  });

  test("the real checks cover ffmpeg, ffprobe and the analyser's imports", () => {
    assert.deepEqual(
      defaultChecks().map((check) => check.name),
      ["ffmpeg", "ffprobe", "analyser"],
    );
  });
});

describe("readiness is cached", () => {
  test("checks run once, and a stale result is served while it is re-checked", async () => {
    let clock = 0;
    let runs = 0;
    let healthy = true;
    const readiness = createReadiness({
      checks: [
        {
          name: "analyser",
          run: async () => {
            runs += 1;
            if (!healthy) throw new Error("gone");
          },
        },
      ],
      maxAgeMs: 60_000,
      now: () => clock,
    });
    assert.equal((await readiness.report()).ready, true);
    clock += 30_000;
    await readiness.report();
    assert.equal(runs, 1, "re-checked within 60 s");

    healthy = false;
    clock += 31_000;
    // Stale: the cached answer now, a fresh check behind it.
    assert.equal((await readiness.report()).ready, true);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(runs, 2);
    assert.equal((await readiness.report()).ready, false);
  });
});
