/**
 * Demo-path regression coverage for POST /api/insert-sections.
 *
 * These drive the real Express app over a real socket with a real upload, so
 * they cover the thing a unit test cannot: that the route, the middleware, the
 * analyser subprocess and the response shape agree. They run in heuristic mode
 * (no model, no network, no keys), which is exactly the credential-free path CI
 * has to be able to run.
 *
 * The analyser is a Python subprocess, so these are slower than the rest of the
 * suite. They are worth it: every failure mode below is one the product used to
 * have and shipped to a user as three confident recommendations.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, test } from "node:test";
import type { Server } from "node:http";

import { BACKEND_DIR } from "../config";

const REPO_ROOT = path.resolve(BACKEND_DIR, "..");
const SAMPLE_AUDIO = path.join(BACKEND_DIR, "audio_tests", "rogan-test1.mp3");

/** Whether the Python analyser can run here. Skips rather than fails if not. */
const analyserAvailable = (): boolean => {
  if (!fs.existsSync(SAMPLE_AUDIO)) return false;
  const probe = spawnSync(process.env.PYTHON_BIN ?? "python", ["-c", "import pydub"], {
    cwd: BACKEND_DIR,
    timeout: 60_000,
  });
  return probe.status === 0;
};

const available = analyserAvailable();

let server: Server;
let baseUrl: string;

const post = async (form: FormData): Promise<{ status: number; body: any }> => {
  const response = await fetch(`${baseUrl}/api/insert-sections`, {
    method: "POST",
    body: form,
  });
  return { status: response.status, body: await response.json() };
};

const audioForm = (filePath: string, filename = "audio.mp3"): FormData => {
  const form = new FormData();
  form.append("audio", new Blob([fs.readFileSync(filePath)]), filename);
  form.append("count", "3");
  return form;
};

describe("POST /api/insert-sections", { skip: !available && "python analyser unavailable" }, () => {
  before(async () => {
    process.env.RANKER_MODE = "heuristic";
    delete process.env.ELEVENLABS_API_KEY;
    delete process.env.OPENAI_API_KEY;

    const express = (await import("express")).default;
    const { insertSectionsRouter } = await import("./insert-sections");
    const { capabilitiesRouter } = await import("./capabilities");
    const app = express();
    app.use(express.json());
    app.use(capabilitiesRouter);
    app.use(insertSectionsRouter);
    await new Promise<void>((resolve) => {
      server = app.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    baseUrl =
      typeof address === "object" && address
        ? `http://127.0.0.1:${address.port}`
        : "http://127.0.0.1:0";
  });

  after(() => {
    server?.close();
  });

  test("a real upload is ranked, and every slot is provenance-stamped", async () => {
    const { status, body } = await post(audioForm(SAMPLE_AUDIO));
    assert.equal(status, 200);
    assert.equal(body.placementStatus, "ok");
    assert.ok(body.slots.length >= 1);

    assert.equal(body.provenance.source, "heuristic_offline_v1");
    assert.equal(body.provenance.mode, "heuristic");
    assert.equal(body.provenance.isCalibratedProbability, false);
    assert.equal(body.provenance.candidateGeneration, "signal_offline_v1");

    for (const slot of body.slots) {
      assert.equal(slot.source, "heuristic_offline_v1");
      assert.ok(Number.isInteger(slot.placement_score));
      assert.ok(slot.placement_score >= 0 && slot.placement_score <= 100);
      assert.ok(Number.isFinite(slot.raw_score));
      assert.ok(Array.isArray(slot.signals));
      assert.ok(slot.candidate_id.startsWith("upload:"));
    }
  });

  test("the response never claims more slots than candidates", async () => {
    const { body } = await post(audioForm(SAMPLE_AUDIO));
    assert.ok(
      body.slots.length <= body.candidateCount,
      `${body.slots.length} slots from ${body.candidateCount} candidates`,
    );
  });

  test("no slot lands on a removed fallback position", async () => {
    // The old route substituted 25/50/75 % of duration on analyser failure, and
    // the old selector padded with 22/50/78 %. Neither may reappear.
    const { body } = await post(audioForm(SAMPLE_AUDIO));
    const duration = body.duration as number;
    const forbidden = [0.22, 0.25, 0.5, 0.75, 0.78].map((ratio) => ratio * duration);
    for (const slot of body.slots) {
      for (const position of forbidden) {
        assert.ok(
          Math.abs(slot.insertion_time_seconds - position) > 0.01,
          `slot at ${slot.insertion_time_seconds}s matches removed fallback ${position}s`,
        );
      }
    }
  });

  test("no slot carries a hard-coded 92/85/78 score", async () => {
    const { body } = await post(audioForm(SAMPLE_AUDIO));
    const scores = body.slots.map((slot: any) => slot.placement_score);
    assert.notDeepEqual(scores.slice(0, 3), [92, 85, 78]);
  });

  test("undecodable audio is a 502 that reports the failure", async () => {
    const form = new FormData();
    form.append("audio", new Blob([Buffer.from("this is not audio")]), "broken.mp3");
    const { status, body } = await post(form);

    assert.equal(status, 502);
    assert.equal(body.placementStatus, "unavailable");
    assert.deepEqual(body.slots, []);
    assert.deepEqual(body.points, []);
    assert.ok(body.error.includes("no insertion points"));
    // The client gets the failure, not the repository's internal paths.
    assert.ok(!String(body.detail).includes("Traceback"));
  });

  test("a request with no audio is rejected before anything runs", async () => {
    const { status, body } = await post(new FormData());
    assert.equal(status, 400);
    assert.equal(body.placementStatus, "unavailable");
    assert.deepEqual(body.slots, []);
  });

  test("the temp upload directory is cleaned up", async () => {
    const before = fs.readdirSync(os.tmpdir()).filter((entry) =>
      entry.startsWith("insert-sections-"),
    ).length;
    await post(audioForm(SAMPLE_AUDIO));
    const after = fs.readdirSync(os.tmpdir()).filter((entry) =>
      entry.startsWith("insert-sections-"),
    ).length;
    assert.ok(after <= before, `${after - before} temp dir(s) leaked`);
  });

  test("capabilities report placement without any credentials", async () => {
    const response = await fetch(`${baseUrl}/api/capabilities`);
    const body = (await response.json()) as any;
    assert.equal(body.placement, true);
    assert.equal(body.tts, false);
    assert.equal(body.voiceCloning, false);
  });
});

describe("the parity padding is unreachable from the product", () => {
  test("no route imports baseline-parity", () => {
    const routesDir = path.join(BACKEND_DIR, "src", "routes");
    const libDir = path.join(BACKEND_DIR, "src", "lib");
    const servicesDir = path.join(BACKEND_DIR, "src", "services");
    const offenders: string[] = [];
    for (const directory of [routesDir, libDir, servicesDir]) {
      for (const entry of fs.readdirSync(directory)) {
        if (!entry.endsWith(".ts") || entry.endsWith(".test.ts")) continue;
        if (entry === "baseline-parity.ts") continue;
        const source = fs.readFileSync(path.join(directory, entry), "utf8");
        if (source.includes('from "./baseline-parity"') || source.includes('from "../lib/baseline-parity"')) {
          offenders.push(path.relative(REPO_ROOT, path.join(directory, entry)));
        }
      }
    }
    assert.deepEqual(
      offenders,
      [],
      `baseline-parity is fixture-only; imported by ${offenders.join(", ")}`,
    );
  });
});
