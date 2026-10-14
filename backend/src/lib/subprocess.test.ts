import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { Server } from "node:http";
import express from "express";

import { internalToken } from "../config";
import { clientAbortSignal } from "./http";
import { pythonEnv, runProcess, SubprocessError } from "./subprocess";

const node = process.execPath;
const SLEEP = ["-e", "setTimeout(() => {}, 30000)"];

describe("runProcess", () => {
  test("resolves with stdout on a clean exit", async () => {
    const { stdout } = await runProcess(node, ["-e", "process.stdout.write('ok')"], {
      label: "node",
      timeoutMs: 10_000,
    });
    assert.equal(stdout, "ok");
  });

  test("a non-zero exit keeps stderr off the message", async () => {
    await assert.rejects(
      runProcess(node, ["-e", "console.error('/srv/secret/trace'); process.exit(3)"], {
        label: "node",
        timeoutMs: 10_000,
      }),
      (error: unknown) => {
        assert.ok(error instanceof SubprocessError);
        assert.equal(error.reason, "exit");
        assert.equal(error.message, "node exited with code 3.");
        assert.match(error.stderr, /\/srv\/secret\/trace/);
        return true;
      },
    );
  });

  test("a timeout SIGKILLs the child and says which env var raises it", async () => {
    const started = Date.now();
    await assert.rejects(
      runProcess(node, SLEEP, {
        label: "sleeper",
        timeoutMs: 200,
        timeoutEnvVar: "SLOTIFY_FFMPEG_TIMEOUT_MS",
      }),
      (error: unknown) => {
        assert.ok(error instanceof SubprocessError);
        assert.equal(error.reason, "timeout");
        assert.match(error.message, /within 0.2 s.*SLOTIFY_FFMPEG_TIMEOUT_MS/);
        return true;
      },
    );
    assert.ok(Date.now() - started < 5_000);
  });

  test("an abort kills the child", async () => {
    const controller = new AbortController();
    const running = runProcess(node, SLEEP, { label: "sleeper", timeoutMs: 30_000, signal: controller.signal });
    setTimeout(() => controller.abort(), 100);
    await assert.rejects(running, (error: unknown) => {
      assert.ok(error instanceof SubprocessError);
      assert.equal(error.reason, "aborted");
      return true;
    });
  });

  test("a missing binary is a spawn failure, not a crash", async () => {
    await assert.rejects(
      runProcess("/nonexistent/slotify-binary", [], { label: "missing", timeoutMs: 1_000 }),
      (error: unknown) => error instanceof SubprocessError && error.reason === "spawn",
    );
  });

  test("Python children get the internal callback token", () => {
    const env = pythonEnv({ EXTRA: "1" });
    assert.equal(env.SLOTIFY_INTERNAL_TOKEN, internalToken);
    assert.equal(env.EXTRA, "1");
    assert.equal(env.PATH, process.env.PATH);
    assert.ok(internalToken.length >= 64);
  });
});

describe("clientAbortSignal", () => {
  test("a client that disconnects mid-request stops the subprocess", async () => {
    let outcome: Promise<string> | undefined;
    const app = express();
    app.get("/slow", (req, res) => {
      const signal = clientAbortSignal(req, res);
      outcome = runProcess(node, SLEEP, { label: "slow", timeoutMs: 30_000, signal }).then(
        () => "finished",
        (error: SubprocessError) => error.reason,
      );
    });
    const server = await new Promise<Server>((resolve) => {
      const instance = app.listen(0, "127.0.0.1", () => resolve(instance));
    });
    try {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      const controller = new AbortController();
      const request = fetch(`http://127.0.0.1:${port}/slow`, { signal: controller.signal }).catch(
        () => undefined,
      );
      await new Promise((resolve) => setTimeout(resolve, 200));
      controller.abort();
      await request;
      const deadline = Date.now() + 5_000;
      while (!outcome && Date.now() < deadline) await new Promise((r) => setTimeout(r, 20));
      assert.equal(await outcome, "aborted");
    } finally {
      server.closeAllConnections();
      server.close();
    }
  });
});
