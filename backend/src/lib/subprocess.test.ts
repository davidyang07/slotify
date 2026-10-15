import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, describe, test } from "node:test";
import type { Server } from "node:http";
import express from "express";

import { internalToken } from "../config";
import { clientAbortSignal } from "./http";
import {
  childEnv,
  liveChildCount,
  pythonEnv,
  runProcess,
  SubprocessError,
  terminateAllChildren,
} from "./subprocess";

const node = process.execPath;
const SLEEP = ["-e", "setTimeout(() => {}, 30000)"];

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "subprocess-test-"));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));

const isAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

const waitFor = async (condition: () => boolean, timeoutMs = 5_000) => {
  const deadline = Date.now() + timeoutMs;
  while (!condition() && Date.now() < deadline) await new Promise((r) => setTimeout(r, 20));
  return condition();
};

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

  test("a timeout stops the child and says which env var raises it", async () => {
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

  test("a stop sends SIGTERM first, so the child's cleanup runs, and settles after it exits", async () => {
    const marker = path.join(scratch, "cleaned-up");
    const controller = new AbortController();
    const script =
      `process.on("SIGTERM", () => { setTimeout(() => {` +
      ` require("fs").writeFileSync(${JSON.stringify(marker)}, "1"); process.exit(0); }, 200); });` +
      ` console.log("ready"); setTimeout(() => {}, 30000);`;
    const running = runProcess(node, ["-e", script], {
      label: "cleaner",
      timeoutMs: 30_000,
      signal: controller.signal,
      killGraceMs: 5_000,
    });
    await new Promise((resolve) => setTimeout(resolve, 300));
    controller.abort();
    await assert.rejects(running, (error: unknown) => error instanceof SubprocessError && error.reason === "aborted");
    // Settled only once the cleanup had run and the child had exited.
    assert.ok(fs.existsSync(marker));
  });

  test("a child that ignores SIGTERM is SIGKILLed after the grace period", async () => {
    const controller = new AbortController();
    const script = `process.on("SIGTERM", () => {}); setTimeout(() => {}, 30000);`;
    const running = runProcess(node, ["-e", script], {
      label: "stubborn",
      timeoutMs: 30_000,
      signal: controller.signal,
      killGraceMs: 300,
    });
    await new Promise((resolve) => setTimeout(resolve, 200));
    const started = Date.now();
    controller.abort();
    await assert.rejects(running, (error: unknown) => error instanceof SubprocessError);
    const elapsed = Date.now() - started;
    assert.ok(elapsed >= 250 && elapsed < 5_000, `settled after ${elapsed} ms`);
  });

  test("stopping a child also stops what it started", { skip: process.platform === "win32" }, async () => {
    const pidFile = path.join(scratch, "grandchild.pid");
    const controller = new AbortController();
    const script =
      `const { spawn } = require("child_process");` +
      ` const grandchild = spawn(process.execPath, ["-e", "setTimeout(() => {}, 30000)"], { stdio: "ignore" });` +
      ` require("fs").writeFileSync(${JSON.stringify(pidFile)}, String(grandchild.pid));` +
      ` setTimeout(() => {}, 30000);`;
    const running = runProcess(node, ["-e", script], {
      label: "parent",
      timeoutMs: 30_000,
      signal: controller.signal,
    }).catch((error) => error);
    assert.ok(await waitFor(() => fs.existsSync(pidFile)));
    const grandchild = Number(fs.readFileSync(pidFile, "utf8"));
    assert.ok(isAlive(grandchild));
    controller.abort();
    await running;
    assert.ok(await waitFor(() => !isAlive(grandchild)), "the grandchild outlived its parent");
  });

  test("the shutdown hook stops every live child", async () => {
    const first = runProcess(node, SLEEP, { label: "one", timeoutMs: 30_000 }).catch((e) => e);
    const second = runProcess(node, SLEEP, { label: "two", timeoutMs: 30_000 }).catch((e) => e);
    await waitFor(() => liveChildCount() === 2);
    await terminateAllChildren();
    for (const outcome of [await first, await second]) {
      assert.ok(outcome instanceof SubprocessError);
      assert.equal(outcome.reason, "shutdown");
      assert.match(outcome.message, /shutting down/);
    }
    assert.equal(liveChildCount(), 0);
  });
});

describe("child environments are allow-lists", () => {
  const SECRETS = {
    SLOTIFY_API_TOKEN: "operator-secret",
    ELEVENLABS_API_KEY: "el-key",
    OPENAI_API_KEY: "oa-key",
    HUGGINGFACE_TOKEN: "hf-key",
    AWS_SECRET_ACCESS_KEY: "aws",
  };
  const saved = Object.fromEntries(Object.keys(SECRETS).map((key) => [key, process.env[key]]));
  const withSecrets = <T>(run: () => T): T => {
    Object.assign(process.env, SECRETS);
    try {
      return run();
    } finally {
      for (const [key, value] of Object.entries(saved)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
  };

  test("ffmpeg and the analyser get the basics and no credential", () => {
    withSecrets(() => {
      for (const env of [childEnv("media"), childEnv("python"), pythonEnv()]) {
        assert.equal(env.PATH, process.env.PATH);
        for (const key of [...Object.keys(SECRETS), "SLOTIFY_INTERNAL_TOKEN"]) {
          assert.equal(env[key], undefined, key);
        }
      }
    });
  });

  test("the generate pipeline gets OpenAI and the callback token, not ElevenLabs", () => {
    withSecrets(() => {
      const env = childEnv("generate");
      assert.equal(env.OPENAI_API_KEY, "oa-key");
      assert.equal(env.SLOTIFY_INTERNAL_TOKEN, internalToken);
      assert.ok(env.SLOTIFY_OPENAI_MODELS);
      // Callback URLs are arguments; no CLI reads it.
      assert.equal(env.API_BASE_URL, undefined);
      assert.equal(env.ELEVENLABS_API_KEY, undefined);
      assert.equal(env.HUGGINGFACE_TOKEN, undefined);
      assert.equal(env.SLOTIFY_API_TOKEN, undefined);
      assert.equal(env.AWS_SECRET_ACCESS_KEY, undefined);
    });
  });

  test("the two-speaker pipeline also gets ElevenLabs and Hugging Face", () => {
    withSecrets(() => {
      const env = childEnv("ad-insert");
      assert.equal(env.ELEVENLABS_API_KEY, "el-key");
      assert.equal(env.HUGGINGFACE_TOKEN, "hf-key");
      assert.equal(env.SLOTIFY_INTERNAL_TOKEN, internalToken);
      assert.equal(env.SLOTIFY_API_TOKEN, undefined);
    });
  });

  test("every Python child gets Whisper's cache location; ffmpeg does not need it", () => {
    const saved = process.env.XDG_CACHE_HOME;
    process.env.XDG_CACHE_HOME = "/var/cache/slotify";
    try {
      for (const kind of ["python", "generate", "ad-insert"] as const) {
        assert.equal(childEnv(kind).XDG_CACHE_HOME, "/var/cache/slotify", kind);
      }
      assert.equal(childEnv("media").XDG_CACHE_HOME, undefined);
    } finally {
      if (saved === undefined) delete process.env.XDG_CACHE_HOME;
      else process.env.XDG_CACHE_HOME = saved;
    }
  });

  test("the operator token never reaches a child, even through `extra`", () => {
    assert.equal(childEnv("ad-insert", { SLOTIFY_API_TOKEN: "x" }).SLOTIFY_API_TOKEN, undefined);
    assert.equal(pythonEnv({ EXTRA: "1" }).EXTRA, "1");
  });

  test("runProcess does not hand a child process.env by default", async () => {
    const { stdout } = await withSecrets(() =>
      runProcess(node, ["-e", "process.stdout.write(JSON.stringify(process.env))"], {
        label: "env",
        timeoutMs: 10_000,
      }),
    );
    const env = JSON.parse(stdout) as Record<string, string>;
    for (const key of Object.keys(SECRETS)) assert.equal(env[key], undefined, key);
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
