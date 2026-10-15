/**
 * Graceful shutdown of the real server process: SIGTERM during an analysis
 * must leave no analyser process and no temp dir behind, and exit 0.
 *
 * The server runs with its own TMPDIR, so "no temp dir behind" and "no child
 * behind" can be checked exactly: every work dir, and every analyser's
 * command line, contains that path.
 */

import assert from "node:assert/strict";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { after, describe, test } from "node:test";

import { BACKEND_DIR } from "./config";

const SAMPLE_AUDIO = path.join(BACKEND_DIR, "audio_tests", "ryan-test1.mp3");
const python = process.env.PYTHON_BIN ?? "python";
const analyserAvailable =
  fs.existsSync(SAMPLE_AUDIO) &&
  spawnSync(python, ["-c", "import pydub"], { cwd: BACKEND_DIR, timeout: 60_000 }).status === 0;
const requireAnalyser = process.env.SLOTIFY_REQUIRE_ANALYSER === "1";
const unsupported = process.platform !== "linux" && "reads /proc";
const skip = unsupported || (!analyserAvailable && !requireAnalyser && "python analyser unavailable");

const freePort = () =>
  new Promise<number>((resolve, reject) => {
    const probe = net.createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      probe.close(() => resolve(typeof address === "object" && address ? address.port : 0));
    });
  });

/** Processes whose command line mentions `needle`. */
const processesMentioning = (needle: string): number[] =>
  fs
    .readdirSync("/proc")
    .filter((entry) => /^\d+$/.test(entry))
    .filter((pid) => {
      try {
        return fs.readFileSync(`/proc/${pid}/cmdline`, "utf8").includes(needle);
      } catch {
        return false;
      }
    })
    .map(Number);

const waitFor = async <T>(probe: () => T | null | undefined | false, timeoutMs: number) => {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = probe();
    if (value) return value;
    if (Date.now() > deadline) return null;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
};

describe("SIGTERM", { skip }, () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "slotify-shutdown-test-"));
  let server: ChildProcess | null = null;

  after(() => {
    if (server && server.exitCode === null) server.kill("SIGKILL");
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  test("stops the analyser, removes its work dir and exits 0", async () => {
    assert.ok(analyserAvailable, "SLOTIFY_REQUIRE_ANALYSER=1 but the analyser cannot run");
    const port = await freePort();
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      PORT: String(port),
      TMPDIR: tmp,
      RANKER_MODE: "heuristic",
      SLOTIFY_KILL_GRACE_MS: "3000",
      SLOTIFY_RATE_LIMIT_PAID_PER_MIN: "0",
      SLOTIFY_RATE_LIMIT_COMPUTE_PER_MIN: "0",
    };
    for (const key of ["SLOTIFY_API_TOKEN", "OPENAI_API_KEY", "ELEVENLABS_API_KEY"]) delete env[key];
    server = spawn(process.execPath, ["--import", "tsx", "src/index.ts"], {
      cwd: BACKEND_DIR,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    server.stdout!.on("data", (chunk) => (output += chunk));
    server.stderr!.on("data", (chunk) => (output += chunk));
    const exited = new Promise<number | null>((resolve) => server!.once("exit", (code) => resolve(code)));

    assert.ok(await waitFor(() => output.includes("API listening"), 30_000), output);

    const form = new FormData();
    form.append("audio", new Blob([fs.readFileSync(SAMPLE_AUDIO)]), "episode.mp3");
    const request = fetch(`http://127.0.0.1:${port}/api/insert-sections`, { method: "POST", body: form })
      .then((response) => response.status)
      .catch(() => "connection closed");

    // The analyser's --work-dir is under this server's TMPDIR.
    const analysers = await waitFor(() => {
      const found = processesMentioning("ad_inserter.analyze_cli").filter((pid) =>
        processesMentioning(tmp).includes(pid),
      );
      return found.length ? found : null;
    }, 30_000);
    assert.ok(analysers, `no analyser started:\n${output}`);
    assert.ok(fs.readdirSync(tmp).some((entry) => entry.startsWith("insert-sections-")));

    const signalled = Date.now();
    server.kill("SIGTERM");
    const code = await Promise.race([
      exited,
      new Promise<string>((resolve) => setTimeout(() => resolve("still running"), 25_000).unref()),
    ]);
    assert.equal(code, 0, output);
    // Hooks (15 s deadline) plus the drain; a clean stop is far quicker.
    assert.ok(Date.now() - signalled < 20_000);

    const outcome = await request;
    assert.ok(outcome === 503 || outcome === "connection closed", `client saw ${outcome}`);
    assert.deepEqual(processesMentioning(tmp), [], "a child outlived the server");
    // tsx keeps its own compile cache in TMPDIR; everything else is ours.
    const left = fs.readdirSync(tmp).filter((entry) => !entry.startsWith("tsx-"));
    assert.deepEqual(left, [], "a temp dir outlived the server");
    assert.match(output, /SIGTERM received; shutting down/);
  });
});
