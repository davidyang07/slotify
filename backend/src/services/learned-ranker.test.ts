/**
 * The learned ranker's client: payload validation, configuration, the child's
 * environment, and the persistent worker's lifecycle.
 *
 * The worker is a stand-in "python" (SLOTIFY_ML_PYTHON) written in Node that
 * speaks the `infer serve` protocol and behaves according to the uploaded
 * file's name -- `hang.mp3` never answers, `crash.mp3` dies, `fail.mp3` gets an
 * `ok: false`, and so on -- and appends what it saw to an event log. Given
 * `infer rank` arguments it acts as the one-shot CLI instead.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, afterEach, before, describe, test } from "node:test";

import { runShutdownHooks } from "../lib/lifecycle";

// Read once, when config.ts is first imported: keep stop-and-kill quick here.
process.env.SLOTIFY_KILL_GRACE_MS = "300";
const {
  LearnedRankerError,
  parseLearnedRanking,
  preloadEnabled,
  preloadLearnedRanker,
  rankerEnv,
  rankerTimeoutMsFromEnv,
  runLearnedRanker,
  stopLearnedRankerWorker,
  validateLearnedRanking,
  workerEnabled,
} = await import("./learned-ranker");
const { rankCandidates } = await import("./ranker");

const validPayload = {
  schema_version: "inference-v1.0.0",
  episode_id: "upload",
  duration_seconds: 600,
  score_scale: "episode_relative_min_max",
  is_calibrated_probability: false,
  candidate_count: 2,
  ranked: [
    {
      candidate_id: "upload:000060000",
      insertion_time_seconds: 60,
      insertion_ms: 60_000,
      raw_score: 1.2,
      normalized_score: 100,
      rank: 1,
      acceptability_logit: 0.3,
      audio_available: true,
      text_available: true,
      feature_status: "complete",
    },
  ],
  excluded: [],
  warnings: [],
  timings_seconds: { score: 0.01 },
  model: {
    model_variant: "gated",
    model_run_id: "gated-abc",
    parameter_count: 489_477,
    training_label_source: "weak_heuristic",
    training_data_provenance: "weak_supervision",
  },
};

describe("parsing the learned ranker's output", () => {
  test("a well-formed payload is accepted", () => {
    const parsed = parseLearnedRanking(JSON.stringify(validPayload));
    assert.equal(parsed.ranked.length, 1);
    assert.equal(parsed.model.model_variant, "gated");
  });

  test("non-JSON output is a failure, not an empty ranking", () => {
    assert.throws(
      () => parseLearnedRanking("Traceback (most recent call last): ..."),
      LearnedRankerError,
    );
  });

  test("an unknown schema version is rejected", () => {
    const payload = { ...validPayload, schema_version: "inference-v9.0.0" };
    assert.throws(
      () => parseLearnedRanking(JSON.stringify(payload)),
      /Unsupported inference schema_version/,
    );
  });

  test("a payload with no ranked list is rejected rather than treated as empty", () => {
    const { ranked: _ranked, ...withoutRanked } = validPayload;
    assert.throws(
      () => parseLearnedRanking(JSON.stringify(withoutRanked)),
      /no ranked list/,
    );
  });

  test("a payload claiming calibrated probabilities is refused", () => {
    // Nothing here has been calibrated against human labels. A payload saying
    // otherwise must never reach a UI that would render it as a percentage.
    const payload = { ...validPayload, is_calibrated_probability: true };
    assert.throws(
      () => parseLearnedRanking(JSON.stringify(payload)),
      /claims calibrated probabilities/,
    );
  });

  test("an empty ranked list is a legitimate answer", () => {
    const payload = { ...validPayload, ranked: [], candidate_count: 0 };
    const parsed = parseLearnedRanking(JSON.stringify(payload));
    assert.deepEqual(parsed.ranked, []);
  });

  test("a worker's payload object goes through the same validation", () => {
    assert.equal(validateLearnedRanking(validPayload).model.model_run_id, "gated-abc");
    assert.throws(() => validateLearnedRanking(undefined), /not a JSON object/);
    assert.throws(
      () => validateLearnedRanking({ ...validPayload, warnings: undefined }),
      /no warnings or excluded list/,
    );
  });
});

describe("the payload must say what the model is", () => {
  test("a payload without the training label source is refused", () => {
    const { training_label_source: _source, ...model } = validPayload.model;
    assert.throws(
      () => parseLearnedRanking(JSON.stringify({ ...validPayload, model })),
      /training\s+label source/,
    );
  });

  test("a payload with no model block is refused, not a TypeError later", () => {
    const { model: _model, ...withoutModel } = validPayload;
    assert.throws(() => parseLearnedRanking(JSON.stringify(withoutModel)), LearnedRankerError);
  });

  test("a JSON value that is not an object is refused", () => {
    assert.throws(() => parseLearnedRanking("null"), LearnedRankerError);
    assert.throws(() => parseLearnedRanking('{"schema_version": {"toString": 1}}'), LearnedRankerError);
  });
});

// ---------------------------------------------------------------------------
// Configuration

describe("configuration", () => {
  test("the timeout defaults to ten minutes and accepts a positive number", () => {
    assert.equal(rankerTimeoutMsFromEnv(undefined), 600_000);
    assert.equal(rankerTimeoutMsFromEnv(""), 600_000);
    assert.equal(rankerTimeoutMsFromEnv(" 1500 "), 1_500);
    assert.equal(rankerTimeoutMsFromEnv("2.5"), 3);
  });

  test("a timeout that is not a positive timer value is a configuration error", () => {
    // Number.parseInt("10m") is 10 and Number("10m") is NaN, which setTimeout
    // treats as 1 ms -- auto mode then fell back to the heuristic every time.
    for (const raw of ["10m", "abc", "0", "-5", "NaN", "Infinity", "1e10"]) {
      assert.throws(() => rankerTimeoutMsFromEnv(raw), /SLOTIFY_RANKER_TIMEOUT_MS must be/, raw);
    }
  });

  test("an invalid timeout passed by a caller is refused before anything starts", async () => {
    await assert.rejects(
      runLearnedRanker({
        audioPath: "/dev/null",
        checkpointPath: "/x.pt",
        normalizerPath: null,
        episodeId: "upload",
        timeoutMs: Number.NaN,
      }),
      /Invalid learned-ranker timeout/,
    );
  });

  test("the worker and preload switches accept only 0 and 1", () => {
    const saved = { worker: process.env.SLOTIFY_RANKER_WORKER, preload: process.env.SLOTIFY_RANKER_PRELOAD };
    try {
      delete process.env.SLOTIFY_RANKER_WORKER;
      delete process.env.SLOTIFY_RANKER_PRELOAD;
      assert.equal(workerEnabled(), true);
      assert.equal(preloadEnabled(), false);
      process.env.SLOTIFY_RANKER_WORKER = "0";
      process.env.SLOTIFY_RANKER_PRELOAD = "1";
      assert.equal(workerEnabled(), false);
      assert.equal(preloadEnabled(), true);
      process.env.SLOTIFY_RANKER_WORKER = "yes";
      assert.throws(() => workerEnabled(), /SLOTIFY_RANKER_WORKER must be 0 or 1/);
    } finally {
      for (const [key, value] of [
        ["SLOTIFY_RANKER_WORKER", saved.worker],
        ["SLOTIFY_RANKER_PRELOAD", saved.preload],
      ] as const) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
  });
});

describe("the ranker child's environment", () => {
  const source: NodeJS.ProcessEnv = {
    PATH: "/usr/bin",
    HOME: "/home/x",
    LANG: "C.UTF-8",
    LC_ALL: "C.UTF-8",
    TMPDIR: "/tmp",
    PYTHONPATH: "/opt/py",
    PYTHONHASHSEED: "0",
    HF_HOME: "/cache/hf",
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_CACHE: "/cache/tf",
    SENTENCE_TRANSFORMERS_HOME: "/cache/st",
    TORCH_HOME: "/cache/torch",
    OMP_NUM_THREADS: "2",
    MKL_NUM_THREADS: "2",
    SLOTIFY_RANKER_SKIP_TRANSCRIPTION: "1",
    XDG_CACHE_HOME: "/cache",
    FFMPEG_BIN: "/opt/ffmpeg",
    SLOTIFY_API_TOKEN: "operator-secret",
    SLOTIFY_INTERNAL_TOKEN: "callback-secret",
    OPENAI_API_KEY: "sk-test",
    ELEVENLABS_API_KEY: "el-test",
    HF_TOKEN: "hf-secret",
    HUGGINGFACE_TOKEN: "hf-secret",
    SLOTIFY_RANKER_FAKE_API_KEY: "never",
    AWS_SECRET_ACCESS_KEY: "aws",
    NODE_OPTIONS: "--inspect",
    RANDOM_UNRELATED: "x",
  };

  test("only the allow-list reaches the child", () => {
    const env = rankerEnv(source);
    assert.deepEqual(Object.keys(env).sort(), [
      "FFMPEG_BIN",
      "HF_HOME",
      "HF_HUB_OFFLINE",
      "HOME",
      "LANG",
      "LC_ALL",
      "MKL_NUM_THREADS",
      "OMP_NUM_THREADS",
      "PATH",
      "PYTHONHASHSEED",
      "PYTHONPATH",
      "SENTENCE_TRANSFORMERS_HOME",
      "SLOTIFY_RANKER_SKIP_TRANSCRIPTION",
      "TMPDIR",
      "TORCH_HOME",
      "TRANSFORMERS_CACHE",
      "XDG_CACHE_HOME",
    ]);
  });

  test("extra values are added on top", () => {
    assert.equal(rankerEnv(source, { PYTHONPATH: "/ml/src" }).PYTHONPATH, "/ml/src");
  });
});

// ---------------------------------------------------------------------------
// The worker, against a stand-in

const FAKE_WORKER = String.raw`#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

const log = process.env.SLOTIFY_RANKER_FAKE_LOG;
const record = (event) => {
  if (log) fs.appendFileSync(log, JSON.stringify({ pid: process.pid, at: Date.now(), ...event }) + "\n");
};
const argv = process.argv.slice(2);
record({ event: "start", argv, cwd: process.cwd(), env: process.env });

process.on("SIGTERM", () => {
  record({ event: "sigterm" });
  if (process.env.SLOTIFY_RANKER_FAKE_IGNORE_TERM !== "1") process.exit(0);
});

const PAYLOAD = ${JSON.stringify(JSON.stringify(validPayload))};
const payload = () => {
  const body = JSON.parse(PAYLOAD);
  body.warnings = ["fake-pid:" + process.pid];
  return body;
};
const arg = (name) => argv[argv.indexOf(name) + 1];

if (argv.includes("rank")) {
  const workDir = arg("--work-dir");
  record({ event: "rank", workDirExists: fs.existsSync(workDir) });
  fs.writeFileSync(path.join(workDir, "scratch.txt"), "upload copy");
  process.stdout.write(JSON.stringify(payload()));
  process.exit(0);
}

if (process.env.SLOTIFY_RANKER_FAKE_STARTUP_FAIL === "1") {
  process.stderr.write("  models: checking the cache\n");
  process.stderr.write(
    "error: Pinned model snapshot(s) are not in the local Hugging Face cache; run " +
      "\x60python -m slotify_rank.cli infer warm\x60 (from ml/, with network access) to fetch them.\n",
  );
  process.exit(1);
}

const respond = (message) => {
  record({ event: "respond", id: message.id });
  process.stdout.write(JSON.stringify(message) + "\n");
};

readline.createInterface({ input: process.stdin }).on("line", (line) => {
  const request = JSON.parse(line);
  const name = path.basename(request.audio_path);
  record({ event: "request", id: request.id, name, request, workDirExists: fs.existsSync(request.work_dir) });
  fs.writeFileSync(path.join(request.work_dir, "scratch.txt"), "upload copy");
  process.stderr.write("  progress for " + name + "\n");
  switch (name) {
    case "hang.mp3":
      return;
    case "crash.mp3":
      process.stderr.write('Traceback (most recent call last):\n  File "/srv/ml/slotify_rank/cli.py", line 9\n');
      process.stderr.write("RuntimeError: CUDA out of memory\n");
      process.exit(1);
    case "fail.mp3":
      return respond({ id: request.id, ok: false, error: "[Errno 2] No such file or directory: '/srv/uploads/a/fail.mp3'" });
    case "garbage.mp3":
      return process.stdout.write("this is not json\n");
    case "wrong-id.mp3":
      return respond({ id: "someone-else", ok: true, payload: payload() });
    case "no-ok.mp3":
      return respond({ id: request.id, status: "done" });
    case "calibrated.mp3":
      return respond({ id: request.id, ok: true, payload: { ...payload(), is_calibrated_probability: true } });
    case "slow.mp3":
      return setTimeout(() => respond({ id: request.id, ok: true, payload: payload() }), 250);
    default:
      return respond({ id: request.id, ok: true, payload: payload() });
  }
}).on("close", () => {
  record({ event: "eof" });
  process.exit(0);
});
`;

const WORKER_ENV_KEYS = [
  "SLOTIFY_ML_PYTHON",
  "SLOTIFY_RANKER_WORKER",
  "SLOTIFY_RANKER_FAKE_LOG",
  "SLOTIFY_RANKER_FAKE_IGNORE_TERM",
  "SLOTIFY_RANKER_FAKE_STARTUP_FAIL",
  "RANKER_MODE",
  "SLOTIFY_RANKER_CHECKPOINT",
  "SLOTIFY_RANKER_NORMALIZER",
  "SLOTIFY_API_TOKEN",
  "SLOTIFY_INTERNAL_TOKEN",
  "OPENAI_API_KEY",
  "HF_HOME",
];
const savedEnv = Object.fromEntries(WORKER_ENV_KEYS.map((key) => [key, process.env[key]]));

let dir: string;
let logFile: string;
let checkpoint: string;

interface FakeEvent {
  pid: number;
  at: number;
  event: string;
  id?: number;
  name?: string;
  argv?: string[];
  cwd?: string;
  env?: Record<string, string>;
  request?: { work_dir: string; [key: string]: unknown };
  workDirExists?: boolean;
}

const events = (): FakeEvent[] =>
  fs.existsSync(logFile)
    ? fs
        .readFileSync(logFile, "utf8")
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line) as FakeEvent)
    : [];

const starts = () => events().filter((entry) => entry.event === "start");

const waitFor = async (predicate: () => boolean, what: string, ms = 5_000) => {
  const deadline = Date.now() + ms;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
};

const isAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

const rank = (name: string, extra: { timeoutMs?: number; signal?: AbortSignal; checkpointPath?: string } = {}) =>
  runLearnedRanker({
    audioPath: path.join(dir, name),
    checkpointPath: extra.checkpointPath ?? checkpoint,
    normalizerPath: path.join(dir, "normalizer.json"),
    episodeId: "upload",
    timeoutMs: extra.timeoutMs,
    signal: extra.signal,
  });

/** A promise awaited later in a test: handled from the start. */
const pending = <T>(promise: Promise<T>): Promise<T> => {
  promise.catch(() => undefined);
  return promise;
};

const pidOf = (payload: { warnings: string[] }): number =>
  Number(payload.warnings.find((warning) => warning.startsWith("fake-pid:"))?.slice("fake-pid:".length));

const workDirsSeen = (): string[] =>
  events()
    .filter((entry) => entry.event === "request" && entry.request)
    .map((entry) => entry.request!.work_dir);

describe("the persistent worker", { skip: process.platform === "win32" && "uses a POSIX stand-in" }, () => {
  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "learned-ranker-test-"));
    logFile = path.join(dir, "events.jsonl");
    checkpoint = path.join(dir, "best_checkpoint.pt");
    fs.writeFileSync(checkpoint, "");
    fs.writeFileSync(path.join(dir, "other.pt"), "");
    fs.writeFileSync(path.join(dir, "normalizer.json"), "{}");
    const fake = path.join(dir, "fake-worker.mjs");
    fs.writeFileSync(fake, FAKE_WORKER, { mode: 0o755 });
    process.env.SLOTIFY_ML_PYTHON = fake;
    process.env.SLOTIFY_RANKER_FAKE_LOG = logFile;
    delete process.env.SLOTIFY_RANKER_WORKER;
  });
  afterEach(async () => {
    await stopLearnedRankerWorker();
    fs.rmSync(logFile, { force: true });
    delete process.env.SLOTIFY_RANKER_FAKE_IGNORE_TERM;
    delete process.env.SLOTIFY_RANKER_FAKE_STARTUP_FAIL;
    delete process.env.SLOTIFY_RANKER_WORKER;
  });
  after(() => {
    for (const key of WORKER_ENV_KEYS) {
      if (savedEnv[key] === undefined) delete process.env[key];
      else process.env[key] = savedEnv[key];
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("one process serves request after request", async () => {
    const pids = [];
    for (let index = 0; index < 3; index += 1) pids.push(pidOf(await rank("ok.mp3")));
    assert.equal(new Set(pids).size, 1);
    assert.equal(starts().length, 1);
    const [start] = starts();
    assert.deepEqual(start.argv, [
      "-m", "slotify_rank.cli", "infer", "serve",
      "--checkpoint", checkpoint,
      "--normalizer", path.join(dir, "normalizer.json"),
    ]);
    assert.equal(path.basename(start.cwd ?? ""), "ml");
  });

  test("the worker gets the allow-list environment, never a credential", async () => {
    process.env.SLOTIFY_API_TOKEN = "operator-secret";
    process.env.SLOTIFY_INTERNAL_TOKEN = "callback-secret";
    process.env.OPENAI_API_KEY = "sk-test";
    process.env.HF_HOME = "/cache/hf";
    await rank("ok.mp3");
    const env = starts()[0].env ?? {};
    assert.equal(env.SLOTIFY_API_TOKEN, undefined);
    assert.equal(env.SLOTIFY_INTERNAL_TOKEN, undefined);
    assert.equal(env.OPENAI_API_KEY, undefined);
    assert.equal(env.HF_HOME, "/cache/hf");
    assert.ok(env.PYTHONPATH?.split(path.delimiter)[0].endsWith(path.join("ml", "src")), env.PYTHONPATH);
  });

  test("each request runs in its own work dir, removed when it settles", async () => {
    await rank("ok.mp3");
    await rank("ok.mp3");
    const dirs = workDirsSeen();
    assert.equal(dirs.length, 2);
    assert.notEqual(dirs[0], dirs[1]);
    assert.ok(events().filter((entry) => entry.event === "request").every((entry) => entry.workDirExists));
    for (const workDir of dirs) {
      assert.equal(path.dirname(workDir), os.tmpdir());
      assert.ok(!fs.existsSync(workDir), `${workDir} was left behind`);
    }
  });

  test("concurrent callers are queued and sent one at a time, in call order", async () => {
    const results = await Promise.all([rank("slow.mp3"), rank("first.mp3"), rank("second.mp3")]);
    assert.equal(new Set(results.map(pidOf)).size, 1);
    const sequence = events()
      .filter((entry) => entry.event === "request" || entry.event === "respond")
      .map((entry) => (entry.event === "request" ? `send ${entry.name}` : "answer"));
    assert.deepEqual(sequence, [
      "send slow.mp3", "answer", "send first.mp3", "answer", "send second.mp3", "answer",
    ]);
  });

  test("a timeout stops the worker and the next request gets a fresh one", async () => {
    await assert.rejects(rank("hang.mp3", { timeoutMs: 200 }), /did not finish within 200 ms/);
    const [first] = starts();
    assert.ok(!isAlive(first.pid), "the timed-out worker is still running");
    assert.ok(events().some((entry) => entry.event === "sigterm"), "SIGTERM came first");
    const next = await rank("ok.mp3");
    assert.notEqual(pidOf(next), first.pid);
    assert.ok(!workDirsSeen().some((workDir) => fs.existsSync(workDir)));
  });

  test("a worker that ignores SIGTERM is killed after the grace period", async () => {
    process.env.SLOTIFY_RANKER_FAKE_IGNORE_TERM = "1";
    const started = Date.now();
    await assert.rejects(rank("hang.mp3", { timeoutMs: 100 }), /did not finish/);
    // Rejected only once the process is gone: 100 ms timeout + 300 ms grace.
    assert.ok(Date.now() - started >= 380, `${Date.now() - started} ms`);
    assert.ok(!isAlive(starts()[0].pid));
  });

  test("a crash is reported by its last line, without paths, and the worker restarts", async () => {
    await assert.rejects(rank("crash.mp3"), (error: unknown) => {
      assert.ok(error instanceof LearnedRankerError);
      assert.match(error.message, /worker exited with code 1: RuntimeError: CUDA out of memory/);
      assert.ok(!error.message.includes("/srv/ml"), error.message);
      assert.match(error.detail, /Traceback/);
      return true;
    });
    const crashed = starts()[0].pid;
    assert.notEqual(pidOf(await rank("ok.mp3")), crashed);
  });

  test("a request the worker refuses is an error, and the worker keeps serving", async () => {
    const before = pidOf(await rank("ok.mp3"));
    await assert.rejects(rank("fail.mp3"), (error: unknown) => {
      assert.ok(error instanceof LearnedRankerError);
      assert.match(error.message, /^The learned ranker failed: .*No such file or directory/);
      assert.ok(!error.message.includes("/srv/uploads"), error.message);
      return true;
    });
    assert.equal(pidOf(await rank("ok.mp3")), before);
    assert.equal(starts().length, 1);
  });

  test("strict payload validation still applies, and keeps the worker", async () => {
    await assert.rejects(rank("calibrated.mp3"), /claims calibrated probabilities/);
    await rank("ok.mp3");
    assert.equal(starts().length, 1);
  });

  for (const [name, why] of [
    ["garbage.mp3", "a line that is not JSON"],
    ["wrong-id.mp3", "a response for another request"],
    ["no-ok.mp3", "an envelope without ok"],
  ] as const) {
    test(`a protocol violation (${why}) restarts the worker`, async () => {
      const original = console.warn;
      console.warn = () => undefined;
      try {
        await assert.rejects(rank(name), /broke its output protocol/);
      } finally {
        console.warn = original;
      }
      const [first] = starts();
      assert.ok(!isAlive(first.pid));
      assert.notEqual(pidOf(await rank("ok.mp3")), first.pid);
    });
  }

  test("an aborted queued request is dropped before the worker sees it", async () => {
    const controller = new AbortController();
    const running = pending(rank("slow.mp3"));
    const queued = pending(rank("ok.mp3", { signal: controller.signal }));
    await waitFor(() => events().some((entry) => entry.event === "request"), "the first request");
    controller.abort();
    await assert.rejects(queued, /not started: the client disconnected/);
    await running;
    const names = events().filter((entry) => entry.event === "request").map((entry) => entry.name);
    assert.deepEqual(names, ["slow.mp3"]);
    assert.equal(starts().length, 1);
    // The worker is still the same, still serving.
    assert.equal(pidOf(await rank("ok.mp3")), starts()[0].pid);
  });

  test("aborting the running request stops the worker", async () => {
    const controller = new AbortController();
    const running = pending(rank("hang.mp3", { signal: controller.signal }));
    await waitFor(() => events().some((entry) => entry.name === "hang.mp3"), "the hung request");
    controller.abort();
    await assert.rejects(running, /stopped because the client disconnected/);
    const [first] = starts();
    assert.ok(!isAlive(first.pid));
    assert.ok(!workDirsSeen().some((workDir) => fs.existsSync(workDir)));
    assert.notEqual(pidOf(await rank("ok.mp3")), first.pid);
  });

  test("an already-aborted request never starts a worker", async () => {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(rank("ok.mp3", { signal: controller.signal }), /not started: the client disconnected/);
    assert.equal(starts().length, 0);
  });

  test("a worker that cannot start says why, naming the fix", async () => {
    process.env.SLOTIFY_RANKER_FAKE_STARTUP_FAIL = "1";
    await assert.rejects(rank("ok.mp3"), (error: unknown) => {
      assert.ok(error instanceof LearnedRankerError);
      assert.match(error.message, /worker exited with code 1: error: Pinned model snapshot/);
      assert.match(error.message, /infer warm/);
      return true;
    });
  });

  test("an interpreter that cannot be spawned is a clear error", async () => {
    const saved = process.env.SLOTIFY_ML_PYTHON;
    process.env.SLOTIFY_ML_PYTHON = path.join(dir, "no-such-python");
    try {
      await assert.rejects(rank("ok.mp3"), /Could not start the learned ranker \(no-such-python\)/);
    } finally {
      process.env.SLOTIFY_ML_PYTHON = saved;
    }
  });

  test("a different checkpoint replaces the worker", async () => {
    const first = pidOf(await rank("ok.mp3"));
    const second = pidOf(await rank("ok.mp3", { checkpointPath: path.join(dir, "other.pt") }));
    assert.notEqual(first, second);
    assert.ok(!isAlive(first));
    assert.equal(starts()[1].argv?.[5], path.join(dir, "other.pt"));
  });

  test("preload starts the worker the first request then uses", async () => {
    process.env.RANKER_MODE = "learned";
    process.env.SLOTIFY_RANKER_CHECKPOINT = checkpoint;
    delete process.env.SLOTIFY_RANKER_NORMALIZER;
    const original = console.log;
    console.log = () => undefined;
    try {
      const pid = await preloadLearnedRanker();
      assert.ok(pid);
      assert.equal(starts().length <= 1, true);
      assert.equal(pidOf(await rank("ok.mp3")), pid);
      assert.equal(starts().length, 1);
    } finally {
      console.log = original;
      delete process.env.RANKER_MODE;
      delete process.env.SLOTIFY_RANKER_CHECKPOINT;
    }
  });

  test("preload is a no-op when the learned ranker is not configured", async () => {
    process.env.RANKER_MODE = "heuristic";
    try {
      assert.equal(await preloadLearnedRanker(), null);
      assert.equal(starts().length, 0);
    } finally {
      delete process.env.RANKER_MODE;
    }
  });

  test("SLOTIFY_RANKER_WORKER=0 runs infer rank per request, in its own work dir", async () => {
    process.env.SLOTIFY_RANKER_WORKER = "0";
    process.env.SLOTIFY_API_TOKEN = "operator-secret";
    const pids = [pidOf(await rank("ok.mp3")), pidOf(await rank("ok.mp3"))];
    assert.notEqual(pids[0], pids[1]);
    const [start] = starts();
    assert.deepEqual(start.argv?.slice(0, 4), ["-m", "slotify_rank.cli", "infer", "rank"]);
    const workDir = start.argv?.[start.argv.indexOf("--work-dir") + 1] ?? "";
    assert.ok(events().some((entry) => entry.event === "rank" && entry.workDirExists));
    assert.ok(!fs.existsSync(workDir), `${workDir} was left behind`);
    assert.equal(start.env?.SLOTIFY_API_TOKEN, undefined);
  });

  describe("through rankCandidates", () => {
    const request = () => ({
      candidates: [{ ms: 30_000, silenceMs: 900, snippet: "First point.", generatedBy: "analyser" as const }],
      episodeId: "upload",
      audioPath: path.join(dir, "crash.mp3"),
      durationSeconds: 120,
      mode: "podcast" as const,
      minSeparationSeconds: 6,
      count: 3,
    });
    before(() => {
      process.env.SLOTIFY_RANKER_CHECKPOINT = checkpoint;
    });
    after(() => {
      delete process.env.SLOTIFY_RANKER_CHECKPOINT;
      delete process.env.RANKER_MODE;
    });

    test("auto: a crashed worker falls back to the heuristic and says why", async () => {
      process.env.RANKER_MODE = "auto";
      const original = console.warn;
      console.warn = () => undefined;
      try {
        const result = await rankCandidates(request());
        assert.equal(result.source, "heuristic_offline_v1");
        assert.match(result.warnings[0], /fell back to the heuristic baseline/);
        assert.match(result.warnings[0], /CUDA out of memory/);
      } finally {
        console.warn = original;
      }
    });

    test("learned: the same crash fails loudly", async () => {
      process.env.RANKER_MODE = "learned";
      await assert.rejects(rankCandidates(request()), LearnedRankerError);
    });

    test("learned: a worker ranking is reported as the learned ranker's", async () => {
      process.env.RANKER_MODE = "learned";
      const result = await rankCandidates({ ...request(), audioPath: path.join(dir, "ok.mp3") });
      assert.equal(result.source, "learned_ranker");
      assert.equal(result.modelRunId, "gated-abc");
    });
  });

  // Last: shutdown is one-way for the whole process.
  test("shutdown stops the worker and refuses queued and new requests", async () => {
    // Both settle during the hooks, so their expectations are attached first.
    const running = assert.rejects(rank("hang.mp3"), /stopped because the server is shutting down/);
    const queued = assert.rejects(rank("ok.mp3"), /not started: the server is shutting down/);
    await waitFor(() => events().some((entry) => entry.name === "hang.mp3"), "the hung request");
    const [start] = starts();
    await runShutdownHooks(5_000);
    await running;
    await queued;
    assert.ok(!isAlive(start.pid));
    await assert.rejects(rank("ok.mp3"), /not started: the server is shutting down/);
    assert.equal(starts().length, 1);
  });
});
