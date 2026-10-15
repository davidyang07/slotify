/**
 * Bridge from the Express API to the PyTorch ranker in `ml/`.
 *
 * **One persistent worker.** `python -m slotify_rank.cli infer serve` loads the
 * checkpoint, its normalizer, Whisper, the Whisper encoder and MiniLM once and
 * then answers newline-delimited JSON requests (`slotify_rank.inference.worker`
 * documents the protocol). The per-request CLI it replaces paid a torch import
 * and three model loads on every request -- most of a short clip's latency.
 * This module owns exactly one such process:
 *
 *   - started on first use, or at startup with SLOTIFY_RANKER_PRELOAD=1;
 *   - requests are queued and sent one at a time, in arrival order;
 *   - each request gets SLOTIFY_RANKER_TIMEOUT_MS from the moment it is sent;
 *   - on a timeout, a crash, a protocol violation or an abort of the running
 *     request, the worker is stopped (SIGTERM, SLOTIFY_KILL_GRACE_MS, SIGKILL)
 *     and the next request starts a fresh one; an aborted *queued* request is
 *     simply dropped -- the worker never sees it;
 *   - a request is rejected only after any worker it was running on has exited,
 *     so the caller's `finally` never removes a work dir out from under it;
 *   - an `onShutdown` hook stops it with the API.
 *
 * SLOTIFY_RANKER_WORKER=0 keeps the old path: one `infer rank` process per
 * request. Both paths hand the Python side a per-request work dir that this
 * module creates and removes; the upload's scratch copy lives there, never in a
 * temp dir the Python side would have to clean up after being killed.
 *
 * **Environment.** The child gets an allow-list (`rankerEnv`), never
 * process.env: it needs binaries, locale, temp, Python, the Hugging Face / torch
 * caches and thread settings, and nothing else -- above all not
 * SLOTIFY_API_TOKEN, SLOTIFY_INTERNAL_TOKEN or a provider key. The ranker never
 * calls back into this API.
 *
 * Contract with the ML side, either path: a ranking document that passes
 * `validateLearnedRanking`, or an error. Nothing here repairs a malformed
 * payload -- a ranking that cannot be parsed is a failure, not something to
 * substitute for. `services/ranker.ts` decides what a failure means (RANKER_MODE
 * =learned fails loudly; auto falls back to the heuristic and says why).
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";
import { killGraceMs } from "../config";
import { clientSafeDetail } from "../lib/errors";
import { isShuttingDown, onShutdown } from "../lib/lifecycle";
import { mlDir, mlPythonBin, REPO_ROOT, resolveRankerMode } from "../lib/ranker-mode";
import { runProcess, SubprocessError } from "../lib/subprocess";
import { makeTempDir, removeTempDir } from "../lib/temp";

export interface LearnedRankedCandidate {
  candidate_id: string;
  insertion_time_seconds: number;
  insertion_ms: number;
  raw_score: number;
  normalized_score: number;
  rank: number;
  acceptability_logit: number | null;
  audio_available: boolean;
  text_available: boolean;
  feature_status: string;
}

export interface LearnedRankingPayload {
  schema_version: string;
  episode_id: string;
  duration_seconds: number | null;
  score_scale: string;
  is_calibrated_probability: boolean;
  candidate_count: number;
  ranked: LearnedRankedCandidate[];
  excluded: Array<{ candidate_id: string; reason: string; detail: string }>;
  warnings: string[];
  timings_seconds: Record<string, number>;
  model: {
    model_variant: string;
    model_run_id: string;
    parameter_count: number;
    training_label_source: string;
    training_data_provenance: string;
    [key: string]: unknown;
  };
}

/** Schema versions this client knows how to read. */
const SUPPORTED_SCHEMA_VERSIONS = new Set(["inference-v1.0.0"]);

/** `message` is safe to show a client; `detail` (stderr) is for the server log. */
export class LearnedRankerError extends Error {
  constructor(
    message: string,
    readonly detail = "",
  ) {
    super(message);
    this.name = "LearnedRankerError";
  }
}

export interface RunLearnedRankerOptions {
  audioPath: string;
  checkpointPath: string;
  normalizerPath: string | null;
  episodeId: string;
  /** Skip Whisper transcription. Faster; every candidate is then text-masked. */
  skipTranscription?: boolean;
  timeoutMs?: number;
  /** Abandons the request when aborted (the client went away). */
  signal?: AbortSignal;
}

// ---------------------------------------------------------------------------
// Configuration

/** setTimeout treats anything larger as 1 ms. */
const MAX_TIMER_MS = 2 ** 31 - 1;
const DEFAULT_TIMEOUT_MS = 600_000;

/**
 * SLOTIFY_RANKER_TIMEOUT_MS, validated. Unset means 10 minutes. Anything that is
 * not a positive number of milliseconds a timer can hold is a configuration
 * error: parsed leniently, "10m" became NaN, NaN became a 1 ms timer, and
 * RANKER_MODE=auto then fell back to the heuristic on every request.
 */
export const rankerTimeoutMsFromEnv = (
  raw: string | undefined = process.env.SLOTIFY_RANKER_TIMEOUT_MS,
): number => {
  const text = raw?.trim();
  if (!text) return DEFAULT_TIMEOUT_MS;
  const value = Number(text);
  if (!Number.isFinite(value) || value <= 0 || value > MAX_TIMER_MS) {
    throw new Error(
      `SLOTIFY_RANKER_TIMEOUT_MS must be a positive number of milliseconds no larger ` +
        `than ${MAX_TIMER_MS}, got ${JSON.stringify(raw)}.`,
    );
  }
  return Math.ceil(value);
};

/** A 0/1 switch from the environment; anything else is a configuration error. */
const flagFromEnv = (name: string, fallback: boolean): boolean => {
  const raw = process.env[name]?.trim();
  if (!raw) return fallback;
  if (raw === "1") return true;
  if (raw === "0") return false;
  throw new Error(`${name} must be 0 or 1, got ${JSON.stringify(process.env[name])}.`);
};

/** SLOTIFY_RANKER_WORKER: 1 (default) the persistent worker, 0 one process per request. */
export const workerEnabled = (): boolean => flagFromEnv("SLOTIFY_RANKER_WORKER", true);

/** SLOTIFY_RANKER_PRELOAD=1: start the worker at startup instead of on first use. */
export const preloadEnabled = (): boolean => flagFromEnv("SLOTIFY_RANKER_PRELOAD", false);

// Read at startup, so a bad value stops the API before it serves a request
// rather than failing (or, in auto mode, quietly falling back on) every one.
const STARTUP_TIMEOUT_MS = rankerTimeoutMsFromEnv();
workerEnabled();
preloadEnabled();

const assertTimeout = (timeoutMs: number): void => {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > MAX_TIMER_MS) {
    throw new LearnedRankerError(`Invalid learned-ranker timeout ${timeoutMs} ms.`);
  }
};

// Exact names, then prefixes: what the ranker reads, and nothing else.
const ENV_KEYS = new Set([
  "PATH",
  "HOME",
  "LANG",
  "TMPDIR",
  // Where the Hugging Face caches live when HF_HOME is unset.
  "XDG_CACHE_HOME",
  "HUGGINGFACE_HUB_CACHE",
  // Read by the ML package itself: its ffmpeg, and its repository root.
  "FFMPEG_BIN",
  "FFPROBE_BIN",
  "SLOTIFY_REPO_ROOT",
]);
// Windows will not start Python without these.
const WINDOWS_ENV_KEYS = new Set([
  "SYSTEMROOT",
  "WINDIR",
  "COMSPEC",
  "PATHEXT",
  "TEMP",
  "TMP",
  "USERPROFILE",
  "APPDATA",
  "LOCALAPPDATA",
]);
const ENV_PREFIXES = [
  "LC_",
  "PYTHON",
  "HF_",
  "TRANSFORMERS_",
  "SENTENCE_TRANSFORMERS_",
  "TORCH",
  "OMP_",
  "MKL_",
  "SLOTIFY_RANKER_",
];
// A credential is never forwarded, whatever prefix it shares (HF_TOKEN).
const CREDENTIAL = /TOKEN|SECRET|PASSWORD|CREDENTIAL|API_?KEY/i;

/**
 * The ranker child's environment: an allow-list over `source`, plus `extra`.
 * Exported for tests.
 */
export const rankerEnv = (
  source: NodeJS.ProcessEnv = process.env,
  extra: NodeJS.ProcessEnv = {},
): NodeJS.ProcessEnv => {
  const windows = process.platform === "win32";
  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(source)) {
    if (value === undefined) continue;
    // Windows environment names are case-insensitive ("Path").
    const name = windows ? key.toUpperCase() : key;
    const allowed =
      ENV_KEYS.has(name) ||
      (windows && WINDOWS_ENV_KEYS.has(name)) ||
      ENV_PREFIXES.some((prefix) => name.startsWith(prefix));
    if (allowed && !CREDENTIAL.test(name)) env[key] = value;
  }
  return { ...env, ...extra };
};

const childEnvironment = (): NodeJS.ProcessEnv =>
  rankerEnv(process.env, {
    // The ml package is installed into ml/.venv in editable mode, but a bare
    // `python` on PATH would not see it; src/ on PYTHONPATH makes the child
    // work either way.
    PYTHONPATH: [path.join(mlDir, "src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
  });

// ---------------------------------------------------------------------------
// Entry point

export const runLearnedRanker = async ({
  audioPath,
  checkpointPath,
  normalizerPath,
  episodeId,
  skipTranscription = false,
  timeoutMs = STARTUP_TIMEOUT_MS,
  signal,
}: RunLearnedRankerOptions): Promise<LearnedRankingPayload> => {
  assertTimeout(timeoutMs);
  if (signal?.aborted) throw notStarted("aborted");
  if (isShuttingDown()) throw notStarted("shutdown");
  const job = { audioPath, checkpointPath, normalizerPath, episodeId, skipTranscription };
  // Queued synchronously, so concurrent callers are served in call order.
  if (workerEnabled()) return rankWithWorker(job, timeoutMs, signal);
  return inWorkDir((workDir) => rankWithCli({ ...job, workDir }, timeoutMs, signal));
};

interface RankJob {
  audioPath: string;
  checkpointPath: string;
  normalizerPath: string | null;
  episodeId: string;
  skipTranscription: boolean;
}

/**
 * Run `body` with a fresh per-request work dir that is removed when it settles.
 * Every scratch file the Python side writes goes inside it. Created when the
 * request is actually sent, so a queued request holds none.
 */
const inWorkDir = async <T>(body: (workDir: string) => Promise<T>): Promise<T> => {
  const workDir = await makeTempDir("slotify-rank-");
  try {
    return await body(workDir);
  } finally {
    await removeTempDir(workDir);
  }
};

type StopReason = "timeout" | "aborted" | "shutdown" | "protocol" | "replaced";

const notStarted = (reason: "aborted" | "shutdown"): LearnedRankerError =>
  new LearnedRankerError(
    reason === "aborted"
      ? "The learned ranker was not started: the client disconnected."
      : "The learned ranker was not started: the server is shutting down.",
  );

const timeoutError = (timeoutMs: number, detail: string): LearnedRankerError =>
  new LearnedRankerError(
    `The learned ranker did not finish within ${timeoutMs} ms. Transcription ` +
      "dominates the runtime on long audio; raise SLOTIFY_RANKER_TIMEOUT_MS " +
      "or use a shorter clip.",
    detail,
  );

// ---------------------------------------------------------------------------
// The per-request CLI (SLOTIFY_RANKER_WORKER=0)

const rankWithCli = async (
  job: RankJob & { workDir: string },
  timeoutMs: number,
  signal: AbortSignal | undefined,
): Promise<LearnedRankingPayload> => {
  const args = [
    "-m",
    "slotify_rank.cli",
    "infer",
    "rank",
    "--audio",
    job.audioPath,
    "--checkpoint",
    job.checkpointPath,
    "--episode-id",
    job.episodeId,
    "--work-dir",
    job.workDir,
    "--output",
    "-",
  ];
  if (job.normalizerPath) args.push("--normalizer", job.normalizerPath);
  if (job.skipTranscription) args.push("--no-transcribe");

  let stdout: string;
  try {
    ({ stdout } = await runProcess(mlPythonBin(), args, {
      label: "the learned ranker",
      cwd: mlDir,
      env: childEnvironment(),
      timeoutMs,
      signal,
    }));
  } catch (error) {
    if (!(error instanceof SubprocessError)) throw error;
    if (error.reason === "timeout") throw timeoutError(timeoutMs, error.stderr);
    if (error.reason === "exit") {
      const last = clientSafeDetail(error.stderr);
      throw new LearnedRankerError(
        last ? `${error.message.replace(/\.$/, "")}: ${last}` : error.message,
        error.stderr,
      );
    }
    throw new LearnedRankerError(
      error.reason === "spawn"
        ? `Could not start the learned ranker (${path.basename(mlPythonBin())}).`
        : error.message,
      `${error.message}\n${error.stderr}`,
    );
  }
  return parseLearnedRanking(stdout);
};

// ---------------------------------------------------------------------------
// The persistent worker

// Each worker leads its own process group, so stopping it also stops the
// ffmpeg it may be running instead of orphaning it.
const USE_PROCESS_GROUPS = process.platform !== "win32";
const MAX_STDERR_CHARS = 64 * 1024;
/** No response line is anywhere near this; a longer one is a broken worker. */
const MAX_LINE_CHARS = 16 * 1024 * 1024;

/** What a worker is bound to. A request for anything else needs a new worker. */
interface WorkerKey {
  python: string;
  checkpointPath: string;
  normalizerPath: string | null;
}

const sameKey = (a: WorkerKey, b: WorkerKey): boolean =>
  a.python === b.python && a.checkpointPath === b.checkpointPath && a.normalizerPath === b.normalizerPath;

interface Envelope {
  id: unknown;
  ok?: unknown;
  payload?: unknown;
  error?: unknown;
}

interface InFlight {
  id: number;
  resolve: (envelope: Envelope) => void;
  reject: (error: LearnedRankerError) => void;
  timeoutMs: number;
}

/** One `infer serve` process. */
class RankerWorker {
  readonly child: ChildProcessWithoutNullStreams;
  readonly exited: Promise<void>;
  private markExited!: () => void;
  private hasExited = false;
  private exitStatus: string | null = null;
  private spawnError: Error | null = null;
  private stdoutBuffer = "";
  /** stderr since the current request was sent (or since spawn). */
  private stderrTail = "";
  private inFlight: InFlight | null = null;
  private stopReason: StopReason | null = null;
  private protocolDetail = "";
  private graceTimer: NodeJS.Timeout | undefined;
  private closeWaitTimer: NodeJS.Timeout | undefined;
  private readonly unregisterShutdown: () => void;

  constructor(readonly key: WorkerKey) {
    const args = ["-m", "slotify_rank.cli", "infer", "serve", "--checkpoint", key.checkpointPath];
    if (key.normalizerPath) args.push("--normalizer", key.normalizerPath);
    this.exited = new Promise<void>((done) => {
      this.markExited = done;
    });
    this.child = spawn(key.python, args, {
      cwd: mlDir,
      env: childEnvironment(),
      stdio: ["pipe", "pipe", "pipe"],
      detached: USE_PROCESS_GROUPS,
    });
    // An idle worker never keeps the API alive by itself (it exits on its own
    // when its stdin closes); one with a request in flight or being stopped does.
    this.holdEventLoop(false);
    this.unregisterShutdown = onShutdown(() => shutdownWorker());

    this.child.stdin.on("error", () => undefined); // EPIPE: the exit reports it
    this.child.stdout.setEncoding("utf8");
    this.child.stderr.setEncoding("utf8");
    this.child.stdout.on("data", (chunk: string) => this.onStdout(chunk));
    this.child.stderr.on("data", (chunk: string) => {
      this.stderrTail += chunk;
      if (this.stderrTail.length > MAX_STDERR_CHARS) {
        this.stderrTail = this.stderrTail.slice(-MAX_STDERR_CHARS);
      }
    });
    this.child.on("error", (error) => {
      // Never spawned (ENOENT, EACCES): there is no process to wait for.
      if (this.child.pid === undefined) {
        this.spawnError = error;
        this.finalize();
      }
    });
    this.child.on("exit", (code, signal) => {
      this.exitStatus = code !== null ? `code ${code}` : `signal ${signal}`;
      clearTimeout(this.graceTimer);
      // Whatever the worker started (an ffmpeg mid-normalize) goes with it.
      this.signal("SIGKILL");
      // `close` follows once stdout is drained -- a last response line written
      // just before exiting is read first. Do not wait forever for pipes a
      // stray grandchild might hold.
      this.closeWaitTimer = setTimeout(() => this.finalize(), 1_000);
    });
    this.child.on("close", () => this.finalize());
  }

  get alive(): boolean {
    return !this.hasExited && this.exitStatus === null && this.stopReason === null;
  }

  get pid(): number | undefined {
    return this.child.pid;
  }

  private holdEventLoop(hold: boolean): void {
    const method = hold ? "ref" : "unref";
    this.child[method]();
    for (const stream of [this.child.stdin, this.child.stdout, this.child.stderr]) {
      (stream as unknown as Record<string, (() => void) | undefined>)[method]?.();
    }
  }

  /** What the worker wrote to stderr since the current request was sent. */
  get stderr(): string {
    return this.stderrTail;
  }

  /** Send one request; settles with its response envelope or a failure. */
  request(
    id: number,
    job: RankJob & { workDir: string },
    timeoutMs: number,
    signal: AbortSignal | undefined,
  ): Promise<Envelope> {
    return new Promise<Envelope>((resolve, reject) => {
      // Aborted after it left the queue but before it was sent: never send it.
      if (signal?.aborted) {
        reject(notStarted("aborted"));
        return;
      }
      if (!this.alive) {
        // Died (or is being stopped) before this request was sent: say why.
        reject(
          this.hasExited
            ? this.exitError()
            : new LearnedRankerError("The learned ranker worker is stopping.", this.stderrTail),
        );
        return;
      }
      let timer: NodeJS.Timeout | undefined;
      const onAbort = () => void this.stop("aborted");
      const finish = () => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        this.inFlight = null;
        if (this.alive) this.holdEventLoop(false);
      };
      this.inFlight = {
        id,
        timeoutMs,
        resolve: (envelope) => {
          finish();
          resolve(envelope);
        },
        reject: (error) => {
          finish();
          reject(error);
        },
      };
      this.stderrTail = "";
      this.holdEventLoop(true);
      timer = setTimeout(() => void this.stop("timeout"), timeoutMs);
      signal?.addEventListener("abort", onAbort, { once: true });
      this.child.stdin.write(
        JSON.stringify({
          id,
          audio_path: job.audioPath,
          work_dir: job.workDir,
          episode_id: job.episodeId,
          skip_transcription: job.skipTranscription,
        }) + "\n",
      );
    });
  }

  /**
   * SIGTERM (the worker removes its scratch files and exits), then SIGKILL
   * after SLOTIFY_KILL_GRACE_MS. Resolves once the process has exited; a
   * request in flight is rejected at that point, for `reason`.
   */
  stop(reason: StopReason): Promise<void> {
    if (this.hasExited || this.stopReason !== null) return this.exited;
    this.stopReason = reason;
    // Stay alive until the exit has been seen, so the request can be settled.
    this.holdEventLoop(true);
    if (this.exitStatus !== null) return this.exited; // exiting already
    if (killGraceMs <= 0) {
      this.signal("SIGKILL");
    } else {
      this.signal("SIGTERM");
      this.graceTimer = setTimeout(() => this.signal("SIGKILL"), killGraceMs);
    }
    return this.exited;
  }

  private signal(name: NodeJS.Signals): void {
    try {
      if (USE_PROCESS_GROUPS && this.child.pid) process.kill(-this.child.pid, name);
      else this.child.kill(name);
    } catch {
      // Already gone.
    }
  }

  /** The process is gone and its output read: settle whatever was in flight. */
  private finalize(): void {
    if (this.hasExited) return;
    this.hasExited = true;
    clearTimeout(this.graceTimer);
    clearTimeout(this.closeWaitTimer);
    this.unregisterShutdown();
    const error = this.exitError();
    if (this.inFlight) {
      this.inFlight.reject(error);
    } else if (this.stopReason === null) {
      // Died between requests (or while starting up, under SLOTIFY_RANKER_PRELOAD).
      console.warn(`Learned ranker worker (pid ${this.child.pid}): ${error.message}`);
    }
    this.markExited();
  }

  private exitError(): LearnedRankerError {
    const detail = this.stderrTail;
    if (this.spawnError) {
      return new LearnedRankerError(
        `Could not start the learned ranker (${path.basename(this.key.python)}).`,
        `${this.spawnError.message}\n${detail}`,
      );
    }
    switch (this.stopReason) {
      case "timeout":
        return timeoutError(this.inFlight?.timeoutMs ?? 0, detail);
      case "aborted":
        return new LearnedRankerError(
          "The learned ranker was stopped because the client disconnected.",
          detail,
        );
      case "shutdown":
        return new LearnedRankerError(
          "The learned ranker was stopped because the server is shutting down.",
          detail,
        );
      case "replaced":
        return new LearnedRankerError(
          "The learned ranker was restarted for a different checkpoint.",
          detail,
        );
      case "protocol":
        return new LearnedRankerError(
          "The learned ranker worker broke its output protocol and was restarted.",
          `${this.protocolDetail}\n${detail}`,
        );
      case null: {
        const last = clientSafeDetail(detail);
        return new LearnedRankerError(
          `The learned ranker worker exited with ${this.exitStatus ?? "an unknown status"}` +
            (last ? `: ${last}` : "."),
          detail,
        );
      }
    }
  }

  private onStdout(chunk: string): void {
    this.stdoutBuffer += chunk;
    let newline = this.stdoutBuffer.indexOf("\n");
    while (newline !== -1) {
      const line = this.stdoutBuffer.slice(0, newline).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
      if (line) this.onLine(line);
      newline = this.stdoutBuffer.indexOf("\n");
    }
    if (this.stdoutBuffer.length > MAX_LINE_CHARS) {
      this.violation(`a response line longer than ${MAX_LINE_CHARS} characters`);
    }
  }

  private onLine(line: string): void {
    if (this.stopReason !== null) return;
    let envelope: unknown;
    try {
      envelope = JSON.parse(line);
    } catch {
      this.violation(`a stdout line that is not JSON: ${line.slice(0, 200)}`);
      return;
    }
    if (envelope === null || typeof envelope !== "object" || Array.isArray(envelope)) {
      this.violation(`a stdout line that is not a JSON object: ${line.slice(0, 200)}`);
      return;
    }
    const message = envelope as Envelope;
    if (!this.inFlight) {
      this.violation(`a response with no request in flight (id ${JSON.stringify(message.id)})`);
      return;
    }
    if (message.id !== this.inFlight.id) {
      this.violation(
        `a response for request ${JSON.stringify(message.id)} while ${this.inFlight.id} was in flight`,
      );
      return;
    }
    this.inFlight.resolve(message);
  }

  /** Called with a protocol problem. Stops the worker, then rejects the request. */
  violation(what: string): void {
    if (this.stopReason !== null) return;
    this.protocolDetail = `The worker wrote ${what}.`;
    console.warn(`Learned ranker worker (pid ${this.child.pid}): ${this.protocolDetail}`);
    void this.stop("protocol");
  }
}

let worker: RankerWorker | null = null;
let nextRequestId = 1;

interface Queued {
  job: RankJob;
  key: WorkerKey;
  timeoutMs: number;
  signal: AbortSignal | undefined;
  resolve: (payload: LearnedRankingPayload) => void;
  reject: (error: unknown) => void;
  onAbort: () => void;
}

const queue: Queued[] = [];
let running = false;

/** The worker for `key`, starting (or replacing) it if need be. */
const workerFor = async (key: WorkerKey): Promise<RankerWorker> => {
  if (worker && worker.alive && sameKey(worker.key, key)) return worker;
  if (worker) await worker.stop("replaced");
  // Shutdown may have begun while the old worker was stopping. A worker started
  // now would register its hook after the hooks ran, and outlive them.
  if (isShuttingDown()) throw notStarted("shutdown");
  worker = new RankerWorker(key);
  return worker;
};

const rankWithWorker = (
  job: RankJob,
  timeoutMs: number,
  signal: AbortSignal | undefined,
): Promise<LearnedRankingPayload> =>
  new Promise<LearnedRankingPayload>((resolve, reject) => {
    const entry: Queued = {
      job,
      key: { python: mlPythonBin(), checkpointPath: job.checkpointPath, normalizerPath: job.normalizerPath },
      timeoutMs,
      signal,
      resolve,
      reject,
      // Aborted while waiting: leave the queue; the worker never sees it.
      onAbort: () => {
        const index = queue.indexOf(entry);
        if (index === -1) return;
        queue.splice(index, 1);
        reject(notStarted("aborted"));
      },
    };
    signal?.addEventListener("abort", entry.onAbort, { once: true });
    queue.push(entry);
    void drain();
  });

/** Run queued requests one at a time until the queue is empty. */
const drain = async (): Promise<void> => {
  if (running) return;
  running = true;
  try {
    for (let entry = queue.shift(); entry; entry = queue.shift()) {
      entry.signal?.removeEventListener("abort", entry.onAbort);
      try {
        entry.resolve(await runOnWorker(entry));
      } catch (error) {
        entry.reject(error);
      }
    }
  } finally {
    running = false;
  }
};

const runOnWorker = async ({ job, key, timeoutMs, signal }: Queued): Promise<LearnedRankingPayload> => {
  if (signal?.aborted) throw notStarted("aborted");
  if (isShuttingDown()) throw notStarted("shutdown");
  // A failed request settles only after its worker has exited, so the work
  // dir is never removed out from under a running process.
  return inWorkDir(async (workDir) => {
    const current = await workerFor(key);
    return sendTo(current, { ...job, workDir }, timeoutMs, signal);
  });
};

const sendTo = async (
  current: RankerWorker,
  job: RankJob & { workDir: string },
  timeoutMs: number,
  signal: AbortSignal | undefined,
): Promise<LearnedRankingPayload> => {
  const envelope = await current.request(nextRequestId++, job, timeoutMs, signal);
  if (envelope.ok === true) return validateLearnedRanking(envelope.payload);
  if (envelope.ok === false && typeof envelope.error === "string") {
    // A request-level failure (unreadable audio, an incompatible checkpoint):
    // the worker answered, and unless the failure was unexpected it is still
    // serving. Its stderr has the full message for the server log.
    const last = clientSafeDetail(envelope.error);
    throw new LearnedRankerError(
      last ? `The learned ranker failed: ${last}` : "The learned ranker failed.",
      current.stderr || envelope.error,
    );
  }
  // Answered, but not with anything this client can read. The request is
  // rejected once the worker has stopped, like every other failure.
  current.violation(`a response envelope without ok/payload/error: ${JSON.stringify(envelope).slice(0, 200)}`);
  await current.exited;
  throw new LearnedRankerError(
    "The learned ranker worker broke its output protocol and was restarted.",
    current.stderr,
  );
};

/**
 * Stop the worker and refuse whatever is queued. The shutdown hook; also used
 * by tests. Resolves once the process has exited.
 */
export const stopLearnedRankerWorker = async (): Promise<void> => {
  const current = worker;
  worker = null;
  if (current) await current.stop("shutdown");
};

const shutdownWorker = async (): Promise<void> => {
  for (const entry of queue.splice(0)) {
    entry.signal?.removeEventListener("abort", entry.onAbort);
    entry.reject(notStarted("shutdown"));
  }
  await stopLearnedRankerWorker();
};

/**
 * Start the worker now, for the configured checkpoint, so the first request
 * does not pay for loading it. A no-op unless the learned ranker is configured
 * and the worker path is on; never throws (a failed start surfaces on the
 * first request). Returns the worker's pid, or null when nothing was started.
 */
export const preloadLearnedRanker = async (): Promise<number | null> => {
  try {
    if (!workerEnabled() || isShuttingDown()) return null;
    const resolved = resolveRankerMode();
    if (resolved.mode !== "learned" || !resolved.checkpointPath) return null;
    const started = await workerFor({
      python: mlPythonBin(),
      checkpointPath: resolved.checkpointPath,
      normalizerPath: resolved.normalizerPath,
    });
    console.log(`Learned ranker worker started ahead of the first request (pid ${started.pid}).`);
    return started.pid ?? null;
  } catch (error) {
    console.warn("Learned ranker preload skipped:", error instanceof Error ? error.message : error);
    return null;
  }
};

if (preloadEnabled()) {
  // After the importing modules have finished loading, not during.
  setImmediate(() => {
    void preloadLearnedRanker();
  });
}

// ---------------------------------------------------------------------------
// Validation

/** Parse and validate the CLI's stdout. Exported so it can be tested directly. */
export const parseLearnedRanking = (stdout: string): LearnedRankingPayload => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout.trim() || "{}");
  } catch (error) {
    throw new LearnedRankerError(
      `The learned ranker produced output that is not JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  return validateLearnedRanking(parsed);
};

/**
 * Validate a ranking document, however it arrived (the CLI's stdout or a
 * worker response's `payload`). Exported so it can be tested directly.
 */
export const validateLearnedRanking = (parsed: unknown): LearnedRankingPayload => {
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new LearnedRankerError("The learned ranker's output is not a JSON object.");
  }
  const payload = parsed as Partial<LearnedRankingPayload>;
  if (
    typeof payload.schema_version !== "string" ||
    !SUPPORTED_SCHEMA_VERSIONS.has(payload.schema_version)
  ) {
    throw new LearnedRankerError(
      `Unsupported inference schema_version ${JSON.stringify(payload.schema_version)}; ` +
        `this API reads ${[...SUPPORTED_SCHEMA_VERSIONS].join(", ")}.`,
    );
  }
  if (!Array.isArray(payload.ranked)) {
    throw new LearnedRankerError(
      "The learned ranker's output has no ranked list; refusing to report a ranking.",
    );
  }
  if (payload.is_calibrated_probability === true) {
    // Nothing in this repository is calibrated. A payload claiming otherwise is
    // either a bug or a different model, and either way must not reach the UI.
    throw new LearnedRankerError(
      "The learned ranker claims calibrated probabilities, which no model in this " +
        "repository has been calibrated to produce.",
    );
  }
  if (!Array.isArray(payload.warnings) || !Array.isArray(payload.excluded)) {
    throw new LearnedRankerError(
      "The learned ranker's output has no warnings or excluded list; refusing to report a ranking.",
    );
  }
  const model = payload.model;
  if (
    !model ||
    typeof model !== "object" ||
    typeof model.model_variant !== "string" ||
    typeof model.model_run_id !== "string" ||
    typeof model.training_label_source !== "string" ||
    !model.training_label_source
  ) {
    // The provenance a response reports -- above all what the model was trained
    // on -- comes from here. A ranking that cannot say is not reported.
    throw new LearnedRankerError(
      "The learned ranker's output does not identify the model and its training " +
        "label source; refusing to report a ranking.",
    );
  }
  return payload as LearnedRankingPayload;
};

/** Path relative to the repository root, for logging. */
export const relativeToRepo = (target: string): string =>
  path.relative(REPO_ROOT, target);
