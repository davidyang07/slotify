import { spawn, type ChildProcess } from "node:child_process";
import { allowedOpenAiModels, internalToken, killGraceMs } from "../config";
import { isShuttingDown, onShutdown } from "./lifecycle";

export type SubprocessFailure = "spawn" | "exit" | "timeout" | "aborted" | "output" | "shutdown";

/**
 * A child process that did not finish cleanly. `message` is ours and safe to
 * show; `stderr` is the child's own output and belongs in the server log.
 */
export class SubprocessError extends Error {
  constructor(
    message: string,
    readonly label: string,
    readonly reason: SubprocessFailure,
    readonly stderr: string,
  ) {
    super(message);
    this.name = "SubprocessError";
  }
}

// ---------------------------------------------------------------------------
// Environment
//
// Children used to inherit all of process.env: every provider key, and the
// operator's SLOTIFY_API_TOKEN, reached ffmpeg and the analyser, which need
// none of them. Each kind of child now gets a base environment plus only what
// it uses.

/** What every child needs to run at all: binaries, locale, temp dir. */
const BASE_ENV_KEYS = ["PATH", "HOME", "LANG", "LANGUAGE", "TMPDIR", "TZ"];
const BASE_ENV_PREFIXES = ["LC_", "PYTHON"];
// Windows will not start Python without these.
const WINDOWS_ENV_KEYS = [
  "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
];
// Thread counts and caches: how an operator keeps numpy/librosa from
// oversubscribing a small box, and where numba and Whisper keep their caches
// ($XDG_CACHE_HOME/whisper) when the home dir is read-only.
const PYTHON_RUNTIME_ENV_KEYS = [
  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_CACHE_DIR",
  "XDG_CACHE_HOME",
];
// Only for children that reach a provider: an egress proxy and its CA, and
// NO_PROXY so callbacks to this API are not sent through the proxy.
const NETWORK_ENV_KEYS = [
  "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "no_proxy",
  "all_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
];
const OPENAI_ENV_KEYS = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"];

export type ChildKind =
  /** ffmpeg / ffprobe: local files only. */
  | "media"
  /** A Python child with no provider and no callback (the analyser). */
  | "python"
  /** `ad_inserter.cli`: writes copy with OpenAI, calls back for TTS and merge. */
  | "generate"
  /** `ad_inserter.insert_ad`: also diarizes (Hugging Face) and deletes the voices it clones. */
  | "ad-insert";

interface ChildEnvSpec {
  keys: readonly string[];
  python: boolean;
  network: boolean;
  /**
   * Gets SLOTIFY_INTERNAL_TOKEN to call back into this API. The callback URLs
   * themselves are arguments (--tts-url etc.); no CLI reads API_BASE_URL.
   */
  callback: boolean;
}

const CHILD_ENV: Record<ChildKind, ChildEnvSpec> = {
  media: { keys: [], python: false, network: false, callback: false },
  python: { keys: [], python: true, network: false, callback: false },
  generate: {
    keys: [...OPENAI_ENV_KEYS, "SLOTIFY_CALLBACK_TIMEOUT_S"],
    python: true,
    network: true,
    callback: true,
  },
  "ad-insert": {
    keys: [
      ...OPENAI_ENV_KEYS,
      "SLOTIFY_CALLBACK_TIMEOUT_S",
      "ELEVENLABS_API_KEY",
      "ELEVENLABS_VOICE_ID_A",
      "ELEVENLABS_VOICE_ID_B",
      "ELEVENLABS_DEFAULT_VOICE_ID",
      "HUGGINGFACE_TOKEN",
      "PYANNOTE_TOKEN",
      "HF_TOKEN",
      "HF_HOME",
      "HF_HUB_CACHE",
      "TORCH_HOME",
    ],
    python: true,
    network: true,
    callback: true,
  },
};

const pick = (keys: readonly string[], into: NodeJS.ProcessEnv) => {
  for (const key of keys) {
    const value = process.env[key];
    if (value !== undefined) into[key] = value;
  }
};

/** The environment for one kind of child: an allow-list, never process.env. */
export const childEnv = (kind: ChildKind, extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv => {
  const spec = CHILD_ENV[kind];
  const env: NodeJS.ProcessEnv = {};
  pick(BASE_ENV_KEYS, env);
  if (process.platform === "win32") pick(WINDOWS_ENV_KEYS, env);
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined && BASE_ENV_PREFIXES.some((prefix) => key.startsWith(prefix))) {
      env[key] = value;
    }
  }
  if (spec.python) pick(PYTHON_RUNTIME_ENV_KEYS, env);
  if (spec.network) pick(NETWORK_ENV_KEYS, env);
  pick(spec.keys, env);
  if (spec.callback) {
    env.SLOTIFY_INTERNAL_TOKEN = internalToken;
    // The same allow-list /ad/insert enforces, so both sides agree.
    env.SLOTIFY_OPENAI_MODELS = allowedOpenAiModels().join(",");
  }
  Object.assign(env, extra);
  // The operator's credential is for clients; no child ever needs it.
  delete env.SLOTIFY_API_TOKEN;
  return env;
};

/**
 * The base environment for a Python child with no provider and no callback.
 * Callers add what their child needs through `extra`.
 */
export const pythonEnv = (extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv =>
  childEnv("python", extra);

// ---------------------------------------------------------------------------
// Running and stopping children

export interface RunProcessOptions {
  /** Human name for errors and logs, e.g. "ffmpeg". */
  label: string;
  timeoutMs: number;
  /** The env var that sets `timeoutMs`, named in the timeout error. */
  timeoutEnvVar?: string;
  /** Stops the child when aborted (the client went away). */
  signal?: AbortSignal;
  cwd?: string;
  /** Defaults to the "media" allow-list: nothing from process.env but the basics. */
  env?: NodeJS.ProcessEnv;
  /** SIGTERM-to-SIGKILL grace; defaults to SLOTIFY_KILL_GRACE_MS. */
  killGraceMs?: number;
}

// ffmpeg logs progress to stderr for the whole run; only the tail matters.
const MAX_STDERR_CHARS = 64 * 1024;
// After the child itself has exited, how long to wait for its stdio to close
// (a grandchild may still hold the pipes) before giving up on it.
const STDIO_CLOSE_WAIT_MS = 1_000;

// Each child leads its own process group, so stopping it also stops whatever
// it started (pydub's ffmpeg, Whisper's workers) instead of orphaning them:
// the Python process does not stop its own ffmpeg children. Windows has no
// process groups to signal; `taskkill /T` ends the tree instead.
const IS_WINDOWS = process.platform === "win32";
const USE_PROCESS_GROUPS = !IS_WINDOWS;

interface LiveChild {
  stop: (reason: SubprocessFailure) => void;
  exited: Promise<void>;
}

const liveChildren = new Set<LiveChild>();

/** Children started by `runProcess` that have not exited yet. */
export const liveChildCount = (): number => liveChildren.size;

const signalTree = (child: ChildProcess, signal: NodeJS.Signals) => {
  if (!child.pid) return;
  try {
    if (USE_PROCESS_GROUPS) {
      process.kill(-child.pid, signal);
    } else if (IS_WINDOWS) {
      const force = signal === "SIGKILL" ? ["/F"] : [];
      spawn("taskkill", ["/pid", String(child.pid), "/T", ...force], {
        stdio: "ignore",
        windowsHide: true,
      }).on("error", () => child.kill(signal));
    } else {
      child.kill(signal);
    }
  } catch {
    // Already gone.
  }
};

/**
 * Stop every live child (SIGTERM, grace, SIGKILL) and wait for them to exit.
 * Registered as a shutdown hook; exported for tests.
 */
export const terminateAllChildren = async (): Promise<void> => {
  const children = [...liveChildren];
  for (const entry of children) entry.stop("shutdown");
  await Promise.all(children.map((entry) => entry.exited));
};

onShutdown(terminateAllChildren);

const stopMessage = (label: string, reason: SubprocessFailure, timeoutMs: number, envVar?: string) => {
  if (reason === "timeout") {
    return (
      `${label} did not finish within ${timeoutMs / 1000} s and was stopped` +
      (envVar ? `; raise ${envVar} for longer audio.` : ".")
    );
  }
  if (reason === "shutdown") return `${label} was stopped because the server is shutting down.`;
  return `${label} was stopped because the client disconnected.`;
};

/**
 * Run a command to completion. On timeout, abort or shutdown the child (and its
 * process group) gets SIGTERM, so a Python pipeline's `finally` blocks can
 * remove what it wrote; after the grace period it gets SIGKILL, because a hung
 * ffmpeg or a Whisper pass on a three-hour upload can ignore SIGTERM for long
 * enough to matter. The promise settles only once the child has exited, so a
 * caller's `finally` never deletes a work dir out from under a running child.
 */
export const runProcess = (
  command: string,
  args: string[],
  {
    label,
    timeoutMs,
    timeoutEnvVar,
    signal,
    cwd,
    env,
    killGraceMs: graceMs = killGraceMs,
  }: RunProcessOptions,
): Promise<{ stdout: string; stderr: string }> =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(
        new SubprocessError(`${label} was not started: the client disconnected.`, label, "aborted", ""),
      );
      return;
    }
    if (isShuttingDown()) {
      reject(
        new SubprocessError(`${label} was not started: the server is shutting down.`, label, "shutdown", ""),
      );
      return;
    }

    const child = spawn(command, args, {
      cwd,
      env: env ?? childEnv("media"),
      stdio: ["ignore", "pipe", "pipe"],
      detached: USE_PROCESS_GROUPS,
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    let stopReason: SubprocessFailure | null = null;
    let graceTimer: NodeJS.Timeout | undefined;
    let closeWaitTimer: NodeJS.Timeout | undefined;

    let markExited!: () => void;
    const entry: LiveChild = {
      stop: (reason) => stop(reason),
      exited: new Promise<void>((done) => {
        markExited = done;
      }),
    };
    liveChildren.add(entry);

    const settle = (error?: SubprocessError) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearTimeout(graceTimer);
      clearTimeout(closeWaitTimer);
      signal?.removeEventListener("abort", onAbort);
      liveChildren.delete(entry);
      markExited();
      if (error) reject(error);
      else resolve({ stdout, stderr });
    };
    const stopped = () =>
      new SubprocessError(
        stopMessage(label, stopReason ?? "aborted", timeoutMs, timeoutEnvVar),
        label,
        stopReason ?? "aborted",
        stderr,
      );

    function stop(reason: SubprocessFailure) {
      // A child that already exited finished on its own; let `close` report it.
      if (stopReason || settled || child.exitCode !== null || child.signalCode !== null) return;
      stopReason = reason;
      if (graceMs <= 0) {
        signalTree(child, "SIGKILL");
        return;
      }
      signalTree(child, "SIGTERM");
      graceTimer = setTimeout(() => signalTree(child, "SIGKILL"), graceMs);
    }
    const onAbort = () => stop("aborted");
    const timer = setTimeout(() => stop("timeout"), timeoutMs);
    signal?.addEventListener("abort", onAbort, { once: true });

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      if (stderr.length > MAX_STDERR_CHARS) stderr = stderr.slice(-MAX_STDERR_CHARS);
    });
    child.on("error", (error) => {
      settle(
        new SubprocessError(
          `Could not start ${label} (${command}): ${error.message}`,
          label,
          "spawn",
          stderr,
        ),
      );
    });
    child.on("exit", () => {
      clearTimeout(graceTimer);
      if (stopReason) {
        // Make sure nothing it started outlives it, then stop waiting for
        // pipes a stray grandchild might hold open.
        signalTree(child, "SIGKILL");
        closeWaitTimer = setTimeout(() => settle(stopped()), STDIO_CLOSE_WAIT_MS);
      }
    });
    child.on("close", (code, killSignal) => {
      if (stopReason) {
        settle(stopped());
        return;
      }
      if (code === 0) {
        settle();
        return;
      }
      settle(
        new SubprocessError(
          `${label} exited with ${code !== null ? `code ${code}` : `signal ${killSignal}`}.`,
          label,
          "exit",
          stderr,
        ),
      );
    });
  });
