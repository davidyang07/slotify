import { spawn } from "node:child_process";
import { internalToken } from "../config";

export type SubprocessFailure = "spawn" | "exit" | "timeout" | "aborted" | "output";

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

export interface RunProcessOptions {
  /** Human name for errors and logs, e.g. "ffmpeg". */
  label: string;
  timeoutMs: number;
  /** The env var that sets `timeoutMs`, named in the timeout error. */
  timeoutEnvVar?: string;
  /** Kills the child when aborted (the client went away). */
  signal?: AbortSignal;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
}

// ffmpeg logs progress to stderr for the whole run; only the tail matters.
const MAX_STDERR_CHARS = 64 * 1024;

/** Environment for every Python child: inherited, plus the callback token. */
export const pythonEnv = (extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv => ({
  ...process.env,
  ...extra,
  SLOTIFY_INTERNAL_TOKEN: internalToken,
});

/**
 * Run a command to completion. The child is SIGKILLed on timeout or abort --
 * a hung ffmpeg or a Whisper pass on a three-hour upload ignores politer
 * signals for long enough to matter.
 */
export const runProcess = (
  command: string,
  args: string[],
  { label, timeoutMs, timeoutEnvVar, signal, cwd, env }: RunProcessOptions,
): Promise<{ stdout: string; stderr: string }> =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(
        new SubprocessError(`${label} was not started: the client disconnected.`, label, "aborted", ""),
      );
      return;
    }

    const child = spawn(command, args, {
      cwd,
      env: env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let settled = false;

    const kill = () => {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    };
    const settle = (error?: SubprocessError) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else resolve({ stdout, stderr });
    };
    const onAbort = () => {
      kill();
      settle(
        new SubprocessError(
          `${label} was stopped because the client disconnected.`,
          label,
          "aborted",
          stderr,
        ),
      );
    };
    const timer = setTimeout(() => {
      kill();
      settle(
        new SubprocessError(
          `${label} did not finish within ${timeoutMs / 1000} s and was stopped` +
            (timeoutEnvVar ? `; raise ${timeoutEnvVar} for longer audio.` : "."),
          label,
          "timeout",
          stderr,
        ),
      );
    }, timeoutMs);
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
    child.on("close", (code, killSignal) => {
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
