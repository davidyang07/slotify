#!/usr/bin/env node
/**
 * Start the demo: preflight, then the API and the UI together.
 *
 * One command instead of two terminals, because the two terminals are where a
 * live demo goes wrong. Preflight runs first and refuses to start anything if a
 * REQUIRED check fails -- discovering that ffmpeg is missing after the browser
 * is already open is worse than not starting.
 *
 *   npm run demo                  RANKER_MODE=auto (the committed checkpoint)
 *   npm run demo -- --heuristic   force the offline baseline
 *   npm run demo -- --no-open     do not print the browser hint
 *
 * The UI is Vite on port 5173 with --strictPort: if the port is taken it fails
 * instead of quietly moving to 5174, and the URL printed is the one Vite reports
 * serving. The browser talks only to Vite, which proxies /api to the API on PORT
 * (VITE_PROXY_TARGET, set here from PORT unless already set), so every request
 * the browser makes is same-origin and none depends on the API's CORS settings.
 *
 * Ctrl-C stops both children. If either child stops on its own the other is
 * stopped too, and the demo exits non-zero unless that child exited cleanly.
 */

import { spawn, spawnSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { apiPort, backendEnv, UI_PORT } from "./lib/demo-env.mjs";
import { selectCheckpoint } from "./lib/select-checkpoint.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/**
 * Start one workspace's dev script.
 *
 * On Windows the command goes through a shell as a single string. Node >= 18.20
 * refuses to spawn a `.cmd` without one (CVE-2024-27980), and passing an args
 * array alongside `shell: true` earns a deprecation warning -- so the command is
 * built here instead. Nothing in it comes from user input; the workspace and
 * script names are literals below.
 *
 * Running the packages' own bin files under this Node would avoid the shell
 * entirely, but tsx re-execs itself and does not survive being invoked that way,
 * so npm stays the entry point. `extra` is forwarded to the script after `--`;
 * it is literals and integers only, never user input.
 */
const npmRun = (script, extra = []) => {
  const args = ["run", script, ...(extra.length ? ["--", ...extra] : [])];
  return process.platform === "win32"
    ? { command: `npm ${args.join(" ")}`, options: { shell: true } }
    : { command: "npm", args, options: {} };
};

const heuristicOnly = process.argv.includes("--heuristic");
const skipPreflight = process.argv.includes("--skip-preflight");

if (!skipPreflight) {
  const preflight = spawnSync(process.execPath, [path.join(REPO_ROOT, "scripts", "preflight.mjs")], {
    stdio: "inherit",
    cwd: REPO_ROOT,
  });
  if (preflight.status !== 0) {
    console.error("\nPreflight failed. Fix the REQUIRED checks above, then re-run.");
    process.exit(1);
  }
}

const env = { ...process.env };

// Point the UI's /api proxy at the port the API will really listen on --
// PORT may come from backend/.env, which only the API reads -- unless the
// operator has aimed it somewhere on purpose.
const API_PORT = apiPort(backendEnv(REPO_ROOT));
env.VITE_PROXY_TARGET = env.VITE_PROXY_TARGET ?? `http://localhost:${API_PORT}`;

if (heuristicOnly) {
  env.RANKER_MODE = "heuristic";
} else {
  env.RANKER_MODE = env.RANKER_MODE ?? "auto";
  // Serve a human-trained checkpoint when one exists, and the committed
  // bootstrap otherwise -- saying which, because the two mean very different
  // things. An operator's own SLOTIFY_RANKER_CHECKPOINT always wins.
  if (!env.SLOTIFY_RANKER_CHECKPOINT) {
    const selected = selectCheckpoint(REPO_ROOT);
    if (selected.checkpoint) {
      env.SLOTIFY_RANKER_CHECKPOINT = selected.checkpoint;
      console.log(`Ranker checkpoint: ${selected.checkpoint}`);
      console.log(`  label_source=${selected.labelSource} -- ${selected.reason}`);
      if (!selected.isHumanTrained) {
        console.log(
          "  Every /api/insert-sections response will carry that caveat too.",
        );
      }
    } else {
      console.log(`Ranker checkpoint: none (${selected.reason})`);
    }
  }
}

const children = [];

/**
 * The exit status to report for a child. A child killed by a signal has a null
 * code; reporting that as 0 would make an OOM kill (SIGKILL) look like a clean
 * stop, so it is 128 + the signal number, as a shell reports it.
 */
const exitStatus = (code, signal) => {
  if (code !== null) return code;
  const number = signal ? os.constants.signals[signal] : undefined;
  return number ? 128 + number : 1;
};

const start = (name, script, cwd, { extra = [], onLine } = {}) => {
  const { command, args, options } = npmRun(script, extra);
  const child = spawn(command, args ?? [], {
    cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  const prefix = `[${name}] `;
  const forward = (stream, target) => {
    stream.setEncoding("utf8");
    let buffer = "";
    stream.on("data", (chunk) => {
      buffer += chunk;
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        target.write(prefix + line + "\n");
        onLine?.(line);
      }
    });
  };
  forward(child.stdout, process.stdout);
  forward(child.stderr, process.stderr);
  child.on("exit", (code, signal) => {
    if (shuttingDown) return;
    if (signal) {
      console.error(`${prefix}killed by ${signal}`);
    } else if (code !== 0) {
      console.error(`${prefix}exited with code ${code}`);
    }
    shutdown(exitStatus(code, signal));
  });
  children.push(child);
  return child;
};

/**
 * A process's descendants, found before any of them is signalled: once a parent
 * dies its children are re-parented to init and can no longer be traced.
 * Empty when pgrep is unavailable.
 */
const descendants = (pid) => {
  const found = spawnSync("pgrep", ["-P", String(pid)], { encoding: "utf8" });
  if (found.error || typeof found.stdout !== "string") return [];
  return found.stdout
    .split(/\s+/)
    .filter(Boolean)
    .map(Number)
    .flatMap((child) => [child, ...descendants(child)]);
};

/**
 * Stop a child and everything it started.
 *
 * Signalling only the direct child is not enough on either platform. On Windows
 * the child is cmd.exe (`shell: true`), and `child.kill()` ends only that: npm,
 * node and Vite carry on holding ports 3001 and 5173, and the next `npm run demo`
 * fails preflight. `taskkill /T` ends the whole tree. On POSIX the child is npm,
 * which runs the script under `sh -c`: SIGTERM to npm ends npm and the shell but
 * not the Vite or tsx process beneath them, so Vite outlived the demo whenever
 * the API exited first. Every process in the tree is signalled instead. (Ctrl-C
 * was never affected: the terminal delivers SIGINT to the whole foreground
 * process group, children included.)
 */
const stop = (child) => {
  if (child.exitCode !== null || child.signalCode !== null) return;
  if (process.platform === "win32") {
    spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
    });
    return;
  }
  for (const pid of [child.pid, ...descendants(child.pid)]) {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      // Already gone.
    }
  }
};

let shuttingDown = false;
const shutdown = (code) => {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) stop(child);
  process.exit(code);
};

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

console.log(
  `\nStarting Slotify (ranker mode: ${env.RANKER_MODE}${
    env.SLOTIFY_RANKER_CHECKPOINT ? `, checkpoint: ${env.SLOTIFY_RANKER_CHECKPOINT}` : ""
  })\n`,
);

// Vite prints "Local:   http://localhost:5173/" once it is serving. That line,
// not a guess at the port, is what the hint repeats.
const ansi = /\u001b\[[0-9;]*m/g;
let announced = process.argv.includes("--no-open");
const announceUrl = (line) => {
  if (announced) return;
  const served = /Local:\s+(https?:\/\/\S+)/.exec(line.replace(ansi, ""))?.[1];
  if (!served) return;
  announced = true;
  console.log("");
  console.log(`  Open ${served} -- upload audio and click Analyze.`);
  console.log(`  (/api is proxied to ${env.VITE_PROXY_TARGET})`);
  console.log("");
};

start("api", "dev", path.join(REPO_ROOT, "backend"));
start("ui", "dev", path.join(REPO_ROOT, "frontend"), {
  extra: ["--port", String(UI_PORT), "--strictPort"],
  onLine: announceUrl,
});
