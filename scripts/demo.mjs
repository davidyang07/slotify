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
 * Ctrl-C stops both children.
 */

import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

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
 * so npm stays the entry point.
 */
const npmRun = (script) =>
  process.platform === "win32"
    ? { command: `npm run ${script}`, options: { shell: true } }
    : { command: "npm", args: ["run", script], options: {} };

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

const defaultCheckpoint = path.join(
  REPO_ROOT,
  "artifacts",
  "training",
  "gated-d8ed976101aa4c3b",
  "best_checkpoint.pt",
);

const env = { ...process.env };
if (heuristicOnly) {
  env.RANKER_MODE = "heuristic";
} else {
  env.RANKER_MODE = env.RANKER_MODE ?? "auto";
  // Point at the committed bootstrap checkpoint unless the operator chose one.
  // auto still falls back to the heuristic if this file is absent, and says so.
  if (!env.SLOTIFY_RANKER_CHECKPOINT && fs.existsSync(defaultCheckpoint)) {
    env.SLOTIFY_RANKER_CHECKPOINT = path.relative(REPO_ROOT, defaultCheckpoint);
  }
}

const children = [];

const start = (name, script, cwd) => {
  const { command, args, options } = npmRun(script);
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
      for (const line of lines) target.write(prefix + line + "\n");
    });
  };
  forward(child.stdout, process.stdout);
  forward(child.stderr, process.stderr);
  child.on("exit", (code) => {
    if (code !== 0 && code !== null) {
      console.error(`${prefix}exited with code ${code}`);
    }
    shutdown(code ?? 0);
  });
  children.push(child);
  return child;
};

let shuttingDown = false;
const shutdown = (code) => {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) child.kill();
  }
  process.exit(code);
};

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

console.log(
  `\nStarting Slotify (ranker mode: ${env.RANKER_MODE}${
    env.SLOTIFY_RANKER_CHECKPOINT ? `, checkpoint: ${env.SLOTIFY_RANKER_CHECKPOINT}` : ""
  })\n`,
);

start("api", "dev", path.join(REPO_ROOT, "backend"));
start("ui", "dev", path.join(REPO_ROOT, "frontend"));

if (!process.argv.includes("--no-open")) {
  setTimeout(() => {
    console.log("");
    console.log("  Open http://localhost:5173 -- upload audio and click Analyze.");
    console.log("");
  }, 4000);
}
