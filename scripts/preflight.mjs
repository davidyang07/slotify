#!/usr/bin/env node
/**
 * Slotify demo preflight.
 *
 * Answers one question before a demo, not during it: will the demo work, and
 * if something is missing, does it matter?
 *
 * Every check reports REQUIRED or OPTIONAL. A missing OPTIONAL is a WARN and
 * exits 0 -- the placement demo needs no API keys, and treating an absent
 * ElevenLabs key as a failure would teach the reader to ignore failures. A
 * missing REQUIRED is a FAIL and exits 1.
 *
 * Node rather than a shell script so one file works on PowerShell, bash and
 * zsh; the repository already requires Node for both workspaces.
 *
 *   node scripts/preflight.mjs [--json]
 */

import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  BOOTSTRAP_CHECKPOINT,
  selectCheckpoint,
} from "./lib/select-checkpoint.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const asJson = process.argv.includes("--json");

/** @type {{name: string, level: "REQUIRED"|"OPTIONAL", status: "OK"|"WARN"|"FAIL", detail: string}[]} */
const results = [];

const record = (name, level, status, detail) => {
  results.push({ name, level, status, detail });
};

const run = (command, args, options = {}) => {
  try {
    return execFileSync(command, args, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 30_000,
      ...options,
    }).trim();
  } catch {
    return null;
  }
};

const exists = (relative) => fs.existsSync(path.join(REPO_ROOT, relative));

const portFree = (port) =>
  new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => server.close(() => resolve(true)));
    server.listen(port, "127.0.0.1");
  });

// -- runtimes ---------------------------------------------------------------

const nodeMajor = Number.parseInt(process.versions.node.split(".")[0], 10);
record(
  "Node.js",
  "REQUIRED",
  nodeMajor >= 20 ? "OK" : "FAIL",
  `v${process.versions.node}${nodeMajor >= 20 ? "" : " (need >= 20 for the built-in test runner)"}`,
);

const mlPython = (() => {
  if (process.env.SLOTIFY_ML_PYTHON) return process.env.SLOTIFY_ML_PYTHON;
  const venv =
    process.platform === "win32"
      ? path.join(REPO_ROOT, "ml", ".venv", "Scripts", "python.exe")
      : path.join(REPO_ROOT, "ml", ".venv", "bin", "python");
  return fs.existsSync(venv) ? venv : process.env.PYTHON_BIN ?? "python";
})();

const pythonVersion = run(mlPython, ["--version"]);
record(
  "Python (ml)",
  "REQUIRED",
  pythonVersion ? "OK" : "FAIL",
  pythonVersion ? `${pythonVersion} at ${path.relative(REPO_ROOT, mlPython) || mlPython}` : `not runnable: ${mlPython}`,
);

const adInserterPython = process.env.PYTHON_BIN ?? "python";
const adInserterVersion = run(adInserterPython, ["--version"]);
record(
  "Python (ad_inserter)",
  "REQUIRED",
  adInserterVersion ? "OK" : "FAIL",
  adInserterVersion ?? `not runnable: ${adInserterPython} (set PYTHON_BIN)`,
);

for (const tool of ["ffmpeg", "ffprobe"]) {
  const version = run(tool, ["-version"]);
  record(
    tool,
    "REQUIRED",
    version ? "OK" : "FAIL",
    version ? version.split("\n")[0] : "not on PATH",
  );
}

// -- dependencies -----------------------------------------------------------

record(
  "frontend dependencies",
  "REQUIRED",
  exists("frontend/node_modules") ? "OK" : "FAIL",
  exists("frontend/node_modules") ? "installed" : "run: cd frontend && npm ci",
);
record(
  "backend dependencies",
  "REQUIRED",
  exists("backend/node_modules") ? "OK" : "FAIL",
  exists("backend/node_modules") ? "installed" : "run: cd backend && npm ci",
);

// find_spec rather than import: importing torch, transformers,
// sentence-transformers and librosa costs ~40 s cold, which is far too long for
// a check whose whole purpose is to answer quickly before a demo.
const pipeline = spawnSync(
  mlPython,
  [
    "-c",
    "import importlib.util as u, json;" +
      "print(json.dumps([n for n in ('torch','transformers','sentence_transformers','librosa','slotify_rank')" +
      " if u.find_spec(n) is None]))",
  ],
  { encoding: "utf8", timeout: 60_000 },
);
let missingModules = ["torch"];
if (pipeline.status === 0) {
  try {
    missingModules = JSON.parse(pipeline.stdout.trim());
  } catch {
    missingModules = ["torch"];
  }
}
const mlExtras = pipeline.status === 0 && missingModules.length === 0;
record(
  "ml learned-feature extras",
  "OPTIONAL",
  mlExtras ? "OK" : "WARN",
  mlExtras
    ? "torch, transformers, sentence-transformers, librosa, slotify_rank importable"
    : `missing: ${missingModules.join(", ") || "unknown"} -- the learned ranker ` +
      "cannot run; the heuristic baseline still can. Install with: " +
      "cd ml && pip install -e .[dev,label,features]",
);

// -- the learned ranker -----------------------------------------------------

const selectedCheckpoint = selectCheckpoint(REPO_ROOT);
const checkpoint =
  process.env.SLOTIFY_RANKER_CHECKPOINT ??
  selectedCheckpoint.checkpoint ??
  BOOTSTRAP_CHECKPOINT;
const checkpointPath = path.resolve(REPO_ROOT, checkpoint);
const normalizerPath = path.join(path.dirname(checkpointPath), "normalizer.json");
const checkpointReady =
  fs.existsSync(checkpointPath) && fs.existsSync(normalizerPath);
record(
  "learned ranker checkpoint",
  "OPTIONAL",
  checkpointReady ? "OK" : "WARN",
  checkpointReady
    ? // The label source is the whole difference between a checkpoint that
      // demonstrates the plumbing and one an evaluation result describes, so it
      // is on the same line as the path rather than a paragraph away.
      `${path.relative(REPO_ROOT, checkpointPath)} (label_source=${
        selectedCheckpoint.labelSource ?? "unknown"
      })`
    : `${checkpoint} not found -- RANKER_MODE=auto will use the heuristic baseline`,
);

if (checkpointReady && mlExtras) {
  const describe = spawnSync(
    mlPython,
    [
      "-m",
      "slotify_rank.cli",
      "infer",
      "describe",
      "--checkpoint",
      checkpointPath,
    ],
    {
      cwd: path.join(REPO_ROOT, "ml"),
      encoding: "utf8",
      timeout: 180_000,
      env: {
        ...process.env,
        PYTHONPATH: [path.join(REPO_ROOT, "ml", "src"), process.env.PYTHONPATH]
          .filter(Boolean)
          .join(path.delimiter),
      },
    },
  );
  if (describe.status === 0) {
    let identity = {};
    try {
      identity = JSON.parse(describe.stdout);
    } catch {
      identity = {};
    }
    const source = identity.training_label_source ?? "unknown";
    record(
      "learned ranker loads",
      "OPTIONAL",
      "OK",
      `${identity.model_variant} (${identity.parameter_count} params), trained on label_source=${source}` +
        (source === "human"
          ? ""
          : " -- a bootstrap distillation, not evidence of ranking quality"),
    );
  } else {
    record(
      "learned ranker loads",
      "OPTIONAL",
      "WARN",
      (describe.stderr || "").trim().split("\n").slice(-1)[0] ||
        "the checkpoint could not be loaded",
    );
  }
}

// -- credentials ------------------------------------------------------------

record(
  "ELEVENLABS_API_KEY",
  "OPTIONAL",
  process.env.ELEVENLABS_API_KEY ? "OK" : "WARN",
  process.env.ELEVENLABS_API_KEY
    ? "set -- voice cloning and TTS enabled"
    : "not set -- ranking works, ad generation disabled",
);
record(
  "OPENAI_API_KEY",
  "OPTIONAL",
  process.env.OPENAI_API_KEY ? "OK" : "WARN",
  process.env.OPENAI_API_KEY
    ? "set -- sponsor copy and slot narration enabled"
    : "not set -- optional enrichment disabled",
);

// -- ports ------------------------------------------------------------------

const ports = [
  { port: Number.parseInt(process.env.PORT ?? "3001", 10), label: "backend" },
  { port: 5173, label: "frontend" },
];
for (const { port, label } of ports) {
  // eslint-disable-next-line no-await-in-loop
  const free = await portFree(port);
  record(
    `port ${port} (${label})`,
    "REQUIRED",
    free ? "OK" : "WARN",
    free ? "available" : "in use -- stop the other process or change the port",
  );
}

// -- report -----------------------------------------------------------------

const failed = results.filter((entry) => entry.status === "FAIL");

if (asJson) {
  console.log(JSON.stringify({ ok: failed.length === 0, checks: results }, null, 2));
} else {
  console.log("Slotify demo preflight\n");
  const width = Math.max(...results.map((entry) => entry.name.length));
  for (const entry of results) {
    const tag = entry.status === "OK" ? "[OK]  " : entry.status === "WARN" ? "[WARN]" : "[FAIL]";
    console.log(`${tag} ${entry.name.padEnd(width)}  ${entry.detail}`);
  }
  console.log("");
  if (failed.length) {
    console.log(`${failed.length} required check(s) failed. The demo will not run.`);
  } else {
    const warned = results.filter((entry) => entry.status === "WARN").length;
    console.log(
      warned
        ? `Ready. ${warned} optional check(s) unavailable (listed above); the placement demo does not need them.`
        : "Ready. Everything available.",
    );
  }
}

process.exit(failed.length ? 1 : 0);
