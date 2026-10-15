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
 * missing REQUIRED is a FAIL and exits 1. The level alone decides which, so a
 * REQUIRED check cannot be recorded as a warning.
 *
 * Settings the API reads (PYTHON_BIN, PORT, the API keys) are resolved the way
 * the API resolves them, backend/.env included, so preflight checks the Python
 * and the port the API will actually use.
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
import { apiPort, backendEnv, UI_PORT } from "./lib/demo-env.mjs";
import {
  BOOTSTRAP_CHECKPOINT,
  selectCheckpoint,
} from "./lib/select-checkpoint.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const asJson = process.argv.includes("--json");
const apiEnv = backendEnv(REPO_ROOT);

/** @type {{name: string, level: "REQUIRED"|"OPTIONAL", status: "OK"|"WARN"|"FAIL", detail: string}[]} */
const results = [];

/** @param {"REQUIRED"|"OPTIONAL"} level @param {boolean} ok */
const record = (name, level, ok, detail) => {
  const status = ok ? "OK" : level === "REQUIRED" ? "FAIL" : "WARN";
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

/** Resolves to null when `host:port` can be bound, else the error code. */
const bindError = (port, host) =>
  new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", (error) => resolve(error.code ?? "error"));
    server.once("listening", () => server.close(() => resolve(null)));
    server.listen(port, host);
  });

/**
 * Null when the port is free on both loopbacks, else why not. Both, because
 * Vite binds `localhost`, which Node may resolve to ::1, and a server on ::1
 * alone leaves 127.0.0.1 bindable. A machine without IPv6 loopback cannot bind
 * ::1 at all, which is not a collision.
 */
const portBusy = async (port) => {
  const v4 = await bindError(port, "127.0.0.1");
  if (v4) return v4;
  const v6 = await bindError(port, "::1");
  return v6 && !["EADDRNOTAVAIL", "EAFNOSUPPORT", "EINVAL"].includes(v6) ? v6 : null;
};

/**
 * Best effort: the process listening on `port`, so the failure can say what to
 * stop. Null when the platform tool is missing or will not say (a process owned
 * by another user, for one) -- the check still fails, it just names only the port.
 */
const portHolder = (port) => {
  if (process.platform === "win32") {
    const pid = (run("netstat", ["-ano"]) ?? "")
      .split(/\r?\n/)
      .map((line) => line.trim().split(/\s+/))
      .find((cols) => cols[0] === "TCP" && cols[1]?.endsWith(`:${port}`) && cols[3] === "LISTENING")?.[4];
    if (!pid) return null;
    const task = run("tasklist", ["/FI", `PID eq ${pid}`, "/FO", "CSV", "/NH"]);
    return { pid, name: task?.match(/^"([^"]+)"/)?.[1] ?? null };
  }
  // lsof -F prints one field per line: p<pid>, then c<command>.
  const lsof = run("lsof", ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN", "-Fpc"]);
  const pid = lsof?.match(/^p(\d+)$/m)?.[1];
  if (pid) return { pid, name: lsof.match(/^c(.+)$/m)?.[1] ?? null };
  const ss = process.platform === "linux" ? run("ss", ["-ltnpH", `sport = :${port}`]) : null;
  const users = ss?.match(/users:\(\("([^"]+)",pid=(\d+)/);
  return users ? { pid: users[2], name: users[1] } : null;
};

// -- runtimes ---------------------------------------------------------------

// The same floor as `engines` in package.json. Every workspace's test script
// hands `node --test` a quoted glob, which Node expands itself only from 21;
// 22 is the LTS line CI runs.
const nodeMajor = Number.parseInt(process.versions.node.split(".")[0], 10);
record(
  "Node.js",
  "REQUIRED",
  nodeMajor >= 22,
  `v${process.versions.node}${nodeMajor >= 22 ? "" : " (need >= 22: `node --test` expands the test globs only from Node 21, and CI runs 22)"}`,
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
  Boolean(pythonVersion),
  pythonVersion ? `${pythonVersion} at ${path.relative(REPO_ROOT, mlPython) || mlPython}` : `not runnable: ${mlPython}`,
);

// The API runs every analysis as `PYTHON_BIN -m ad_inserter.analyze_cli`, so an
// interpreter that starts but lacks backend/requirements.txt passes a version
// check and then fails every request with ModuleNotFoundError. The modules are
// imported, not just located: soundfile, for one, is installed but unusable
// without its native library, and only an import says so. librosa is the slow
// one, a few seconds cold.
const AD_INSERTER_MODULES = ["pydub", "numpy", "pyloudnorm", "librosa", "soundfile"];
const adInserterPython = apiEnv.PYTHON_BIN ?? "python";
const adInserterProbe = spawnSync(
  adInserterPython,
  [
    "-c",
    [
      "import importlib, json, platform, sys",
      "broken = []",
      "for name in sys.argv[1:]:",
      "    try:",
      "        importlib.import_module(name)",
      "    except Exception as error:",
      "        broken.append(f'{name} ({type(error).__name__}: {error})')",
      "print(json.dumps({'version': platform.python_version(), 'broken': broken}))",
    ].join("\n"),
    ...AD_INSERTER_MODULES,
  ],
  { cwd: path.join(REPO_ROOT, "backend"), encoding: "utf8", timeout: 120_000 },
);
let adInserter = null;
try {
  adInserter = adInserterProbe.status === 0 ? JSON.parse(adInserterProbe.stdout.trim()) : null;
} catch {
  adInserter = null;
}
const installHint = `install with: ${adInserterPython} -m pip install -r backend/requirements.txt`;
record(
  "Python (ad_inserter)",
  "REQUIRED",
  adInserter !== null && adInserter.broken.length === 0,
  adInserter === null
    ? adInserterProbe.error
      ? `not runnable: ${adInserterPython} (set PYTHON_BIN)`
      : `${adInserterPython} could not run the import check: ${
          (adInserterProbe.stderr || "").trim().split("\n").slice(-1)[0] || "no output"
        }`
    : adInserter.broken.length === 0
      ? `Python ${adInserter.version} at ${adInserterPython}; ${AD_INSERTER_MODULES.join(", ")} import`
      : `Python ${adInserter.version} at ${adInserterPython} cannot import ` +
        `${adInserter.broken.join("; ")} -- every analysis would fail. ${installHint}`,
);

for (const tool of ["ffmpeg", "ffprobe"]) {
  const version = run(tool, ["-version"]);
  record(
    tool,
    "REQUIRED",
    Boolean(version),
    version ? version.split("\n")[0] : "not on PATH",
  );
}

// -- dependencies -----------------------------------------------------------

record(
  "frontend dependencies",
  "REQUIRED",
  exists("frontend/node_modules"),
  exists("frontend/node_modules") ? "installed" : "run: cd frontend && npm ci",
);
record(
  "backend dependencies",
  "REQUIRED",
  exists("backend/node_modules"),
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
  mlExtras,
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
  checkpointReady,
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
      true,
      `${identity.model_variant} (${identity.parameter_count} params), trained on label_source=${source}` +
        (source === "human"
          ? ""
          : " -- a bootstrap distillation, not evidence of ranking quality"),
    );
  } else {
    record(
      "learned ranker loads",
      "OPTIONAL",
      false,
      (describe.stderr || "").trim().split("\n").slice(-1)[0] ||
        "the checkpoint could not be loaded",
    );
  }

  // The ranker worker runs offline against pinned Whisper and MiniLM commits.
  // Without the snapshots it refuses to start and auto mode falls back to the
  // heuristic, so say so here rather than on the first upload.
  const warm = spawnSync(
    mlPython,
    ["-m", "slotify_rank.cli", "infer", "warm", "--check"],
    {
      cwd: path.join(REPO_ROOT, "ml"),
      encoding: "utf8",
      timeout: 60_000,
      env: {
        ...process.env,
        PYTHONPATH: [path.join(REPO_ROOT, "ml", "src"), process.env.PYTHONPATH]
          .filter(Boolean)
          .join(path.delimiter),
      },
    },
  );
  record(
    "pinned model snapshots",
    "OPTIONAL",
    warm.status === 0,
    warm.status === 0
      ? "Whisper and MiniLM are local -- the learned ranker runs offline"
      : "missing -- run `cd ml && python -m slotify_rank.cli infer warm` once (RANKER_MODE=auto falls back to the heuristic until then)",
  );
}

// -- credentials ------------------------------------------------------------

record(
  "ELEVENLABS_API_KEY",
  "OPTIONAL",
  Boolean(apiEnv.ELEVENLABS_API_KEY),
  apiEnv.ELEVENLABS_API_KEY
    ? "set -- voice cloning and TTS enabled"
    : "not set -- ranking works, ad generation disabled",
);
record(
  "OPENAI_API_KEY",
  "OPTIONAL",
  Boolean(apiEnv.OPENAI_API_KEY),
  apiEnv.OPENAI_API_KEY
    ? "set -- sponsor copy and slot narration enabled"
    : "not set -- optional enrichment disabled",
);

// -- ports ------------------------------------------------------------------

// REQUIRED: neither server can start on a taken port. The UI port has no
// fallback either -- the demo starts Vite with --strictPort, so it fails rather
// than drifting to 5174 behind a URL that points at whatever holds 5173.
const ports = [
  {
    port: apiPort(apiEnv),
    label: "backend",
    otherwise: ", or set PORT to a free one (the demo points the UI's /api proxy at it)",
  },
  {
    port: UI_PORT,
    label: "frontend",
    otherwise: "; the demo starts Vite with --strictPort, so it will not move to another port",
  },
];
const findHolder = (port) =>
  process.platform === "win32"
    ? `\`netstat -ano | findstr :${port}\``
    : `\`lsof -nP -iTCP:${port} -sTCP:LISTEN\``;
const stopCommand = (pid) =>
  process.platform === "win32" ? `taskkill /pid ${pid} /T /F` : `kill ${pid}`;
for (const { port, label, otherwise } of ports) {
  // eslint-disable-next-line no-await-in-loop
  const busy = await portBusy(port);
  const holder = busy === "EADDRINUSE" ? portHolder(port) : null;
  let detail = "available";
  if (busy && busy !== "EADDRINUSE") {
    detail = `cannot be bound (${busy})${otherwise}`;
  } else if (holder) {
    detail =
      `in use by ${holder.name ?? "a process"} (pid ${holder.pid}) -- ` +
      `stop it with \`${stopCommand(holder.pid)}\`${otherwise}`;
  } else if (busy) {
    detail = `in use -- find the process with ${findHolder(port)} and stop it${otherwise}`;
  }
  record(`port ${port} (${label})`, "REQUIRED", !busy, detail);
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
