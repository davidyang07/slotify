#!/usr/bin/env node
/**
 * Regenerate every statistical artifact, in dependency order, then print the
 * model-evidence status for every capability.
 *
 * One command so the numbers in the repository can be re-derived rather than
 * trusted. It runs only the cheap, offline reporting steps -- it does not
 * re-transcribe the corpus, re-embed it or retrain anything, because those cost
 * tens of minutes and their outputs are already cached and checksummed.
 *
 *   npm run evidence
 *   npm run evidence -- --split-version v2
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ML_DIR = path.join(REPO_ROOT, "ml");

const argIndex = process.argv.indexOf("--split-version");
const splitVersion = argIndex === -1 ? "v2" : process.argv[argIndex + 1] ?? "v2";

const pythonBin = (() => {
  if (process.env.SLOTIFY_ML_PYTHON) return process.env.SLOTIFY_ML_PYTHON;
  const venv =
    process.platform === "win32"
      ? path.join(ML_DIR, ".venv", "Scripts", "python.exe")
      : path.join(ML_DIR, ".venv", "bin", "python");
  return fs.existsSync(venv) ? venv : process.env.PYTHON_BIN ?? "python";
})();

const steps = [
  {
    name: "dataset statistics",
    args: ["dataset", "stats", "--split-version", splitVersion],
  },
  {
    name: "feature statistics",
    args: ["features", "stats", "--split-version", splitVersion],
  },
  {
    name: "model evidence",
    args: ["report", "model-evidence"],
  },
];

let failed = 0;
for (const step of steps) {
  console.log(`\n=== ${step.name} ===`);
  const result = spawnSync(pythonBin, ["-m", "slotify_rank.cli", ...step.args], {
    cwd: ML_DIR,
    stdio: "inherit",
    env: {
      ...process.env,
      PYTHONPATH: [path.join(ML_DIR, "src"), process.env.PYTHONPATH]
        .filter(Boolean)
        .join(path.delimiter),
    },
  });
  if (result.status !== 0) {
    failed += 1;
    console.error(`  ${step.name} failed (exit ${result.status}).`);
  }
}

console.log(
  failed
    ? `\n${failed} step(s) failed. The artifacts may be stale.`
    : "\nArtifacts regenerated. See artifacts/reports/model_evidence.md.",
);
process.exit(failed ? 1 : 0);
